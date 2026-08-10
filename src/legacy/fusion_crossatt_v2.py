"""
Architecture 2 — Cross-Attention Fusion (Improved)
Three key fixes over previous version:
  1. Uses 2048-dim spatial features instead of 18-dim (richer signal)
  2. Early stopping (patience=5) — stops when val AUC plateaus
  3. Stronger regularization (weight_decay=1e-3, dropout=0.5)

Flow:
  Face X-Ray → final_layer spatial map [N x 2048]
  GenD       → CLS token [1024]
  Cross-Attention(Q=GenD[512], K=V=XRay patches[512])
  → Attended [512] + residual → classifier
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import random

# ── CONFIG ────────────────────────────────────────────────────────────────────
FACE_XRAY_CKPT  = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\Face-X-Ray-master\Face-X-Ray-master\result\result_default\best_model.pth.tar"
FACE_XRAY_ROOT  = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\Face-X-Ray-master\Face-X-Ray-master"
HRNET_CONFIG    = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\Face-X-Ray-master\Face-X-Ray-master\HRNet\hrnet_config\experiments\cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
GEND_MODEL      = "yermandy/GenD_DINOv3_L"
HF_TOKEN        = None

FF_REAL_IMAGES  = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\extract\original_sequences\youtube\c23\images"
FF_FAKE_BASE    = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\extract\manipulated_sequences"
SPLITS_DIR      = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\splits"
FAKE_DATASETS   = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

# Training hyperparameters
EPOCHS          = 50     # more epochs since early stopping handles overfitting
LR              = 3e-4
BATCH_SIZE      = 16
MAX_TRAIN       = 20000
MAX_VAL         = 3000
PROJ_DIM        = 512
NUM_HEADS       = 8
PATIENCE        = 5      # early stopping patience
WEIGHT_DECAY    = 1e-3   # stronger regularization
DROPOUT         = 0.5    # stronger dropout

SAVE_DIR        = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\GenD\GenD\fusion_weights"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, FACE_XRAY_ROOT)

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Using device: {DEVICE}')


# ══════════════════════════════════════════════════════════════════════════════
# Improved Cross-Attention Fusion Model
# Uses 2048-dim spatial patches instead of 18-dim
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionFusionV2(nn.Module):
    """
    Improved Architecture 2.
    Face X-Ray final_layer: [B, 2048, H, W] → patches [B, N, 2048]
    GenD CLS token: [B, 1024]

    Cross-attention:
        Q = GenD token projected to proj_dim
        K = V = Face X-Ray patches projected to proj_dim
    """
    def __init__(self, xray_dim=2048, gend_dim=1024,
                 proj_dim=512, num_heads=8, dropout=0.5):
        super().__init__()

        self.proj_dim = proj_dim

        # Project Face X-Ray 2048-dim patches to proj_dim
        self.xray_patch_proj = nn.Sequential(
            nn.Linear(xray_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )

        # Project GenD 1024-dim token to proj_dim
        self.gend_proj = nn.Sequential(
            nn.Linear(gend_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )

        # Cross-attention: Q from GenD, K/V from Face X-Ray
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout * 0.5,  # lighter dropout inside attention
            batch_first=True
        )

        # Post-attention norm and feed-forward
        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(proj_dim * 2, proj_dim)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2)
        )

    def forward(self, xray_spatial, gend_feat):
        """
        xray_spatial: [B, N, 2048] — N spatial patches
        gend_feat:    [B, 1024]    — GenD CLS token
        """
        # Project Face X-Ray patches: [B, N, 2048] → [B, N, proj_dim]
        patches = self.xray_patch_proj(xray_spatial)

        # Project GenD token: [B, 1024] → [B, 1, proj_dim]
        query = self.gend_proj(gend_feat).unsqueeze(1)

        # Cross-attention
        attended, attn_weights = self.cross_attention(
            query=query,
            key=patches,
            value=patches
        )

        # Post-attention: residual + norm
        attended = attended.squeeze(1)
        query_sq = query.squeeze(1)
        fused = self.norm1(attended + query_sq)

        # Feed-forward with residual
        fused = self.norm2(fused + self.feed_forward(fused))

        return self.classifier(fused), attn_weights


# ══════════════════════════════════════════════════════════════════════════════
# Load backbone models (frozen)
# ══════════════════════════════════════════════════════════════════════════════

def load_face_xray():
    print('\nLoading Face X-Ray...')
    from HRNet import get_net
    net = get_net(cfg_file=HRNET_CONFIG, devices=[torch.device('cuda:0')])
    net.classifier = nn.Linear(2048, 2)
    ckpt = torch.load(FACE_XRAY_CKPT, map_location='cpu', weights_only=False)
    new_sd = {}
    for k, v in ckpt['state_dict'].items():
        if k.startswith('HRNet_layer.'):
            k = k[len('HRNet_layer.'):]
        new_sd[k] = v
    new_sd = {k: v for k, v in new_sd.items() if not k.startswith('classifier')}
    net.load_state_dict(new_sd, strict=False)
    net = net.to(DEVICE)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    print(f'  Loaded. Checkpoint AUC: {ckpt["best_auc"]:.3f}')
    return net


def load_gend():
    print('\nLoading GenD (DINOv3)...')
    from src.hf.modeling_gend import GenD
    token = HF_TOKEN or os.environ.get('HUGGINGFACE_TOKEN', None)
    torch.set_default_device('cpu')
    model = GenD.from_pretrained(GEND_MODEL, token=token)
    torch.set_default_device(None)
    model = model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print('  GenD loaded.')
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction — KEY CHANGE: use final_layer (2048-dim) not first stream
# ══════════════════════════════════════════════════════════════════════════════

def extract_xray_spatial(net, image_path):
    """
    Extract 2048-dim spatial patches from Face X-Ray final_layer.
    Returns [N, 2048] where N = H*W spatial locations.
    Much richer than the 18-dim first stream used before.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        hrnet = net.HRNet_layer

        # Run through all 4 parallel streams
        y_list = hrnet(tensor)

        # Merge streams using incre_modules and downsamp_modules
        y = hrnet.incre_modules[0](y_list[0])
        for i in range(len(hrnet.downsamp_modules)):
            y = hrnet.incre_modules[i+1](y_list[i+1]) + hrnet.downsamp_modules[i](y)

        # Apply final_layer → [1, 2048, H, W]
        y = hrnet.final_layer(y)

        # Reshape to spatial patches [N, 2048]
        B, C, H, W = y.shape
        patches = y.view(B, C, H * W).permute(0, 2, 1)  # [1, N, 2048]
        patches = patches.squeeze(0)  # [N, 2048]

    return patches.cpu()


def extract_gend_features(model, image_path):
    """Extract 1024-dim L2-normalized CLS token from GenD."""
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = model.feature_extractor.preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model.feature_extractor(tensor)
            return F.normalize(feat, dim=-1).squeeze(0).cpu()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Data collection and pre-extraction
# ══════════════════════════════════════════════════════════════════════════════

def collect_image_paths(split_name, max_per_class):
    with open(os.path.join(SPLITS_DIR, f'{split_name}.json')) as f:
        pairs = json.load(f)
    split_ids = set()
    for pair in pairs:
        split_ids.add(pair[0])
        split_ids.add(pair[1])

    real_paths, fake_paths = [], []

    if os.path.exists(FF_REAL_IMAGES):
        for vid_dir in sorted(os.listdir(FF_REAL_IMAGES)):
            if vid_dir not in split_ids:
                continue
            vid_path = os.path.join(FF_REAL_IMAGES, vid_dir)
            if os.path.isdir(vid_path):
                for frame in sorted(os.listdir(vid_path)):
                    if frame.endswith('.png'):
                        real_paths.append(os.path.join(vid_path, frame))

    for ds in FAKE_DATASETS:
        fake_base = os.path.join(FF_FAKE_BASE, ds, 'c23', 'images')
        if not os.path.exists(fake_base):
            continue
        for vid_dir in sorted(os.listdir(fake_base)):
            if vid_dir[:3] not in split_ids:
                continue
            vid_path = os.path.join(fake_base, vid_dir)
            if os.path.isdir(vid_path):
                for frame in sorted(os.listdir(vid_path)):
                    if frame.endswith('.png'):
                        fake_paths.append(os.path.join(vid_path, frame))

    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    real_paths = real_paths[:max_per_class]
    fake_paths = fake_paths[:max_per_class]

    print(f'  {split_name}: {len(real_paths)} real, {len(fake_paths)} fake')
    return real_paths, fake_paths


def preextract_features(xray_net, gend_model, real_paths, fake_paths, desc):
    print(f'Pre-extracting features for {desc}...')
    data = []
    all_paths = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    random.shuffle(all_paths)

    for path, label in tqdm(all_paths, desc=f'  {desc}'):
        xf = extract_xray_spatial(xray_net, path)
        gf = extract_gend_features(gend_model, path)
        if xf is not None and gf is not None:
            data.append((xf, gf, label))

    print(f'  Extracted {len(data)} samples')
    if data:
        print(f'  Spatial patch shape : {data[0][0].shape}')
        print(f'  GenD feature shape  : {data[0][1].shape}')
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Training and evaluation
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(fusion, optimizer, data, batch_size):
    fusion.train()
    random.shuffle(data)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        xf     = torch.stack([d[0] for d in batch]).to(DEVICE)
        gf     = torch.stack([d[1] for d in batch]).to(DEVICE)
        labels = torch.tensor([d[2] for d in batch], dtype=torch.long).to(DEVICE)

        optimizer.zero_grad()
        logits, _ = fusion(xf, gf)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(batch)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(batch)

    return total_loss / total, 100 * correct / total


def eval_epoch(fusion, data, batch_size):
    fusion.eval()
    all_labels, all_scores = [], []
    correct = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            xf     = torch.stack([d[0] for d in batch]).to(DEVICE)
            gf     = torch.stack([d[1] for d in batch]).to(DEVICE)
            labels = [d[2] for d in batch]

            logits, _ = fusion(xf, gf)
            probs = torch.softmax(logits, dim=-1)
            all_scores.extend(probs[:, 1].cpu().tolist())
            all_labels.extend(labels)
            correct += sum(p == l for p, l in zip(logits.argmax(dim=1).cpu().tolist(), labels))

    auc = roc_auc_score(all_labels, all_scores) * 100
    acc = 100 * sum(1 for s, l in zip(all_scores, all_labels) if (s > 0.5) == bool(l)) / len(all_labels)
    return auc, acc


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=EPOCHS)
    parser.add_argument('--lr',          type=float, default=LR)
    parser.add_argument('--batch_size',  type=int,   default=BATCH_SIZE)
    parser.add_argument('--proj_dim',    type=int,   default=PROJ_DIM)
    parser.add_argument('--num_heads',   type=int,   default=NUM_HEADS)
    parser.add_argument('--patience',    type=int,   default=PATIENCE)
    parser.add_argument('--max_train',   type=int,   default=MAX_TRAIN)
    parser.add_argument('--max_val',     type=int,   default=MAX_VAL)
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print(f'ARCHITECTURE 2 IMPROVED — CROSS-ATTENTION FUSION V2')
    print(f'Key changes:')
    print(f'  1. 2048-dim spatial patches (was 18-dim)')
    print(f'  2. Early stopping (patience={args.patience})')
    print(f'  3. Stronger regularization (weight_decay={WEIGHT_DECAY})')
    print(f'{"="*60}')

    # Load backbones
    xray_net   = load_face_xray()
    gend_model = load_gend()

    # Collect paths
    print('\nCollecting image paths...')
    train_real, train_fake = collect_image_paths('train', args.max_train)
    val_real,   val_fake   = collect_image_paths('val',   args.max_val)

    # Pre-extract features
    train_data = preextract_features(xray_net, gend_model, train_real, train_fake, 'train')
    val_data   = preextract_features(xray_net, gend_model, val_real,   val_fake,   'val')

    # Get actual spatial dim from data
    actual_xray_dim = train_data[0][0].shape[1]
    print(f'\nSpatial dim per patch : {actual_xray_dim}')
    print(f'Number of patches     : {train_data[0][0].shape[0]}')

    # Create model
    fusion = CrossAttentionFusionV2(
        xray_dim=actual_xray_dim,
        gend_dim=1024,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        dropout=DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in fusion.parameters())
    print(f'Trainable parameters  : {total_params:,}')

    optimizer = torch.optim.Adam(
        fusion.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY  # stronger regularization
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_auc = 0
    no_improve   = 0
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, 'fusion_crossattn_v2_best.pth')

    print(f'\nStarting training (max {args.epochs} epochs, early stop patience={args.patience})...\n')
    print(f'{"Epoch":<8} {"Train Loss":<12} {"Train Acc":<12} {"Val AUC":<12} {"Val Acc":<10}')
    print('-' * 56)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(fusion, optimizer, train_data, args.batch_size)
        val_auc,   val_acc    = eval_epoch(fusion, val_data,   args.batch_size)
        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve   = 0
            marker = ' ← best'
            torch.save({
                'epoch':            epoch,
                'fusion_state_dict': fusion.state_dict(),
                'val_auc':          val_auc,
                'val_acc':          val_acc,
                'proj_dim':         args.proj_dim,
                'num_heads':        args.num_heads,
                'xray_dim':         actual_xray_dim,
            }, save_path)
        else:
            no_improve += 1
            marker = f' (no improve {no_improve}/{args.patience})'

        print(f'{epoch:<8} {train_loss:<12.4f} {train_acc:<12.2f} {val_auc:<12.3f} {val_acc:<10.2f}{marker}')

        if no_improve >= args.patience:
            print(f'\nEarly stopping triggered at epoch {epoch}.')
            break

    print(f'\n{"="*60}')
    print(f'Training complete!')
    print(f'Best validation AUC   : {best_val_auc:.3f}%')
    print(f'Saved to              : {save_path}')
    print(f'\nFull comparison:')
    print(f'  GenD alone          : 92.5%  (no fusion)')
    print(f'  Arch 1 Compressed   : 84.4%')
    print(f'  Arch 1 Full         : 84.6%')
    print(f'  Arch 2 V1 (18-dim)  : 86.2%')
    print(f'  Arch 2 V2 (2048-dim): {best_val_auc:.3f}%  ← this run')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()