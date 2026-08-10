"""
Architecture 2 — Cross-Attention Fusion
Face X-Ray + GenD combined using cross-attention.

GenD's CLS token (Query) attends to Face X-Ray's spatial patches (Keys/Values).
This lets semantic forgery knowledge guide where to look for spatial evidence.

Flow:
  Face X-Ray → spatial feature map [N x 512] (projected patches)
  GenD       → CLS token [512] (projected)
  Cross-Attention(Q=GenD, K=V=FaceXRay patches)
  → Attended features [512] + residual → classifier
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import random
import math

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

# Hyperparameters
EPOCHS          = 20
LR              = 3e-4
BATCH_SIZE      = 16   # smaller batch for cross-attention (more memory)
MAX_TRAIN       = 20000
MAX_VAL         = 3000
PROJ_DIM        = 512  # projection dimension for both models
NUM_HEADS       = 8    # number of attention heads

SAVE_DIR        = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\GenD\GenD\fusion_weights"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, FACE_XRAY_ROOT)

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Using device: {DEVICE}')


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 2 — Cross-Attention Fusion Model
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    GenD CLS token queries Face X-Ray spatial patches via cross-attention.

    Face X-Ray spatial map: [B, C, H, W] → flatten → [B, N, proj_dim]
    GenD CLS token:         [B, gend_dim] → project → [B, 1, proj_dim]

    Cross-attention:
        Query   = GenD token   [B, 1, proj_dim]
        Key     = XRay patches [B, N, proj_dim]
        Value   = XRay patches [B, N, proj_dim]
        Output  = attended features [B, 1, proj_dim]

    Then: attended + GenD residual → LayerNorm → classifier
    """
    def __init__(self, xray_spatial_dim=18, gend_dim=1024,
                 proj_dim=512, num_heads=8):
        super().__init__()

        self.proj_dim = proj_dim

        # Project Face X-Ray spatial patches from C channels to proj_dim
        self.xray_patch_proj = nn.Linear(xray_spatial_dim, proj_dim)

        # Project GenD CLS token to proj_dim
        self.gend_proj = nn.Linear(gend_dim, proj_dim)

        # Cross-attention: Q from GenD, K/V from Face X-Ray patches
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Layer norm after attention
        self.norm = nn.LayerNorm(proj_dim)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, xray_spatial, gend_feat):
        """
        xray_spatial: [B, N, C] — N spatial patches, C channels each
        gend_feat:    [B, gend_dim] — GenD CLS token
        """
        # Project Face X-Ray patches: [B, N, C] → [B, N, proj_dim]
        patches = self.xray_patch_proj(xray_spatial)
        patches = F.normalize(patches, dim=-1)

        # Project GenD token: [B, gend_dim] → [B, 1, proj_dim]
        query = self.gend_proj(gend_feat).unsqueeze(1)
        query = F.normalize(query, dim=-1)

        # Cross-attention: GenD queries the Face X-Ray spatial map
        attended, attention_weights = self.cross_attention(
            query=query,      # [B, 1, proj_dim]
            key=patches,      # [B, N, proj_dim]
            value=patches     # [B, N, proj_dim]
        )
        # attended: [B, 1, proj_dim]

        # Residual: add GenD token back so global semantic signal is preserved
        fused = attended.squeeze(1) + query.squeeze(1)  # [B, proj_dim]

        # Layer norm
        fused = self.norm(fused)

        # Classify
        return self.classifier(fused), attention_weights


# ══════════════════════════════════════════════════════════════════════════════
# Load backbone models
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
    for param in net.parameters():
        param.requires_grad = False
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
    for param in model.parameters():
        param.requires_grad = False
    print('  GenD loaded.')
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# For Architecture 2 we need the SPATIAL map from Face X-Ray
# not just the pooled 2048-dim vector
# ══════════════════════════════════════════════════════════════════════════════

def extract_xray_spatial(net, image_path):
    """
    Extract spatial feature map from Face X-Ray HRNet.
    Returns [N, C] where N = H*W spatial locations, C = channels per location.
    This preserves spatial information for cross-attention.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        hrnet = net.HRNet_layer
        # Get all 4 resolution stream outputs
        y_list = hrnet(tensor)

        # Use the highest resolution stream [B, C, H, W]
        # This preserves spatial information
        spatial_map = y_list[0]  # [1, 18, H, W]

        # Reshape to sequence of patches: [1, H*W, 18]
        B, C, H, W = spatial_map.shape
        patches = spatial_map.view(B, C, H * W).permute(0, 2, 1)  # [1, N, 18]
        patches = patches.squeeze(0)  # [N, 18]

    return patches.cpu()


def extract_gend_features(model, image_path):
    """Extract 1024-dim L2-normalized CLS token from GenD."""
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = model.feature_extractor.preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model.feature_extractor(tensor)
            return F.normalize(feat, dim=-1).squeeze(0).cpu()
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Collect image paths and pre-extract features
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
    """Pre-extract spatial patches from Face X-Ray and CLS token from GenD."""
    print(f'Pre-extracting features for {desc}...')
    data = []
    all_paths = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    random.shuffle(all_paths)

    for path, label in tqdm(all_paths, desc=f'  Extracting {desc}'):
        xf = extract_xray_spatial(xray_net, path)   # [N, 18]
        gf = extract_gend_features(gend_model, path) # [1024]
        if xf is not None and gf is not None:
            data.append((xf, gf, label))

    print(f'  Extracted {len(data)} samples')
    if len(data) > 0:
        print(f'  Spatial patch shape: {data[0][0].shape}')
        print(f'  GenD feature shape:  {data[0][1].shape}')
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Training and evaluation loops
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
        xf     = torch.stack([d[0] for d in batch]).to(DEVICE)  # [B, N, 18]
        gf     = torch.stack([d[1] for d in batch]).to(DEVICE)  # [B, 1024]
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
    total = 0

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
            correct_batch = sum(p == l for p, l in zip(logits.argmax(dim=1).cpu().tolist(), labels))
            correct += correct_batch
            total += len(batch)

    auc = roc_auc_score(all_labels, all_scores) * 100
    acc = 100 * sum(1 for s, l in zip(all_scores, all_labels) if (s > 0.5) == bool(l)) / len(all_labels)
    return auc, acc


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=EPOCHS)
    parser.add_argument('--lr',         type=float, default=LR)
    parser.add_argument('--batch_size', type=int,   default=BATCH_SIZE)
    parser.add_argument('--proj_dim',   type=int,   default=PROJ_DIM)
    parser.add_argument('--num_heads',  type=int,   default=NUM_HEADS)
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print(f'ARCHITECTURE 2 — CROSS-ATTENTION FUSION')
    print(f'GenD semantic token queries Face X-Ray spatial patches')
    print(f'{"="*60}')

    # Load backbones
    xray_net   = load_face_xray()
    gend_model = load_gend()

    # Check spatial map size
    test_img = cv2.imread(cv2.samples.findFile('lena.jpg')) if False else None
    print(f'\nProjection dim: {args.proj_dim}')
    print(f'Attention heads: {args.num_heads}')

    # Collect paths
    print('\nCollecting image paths...')
    train_real, train_fake = collect_image_paths('train', MAX_TRAIN)
    val_real,   val_fake   = collect_image_paths('val',   MAX_VAL)

    # Pre-extract features
    train_data = preextract_features(xray_net, gend_model, train_real, train_fake, 'train')
    val_data   = preextract_features(xray_net, gend_model, val_real,   val_fake,   'val')

    # Get actual spatial channel size from extracted data
    actual_spatial_channels = train_data[0][0].shape[1]  # C from [N, C]
    print(f'\nSpatial channels (C): {actual_spatial_channels}')
    print(f'Spatial patches (N): {train_data[0][0].shape[0]}')

    # Create cross-attention fusion model
    fusion = CrossAttentionFusion(
        xray_spatial_dim=actual_spatial_channels,
        gend_dim=1024,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads
    ).to(DEVICE)

    total_params = sum(p.numel() for p in fusion.parameters())
    print(f'Trainable parameters: {total_params:,}')

    optimizer = torch.optim.Adam(fusion.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, 'fusion_crossattn_best.pth')

    print(f'\nStarting training for {args.epochs} epochs...\n')
    print(f'{"Epoch":<8} {"Train Loss":<12} {"Train Acc":<12} {"Val AUC":<12} {"Val Acc":<10}')
    print('-' * 56)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(fusion, optimizer, train_data, args.batch_size)
        val_auc, val_acc      = eval_epoch(fusion, val_data, args.batch_size)
        scheduler.step()

        marker = ' ← best' if val_auc > best_val_auc else ''
        print(f'{epoch:<8} {train_loss:<12.4f} {train_acc:<12.2f} {val_auc:<12.3f} {val_acc:<10.2f}{marker}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch':              epoch,
                'fusion_state_dict':  fusion.state_dict(),
                'val_auc':            val_auc,
                'val_acc':            val_acc,
                'proj_dim':           args.proj_dim,
                'num_heads':          args.num_heads,
                'spatial_channels':   actual_spatial_channels,
            }, save_path)

    print(f'\n{"="*60}')
    print(f'Training complete!')
    print(f'Best validation AUC : {best_val_auc:.3f}%')
    print(f'Saved to            : {save_path}')
    print(f'\nComparison with Architecture 1:')
    print(f'  Arch 1 Compressed : 84.389%')
    print(f'  Arch 1 Full       : 84.577%')
    print(f'  Arch 2 CrossAttn  : {best_val_auc:.3f}%')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()