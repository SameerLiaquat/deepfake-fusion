"""
Architecture 2b — GATED Cross-Attention Fusion
Face X-Ray + GenD combined using cross-attention, PLUS a learned gate that
lets the model fall back to a GenD-only prediction per sample.

WHY: In the original CrossAttentionFusion, every sample is forced through
the same fused representation, so Face X-Ray's noise on hard/generative
samples (NeuralTextures, Face2Face) can drag down predictions that GenD
alone would have gotten right. This version gives the model an explicit
"trust GenD alone" branch and a gate that decides, per sample, how much to
blend in the Face-X-Ray-informed branch. The gate's bias is initialized
negative so training STARTS mostly trusting GenD, and only opens up where
Face X-Ray demonstrably helps.

Everything else (feature extraction, data collection, training loop shape)
is unchanged from your original fusion_crossattn.py.
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
BATCH_SIZE      = 16
MAX_TRAIN       = 20000
MAX_VAL         = 3000
PROJ_DIM        = 512
NUM_HEADS       = 8

SAVE_DIR        = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\GenD\GenD\fusion_weights"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, FACE_XRAY_ROOT)

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Using device: {DEVICE}')


# ══════════════════════════════════════════════════════════════════════════════
# Architecture 2b — Gated Cross-Attention Fusion Model
# ══════════════════════════════════════════════════════════════════════════════

class GatedCrossAttentionFusion(nn.Module):
    """
    Same GenD-queries-FaceXRay cross-attention as before, but now the final
    prediction is a learned mix:

        logits = g * fused_logits + (1 - g) * gend_only_logits

    where g in (0,1) is predicted per-sample from the attention output.
    gend_only_logits comes from a separate small head trained directly on
    GenD's CLS token, so the model always has a "just trust GenD" fallback
    available instead of being forced through the fused representation.

    Gate bias is initialized negative (gate starts near 0) so training
    begins mostly in "trust GenD" mode and has to learn to open the gate.
    """
    def __init__(self, xray_spatial_dim=18, gend_dim=1024,
                 proj_dim=512, num_heads=8, gate_init_bias=-2.0):
        super().__init__()

        self.proj_dim = proj_dim

        self.xray_patch_proj = nn.Linear(xray_spatial_dim, proj_dim)
        self.gend_proj = nn.Linear(gend_dim, proj_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.norm = nn.LayerNorm(proj_dim)

        # fused branch (same shape as your original classifier)
        self.fused_classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        # GenD-only fallback branch
        self.gend_only_classifier = nn.Sequential(
            nn.Linear(gend_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        # gate: how much to trust the fused branch vs GenD-only
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        nn.init.constant_(self.gate[-1].bias, gate_init_bias)

    def forward(self, xray_spatial, gend_feat):
        """
        xray_spatial: [B, N, C]
        gend_feat:    [B, gend_dim]
        Returns (logits, gate) — gate kept as second return value so your
        existing `logits, _ = fusion(xf, gf)` calls still work unchanged.
        """
        patches = self.xray_patch_proj(xray_spatial)
        patches = F.normalize(patches, dim=-1)

        query = self.gend_proj(gend_feat).unsqueeze(1)
        query = F.normalize(query, dim=-1)

        attended, attention_weights = self.cross_attention(
            query=query, key=patches, value=patches
        )

        fused = attended.squeeze(1) + query.squeeze(1)
        fused = self.norm(fused)

        fused_logits = self.fused_classifier(fused)
        gend_logits = self.gend_only_classifier(gend_feat)

        gate_input = torch.cat([query.squeeze(1), attended.squeeze(1)], dim=-1)
        g = torch.sigmoid(self.gate(gate_input))  # [B, 1]

        logits = g * fused_logits + (1 - g) * gend_logits

        return logits, g.squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Load backbone models (unchanged — both already frozen, this is correct)
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
# Feature extraction (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def extract_xray_spatial(net, image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        hrnet = net.HRNet_layer
        y_list = hrnet(tensor)
        spatial_map = y_list[0]
        B, C, H, W = spatial_map.shape
        patches = spatial_map.view(B, C, H * W).permute(0, 2, 1)
        patches = patches.squeeze(0)

    return patches.cpu()


def extract_gend_features(model, image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = model.feature_extractor.preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model.feature_extractor(tensor)
            return F.normalize(feat, dim=-1).squeeze(0).cpu()
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Collect image paths — NOW TRACKS MANIPULATION METHOD PER SAMPLE
# ══════════════════════════════════════════════════════════════════════════════

def collect_image_paths(split_name, max_per_class):
    with open(os.path.join(SPLITS_DIR, f'{split_name}.json')) as f:
        pairs = json.load(f)
    split_ids = set()
    for pair in pairs:
        split_ids.add(pair[0])
        split_ids.add(pair[1])

    real_paths = []       # list of paths
    fake_paths = []       # list of (path, method) tuples

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
                        fake_paths.append((os.path.join(vid_path, frame), ds))

    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    real_paths = real_paths[:max_per_class]
    fake_paths = fake_paths[:max_per_class]

    print(f'  {split_name}: {len(real_paths)} real, {len(fake_paths)} fake')
    return real_paths, fake_paths


def preextract_features(xray_net, gend_model, real_paths, fake_paths, desc):
    """Pre-extract features. Each sample now stored as (xf, gf, label, method)."""
    print(f'Pre-extracting features for {desc}...')
    data = []
    all_items = [(p, 0, 'real') for p in real_paths] + [(p, 1, m) for p, m in fake_paths]
    random.shuffle(all_items)

    for path, label, method in tqdm(all_items, desc=f'  Extracting {desc}'):
        xf = extract_xray_spatial(xray_net, path)
        gf = extract_gend_features(gend_model, path)
        if xf is not None and gf is not None:
            data.append((xf, gf, label, method))

    print(f'  Extracted {len(data)} samples')
    if len(data) > 0:
        print(f'  Spatial patch shape: {data[0][0].shape}')
        print(f'  GenD feature shape:  {data[0][1].shape}')
    return data


def build_balanced_order(data):
    """
    Resamples `data` so 'real', 'Deepfakes', 'Face2Face', 'FaceSwap', and
    'NeuralTextures' are each drawn with equal probability per epoch,
    instead of proportional to however many of each happened to survive
    truncation. Cheap insurance on top of the already-roughly-balanced
    random shuffle in collect_image_paths.
    """
    methods = [d[3] for d in data]
    unique, counts = np.unique(methods, return_counts=True)
    weight_by_method = {m: 1.0 / c for m, c in zip(unique, counts)}
    weights = np.array([weight_by_method[m] for m in methods], dtype=np.float64)
    weights = weights / weights.sum()
    idx = np.random.choice(len(data), size=len(data), replace=True, p=weights)
    return [data[i] for i in idx]


# ══════════════════════════════════════════════════════════════════════════════
# Training and evaluation loops
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(fusion, optimizer, data, batch_size):
    fusion.train()
    data = build_balanced_order(data)   # was: random.shuffle(data)
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


def analyze_gate_by_method(fusion, data, batch_size):
    """
    Runs after training. Reports mean gate value per manipulation method.
    Your hypothesis predicts gate(Deepfakes)/gate(FaceSwap) should be
    higher than gate(NeuralTextures)/gate(Face2Face) -- i.e. the model
    should learn to lean on Face X-Ray specifically for blend-based fakes.
    """
    fusion.eval()
    from collections import defaultdict
    gate_by_method = defaultdict(list)

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            xf      = torch.stack([d[0] for d in batch]).to(DEVICE)
            gf      = torch.stack([d[1] for d in batch]).to(DEVICE)
            methods = [d[3] for d in batch]

            _, gate = fusion(xf, gf)
            for g_val, m in zip(gate.cpu().tolist(), methods):
                gate_by_method[m].append(g_val)

    print(f'\n{"Method":<16} {"Mean Gate":>10} {"N":>6}')
    print('-' * 34)
    for method in sorted(gate_by_method.keys()):
        vals = gate_by_method[method]
        print(f'{method:<16} {np.mean(vals):>10.3f} {len(vals):>6}')


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
    parser.add_argument('--gate_init_bias', type=float, default=-2.0,
                         help='Negative = gate starts closer to "trust GenD only"')
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print(f'ARCHITECTURE 2b — GATED CROSS-ATTENTION FUSION')
    print(f'Learned gate lets the model fall back to GenD-only per sample')
    print(f'{"="*60}')

    xray_net   = load_face_xray()
    gend_model = load_gend()

    print(f'\nProjection dim: {args.proj_dim}')
    print(f'Attention heads: {args.num_heads}')
    print(f'Gate init bias: {args.gate_init_bias}')

    print('\nCollecting image paths...')
    train_real, train_fake = collect_image_paths('train', MAX_TRAIN)
    val_real,   val_fake   = collect_image_paths('val',   MAX_VAL)

    train_data = preextract_features(xray_net, gend_model, train_real, train_fake, 'train')
    val_data   = preextract_features(xray_net, gend_model, val_real,   val_fake,   'val')

    actual_spatial_channels = train_data[0][0].shape[1]
    print(f'\nSpatial channels (C): {actual_spatial_channels}')
    print(f'Spatial patches (N): {train_data[0][0].shape[0]}')

    fusion = GatedCrossAttentionFusion(
        xray_spatial_dim=actual_spatial_channels,
        gend_dim=1024,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        gate_init_bias=args.gate_init_bias
    ).to(DEVICE)

    total_params = sum(p.numel() for p in fusion.parameters())
    print(f'Trainable parameters: {total_params:,}')

    optimizer = torch.optim.Adam(fusion.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, 'fusion_crossattn_gated_best.pth')

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
                'gate_init_bias':     args.gate_init_bias,
            }, save_path)

    print(f'\n{"="*60}')
    print(f'Training complete!')
    print(f'Best validation AUC : {best_val_auc:.3f}%')
    print(f'Saved to            : {save_path}')
    print(f'\nComparison:')
    print(f'  GenD alone           : 92.5%   (reference)')
    print(f'  Face X-Ray alone     : 52.1%   (reference)')
    print(f'  Arch 1 Full          : 84.577%')
    print(f'  Arch 2 CrossAttn     : 86.170%')
    print(f'  Arch 2b Gated        : {best_val_auc:.3f}%')
    print(f'{"="*60}')

    # Load best checkpoint back and run the diagnostic
    ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
    fusion.load_state_dict(ckpt['fusion_state_dict'])
    print('\nGate analysis by manipulation method (val set, best checkpoint):')
    analyze_gate_by_method(fusion, val_data, args.batch_size)


if __name__ == '__main__':
    main()