"""
Architecture 2b v2 -- GATED Cross-Attention Fusion, WITH FIXES

Three confirmed bugs from diagnostic testing, now fixed:
  1. load_face_xray(): removed the "HRNet_layer." prefix-stripping that was
     silently discarding ~1954 backbone weight tensors (they never loaded).
  2. extract_xray_spatial(): no ImageNet normalization (confirmed to hurt).
  3. extract_xray_spatial(): added a face-crop step (Haar cascade, 1.3x
     margin). Sample images showed full upper-body frames, not tight face
     crops -- standalone AUC went 52.68% -> 87.09% once this was added.

Also added: disk caching of extracted features. Extraction is now slower
per-image (face detection overhead) so caching matters more than ever --
this run pays the cost once, every future run of this script (or any
variant you build from it) loads instantly from cache instead.

Everything else (GatedCrossAttentionFusion architecture, balanced sampling,
training loop, gate diagnostic) is unchanged from fusion_crossattn_gated.py.
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

EPOCHS          = 20
LR              = 3e-4
BATCH_SIZE      = 16
MAX_TRAIN       = 20000
MAX_VAL         = 3000
PROJ_DIM        = 512
NUM_HEADS       = 8

SAVE_DIR        = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\GenD\GenD\fusion_weights"
CACHE_DIR       = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\GenD\GenD\feature_cache"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, FACE_XRAY_ROOT)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Using device: {DEVICE}')


# ══════════════════════════════════════════════════════════════════════════════
# Gated Cross-Attention Fusion Model (unchanged from fusion_crossattn_gated.py)
# ══════════════════════════════════════════════════════════════════════════════

class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, xray_spatial_dim=18, gend_dim=1024,
                 proj_dim=512, num_heads=8, gate_init_bias=-2.0):
        super().__init__()
        self.proj_dim = proj_dim
        self.xray_patch_proj = nn.Linear(xray_spatial_dim, proj_dim)
        self.gend_proj = nn.Linear(gend_dim, proj_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(proj_dim)
        self.fused_classifier = nn.Sequential(
            nn.Linear(proj_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)
        )
        self.gend_only_classifier = nn.Sequential(
            nn.Linear(gend_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)
        )
        self.gate = nn.Sequential(nn.Linear(proj_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.constant_(self.gate[-1].bias, gate_init_bias)

    def forward(self, xray_spatial, gend_feat):
        patches = self.xray_patch_proj(xray_spatial)
        patches = F.normalize(patches, dim=-1)
        query = self.gend_proj(gend_feat).unsqueeze(1)
        query = F.normalize(query, dim=-1)
        attended, _ = self.cross_attention(query=query, key=patches, value=patches)
        fused = self.norm(attended.squeeze(1) + query.squeeze(1))
        fused_logits = self.fused_classifier(fused)
        gend_logits = self.gend_only_classifier(gend_feat)
        gate_input = torch.cat([query.squeeze(1), attended.squeeze(1)], dim=-1)
        g = torch.sigmoid(self.gate(gate_input))
        logits = g * fused_logits + (1 - g) * gend_logits
        return logits, g.squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Load backbones -- FIXED loading for Face X-Ray
# ══════════════════════════════════════════════════════════════════════════════

def load_face_xray():
    print('\nLoading Face X-Ray...')
    from HRNet import get_net
    net = get_net(cfg_file=HRNET_CONFIG, devices=[torch.device('cuda:0')])

    ckpt = torch.load(FACE_XRAY_CKPT, map_location='cpu', weights_only=False)
    raw_sd = ckpt['state_dict']
    # FIX: do NOT strip the "HRNet_layer." prefix -- net.HRNet_layer is a
    # genuinely nested submodule, so checkpoint keys must load AS-IS.
    # Only the irrelevant 1000-class ImageNet head is excluded.
    filtered_sd = {k: v for k, v in raw_sd.items()
                   if not k.startswith('HRNet_layer.classifier')}
    missing, unexpected = net.load_state_dict(filtered_sd, strict=False)
    print(f'  Missing keys after load   : {len(missing)}  (expect ~2)')
    print(f'  Unexpected keys after load: {len(unexpected)}  (expect 0)')

    net = net.to(DEVICE)
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    print(f'  Loaded. Checkpoint self-reported AUC: {ckpt["best_auc"]:.3f}')
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
# Feature extraction -- FIXED: face crop, no normalization
# ══════════════════════════════════════════════════════════════════════════════

def crop_face(img_bgr, margin=1.3):
    """Detect the largest face and crop a square region with a margin.
    Returns None if no face detected -- caller falls back to the full frame."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w / 2, y + h / 2
    half = (max(w, h) * margin) / 2
    H, W = img_bgr.shape[:2]
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, W)), int(min(cy + half, H))
    crop = img_bgr[y0:y1, x0:x1]
    return crop if crop.size > 0 else None


def extract_xray_spatial(net, image_path):
    """FIXED: crops to the face before resizing. Still returns the y_list[0]
    intermediate spatial patches (same representation the fusion head was
    already built around) -- but now from a properly loaded backbone seeing
    a properly framed face, instead of noise."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    cropped = crop_face(img)
    if cropped is not None:
        img = cropped
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)  # no normalization -- confirmed correct

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
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Collect image paths (unchanged -- tracks manipulation method per sample)
# ══════════════════════════════════════════════════════════════════════════════

def collect_image_paths(split_name, max_per_class):
    with open(os.path.join(SPLITS_DIR, f'{split_name}.json')) as f:
        pairs = json.load(f)
    split_ids = set()
    for pair in pairs:
        split_ids.add(pair[0]); split_ids.add(pair[1])

    real_paths = []
    fake_paths = []

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

    random.shuffle(real_paths); random.shuffle(fake_paths)
    real_paths = real_paths[:max_per_class]
    fake_paths = fake_paths[:max_per_class]
    print(f'  {split_name}: {len(real_paths)} real, {len(fake_paths)} fake')
    return real_paths, fake_paths


def preextract_features(xray_net, gend_model, real_paths, fake_paths, desc):
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


def get_or_extract_features(xray_net, gend_model, real_paths, fake_paths, desc, cache_name):
    """Loads from disk cache if present, otherwise extracts and saves for
    next time. This is the key change -- extraction only ever happens once."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(cache_path):
        print(f'Loading cached features from {cache_path} ...')
        data = torch.load(cache_path, weights_only=False)
        print(f'  Loaded {len(data)} cached samples (skipped extraction)')
        return data
    data = preextract_features(xray_net, gend_model, real_paths, fake_paths, desc)
    torch.save(data, cache_path)
    print(f'  Cached to {cache_path}')
    return data


def build_balanced_order(data):
    methods = [d[3] for d in data]
    unique, counts = np.unique(methods, return_counts=True)
    weight_by_method = {m: 1.0 / c for m, c in zip(unique, counts)}
    weights = np.array([weight_by_method[m] for m in methods], dtype=np.float64)
    weights = weights / weights.sum()
    idx = np.random.choice(len(data), size=len(data), replace=True, p=weights)
    return [data[i] for i in idx]


# ══════════════════════════════════════════════════════════════════════════════
# Training and evaluation loops (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(fusion, optimizer, data, batch_size):
    fusion.train()
    data = build_balanced_order(data)
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0

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
    correct, total = 0, 0

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
            total += len(batch)

    auc = roc_auc_score(all_labels, all_scores) * 100
    acc = 100 * sum(1 for s, l in zip(all_scores, all_labels) if (s > 0.5) == bool(l)) / len(all_labels)
    return auc, acc


def analyze_gate_by_method(fusion, data, batch_size):
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
    parser.add_argument('--gate_init_bias', type=float, default=-2.0)
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print(f'ARCHITECTURE 2b v2 -- GATED FUSION, FIXED LOADING + FACE CROP')
    print(f'{"="*60}')

    xray_net   = load_face_xray()
    gend_model = load_gend()

    print(f'\nProjection dim: {args.proj_dim}  Attention heads: {args.num_heads}  Gate init bias: {args.gate_init_bias}')

    print('\nCollecting image paths...')
    train_real, train_fake = collect_image_paths('train', MAX_TRAIN)
    val_real,   val_fake   = collect_image_paths('val',   MAX_VAL)

    train_data = get_or_extract_features(xray_net, gend_model, train_real, train_fake,
                                          'train', 'train_features_facecrop_fixed.pt')
    val_data   = get_or_extract_features(xray_net, gend_model, val_real,   val_fake,
                                          'val',   'val_features_facecrop_fixed.pt')

    actual_spatial_channels = train_data[0][0].shape[1]
    print(f'\nSpatial channels (C): {actual_spatial_channels}')
    print(f'Spatial patches (N): {train_data[0][0].shape[0]}')

    fusion = GatedCrossAttentionFusion(
        xray_spatial_dim=actual_spatial_channels, gend_dim=1024,
        proj_dim=args.proj_dim, num_heads=args.num_heads,
        gate_init_bias=args.gate_init_bias
    ).to(DEVICE)

    total_params = sum(p.numel() for p in fusion.parameters())
    print(f'Trainable parameters: {total_params:,}')

    optimizer = torch.optim.Adam(fusion.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, 'fusion_crossattn_gated_v2_best.pth')

    print(f'\nStarting training for {args.epochs} epochs...\n')
    print(f'{"Epoch":<8} {"Train Loss":<12} {"Train Acc":<12} {"Val AUC":<12} {"Val Acc":<10}')
    print('-' * 56)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(fusion, optimizer, train_data, args.batch_size)
        val_auc, val_acc      = eval_epoch(fusion, val_data, args.batch_size)
        scheduler.step()

        marker = ' <- best' if val_auc > best_val_auc else ''
        print(f'{epoch:<8} {train_loss:<12.4f} {train_acc:<12.2f} {val_auc:<12.3f} {val_acc:<10.2f}{marker}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch, 'fusion_state_dict': fusion.state_dict(),
                'val_auc': val_auc, 'val_acc': val_acc,
                'proj_dim': args.proj_dim, 'num_heads': args.num_heads,
                'spatial_channels': actual_spatial_channels,
            }, save_path)

    print(f'\n{"="*60}')
    print(f'Training complete!')
    print(f'Best validation AUC : {best_val_auc:.3f}%')
    print(f'Saved to            : {save_path}')
    print(f'\nComparison:')
    print(f'  GenD alone                  : 92.5%   (reference)')
    print(f'  Face X-Ray alone (broken)    : 52.1%   (old, pre-fix reference)')
    print(f'  Face X-Ray alone (fixed)     : ~87%    (quick 600-sample check, see note)')
    print(f'  Arch 2 CrossAttn (broken)    : 86.170%')
    print(f'  Arch 2b Gated v2 (fixed)     : {best_val_auc:.3f}%')
    print(f'{"="*60}')

    ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
    fusion.load_state_dict(ckpt['fusion_state_dict'])
    print('\nGate analysis by manipulation method (val set, best checkpoint):')
    analyze_gate_by_method(fusion, val_data, args.batch_size)


if __name__ == '__main__':
    main()