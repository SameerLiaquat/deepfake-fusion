"""
Fusion Training Script — Face X-Ray + GenD
Trains the fusion layers on FF++ training set.

Supports two configurations:
  --mode compressed  : 2048→512 + 1024→512 → concat 1024 → classifier
  --mode full        : 2048 + 1024 → concat 3072 → classifier (no projection)

Both backbones are FROZEN. Only fusion layers are trained.
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
EPOCHS          = 20
LR              = 3e-4
BATCH_SIZE      = 32
MAX_TRAIN       = 20000   # max images per class for training
MAX_VAL         = 3000    # max images per class for validation

SAVE_DIR        = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\GenD\GenD\fusion_weights"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, FACE_XRAY_ROOT)

# ══════════════════════════════════════════════════════════════════════════════
# Fusion Models
# ══════════════════════════════════════════════════════════════════════════════

class FusionCompressed(nn.Module):
    """Architecture 1a — Project both to 512, concatenate to 1024"""
    def __init__(self):
        super().__init__()
        self.xray_proj  = nn.Linear(2048, 512)
        self.gend_proj  = nn.Linear(1024, 512)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, xray_feat, gend_feat):
        x = F.normalize(self.xray_proj(xray_feat), dim=-1)
        g = F.normalize(self.gend_proj(gend_feat),  dim=-1)
        return self.classifier(torch.cat([x, g], dim=-1))


class FusionFull(nn.Module):
    """Architecture 1b — Keep full dimensions, concatenate to 3072"""
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, xray_feat, gend_feat):
        # L2 normalise each separately so neither dominates
        x = F.normalize(xray_feat, dim=-1)
        g = F.normalize(gend_feat,  dim=-1)
        return self.classifier(torch.cat([x, g], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# Load models
# ══════════════════════════════════════════════════════════════════════════════

def load_face_xray():
    print('Loading Face X-Ray...')
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
    print(f'  Face X-Ray loaded. AUC: {ckpt["best_auc"]:.3f}')
    return net


def load_gend():
    print('Loading GenD...')
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
# ══════════════════════════════════════════════════════════════════════════════

def extract_xray_features(net, image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        hrnet = net.HRNet_layer
        y_list = hrnet(tensor)
        y = hrnet.incre_modules[0](y_list[0])
        for i in range(len(hrnet.downsamp_modules)):
            y = hrnet.incre_modules[i+1](y_list[i+1]) + hrnet.downsamp_modules[i](y)
        y = hrnet.final_layer(y)
        feat = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        return feat.squeeze(0)


def extract_gend_features(model, image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = model.feature_extractor.preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model.feature_extractor(tensor)
            return F.normalize(feat, dim=-1).squeeze(0)
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Pre-extract all features and cache them (much faster training)
# ══════════════════════════════════════════════════════════════════════════════

def collect_image_paths(split_name, max_per_class):
    with open(os.path.join(SPLITS_DIR, f'{split_name}.json')) as f:
        pairs = json.load(f)
    split_ids = set()
    for pair in pairs:
        split_ids.add(pair[0])
        split_ids.add(pair[1])

    real_paths = []
    fake_paths = []

    # Real
    if os.path.exists(FF_REAL_IMAGES):
        for vid_dir in sorted(os.listdir(FF_REAL_IMAGES)):
            if vid_dir not in split_ids:
                continue
            vid_path = os.path.join(FF_REAL_IMAGES, vid_dir)
            if os.path.isdir(vid_path):
                for frame in sorted(os.listdir(vid_path)):
                    if frame.endswith('.png'):
                        real_paths.append(os.path.join(vid_path, frame))

    # Fake — all 4 datasets
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
    """Extract and cache all features upfront — avoids re-running backbones each batch"""
    print(f'Pre-extracting features for {desc}...')
    data = []
    all_paths = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    random.shuffle(all_paths)

    for path, label in tqdm(all_paths, desc=f'  Extracting {desc}'):
        xf = extract_xray_features(xray_net, path)
        gf = extract_gend_features(gend_model, path)
        if xf is not None and gf is not None:
            data.append((xf.cpu(), gf.cpu(), label))

    print(f'  Extracted {len(data)} samples')
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
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
        xf = torch.stack([d[0] for d in batch]).to(DEVICE)
        gf = torch.stack([d[1] for d in batch]).to(DEVICE)
        labels = torch.tensor([d[2] for d in batch], dtype=torch.long).to(DEVICE)

        optimizer.zero_grad()
        logits = fusion(xf, gf)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(batch)

    return total_loss / total, 100 * correct / total


def eval_epoch(fusion, data, batch_size):
    fusion.eval()
    all_labels = []
    all_scores = []
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            xf = torch.stack([d[0] for d in batch]).to(DEVICE)
            gf = torch.stack([d[1] for d in batch]).to(DEVICE)
            labels = [d[2] for d in batch]

            logits = fusion(xf, gf)
            probs = torch.softmax(logits, dim=-1)
            fake_probs = probs[:, 1].cpu().tolist()

            all_scores.extend(fake_probs)
            all_labels.extend(labels)

            preds = logits.argmax(dim=1).cpu().tolist()
            correct += sum(p == l for p, l in zip(preds, labels))
            total += len(batch)

    auc = roc_auc_score(all_labels, all_scores) * 100
    acc = 100 * correct / total
    return auc, acc


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['compressed', 'full'], default='compressed',
                        help='compressed=512-dim projection, full=3072 no projection')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print(f'FUSION TRAINING — Mode: {args.mode.upper()}')
    print(f'{"="*60}\n')

    # Load backbones
    xray_net   = load_face_xray()
    gend_model = load_gend()

    # Collect image paths
    print('\nCollecting image paths...')
    train_real, train_fake = collect_image_paths('train', MAX_TRAIN)
    val_real,   val_fake   = collect_image_paths('val',   MAX_VAL)

    # Pre-extract all features
    train_data = preextract_features(xray_net, gend_model, train_real, train_fake, 'train')
    val_data   = preextract_features(xray_net, gend_model, val_real,   val_fake,   'val')

    # Create fusion model
    if args.mode == 'compressed':
        fusion = FusionCompressed().to(DEVICE)
        print(f'\nFusion model: Compressed (2048→512 + 1024→512 → 1024 → 2)')
    else:
        fusion = FusionFull().to(DEVICE)
        print(f'\nFusion model: Full (2048 + 1024 → 3072 → 2)')

    total_params = sum(p.numel() for p in fusion.parameters())
    print(f'Trainable parameters: {total_params:,}')

    optimizer = torch.optim.Adam(fusion.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_auc = 0
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f'fusion_{args.mode}_best.pth')

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
                'epoch': epoch,
                'mode': args.mode,
                'fusion_state_dict': fusion.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
            }, save_path)

    print(f'\n{"="*60}')
    print(f'Training complete!')
    print(f'Best validation AUC: {best_val_auc:.3f}%')
    print(f'Saved to: {save_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()