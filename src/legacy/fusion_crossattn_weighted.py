"""
Architecture 2 Cross-Attention — Improved Training
Based on FP/FN analysis findings:

Key changes:
  1. Weighted loss — FN penalised more than FP
     (32% FN rate vs 16.6% FP rate from analysis)
  2. Score normalization 1-99
     (avoids extreme 0/1 values, better calibration)
  3. Optimal threshold = 82/99 (from ROC analysis, was 0.818)
  4. Label smoothing to prevent overconfident predictions
  5. Focal loss option to focus on hard examples

From FP/FN analysis:
  - NeuralTextures hardest (41.6% miss rate)
  - Deepfakes easiest (13.6% miss rate)
  - FN avg confidence = 0.129 (model very sure fakes are real)
  - Optimal threshold = 0.818 → 82 on 1-99 scale
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
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import random
import numpy as np

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
EPOCHS          = 30
LR              = 3e-4
BATCH_SIZE      = 16
MAX_TRAIN       = 20000
MAX_VAL         = 3000
PROJ_DIM        = 512
NUM_HEADS       = 8
PATIENCE        = 7
WEIGHT_DECAY    = 1e-4

# ── KEY SETTINGS FROM FP/FN ANALYSIS ─────────────────────────────────────────
# FN rate was 32%, FP rate was 16.6%
# We penalise missing fakes (FN) roughly 2x more than false alarms (FP)
# class_weights[0] = weight for real class (FP cost)
# class_weights[1] = weight for fake class (FN cost)
FN_WEIGHT       = 2.0   # penalise missing fakes 2x more
FP_WEIGHT       = 1.0   # standard weight for false alarms

# Score normalization range
SCORE_MIN       = 1
SCORE_MAX       = 99

# Optimal threshold from ROC analysis (0.818 raw → 82 on 1-99 scale)
# Formula: threshold_normalized = (0.818 - 0) * (99-1) + 1 = 81.2 ≈ 82
OPTIMAL_THRESHOLD_RAW = 0.818
OPTIMAL_THRESHOLD_NORM = int(OPTIMAL_THRESHOLD_RAW * (SCORE_MAX - SCORE_MIN) + SCORE_MIN)

# Label smoothing — prevents overconfident predictions
LABEL_SMOOTHING = 0.1

SAVE_DIR        = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\GenD\GenD\fusion_weights"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, FACE_XRAY_ROOT)

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Using device: {DEVICE}')


# ══════════════════════════════════════════════════════════════════════════════
# Score Normalization (1-99)
# Converts raw softmax probability (0-1) to normalized score (1-99)
# Avoids extreme 0 and 1 values, more interpretable
# ══════════════════════════════════════════════════════════════════════════════

def normalize_score(raw_prob):
    """
    Normalize raw fake probability (0-1) to 1-99 scale.
    Formula: score = raw_prob * (99-1) + 1
    So: 0.0 → 1,  0.5 → 50,  0.818 → 81.2,  1.0 → 99
    """
    return raw_prob * (SCORE_MAX - SCORE_MIN) + SCORE_MIN


def denormalize_score(norm_score):
    """Convert 1-99 score back to 0-1 probability."""
    return (norm_score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


# ══════════════════════════════════════════════════════════════════════════════
# Weighted Loss with Label Smoothing
# FN weighted more than FP based on analysis
# ══════════════════════════════════════════════════════════════════════════════

class WeightedFocalLoss(nn.Module):
    """
    Weighted cross-entropy with label smoothing.
    - class_weights: [real_weight, fake_weight]
    - fake_weight > real_weight means we penalise missing fakes more
    - label_smoothing prevents overconfident predictions
    """
    def __init__(self, real_weight=1.0, fake_weight=2.0, label_smoothing=0.1):
        super().__init__()
        self.real_weight  = real_weight
        self.fake_weight  = fake_weight
        self.smoothing    = label_smoothing
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        # Standard cross-entropy per sample
        loss = self.ce(logits, labels)

        # Apply class weights
        weights = torch.where(
            labels == 1,
            torch.tensor(self.fake_weight, device=logits.device),
            torch.tensor(self.real_weight, device=logits.device)
        )
        weighted_loss = (loss * weights).mean()

        # Label smoothing regularization
        # Pushes model away from extreme 0/1 predictions
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_probs.mean()
        total_loss = (1 - self.smoothing) * weighted_loss + self.smoothing * smooth_loss

        return total_loss


# ══════════════════════════════════════════════════════════════════════════════
# Cross-Attention Fusion Model (same V1 architecture that gave 86.17%)
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    def __init__(self, xray_spatial_dim=18, gend_dim=1024,
                 proj_dim=512, num_heads=8, dropout=0.3):
        super().__init__()
        self.xray_patch_proj = nn.Linear(xray_spatial_dim, proj_dim)
        self.gend_proj       = nn.Linear(gend_dim, proj_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, xray_spatial, gend_feat):
        patches = F.normalize(self.xray_patch_proj(xray_spatial), dim=-1)
        query   = F.normalize(self.gend_proj(gend_feat).unsqueeze(1), dim=-1)
        attended, attn_weights = self.cross_attention(query, patches, patches)
        fused = self.norm(attended.squeeze(1) + query.squeeze(1))
        return self.classifier(fused), attn_weights


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
    for p in net.parameters():
        p.requires_grad = False
    print(f'  Loaded. AUC: {ckpt["best_auc"]:.3f}')
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
# Feature extraction
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
        spatial = y_list[0]
        B, C, H, W = spatial.shape
        patches = spatial.view(B, C, H*W).permute(0, 2, 1).squeeze(0)
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
# Data collection
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
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Training and evaluation with normalized scores
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(fusion, optimizer, criterion, data, batch_size):
    fusion.train()
    random.shuffle(data)
    total_loss = 0
    correct = 0
    total = 0

    for i in range(0, len(data), batch_size):
        batch  = data[i:i+batch_size]
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


def eval_epoch(fusion, data, batch_size, threshold_norm=OPTIMAL_THRESHOLD_NORM):
    """
    Evaluates using normalized scores (1-99).
    threshold_norm: decision boundary on 1-99 scale (default 82)
    """
    fusion.eval()
    all_labels  = []
    all_scores_norm = []   # normalized 1-99 scores
    all_scores_raw  = []   # raw 0-1 probabilities for AUC

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch  = data[i:i+batch_size]
            xf     = torch.stack([d[0] for d in batch]).to(DEVICE)
            gf     = torch.stack([d[1] for d in batch]).to(DEVICE)
            labels = [d[2] for d in batch]

            logits, _ = fusion(xf, gf)
            probs = torch.softmax(logits, dim=-1)
            raw_fake_probs = probs[:, 1].cpu().tolist()

            # Normalize to 1-99
            norm_scores = [normalize_score(p) for p in raw_fake_probs]

            all_scores_raw.extend(raw_fake_probs)
            all_scores_norm.extend(norm_scores)
            all_labels.extend(labels)

    # AUC uses raw probabilities (scale invariant)
    auc = roc_auc_score(all_labels, all_scores_raw) * 100

    # Accuracy uses normalized threshold
    preds = [1 if s >= threshold_norm else 0 for s in all_scores_norm]
    acc = 100 * sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)

    # Confusion matrix
    tn = fp = fn = tp = 0
    for p, l in zip(preds, all_labels):
        if l == 0 and p == 0: tn += 1
        elif l == 0 and p == 1: fp += 1
        elif l == 1 and p == 0: fn += 1
        elif l == 1 and p == 1: tp += 1

    sensitivity = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0

    return auc, acc, sensitivity, specificity, all_scores_norm, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=EPOCHS)
    parser.add_argument('--lr',          type=float, default=LR)
    parser.add_argument('--batch_size',  type=int,   default=BATCH_SIZE)
    parser.add_argument('--fn_weight',   type=float, default=FN_WEIGHT,
                        help='Weight for fake class (FN penalty)')
    parser.add_argument('--fp_weight',   type=float, default=FP_WEIGHT,
                        help='Weight for real class (FP penalty)')
    parser.add_argument('--threshold',   type=int,   default=OPTIMAL_THRESHOLD_NORM,
                        help=f'Decision threshold on 1-99 scale (default {OPTIMAL_THRESHOLD_NORM})')
    parser.add_argument('--max_train',   type=int,   default=MAX_TRAIN)
    parser.add_argument('--max_val',     type=int,   default=MAX_VAL)
    parser.add_argument('--patience',    type=int,   default=PATIENCE)
    args = parser.parse_args()

    print(f'\n{"="*65}')
    print(f'CROSS-ATTENTION FUSION — IMPROVED WITH FP/FN ANALYSIS')
    print(f'{"="*65}')
    print(f'  FN weight (fake penalty)   : {args.fn_weight}x')
    print(f'  FP weight (real penalty)   : {args.fp_weight}x')
    print(f'  Score normalization        : {SCORE_MIN}-{SCORE_MAX}')
    print(f'  Decision threshold         : {args.threshold}/99')
    print(f'  Label smoothing            : {LABEL_SMOOTHING}')
    print(f'  From FP/FN analysis:')
    print(f'    NeuralTextures miss rate : 41.6% (hardest)')
    print(f'    Deepfakes miss rate      : 13.6% (easiest)')
    print(f'    Previous FN rate         : 32%')
    print(f'    Previous FP rate         : 16.6%')
    print(f'{"="*65}')

    # Load backbones
    xray_net   = load_face_xray()
    gend_model = load_gend()

    # Collect and extract
    print('\nCollecting image paths...')
    train_real, train_fake = collect_image_paths('train', args.max_train)
    val_real,   val_fake   = collect_image_paths('val',   args.max_val)

    train_data = preextract_features(xray_net, gend_model, train_real, train_fake, 'train')
    val_data   = preextract_features(xray_net, gend_model, val_real,   val_fake,   'val')

    spatial_channels = train_data[0][0].shape[1]

    # Create model
    fusion = CrossAttentionFusion(
        xray_spatial_dim=spatial_channels,
        gend_dim=1024,
        proj_dim=PROJ_DIM,
        num_heads=NUM_HEADS
    ).to(DEVICE)

    total_params = sum(p.numel() for p in fusion.parameters())
    print(f'\nTrainable parameters: {total_params:,}')

    # Weighted loss with label smoothing
    criterion = WeightedFocalLoss(
        real_weight=args.fp_weight,
        fake_weight=args.fn_weight,
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = torch.optim.Adam(
        fusion.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_auc = 0
    no_improve   = 0
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, 'fusion_crossattn_weighted_best.pth')

    print(f'\nTraining with threshold={args.threshold}/99 '
          f'(raw={args.threshold/99:.3f})\n')
    print(f'{"Epoch":<7} {"Loss":<10} {"TrainAcc":<11} '
          f'{"ValAUC":<10} {"ValAcc":<10} {"Sens":<8} {"Spec":<8}')
    print('-' * 66)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            fusion, optimizer, criterion, train_data, args.batch_size
        )
        val_auc, val_acc, sens, spec, norm_scores, val_labels = eval_epoch(
            fusion, val_data, args.batch_size, args.threshold
        )
        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve   = 0
            marker = ' ← best'
            torch.save({
                'epoch':              epoch,
                'fusion_state_dict':  fusion.state_dict(),
                'val_auc':            val_auc,
                'val_acc':            val_acc,
                'sensitivity':        sens,
                'specificity':        spec,
                'threshold_norm':     args.threshold,
                'fn_weight':          args.fn_weight,
                'fp_weight':          args.fp_weight,
                'spatial_channels':   spatial_channels,
                'proj_dim':           PROJ_DIM,
                'num_heads':          NUM_HEADS,
            }, save_path)
        else:
            no_improve += 1
            marker = f' ({no_improve}/{args.patience})'

        print(f'{epoch:<7} {train_loss:<10.4f} {train_acc:<11.2f} '
              f'{val_auc:<10.3f} {val_acc:<10.2f} {sens:<8.1f} {spec:<8.1f}{marker}')

        if no_improve >= args.patience:
            print(f'\nEarly stopping at epoch {epoch}.')
            break

    # Final score distribution analysis
    print(f'\n{"="*65}')
    print('FINAL SCORE DISTRIBUTION (1-99 scale)')
    print(f'{"="*65}')
    real_scores = [s for s, l in zip(norm_scores, val_labels) if l == 0]
    fake_scores = [s for s, l in zip(norm_scores, val_labels) if l == 1]
    print(f'  Real images : mean={np.mean(real_scores):.1f}  '
          f'std={np.std(real_scores):.1f}  '
          f'max={np.max(real_scores):.1f}')
    print(f'  Fake images : mean={np.mean(fake_scores):.1f}  '
          f'std={np.std(fake_scores):.1f}  '
          f'min={np.min(fake_scores):.1f}')
    print(f'  Threshold   : {args.threshold}/99')
    print(f'  Real above threshold (FP) : '
          f'{sum(1 for s in real_scores if s >= args.threshold)}')
    print(f'  Fake below threshold (FN) : '
          f'{sum(1 for s in fake_scores if s < args.threshold)}')

    print(f'\n{"="*65}')
    print('SUMMARY')
    print(f'{"="*65}')
    print(f'  Best Val AUC              : {best_val_auc:.3f}%')
    print(f'\n  Previous results (threshold=0.5):')
    print(f'    Arch 2 V1 AUC           : 86.170%')
    print(f'    FN rate                 : 32.0%')
    print(f'    FP rate                 : 16.6%')
    print(f'\n  This run (threshold={args.threshold}/99, FN weight={args.fn_weight}x):')
    print(f'    Val AUC                 : {best_val_auc:.3f}%')
    print(f'  Saved to: {save_path}')
    print(f'{"="*65}')


if __name__ == '__main__':
    main()