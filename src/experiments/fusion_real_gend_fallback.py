"""
fusion_real_gend_fallback.py

Same GatedCrossAttentionFusion as fusion_crossattn_gated_v3.py, except the
"trust GenD" fallback branch uses GenD's REAL, already-trained classifier
(model.model.linear -- confirmed genuine and strong: 96.6% AUC standalone)
instead of training a fresh substitute MLP from scratch (which, tested in
isolation, only reached 84.5%).

Uses the SAME cached features from the v3 run -- no re-extraction, no
changes to v3's script, checkpoint, or cache. This is purely additive:
a new model variant, trained fresh, saved to its own checkpoint file.

Usage:
    conda activate GenD
    python fusion_real_gend_fallback.py --epochs 30
"""

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

CACHE_DIR = r"C:\Users\y5s86\LocalWork\feature_cache"
SAVE_DIR  = r"C:\Users\y5s86\LocalWork\fusion_weights"
GEND_MODEL = "yermandy/GenD_DINOv3_L"
HF_TOKEN = None

BATCH_SIZE = 16
EPOCHS     = 30
PATIENCE   = 8
PROJ_DIM   = 512
NUM_HEADS  = 8

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


def load_gend_real_linear():
    """Only need GenD's tiny real classifier -- no backbone forward pass,
    since we already have cached CLS-token features."""
    from src.hf.modeling_gend import GenD
    token = HF_TOKEN or os.environ.get('HUGGINGFACE_TOKEN', None)
    torch.set_default_device('cpu')
    model = GenD.from_pretrained(GEND_MODEL, token=token)
    torch.set_default_device(None)
    linear = model.model.linear  # confirmed real trained head: (2,1024) weight, (2,) bias
    linear.eval()
    for p in linear.parameters():
        p.requires_grad = False
    return linear


def load_cached(name):
    path = os.path.join(CACHE_DIR, name)
    print(f'Loading {path} ...')
    data = torch.load(path, weights_only=False)
    print(f'  {len(data)} samples')
    return data


def build_balanced_order(data):
    methods = [d[3] for d in data]
    unique, counts = np.unique(methods, return_counts=True)
    weight_by_method = {m: 1.0 / c for m, c in zip(unique, counts)}
    weights = np.array([weight_by_method[m] for m in methods], dtype=np.float64)
    weights = weights / weights.sum()
    idx = np.random.choice(len(data), size=len(data), replace=True, p=weights)
    return [data[i] for i in idx]


class GatedFusionRealGenD(nn.Module):
    def __init__(self, gend_real_linear, xray_spatial_dim=18, gend_dim=1024,
                 proj_dim=512, num_heads=8, gate_init_bias=-2.0, dropout=0.3):
        super().__init__()
        self.xray_patch_proj = nn.Linear(xray_spatial_dim, proj_dim)
        self.gend_proj = nn.Linear(gend_dim, proj_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        self.fused_classifier = nn.Sequential(
            nn.Linear(proj_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2))
        self.gend_real_linear = gend_real_linear  # REAL, frozen GenD classifier
        self.gate = nn.Sequential(nn.Linear(proj_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.constant_(self.gate[-1].bias, gate_init_bias)

    def forward(self, xray_spatial, gend_feat):
        patches = F.normalize(self.xray_patch_proj(xray_spatial), dim=-1)
        query = F.normalize(self.gend_proj(gend_feat), dim=-1).unsqueeze(1)
        attended, _ = self.cross_attention(query=query, key=patches, value=patches)
        fused = self.norm(attended.squeeze(1) + query.squeeze(1))
        fused_logits = self.fused_classifier(fused)
        with torch.no_grad():
            gend_logits = self.gend_real_linear(gend_feat)  # real, frozen
        gate_input = torch.cat([query.squeeze(1), attended.squeeze(1)], dim=-1)
        g = torch.sigmoid(self.gate(gate_input))
        logits = g * fused_logits + (1 - g) * gend_logits
        return logits, g.squeeze(-1)


def run_epoch(model, optimizer, data, batch_size, train):
    model.train(train)
    data = list(data)
    if train:
        data = build_balanced_order(data)

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0
    all_labels, all_scores = [], []

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        xf     = torch.stack([d[0] for d in batch]).to(DEVICE)
        gf     = torch.stack([d[1] for d in batch]).to(DEVICE)
        labels = torch.tensor([d[2] for d in batch], dtype=torch.long).to(DEVICE)

        with torch.set_grad_enabled(train):
            logits, _ = model(xf, gf)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item() * len(batch)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(batch)
        probs = torch.softmax(logits, dim=-1)
        all_scores.extend(probs[:, 1].detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    auc = roc_auc_score(all_labels, all_scores) * 100
    return total_loss / total, 100 * correct / total, auc


def analyze_gate_by_method(model, data, batch_size):
    model.eval()
    from collections import defaultdict
    gate_by_method = defaultdict(list)
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            xf      = torch.stack([d[0] for d in batch]).to(DEVICE)
            gf      = torch.stack([d[1] for d in batch]).to(DEVICE)
            methods = [d[3] for d in batch]
            _, gate = model(xf, gf)
            for g_val, m in zip(gate.cpu().tolist(), methods):
                gate_by_method[m].append(g_val)
    print(f'\n{"Method":<16} {"Mean Gate":>10} {"N":>6}')
    print('-' * 34)
    for method in sorted(gate_by_method.keys()):
        vals = gate_by_method[method]
        print(f'{method:<16} {np.mean(vals):>10.3f} {len(vals):>6}')


def main():
    train_data = load_cached('train_features_both_cropped.pt')
    val_data   = load_cached('val_features_both_cropped.pt')
    spatial_dim = train_data[0][0].shape[1]

    gend_real_linear = load_gend_real_linear()
    model = GatedFusionRealGenD(gend_real_linear, xray_spatial_dim=spatial_dim,
                                proj_dim=PROJ_DIM, num_heads=NUM_HEADS).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable:,} (GenD classifier frozen, not counted)')

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc, best_state, epochs_since_improve = 0, None, 0
    print(f'\n{"Epoch":<8}{"TrLoss":<10}{"TrAcc":<10}{"ValAUC":<10}')

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _ = run_epoch(model, optimizer, train_data, BATCH_SIZE, train=True)
        _, _, val_auc = run_epoch(model, optimizer, val_data, BATCH_SIZE, train=False)
        scheduler.step()

        marker = ''
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
            marker = ' <- best'
        else:
            epochs_since_improve += 1

        print(f'{epoch:<8}{tr_loss:<10.4f}{tr_acc:<10.2f}{val_auc:<10.3f}{marker}')

        if epochs_since_improve >= PATIENCE:
            print(f'  Early stop at epoch {epoch}')
            break

    model.load_state_dict(best_state)
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'fusion_real_gend_fallback_best.pth'))

    print(f'\n{"="*60}')
    print(f'RESULTS')
    print(f'{"="*60}')
    print(f'GenD alone (real classifier)          : 96.648%')
    print(f'Fusion v3 (trained-from-scratch fallback) : 97.074%')
    print(f'Fusion, REAL GenD fallback (this run) : {best_auc:.3f}%')

    print('\nGate analysis by manipulation method:')
    analyze_gate_by_method(model, val_data, BATCH_SIZE)


if __name__ == '__main__':
    main()