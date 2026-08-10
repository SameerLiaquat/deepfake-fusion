"""
fusion_multiseed_validation.py

Trains the winning architecture (PureFusedCrossAttention -- no gate, no
GenD-only fallback, always 100% fused branch, from fusion_no_gate_pure.py,
which beat every other variant at 97.221%) across 5 different random
seeds, using the same cached features -- no re-extraction. Reports mean
+/- std AUC and how many seeds actually beat GenD alone (96.648%), turning
a single lucky-epoch result into a defensible statistic.

Usage:
    conda activate GenD
    python fusion_multiseed_validation.py
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

BATCH_SIZE = 16
EPOCHS     = 30
PATIENCE   = 8
PROJ_DIM   = 512
NUM_HEADS  = 8
SEEDS      = [0, 1, 2, 3, 4]

GEND_ALONE_AUC = 96.648  # confirmed reference, exact-match test

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class PureFusedCrossAttention(nn.Module):
    """The winning architecture: no gate, no fallback, always 100% fused."""
    def __init__(self, xray_spatial_dim=18, gend_dim=1024, proj_dim=512,
                 num_heads=8, dropout=0.3):
        super().__init__()
        self.xray_patch_proj = nn.Linear(xray_spatial_dim, proj_dim)
        self.gend_proj = nn.Linear(gend_dim, proj_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        self.fused_classifier = nn.Sequential(
            nn.Linear(proj_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, xray_spatial, gend_feat):
        patches = F.normalize(self.xray_patch_proj(xray_spatial), dim=-1)
        query = F.normalize(self.gend_proj(gend_feat), dim=-1).unsqueeze(1)
        attended, _ = self.cross_attention(query=query, key=patches, value=patches)
        fused = self.norm(attended.squeeze(1) + query.squeeze(1))
        logits = self.fused_classifier(fused)
        return logits


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
            logits = model(xf, gf)
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


def train_one_seed(seed, train_data, val_data, spatial_dim):
    set_seed(seed)
    model = PureFusedCrossAttention(xray_spatial_dim=spatial_dim,
                                     proj_dim=PROJ_DIM, num_heads=NUM_HEADS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc, best_state, epochs_since_improve = 0, None, 0

    print(f'\n{"="*60}\nSEED {seed}\n{"="*60}')
    print(f'{"Epoch":<8}{"TrLoss":<10}{"TrAcc":<10}{"ValAUC":<10}')

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

    return best_auc, best_state


def main():
    train_data = load_cached('train_features_both_cropped.pt')
    val_data   = load_cached('val_features_both_cropped.pt')
    spatial_dim = train_data[0][0].shape[1]

    os.makedirs(SAVE_DIR, exist_ok=True)
    results = []

    for seed in SEEDS:
        best_auc, best_state = train_one_seed(seed, train_data, val_data, spatial_dim)
        results.append(best_auc)
        torch.save(best_state, os.path.join(SAVE_DIR, f'fusion_no_gate_seed{seed}_best.pth'))

    results = np.array(results)
    beat_gend = int((results > GEND_ALONE_AUC).sum())

    print(f'\n{"="*60}')
    print(f'MULTI-SEED SUMMARY ({len(SEEDS)} seeds: {SEEDS})')
    print(f'{"="*60}')
    for seed, auc in zip(SEEDS, results):
        marker = '  (beats GenD)' if auc > GEND_ALONE_AUC else '  (below GenD)'
        print(f'  Seed {seed}: {auc:.3f}%{marker}')
    print(f'\n  Mean +/- std : {results.mean():.3f}% +/- {results.std():.3f}%')
    print(f'  Min / Max    : {results.min():.3f}% / {results.max():.3f}%')
    print(f'  GenD alone (reference) : {GEND_ALONE_AUC:.3f}%')
    print(f'  Seeds beating GenD alone: {beat_gend}/{len(SEEDS)}')


if __name__ == '__main__':
    main()