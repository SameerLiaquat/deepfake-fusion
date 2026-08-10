"""
fusion_experiments_v3.py

Uses the CACHED features from fusion_crossattn_gated_v2.py's run -- no need
to reload Face X-Ray or GenD backbones, no re-extraction. Should run in a
few minutes total instead of ~2 hours.

Three experiments, all using the same cached data for a fair comparison:

  A. Gated fusion, same architecture as before, but dropout 0.3->0.5 and
     REAL early stopping (patience=6, keeps the best checkpoint instead of
     training all 30 epochs regardless). Tests whether overfitting was
     costing real AUC.

  B. Standalone GenD-only probe -- the exact architecture of
     gend_only_classifier, trained completely alone with no fusion branch
     at all. Tells us whether the "trust GenD" fallback inside the gated
     model actually reaches GenD's real ~92.5%, or whether joint training
     is starving it.

  C. Residual-correction fusion. Correction head is zero-initialized, so
     training starts mathematically identical to GenD-only and can only
     move away from that if Face X-Ray genuinely helps -- a stronger
     guarantee than gating that fusion won't do worse than GenD alone.

Usage:
    conda activate GenD
    python fusion_experiments_v3.py
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
MAX_EPOCHS = 40
PATIENCE   = 6
PROJ_DIM   = 512
NUM_HEADS  = 8

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


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


# ═══════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════

class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, xray_spatial_dim=18, gend_dim=1024, proj_dim=512,
                 num_heads=8, gate_init_bias=-2.0, dropout=0.5, hidden_dim=128):
        super().__init__()
        self.xray_patch_proj = nn.Linear(xray_spatial_dim, proj_dim)
        self.gend_proj = nn.Linear(gend_dim, proj_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        self.fused_classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        self.gend_only_classifier = nn.Sequential(
            nn.Linear(gend_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        self.gate = nn.Sequential(nn.Linear(proj_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.constant_(self.gate[-1].bias, gate_init_bias)

    def forward(self, xray_spatial, gend_feat):
        patches = F.normalize(self.xray_patch_proj(xray_spatial), dim=-1)
        query = F.normalize(self.gend_proj(gend_feat), dim=-1).unsqueeze(1)
        attended, _ = self.cross_attention(query=query, key=patches, value=patches)
        fused = self.norm(attended.squeeze(1) + query.squeeze(1))
        fused_logits = self.fused_classifier(fused)
        gend_logits = self.gend_only_classifier(gend_feat)
        gate_input = torch.cat([query.squeeze(1), attended.squeeze(1)], dim=-1)
        g = torch.sigmoid(self.gate(gate_input))
        logits = g * fused_logits + (1 - g) * gend_logits
        return logits, g.squeeze(-1)


class GenDOnlyProbe(nn.Module):
    """Isolated copy of gend_only_classifier's architecture, trained alone."""
    def __init__(self, gend_dim=1024, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gend_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, xray_spatial, gend_feat):  # xray_spatial unused, uniform call signature
        return self.net(gend_feat), None


class ResidualCorrectionFusion(nn.Module):
    def __init__(self, xray_spatial_dim=18, gend_dim=1024, proj_dim=512,
                 num_heads=8, dropout=0.5, hidden_dim=128):
        super().__init__()
        self.xray_patch_proj = nn.Linear(xray_spatial_dim, proj_dim)
        self.gend_proj = nn.Linear(gend_dim, proj_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        self.gend_only_head = nn.Sequential(
            nn.Linear(gend_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        self.correction_head = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def forward(self, xray_spatial, gend_feat):
        patches = F.normalize(self.xray_patch_proj(xray_spatial), dim=-1)
        query = F.normalize(self.gend_proj(gend_feat), dim=-1).unsqueeze(1)
        attended, _ = self.cross_attention(query=query, key=patches, value=patches)
        fused = self.norm(attended.squeeze(1) + query.squeeze(1))
        gend_logits = self.gend_only_head(gend_feat)
        correction = self.correction_head(fused)
        logits = gend_logits + correction
        return logits, correction


# ═══════════════════════════════════════════════════════════════════
# Shared train / eval loop with early stopping
# ═══════════════════════════════════════════════════════════════════

def run_epoch(model, optimizer, data, batch_size, train, balanced_sampling=True):
    model.train(train)
    data = list(data)
    if train:
        data = build_balanced_order(data) if balanced_sampling else random.sample(data, len(data))

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


def train_with_early_stopping(model, train_data, val_data, name, lr=3e-4,
                               max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_auc, best_state, epochs_since_improve = 0, None, 0

    print(f'\n{"="*60}\n{name}\n{"="*60}')
    print(f'{"Epoch":<8}{"TrLoss":<10}{"TrAcc":<10}{"ValAUC":<10}')

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc, _ = run_epoch(model, optimizer, train_data, batch_size, train=True)
        _, _, val_auc = run_epoch(model, optimizer, val_data, batch_size, train=False)
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

        if epochs_since_improve >= patience:
            print(f'  Early stop at epoch {epoch} (no improvement for {patience} epochs)')
            break

    model.load_state_dict(best_state)
    return model, best_auc


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    train_data = load_cached('train_features_facecrop_fixed.pt')
    val_data   = load_cached('val_features_facecrop_fixed.pt')
    spatial_dim = train_data[0][0].shape[1]
    results = {}

    model_a = GatedCrossAttentionFusion(xray_spatial_dim=spatial_dim, proj_dim=PROJ_DIM,
                                         num_heads=NUM_HEADS, dropout=0.5, hidden_dim=128)
    model_a, auc_a = train_with_early_stopping(
        model_a, train_data, val_data, 'EXPERIMENT A: Gated fusion (dropout 0.5, early stop)')
    results['A. Gated fusion (regularized)'] = auc_a
    torch.save(model_a.state_dict(), os.path.join(SAVE_DIR, 'exp_a_gated_regularized.pth'))

    model_b = GenDOnlyProbe(hidden_dim=128, dropout=0.3)
    model_b, auc_b = train_with_early_stopping(
        model_b, train_data, val_data, 'EXPERIMENT B: Standalone GenD-only probe')
    results['B. GenD-only probe (isolated)'] = auc_b

    model_c = ResidualCorrectionFusion(xray_spatial_dim=spatial_dim, proj_dim=PROJ_DIM,
                                        num_heads=NUM_HEADS, dropout=0.5, hidden_dim=128)
    model_c, auc_c = train_with_early_stopping(
        model_c, train_data, val_data, 'EXPERIMENT C: Residual-correction fusion')
    results['C. Residual-correction fusion'] = auc_c
    torch.save(model_c.state_dict(), os.path.join(SAVE_DIR, 'exp_c_residual.pth'))

    print(f'\n{"="*60}')
    print('FINAL COMPARISON')
    print(f'{"="*60}')
    print(f'{"GenD alone (reference)":<35}{"92.500":>10}')
    print(f'{"Gated fusion v2 (previous run)":<35}{"90.320":>10}')
    for name, auc in results.items():
        print(f'{name:<35}{auc:>10.3f}')


if __name__ == '__main__':
    main()