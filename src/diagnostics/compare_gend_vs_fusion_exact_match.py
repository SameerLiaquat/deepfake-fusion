"""
compare_gend_vs_fusion_exact_match.py

Uses GenD's REAL trained classifier (model.model.linear -- confirmed via
inspect_and_verify_gend.py) applied directly to the cached GenD CLS-token
features from fusion_crossattn_gated_v3.py's run. Those cached features
are already the L2-normalized CLS token GenD's linear classifier expects,
so this needs no re-extraction, no image loading, no backbone forward
pass -- just one matrix multiply per cached sample. Fast (seconds).

This gives a perfectly matched comparison: GenD-alone and fusion evaluated
on the EXACT SAME val samples, not just "should be similar" samples.

Usage:
    conda activate GenD
    python compare_gend_vs_fusion_exact_match.py
"""

import os
import torch
from sklearn.metrics import roc_auc_score

CACHE_DIR = r"C:\Users\y5s86\LocalWork\feature_cache"
GEND_MODEL = "yermandy/GenD_DINOv3_L"
HF_TOKEN = None


def load_gend_linear_head():
    """Only need the tiny trained classifier -- no backbone forward pass
    required since we already have cached CLS-token features."""
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


def main():
    val_path = os.path.join(CACHE_DIR, 'val_features_both_cropped.pt')
    print(f'Loading {val_path} ...')
    val_data = torch.load(val_path, weights_only=False)
    print(f'  {len(val_data)} samples (exact same val set fusion was measured on)')

    linear = load_gend_linear_head()

    labels, scores = [], []
    with torch.no_grad():
        for xf, gf, label, method in val_data:
            logits = linear(gf.unsqueeze(0))
            prob_fake = torch.softmax(logits, dim=-1)[0, 1].item()
            labels.append(label)
            scores.append(prob_fake)

    auc = roc_auc_score(labels, scores)
    print(f'\n{"="*60}')
    print(f'GenD real classifier -- EXACT SAME val samples as fusion run')
    print(f'{"="*60}')
    print(f'AUC: {auc*100:.3f}%')
    print(f'\nCompare directly to:')
    print(f'  Fusion (same exact val set)      : 97.074%')
    print(f'  GenD, different 120-video subset : 96.140%  (earlier estimate)')


if __name__ == '__main__':
    main()