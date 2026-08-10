"""
FP/FN Analysis — Architecture 2 V1 Cross-Attention
Loads the best saved checkpoint and runs detailed error analysis.

Tells us:
- Which deepfake methods fool the model most
- What confidence scores look like for errors
- Where false positives and false negatives cluster
- Insights to guide next architecture improvements
"""

import os
import sys
import json
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
FUSION_CKPT     = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\GenD\GenD\fusion_weights\fusion_crossattn_best.pth"
HF_TOKEN        = None

FF_REAL_IMAGES  = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\extract\original_sequences\youtube\c23\images"
FF_FAKE_BASE    = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\extract\manipulated_sequences"
SPLITS_JSON     = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\splits\test.json"
FAKE_DATASETS   = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

MAX_PER_CLASS   = 500

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, FACE_XRAY_ROOT)


# ══════════════════════════════════════════════════════════════════════════════
# V1 Fusion Model (same architecture as fusion_crossattn.py)
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    def __init__(self, xray_spatial_dim=18, gend_dim=1024,
                 proj_dim=512, num_heads=8):
        super().__init__()
        self.xray_patch_proj = nn.Linear(xray_spatial_dim, proj_dim)
        self.gend_proj = nn.Linear(gend_dim, proj_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, xray_spatial, gend_feat):
        patches = self.xray_patch_proj(xray_spatial)
        patches = F.normalize(patches, dim=-1)
        query = self.gend_proj(gend_feat).unsqueeze(1)
        query = F.normalize(query, dim=-1)
        attended, attn_weights = self.cross_attention(query, patches, patches)
        fused = attended.squeeze(1) + query.squeeze(1)
        fused = self.norm(fused)
        return self.classifier(fused), attn_weights


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
    for p in net.parameters():
        p.requires_grad = False
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
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_fusion():
    print('Loading fusion checkpoint...')
    ckpt = torch.load(FUSION_CKPT, map_location='cpu', weights_only=False)
    spatial_channels = ckpt.get('spatial_channels', 18)
    proj_dim = ckpt.get('proj_dim', 512)
    num_heads = ckpt.get('num_heads', 8)
    fusion = CrossAttentionFusion(
        xray_spatial_dim=spatial_channels,
        gend_dim=1024,
        proj_dim=proj_dim,
        num_heads=num_heads
    ).to(DEVICE)
    fusion.load_state_dict(ckpt['fusion_state_dict'])
    fusion.eval()
    print(f'  Loaded. Val AUC: {ckpt["val_auc"]:.3f}%  Epoch: {ckpt["epoch"]}')
    return fusion


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
        spatial_map = y_list[0]
        B, C, H, W = spatial_map.shape
        patches = spatial_map.view(B, C, H*W).permute(0, 2, 1).squeeze(0)
    return patches


def extract_gend_features(model, image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = model.feature_extractor.preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model.feature_extractor(tensor)
            return F.normalize(feat, dim=-1).squeeze(0)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Collect test images
# ══════════════════════════════════════════════════════════════════════════════

def collect_test_images():
    with open(SPLITS_JSON) as f:
        pairs = json.load(f)
    test_ids = set()
    for pair in pairs:
        test_ids.add(pair[0])
        test_ids.add(pair[1])

    real_paths = []
    fake_by_method = {ds: [] for ds in FAKE_DATASETS}

    if os.path.exists(FF_REAL_IMAGES):
        for vid_dir in sorted(os.listdir(FF_REAL_IMAGES)):
            if vid_dir not in test_ids:
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
            if vid_dir[:3] not in test_ids:
                continue
            vid_path = os.path.join(fake_base, vid_dir)
            if os.path.isdir(vid_path):
                for frame in sorted(os.listdir(vid_path)):
                    if frame.endswith('.png'):
                        fake_by_method[ds].append(os.path.join(vid_path, frame))

    random.shuffle(real_paths)
    real_paths = real_paths[:MAX_PER_CLASS]

    all_items = [(p, 'real', 0) for p in real_paths]
    per_method = MAX_PER_CLASS // len(FAKE_DATASETS)
    for ds in FAKE_DATASETS:
        random.shuffle(fake_by_method[ds])
        for p in fake_by_method[ds][:per_method]:
            all_items.append((p, ds, 1))

    print(f'Real: {len(real_paths)}')
    for ds in FAKE_DATASETS:
        count = sum(1 for _, m, _ in all_items if m == ds)
        print(f'{ds}: {count}')
    return all_items


# ══════════════════════════════════════════════════════════════════════════════
# Main analysis
# ══════════════════════════════════════════════════════════════════════════════

def main():
    xray_net   = load_face_xray()
    gend_model = load_gend()
    fusion     = load_fusion()

    print('\nCollecting test images...')
    all_items = collect_test_images()

    results = []
    skipped = 0

    print('\nRunning inference...')
    for path, method, label in tqdm(all_items):
        xf = extract_xray_spatial(xray_net, path)
        gf = extract_gend_features(gend_model, path)

        if xf is None or gf is None:
            skipped += 1
            continue

        with torch.no_grad():
            logits, attn = fusion(xf.unsqueeze(0), gf.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)[0]
            fake_prob = float(probs[1].cpu())
            pred = 1 if fake_prob > 0.5 else 0

        results.append({
            'path':      path,
            'method':    method,
            'label':     label,
            'pred':      pred,
            'fake_prob': fake_prob,
            'correct':   pred == label
        })

    print(f'\nTotal: {len(results)}  Skipped: {skipped}')

    # ── Overall metrics ────────────────────────────────────────────────────
    labels     = [r['label']     for r in results]
    scores     = [r['fake_prob'] for r in results]
    preds      = [r['pred']      for r in results]

    auc = roc_auc_score(labels, scores) * 100
    acc = 100 * sum(r['correct'] for r in results) / len(results)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    print(f'\n{"="*60}')
    print('OVERALL RESULTS')
    print(f'{"="*60}')
    print(f'  AUC          : {auc:.3f}%')
    print(f'  Accuracy     : {acc:.2f}%')
    print(f'  True Pos     : {tp}   (fake correctly detected)')
    print(f'  True Neg     : {tn}   (real correctly identified)')
    print(f'  False Pos    : {fp}   (real wrongly called fake)  ← FP')
    print(f'  False Neg    : {fn}   (fake missed by model)      ← FN')
    print(f'  Sensitivity  : {100*tp/(tp+fn):.1f}%  (how well we catch fakes)')
    print(f'  Specificity  : {100*tn/(tn+fp):.1f}%  (how well we identify real)')

    # ── Per method analysis ────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('PER-METHOD BREAKDOWN')
    print(f'{"="*60}')
    print(f'  {"Method":<20} {"Count":>6} {"Detected":>10} {"Missed(FN)":>12} {"Detection%":>12}')
    print(f'  {"─"*62}')

    real_results = [r for r in results if r['method'] == 'real']
    real_fp = sum(1 for r in real_results if r['pred'] == 1)
    print(f'  {"Real (FP rate)":<20} {len(real_results):>6} {len(real_results)-real_fp:>10} {real_fp:>12} {100*(1-real_fp/len(real_results)):>11.1f}%')

    for ds in FAKE_DATASETS:
        ds_results = [r for r in results if r['method'] == ds]
        if not ds_results:
            continue
        detected = sum(1 for r in ds_results if r['pred'] == 1)
        missed   = len(ds_results) - detected
        det_rate = 100 * detected / len(ds_results)
        print(f'  {ds:<20} {len(ds_results):>6} {detected:>10} {missed:>12} {det_rate:>11.1f}%')

    # ── Confidence score analysis ──────────────────────────────────────────
    print(f'\n{"="*60}')
    print('CONFIDENCE SCORE ANALYSIS')
    print(f'{"="*60}')

    fp_results = [r for r in results if r['label'] == 0 and r['pred'] == 1]
    fn_results = [r for r in results if r['label'] == 1 and r['pred'] == 0]
    tp_results = [r for r in results if r['label'] == 1 and r['pred'] == 1]
    tn_results = [r for r in results if r['label'] == 0 and r['pred'] == 0]

    def avg_conf(lst):
        return np.mean([r['fake_prob'] for r in lst]) if lst else 0

    print(f'\n  True Positives  (fake correctly caught) : avg fake_prob = {avg_conf(tp_results):.3f}')
    print(f'  True Negatives  (real correctly kept)   : avg fake_prob = {avg_conf(tn_results):.3f}')
    print(f'  False Positives (real wrongly flagged)  : avg fake_prob = {avg_conf(fp_results):.3f}')
    print(f'  False Negatives (fake wrongly missed)   : avg fake_prob = {avg_conf(fn_results):.3f}')

    # ── Score distribution ─────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('SCORE DISTRIBUTION')
    print(f'{"="*60}')

    real_scores = [r['fake_prob'] for r in results if r['label'] == 0]
    fake_scores = [r['fake_prob'] for r in results if r['label'] == 1]

    print(f'\n  Real images fake_prob distribution:')
    print(f'    Mean   : {np.mean(real_scores):.3f}')
    print(f'    Std    : {np.std(real_scores):.3f}')
    print(f'    Min    : {np.min(real_scores):.3f}')
    print(f'    Max    : {np.max(real_scores):.3f}')
    print(f'    >0.9   : {sum(1 for s in real_scores if s > 0.9)} images (very wrong FP)')

    print(f'\n  Fake images fake_prob distribution:')
    print(f'    Mean   : {np.mean(fake_scores):.3f}')
    print(f'    Std    : {np.std(fake_scores):.3f}')
    print(f'    Min    : {np.min(fake_scores):.3f}')
    print(f'    Max    : {np.max(fake_scores):.3f}')
    print(f'    <0.1   : {sum(1 for s in fake_scores if s < 0.1)} images (very wrong FN)')

    # ── Hardest cases ─────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('HARDEST CASES')
    print(f'{"="*60}')

    fp_sorted = sorted(fp_results, key=lambda r: r['fake_prob'], reverse=True)[:5]
    fn_sorted = sorted(fn_results, key=lambda r: r['fake_prob'])[:5]

    print(f'\n  Top 5 worst False Positives (real images model was MOST sure were fake):')
    for r in fp_sorted:
        print(f'    fake_prob={r["fake_prob"]:.3f}  {os.path.basename(r["path"])}')

    print(f'\n  Top 5 worst False Negatives (fake images model was MOST sure were real):')
    for r in fn_sorted:
        print(f'    fake_prob={r["fake_prob"]:.3f}  [{r["method"]}]  {os.path.basename(r["path"])}')

    # ── FN breakdown by method ─────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('FALSE NEGATIVES BY METHOD (missed fakes)')
    print(f'{"="*60}')
    for ds in FAKE_DATASETS:
        ds_fn = [r for r in fn_results if r['method'] == ds]
        ds_total = [r for r in results if r['method'] == ds]
        if ds_total:
            pct = 100 * len(ds_fn) / len(ds_total)
            print(f'  {ds:<20}: {len(ds_fn):>3} missed out of {len(ds_total):>3} ({pct:.1f}% miss rate)')

    # ── Key insights ──────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('KEY INSIGHTS FOR NEXT ARCHITECTURE')
    print(f'{"="*60}')

    most_missed = max(FAKE_DATASETS,
        key=lambda ds: sum(1 for r in fn_results if r['method'] == ds) /
                       max(1, sum(1 for r in results if r['method'] == ds)))
    least_missed = min(FAKE_DATASETS,
        key=lambda ds: sum(1 for r in fn_results if r['method'] == ds) /
                       max(1, sum(1 for r in results if r['method'] == ds)))

    print(f'\n  Hardest deepfake method  : {most_missed}')
    print(f'  Easiest deepfake method  : {least_missed}')
    print(f'  FP rate (false alarms)   : {100*fp/(fp+tn):.1f}%')
    print(f'  FN rate (missed fakes)   : {100*fn/(fn+tp):.1f}%')

    if fn > fp:
        print(f'\n  → Model misses more fakes than it false-alarms on real')
        print(f'    Suggestion: Lower decision threshold below 0.5')
        print(f'    Or: Train with class weights to penalise missed fakes more')
    else:
        print(f'\n  → Model false-alarms on real more than it misses fakes')
        print(f'    Suggestion: Raise decision threshold above 0.5')
        print(f'    Or: Add more real training data')

    # Optimal threshold
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr_arr - fpr_arr)
    optimal_threshold = thresholds[optimal_idx]
    print(f'\n  Optimal decision threshold: {optimal_threshold:.3f} (instead of 0.5)')
    opt_preds = [1 if s >= optimal_threshold else 0 for s in scores]
    opt_acc = 100 * sum(p == l for p, l in zip(opt_preds, labels)) / len(labels)
    print(f'  Accuracy at optimal threshold: {opt_acc:.2f}%  (vs {acc:.2f}% at 0.5)')

    print(f'\n{"="*60}\n')


if __name__ == '__main__':
    main()