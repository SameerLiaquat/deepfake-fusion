"""
evaluate_celebdf.py

Cross-dataset generalization test: runs the ALREADY-TRAINED fusion model
(no retraining) on Celeb-DF v2, using the exact same face-crop + Face
X-Ray + GenD pipeline validated on FF++. Compares against standalone Face
X-Ray and standalone GenD baselines on the same videos.

By default scans ALL videos in the three folders (Celeb-real, YouTube-real,
Celeb-synthesis) -- the full dataset (~6,529 videos), not just the
official 518-video benchmark subset -- since no training happens on this
data, there's no leakage concern in using all of it. Set
MAX_VIDEOS_PER_CATEGORY below to cap the run if the full ~3 hour scan is
more than you want right now.

Uses the Celeb-DF-v2 folder specifically -- its List_of_testing_videos.txt
was separately confirmed to match the official protocol exactly (518
lines, 178 real, 340 fake), for reference, though this script does not
use that file by default. The separate "Celeb-DF" folder's list did not
match and should not be used.

Usage:
    conda activate GenD
    python evaluate_celebdf.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import defaultdict

# ── CONFIG ──────────────────────────────────────────────────────────────
CELEBDF_ROOT = r"C:\Users\y5s86\Downloads\Celeb-DF-v2"

# Recommended: True uses only the official 518-video test split (178 real,
# 340 fake) -- this is the standard, published-comparable protocol, and
# what GenD's own paper's cross-dataset numbers are measured against.
# Set False to scan the full ~6,529-video pool instead (a slower,
# non-standard, supplementary robustness check -- not the headline number).
USE_OFFICIAL_SPLIT_ONLY = True

# Set to None to use every video in all three folders when
# USE_OFFICIAL_SPLIT_ONLY is False (~6,529 videos, ~3 hours).
MAX_VIDEOS_PER_CATEGORY = None

REAL_FOLDERS = ["Celeb-real", "YouTube-real"]
FAKE_FOLDERS = ["Celeb-synthesis"]

FACE_XRAY_CKPT  = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\Face-X-Ray-master\Face-X-Ray-master\result\result_default\best_model.pth.tar"
FACE_XRAY_ROOT  = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\Face-X-Ray-master\Face-X-Ray-master"
HRNET_CONFIG    = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\Face-X-Ray-master\Face-X-Ray-master\HRNet\hrnet_config\experiments\cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
GEND_MODEL      = "yermandy/GenD_DINOv3_L"
HF_TOKEN        = None

FUSION_CKPT_DIR = r"C:\Users\y5s86\LocalWork\fusion_weights"
FUSION_SEEDS    = [0, 1, 2, 3, 4]

FRAMES_PER_VIDEO = 32  # matches GenD's own paper's video-level aggregation protocol
PROJ_DIM  = 512
NUM_HEADS = 8

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, FACE_XRAY_ROOT)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f'Using device: {DEVICE}')


# ══════════════════════════════════════════════════════════════════════
# Parse the official test list (confirmed: 518 lines, 178 real, 340 fake)
# ══════════════════════════════════════════════════════════════════════

def load_official_test_filenames():
    """Returns dict: folder_name -> set of filenames in the official
    518-video test split (178 real, 340 fake), as separately confirmed
    against Celeb-DF-v2's List_of_testing_videos.txt."""
    test_list_path = os.path.join(CELEBDF_ROOT, "List_of_testing_videos.txt")
    by_folder = defaultdict(set)
    with open(test_list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, rel_path = line.split(' ', 1)
            folder, fname = rel_path.split('/', 1)
            by_folder[folder].add(fname)
    return by_folder


def scan_full_dataset():
    """Scans the three folders directly. If USE_OFFICIAL_SPLIT_ONLY, keeps
    only videos in the official 518-video test list (the standard,
    published-comparable protocol); otherwise uses every video found (or
    up to MAX_VIDEOS_PER_CATEGORY per category if set)."""
    import random
    items = []  # (full_path, label [0=real,1=fake], video_id)
    official = load_official_test_filenames() if USE_OFFICIAL_SPLIT_ONLY else None

    for folder in REAL_FOLDERS:
        folder_path = os.path.join(CELEBDF_ROOT, folder)
        if not os.path.isdir(folder_path):
            print(f'WARNING: folder not found: {folder_path}')
            continue
        videos = sorted(f for f in os.listdir(folder_path) if f.endswith('.mp4'))
        if official is not None:
            videos = [v for v in videos if v in official.get(folder, set())]
        else:
            random.Random(0).shuffle(videos)
            if MAX_VIDEOS_PER_CATEGORY:
                videos = videos[:MAX_VIDEOS_PER_CATEGORY]
        for v in videos:
            items.append((os.path.join(folder_path, v), 0, f'{folder}_{v}'))

    for folder in FAKE_FOLDERS:
        folder_path = os.path.join(CELEBDF_ROOT, folder)
        if not os.path.isdir(folder_path):
            print(f'WARNING: folder not found: {folder_path}')
            continue
        videos = sorted(f for f in os.listdir(folder_path) if f.endswith('.mp4'))
        if official is not None:
            videos = [v for v in videos if v in official.get(folder, set())]
        else:
            random.Random(0).shuffle(videos)
            if MAX_VIDEOS_PER_CATEGORY:
                videos = videos[:MAX_VIDEOS_PER_CATEGORY]
        for v in videos:
            items.append((os.path.join(folder_path, v), 1, f'{folder}_{v}'))

    return items


# ══════════════════════════════════════════════════════════════════════
# Load backbones (same corrected loading as the FF++ pipeline)
# ══════════════════════════════════════════════════════════════════════

def load_face_xray():
    print('\nLoading Face X-Ray...')
    from HRNet import get_net
    net = get_net(cfg_file=HRNET_CONFIG, devices=[torch.device('cuda:0')])
    ckpt = torch.load(FACE_XRAY_CKPT, map_location='cpu', weights_only=False)
    raw_sd = ckpt['state_dict']
    filtered_sd = {k: v for k, v in raw_sd.items() if not k.startswith('HRNet_layer.classifier')}
    missing, unexpected = net.load_state_dict(filtered_sd, strict=False)
    print(f'  Missing: {len(missing)} (expect ~2)  Unexpected: {len(unexpected)} (expect 0)')
    net = net.to(DEVICE)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
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
    gend_real_linear = model.model.linear  # confirmed real trained classifier
    return model, gend_real_linear


class PureFusedCrossAttention(nn.Module):
    """The winning no-gate architecture from fusion_no_gate_pure.py."""
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
        return self.fused_classifier(fused)


def load_fusion_ensemble(spatial_dim):
    models = []
    for seed in FUSION_SEEDS:
        path = os.path.join(FUSION_CKPT_DIR, f'fusion_no_gate_seed{seed}_best.pth')
        if not os.path.exists(path):
            print(f'WARNING: checkpoint not found, skipping: {path}')
            continue
        m = PureFusedCrossAttention(xray_spatial_dim=spatial_dim, proj_dim=PROJ_DIM, num_heads=NUM_HEADS).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        models.append(m)
    if not models:
        raise RuntimeError("No fusion checkpoints found -- check FUSION_CKPT_DIR and filenames.")
    print(f'Loaded {len(models)} fusion checkpoints for ensembling: seeds {FUSION_SEEDS}')
    return models


# ══════════════════════════════════════════════════════════════════════
# Per-frame feature extraction (in-memory frames, not file paths)
# ══════════════════════════════════════════════════════════════════════

def crop_face_bgr(img_bgr, margin=1.3):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return img_bgr
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w / 2, y + h / 2
    half = (max(w, h) * margin) / 2
    H, W = img_bgr.shape[:2]
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, W)), int(min(cy + half, H))
    crop = img_bgr[y0:y1, x0:x1]
    return crop if crop.size > 0 else img_bgr


@torch.no_grad()
def extract_xray_all(net, img_bgr_cropped):
    """Single HRNet forward pass, reused for both the fusion patches AND
    the standalone real mask+classifier pipeline -- avoids computing the
    backbone twice per frame."""
    img = cv2.resize(img_bgr_cropped, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    y_list = net.HRNet_layer(tensor)

    spatial_map = y_list[0]
    B, C, H, W = spatial_map.shape
    patches = spatial_map.view(B, C, H * W).permute(0, 2, 1).squeeze(0)

    resized = [F.interpolate(y, size=(64, 64), mode='bilinear', align_corners=False) for y in y_list]
    concat = torch.cat(resized, dim=1)
    mask_logit = net.one_channel_conv(concat)
    mask_logit = F.interpolate(mask_logit, size=(256, 256), mode='bilinear', align_corners=False)
    mask = torch.sigmoid(mask_logit)
    cls_logits = net.cls_layer(mask).squeeze(0)

    return patches, cls_logits


@torch.no_grad()
def extract_gend_cls(model, img_bgr_cropped):
    img_rgb = cv2.cvtColor(img_bgr_cropped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = model.feature_extractor.preprocess(pil_img).unsqueeze(0).to(DEVICE)
    feat = model.feature_extractor(tensor)
    return F.normalize(feat, dim=-1).squeeze(0)


def sample_frame_indices(total_frames, n):
    if total_frames <= n:
        return set(range(total_frames))
    return set(int(i * total_frames / n) for i in range(n))


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    items = scan_full_dataset()
    n_real = sum(1 for _, l, _ in items if l == 0)
    n_fake = sum(1 for _, l, _ in items if l == 1)
    if USE_OFFICIAL_SPLIT_ONLY:
        mode_note = ' (official 518-video test split -- standard, published-comparable protocol)'
    else:
        mode_note = f' (full dataset scan{f", capped at {MAX_VIDEOS_PER_CATEGORY}/category" if MAX_VIDEOS_PER_CATEGORY else ", no cap"} -- supplementary, non-standard)'
    print(f'Scanned {len(items)} videos{mode_note}: {n_real} real, {n_fake} fake')

    missing_files = [p for p, _, _ in items if not os.path.exists(p)]
    if missing_files:
        print(f'WARNING: {len(missing_files)} of {len(items)} referenced videos missing on disk.')
        print('First few missing:', missing_files[:5])
    items = [(p, l, v) for p, l, v in items if os.path.exists(p)]
    print(f'Proceeding with {len(items)} videos found on disk.\n')

    xray_net = load_face_xray()
    gend_model, gend_real_linear = load_gend()
    fusion_models = load_fusion_ensemble(spatial_dim=18)

    frame_labels = []
    frame_scores_fx, frame_scores_gend, frame_scores_fusion = [], [], []
    video_scores_fx, video_scores_gend, video_scores_fusion = defaultdict(list), defaultdict(list), defaultdict(list)
    video_labels = {}

    for video_path, label, video_id in tqdm(items, desc='Evaluating on Celeb-DF v2'):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue
        indices = sample_frame_indices(total_frames, FRAMES_PER_VIDEO)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in indices:
                cropped = crop_face_bgr(frame)

                fx_patches, fx_cls_logits = extract_xray_all(xray_net, cropped)
                gend_cls = extract_gend_cls(gend_model, cropped)

                fx_prob = torch.softmax(fx_cls_logits, dim=-1)[1].item()
                gend_logits = gend_real_linear(gend_cls.unsqueeze(0)).squeeze(0)
                gend_prob = torch.softmax(gend_logits, dim=-1)[1].item()

                fusion_probs = []
                with torch.no_grad():
                    for m in fusion_models:
                        logits = m(fx_patches.unsqueeze(0), gend_cls.unsqueeze(0))
                        fusion_probs.append(torch.softmax(logits, dim=-1)[0, 1].item())
                fusion_prob = float(np.mean(fusion_probs))

                frame_labels.append(label)
                frame_scores_fx.append(fx_prob)
                frame_scores_gend.append(gend_prob)
                frame_scores_fusion.append(fusion_prob)
                video_scores_fx[video_id].append(fx_prob)
                video_scores_gend[video_id].append(gend_prob)
                video_scores_fusion[video_id].append(fusion_prob)
                video_labels[video_id] = label

            frame_idx += 1
        cap.release()

    def report(name, frame_scores, video_scores):
        frame_auc = roc_auc_score(frame_labels, frame_scores)
        vid_ids = list(video_scores.keys())
        vid_mean = [np.mean(video_scores[v]) for v in vid_ids]
        vid_lab = [video_labels[v] for v in vid_ids]
        video_auc = roc_auc_score(vid_lab, vid_mean)
        print(f'{name:26s} frame-level AUC: {frame_auc*100:6.3f}%   video-level AUC: {video_auc*100:6.3f}%')

    print(f'\n{"="*72}')
    print(f'CELEB-DF V2 CROSS-DATASET RESULTS  ({len(items)} videos, {len(frame_labels)} sampled frames)')
    print(f'{"="*72}')
    report('Face X-Ray alone', frame_scores_fx, video_scores_fx)
    report('GenD alone', frame_scores_gend, video_scores_gend)
    report('Fusion (5-seed ensemble)', frame_scores_fusion, video_scores_fusion)
    print(f'\nFor reference, on FF++ (in-distribution):')
    print(f'  Face X-Ray alone : ~87%')
    print(f'  GenD alone       : 96.648%')
    print(f'  Fusion           : 97.241% (mean, 5 seeds)')


if __name__ == '__main__':
    main()