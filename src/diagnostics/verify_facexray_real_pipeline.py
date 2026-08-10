"""
verify_facexray_real_pipeline.py  (v3)

Backbone loading confirmed correct. Normalization confirmed to HURT (keep
ToTensor()-only). Softmax question resolved (cls_layer already sums to ~0.9,
external softmax barely changes ranking either way).

This version tests the leading remaining suspect: face framing/cropping.
Sample images showed full upper-body shots with lots of background, not
tight face crops -- but Face X-Ray's training data (per the paper, Sec 3.2)
is built from landmark-based face crops. This script adds a quick face-crop
step (OpenCV Haar cascade -- zero extra dependencies) and compares AUC/mask
signal with vs without cropping, on the same small subset, before we commit
to re-running the full ~87-minute feature extraction.

Usage:
    conda activate GenD
    python verify_facexray_real_pipeline.py
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ── CONFIG ──────────────────────────────────────────────────────────────
FACE_XRAY_CKPT  = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\Face-X-Ray-master\Face-X-Ray-master\result\result_default\best_model.pth.tar"
FACE_XRAY_ROOT  = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\Face-X-Ray-master\Face-X-Ray-master"
HRNET_CONFIG    = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\Face-X-Ray-master\Face-X-Ray-master\HRNet\hrnet_config\experiments\cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"

FF_REAL_IMAGES  = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\extract\original_sequences\youtube\c23\images"
FF_FAKE_BASE    = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\extract\manipulated_sequences"
SPLITS_DIR      = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\splits"
FAKE_DATASETS   = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

MAX_EVAL_PER_CLASS = 300

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, FACE_XRAY_ROOT)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f'Using device: {DEVICE}')


def load_face_xray_real_pipeline():
    from HRNet import get_net
    net = get_net(cfg_file=HRNET_CONFIG, devices=[torch.device('cuda:0')])
    ckpt = torch.load(FACE_XRAY_CKPT, map_location='cpu', weights_only=False)
    raw_sd = ckpt['state_dict']
    filtered_sd = {k: v for k, v in raw_sd.items()
                   if not k.startswith('HRNet_layer.classifier')}
    missing, unexpected = net.load_state_dict(filtered_sd, strict=False)
    print(f'  Missing keys after load   : {len(missing)}')
    print(f'  Unexpected keys after load: {len(unexpected)}')
    net = net.to(DEVICE)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    print(f'  Loaded. Checkpoint self-reported AUC: {ckpt["best_auc"]:.3f}')
    return net


def crop_face(img_bgr, margin=1.3):
    """Detect the largest face and crop a square region around it with a
    margin. Returns None if no face is detected (caller falls back to the
    full frame)."""
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


@torch.no_grad()
def predict_real_fake(net, image_path, use_crop):
    img = cv2.imread(image_path)
    if img is None:
        return None
    detected = False
    if use_crop:
        cropped = crop_face(img)
        if cropped is not None:
            img = cropped
            detected = True
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

    y_list = net.HRNet_layer(tensor)
    resized = [F.interpolate(y, size=(64, 64), mode='bilinear', align_corners=False)
               for y in y_list]
    concat = torch.cat(resized, dim=1)
    mask_logit = net.one_channel_conv(concat)
    mask_logit = F.interpolate(mask_logit, size=(256, 256), mode='bilinear', align_corners=False)
    mask = torch.sigmoid(mask_logit)
    cls_out = net.cls_layer(mask).squeeze(0).cpu()
    return cls_out, mask.squeeze().cpu(), detected


def collect_small_val_set():
    with open(os.path.join(SPLITS_DIR, 'val.json')) as f:
        pairs = json.load(f)
    split_ids = set()
    for pair in pairs:
        split_ids.add(pair[0]); split_ids.add(pair[1])

    real_paths = []
    if os.path.exists(FF_REAL_IMAGES):
        for vid_dir in sorted(os.listdir(FF_REAL_IMAGES)):
            if vid_dir not in split_ids:
                continue
            vid_path = os.path.join(FF_REAL_IMAGES, vid_dir)
            if os.path.isdir(vid_path):
                for frame in sorted(os.listdir(vid_path)):
                    if frame.endswith('.png'):
                        real_paths.append(os.path.join(vid_path, frame))

    fake_paths = []
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

    random.shuffle(real_paths); random.shuffle(fake_paths)
    return real_paths[:MAX_EVAL_PER_CLASS], fake_paths[:MAX_EVAL_PER_CLASS]


def run_eval(net, real_paths, fake_paths, use_crop):
    labels, sm1, mask_means = [], [], []
    n_detected = 0
    items = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    random.shuffle(items)

    tag = "WITH face crop" if use_crop else "WITHOUT face crop (original)"
    for path, label in tqdm(items, desc=f'Evaluating ({tag})'):
        result = predict_real_fake(net, path, use_crop)
        if result is None:
            continue
        cls_out, mask, detected = result
        n_detected += int(detected)
        sm = torch.softmax(cls_out, dim=-1)
        labels.append(label)
        sm1.append(sm[1].item())
        mask_means.append(mask.mean().item())

    auc = roc_auc_score(labels, sm1)
    mask_means = np.array(mask_means); labels_arr = np.array(labels)
    real_m = mask_means[labels_arr == 0].mean()
    fake_m = mask_means[labels_arr == 1].mean()

    print(f'\n{"="*60}')
    print(f'RESULTS -- {tag}')
    print(f'{"="*60}')
    if use_crop:
        print(f'Faces detected: {n_detected}/{len(items)}')
    print(f'AUC (softmax[1] as fake-score): {auc*100:.3f}%')
    print(f'Mean mask intensity -- real: {real_m:.5f}   fake: {fake_m:.5f}'
          f'   ratio: {fake_m / max(real_m, 1e-8):.2f}x')


def main():
    net = load_face_xray_real_pipeline()
    print('\nCollecting a small validation subset...')
    real_paths, fake_paths = collect_small_val_set()
    print(f'  {len(real_paths)} real, {len(fake_paths)} fake\n')

    run_eval(net, real_paths, fake_paths, use_crop=False)
    run_eval(net, real_paths, fake_paths, use_crop=True)


if __name__ == '__main__':
    main()