"""
verify_gend_facecrop.py

GenD's own native classifier (model.model.linear, confirmed real) scored
worse (73.6% frame-level) than a freshly-trained substitute (84.5%) -- the
opposite of what should happen if it were seeing properly-framed input.
Leading hypothesis: model.feature_extractor.preprocess() does plain
resize+normalize, NOT the face-detect -> align -> crop pipeline GenD's own
paper describes as its actual preprocessing -- the same missing-crop issue
Face X-Ray had. This tests that directly: same video-grouped subset, same
native classifier, with vs without a face-crop step before GenD's own
preprocess() call.

Usage:
    conda activate GenD
    python verify_gend_facecrop.py
"""

import os
import json
import random
from collections import defaultdict

import numpy as np
import torch
import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

GEND_MODEL = "yermandy/GenD_DINOv3_L"
HF_TOKEN = None

FF_REAL_IMAGES = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\extract\original_sequences\youtube\c23\images"
FF_FAKE_BASE   = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\extract\manipulated_sequences"
SPLITS_DIR     = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\splits"
FAKE_DATASETS  = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

MAX_VIDEOS_PER_CLASS = 60

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f'Using device: {DEVICE}')


def load_gend():
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


def crop_face_pil(pil_img, margin=1.3):
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return pil_img, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w / 2, y + h / 2
    half = (max(w, h) * margin) / 2
    H, W = img_bgr.shape[:2]
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, W)), int(min(cy + half, H))
    crop_bgr = img_bgr[y0:y1, x0:x1]
    if crop_bgr.size == 0:
        return pil_img, False
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb), True


@torch.no_grad()
def get_native_logits(model, image_path, use_crop):
    img = Image.open(image_path).convert('RGB')
    detected = False
    if use_crop:
        img, detected = crop_face_pil(img)
    tensor = model.feature_extractor.preprocess(img).unsqueeze(0).to(DEVICE)
    out = model(tensor)
    logits = out if isinstance(out, torch.Tensor) else out.logits
    return logits.squeeze(0).cpu(), detected


def collect_small_video_set():
    with open(os.path.join(SPLITS_DIR, 'val.json')) as f:
        pairs = json.load(f)
    split_ids = set()
    for pair in pairs:
        split_ids.add(pair[0]); split_ids.add(pair[1])
    split_ids_list = list(split_ids)
    random.shuffle(split_ids_list)

    items = []
    real_videos_used = 0
    for vid in split_ids_list:
        if real_videos_used >= MAX_VIDEOS_PER_CLASS:
            break
        vid_path = os.path.join(FF_REAL_IMAGES, vid)
        if os.path.isdir(vid_path):
            frames = [f for f in sorted(os.listdir(vid_path)) if f.endswith('.png')]
            for frame in frames:
                items.append((os.path.join(vid_path, frame), 0, f'real_{vid}'))
            if frames:
                real_videos_used += 1

    for ds in FAKE_DATASETS:
        fake_base = os.path.join(FF_FAKE_BASE, ds, 'c23', 'images')
        if not os.path.exists(fake_base):
            continue
        count_this_ds = 0
        for vid_dir in sorted(os.listdir(fake_base)):
            if count_this_ds >= MAX_VIDEOS_PER_CLASS // len(FAKE_DATASETS):
                break
            if vid_dir[:3] not in split_ids:
                continue
            vid_path = os.path.join(fake_base, vid_dir)
            if os.path.isdir(vid_path):
                frames = [f for f in sorted(os.listdir(vid_path)) if f.endswith('.png')]
                for frame in frames:
                    items.append((os.path.join(vid_path, frame), 1, f'{ds}_{vid_dir}'))
                if frames:
                    count_this_ds += 1

    return items


def run_eval(model, items, use_crop):
    tag = "WITH face crop" if use_crop else "WITHOUT face crop (current pipeline)"
    frame_labels, frame_scores = [], []
    video_scores = defaultdict(list)
    video_labels = {}
    n_detected = 0

    for path, label, video_id in tqdm(items, desc=f'Evaluating ({tag})'):
        logits, detected = get_native_logits(model, path, use_crop)
        n_detected += int(detected)
        prob_fake = torch.softmax(logits, dim=-1)[1].item()
        frame_labels.append(label)
        frame_scores.append(prob_fake)
        video_scores[video_id].append(prob_fake)
        video_labels[video_id] = label

    frame_auc = roc_auc_score(frame_labels, frame_scores)
    vid_ids = list(video_scores.keys())
    vid_scores_mean = [np.mean(video_scores[v]) for v in vid_ids]
    vid_labels_list = [video_labels[v] for v in vid_ids]
    video_auc = roc_auc_score(vid_labels_list, vid_scores_mean)

    print(f'\n{"="*60}\nRESULTS -- {tag}\n{"="*60}')
    if use_crop:
        print(f'Faces detected: {n_detected}/{len(items)}')
    print(f'Frame-level AUC : {frame_auc*100:.3f}%')
    print(f'Video-level AUC : {video_auc*100:.3f}%')


def main():
    model = load_gend()
    print('Collecting a small video-grouped val set...')
    items = collect_small_video_set()
    print(f'  {len(items)} frames across ~{len(set(v for _, _, v in items))} videos\n')

    run_eval(model, items, use_crop=False)
    run_eval(model, items, use_crop=True)


if __name__ == '__main__':
    main()