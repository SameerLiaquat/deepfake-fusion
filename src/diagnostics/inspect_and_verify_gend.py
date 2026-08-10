"""
inspect_and_verify_gend.py

Experiment B showed a freshly-trained classifier on GenD's extracted
CLS-token features only reaches 84.5% AUC -- 8 points below the "GenD
alone: 92.5%" reference used throughout this project. Two hypotheses,
both tested here:

  1. GenD ships with its OWN real trained classifier (paper Sec 3.1: a
     linear layer W in R^1024x2, b in R^2, trained end-to-end with the
     backbone's fine-tuned LayerNorm params) -- but extract_gend_features()
     only pulls the CLS token and never calls that real classifier. Same
     category of bug as the Face X-Ray classifier issue.

  2. GenD's paper evaluates almost everything at VIDEO level (averaging
     softmax probabilities across 32 frames per video), not frame level.
     If "92.5%" was computed video-level while fusion experiments are all
     frame-level, that's an apples-to-oranges comparison.

This inspects GenD's real module structure, tries calling its native
classifier directly, and reports AUC both frame-level and video-level
(grouped by FF++ video ID) for direct comparison.

Usage:
    conda activate GenD
    python inspect_and_verify_gend.py
"""

import os
import json
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

GEND_MODEL = "yermandy/GenD_DINOv3_L"
HF_TOKEN = None

FF_REAL_IMAGES = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\extract\original_sequences\youtube\c23\images"
FF_FAKE_BASE   = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\extract\manipulated_sequences"
SPLITS_DIR     = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\splits"
FAKE_DATASETS  = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

MAX_VIDEOS_PER_CLASS = 60  # videos, not frames -- keeps this quick

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def inspect_structure(model):
    print("=" * 70)
    print("TOP-LEVEL NAMED CHILDREN OF model")
    print("=" * 70)
    for name, module in model.named_children():
        print(f"  {name:30s} {module.__class__.__name__}")

    print("\n" + "=" * 70)
    print("PARAMETERS MATCHING LIKELY CLASSIFIER NAMES (small tensors only)")
    print("=" * 70)
    flag_terms = ['classif', 'head', 'fc', 'linear', 'logit', 'proj', 'cls']
    found = False
    for name, param in model.named_parameters():
        if any(t in name.lower() for t in flag_terms) and param.numel() < 5000:
            found = True
            print(f"  {name:55s} {tuple(param.shape)}")
    if not found:
        print("  (none found under top-level `model` -- checking feature_extractor...)")
        if hasattr(model, 'feature_extractor'):
            for name, param in model.feature_extractor.named_parameters():
                if any(t in name.lower() for t in flag_terms) and param.numel() < 5000:
                    print(f"  feature_extractor.{name:40s} {tuple(param.shape)}")


@torch.no_grad()
def try_direct_forward(model, image_path):
    """Attempts several plausible calling conventions to get GenD's own
    native classification output, not just the CLS-token embedding."""
    img = Image.open(image_path).convert('RGB')
    tensor = model.feature_extractor.preprocess(img).unsqueeze(0).to(DEVICE)

    attempts = []

    # Attempt 1: call model directly (typical HF PreTrainedModel pattern)
    try:
        out = model(tensor)
        if hasattr(out, 'logits'):
            attempts.append(('model(tensor).logits', out.logits.squeeze(0).cpu()))
        elif isinstance(out, torch.Tensor) and out.shape[-1] == 2:
            attempts.append(('model(tensor) raw tensor', out.squeeze(0).cpu()))
    except Exception as e:
        attempts.append((f'model(tensor) FAILED: {e}', None))

    # Attempt 2: explicit classifier/head attribute on the L2-normalized CLS token
    for attr_name in ['classifier', 'head', 'cls_head', 'fc']:
        if hasattr(model, attr_name):
            try:
                feat = model.feature_extractor(tensor)
                feat = F.normalize(feat, dim=-1)
                out = getattr(model, attr_name)(feat)
                attempts.append((f'model.{attr_name}(L2-norm CLS)', out.squeeze(0).cpu()))
            except Exception as e:
                attempts.append((f'model.{attr_name}(...) FAILED: {e}', None))

    return attempts


def collect_small_video_set():
    with open(os.path.join(SPLITS_DIR, 'val.json')) as f:
        pairs = json.load(f)
    split_ids = set()
    for pair in pairs:
        split_ids.add(pair[0]); split_ids.add(pair[1])
    split_ids = list(split_ids)
    random.shuffle(split_ids)

    items = []  # (path, label, video_id)
    real_videos_used, fake_videos_used = 0, 0

    for vid in split_ids:
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


def main():
    model = load_gend()
    inspect_structure(model)

    print("\n" + "=" * 70)
    print("COLLECTING A SMALL VIDEO-GROUPED VAL SET")
    print("=" * 70)
    items = collect_small_video_set()
    print(f'  {len(items)} frames across ~{len(set(v for _, _, v in items))} videos')

    print("\n" + "=" * 70)
    print("TESTING DIRECT-FORWARD CALLING CONVENTIONS (first image)")
    print("=" * 70)
    first_path = items[0][0]
    attempts = try_direct_forward(model, first_path)
    for name, result in attempts:
        if result is not None:
            print(f'  SUCCESS: {name} -> shape {tuple(result.shape)}, values {result.tolist()}')
        else:
            print(f'  {name}')

    working_method = None
    for name, result in attempts:
        if result is not None and hasattr(result, 'shape') and result.shape[-1] == 2:
            working_method = name
            break

    if working_method is None:
        print("\nNo direct classification output found via the attempted conventions.")
        print("Share src/hf/modeling_gend.py (or just its forward()/class definition)")
        print("and I'll write the exact correct call.")
        return

    print(f"\nUsing '{working_method}' for full evaluation...\n")

    frame_labels, frame_scores = [], []
    video_scores = defaultdict(list)
    video_labels = {}

    for path, label, video_id in tqdm(items, desc='Evaluating with GenD native output'):
        attempts = try_direct_forward(model, path)
        result = dict(a for a in attempts if a[1] is not None)[working_method]
        prob_fake = torch.softmax(result, dim=-1)[1].item()

        frame_labels.append(label)
        frame_scores.append(prob_fake)
        video_scores[video_id].append(prob_fake)
        video_labels[video_id] = label

    frame_auc = roc_auc_score(frame_labels, frame_scores)

    vid_ids = list(video_scores.keys())
    vid_scores_mean = [np.mean(video_scores[v]) for v in vid_ids]
    vid_labels_list = [video_labels[v] for v in vid_ids]
    video_auc = roc_auc_score(vid_labels_list, vid_scores_mean)

    print(f'\n{"="*60}')
    print(f'RESULTS -- using GenD\'s own native classifier output')
    print(f'{"="*60}')
    print(f'Frame-level AUC : {frame_auc*100:.3f}%   (n={len(frame_labels)} frames)')
    print(f'Video-level AUC : {video_auc*100:.3f}%   (n={len(vid_ids)} videos, mean-pooled)')
    print(f'\nFor comparison:')
    print(f'  "GenD alone" reference (unknown methodology) : 92.500%')
    print(f'  Experiment B (fresh probe, frame-level)       : 84.548%')


if __name__ == '__main__':
    main()