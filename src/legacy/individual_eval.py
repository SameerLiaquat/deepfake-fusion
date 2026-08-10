"""
Individual Model Evaluation
Tests Face X-Ray alone and GenD alone on FF++ test set.
Gives AUC and accuracy for each model separately.
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
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
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
SPLITS_JSON     = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\splits\test.json"
FAKE_DATASETS   = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

MAX_SAMPLES     = 500  # per class — increase for more thorough evaluation

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, FACE_XRAY_ROOT)

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
    print(f'  Loaded. Checkpoint AUC: {ckpt["best_auc"]:.3f}')
    return net


def load_gend():
    print('Loading GenD (DINOv3)...')
    from src.hf.modeling_gend import GenD
    token = HF_TOKEN or os.environ.get('HUGGINGFACE_TOKEN', None)
    torch.set_default_device('cpu')
    model = GenD.from_pretrained(GEND_MODEL, token=token)
    torch.set_default_device(None)
    model = model.to(DEVICE)
    model.eval()
    print('  GenD loaded.')
    return model

# ══════════════════════════════════════════════════════════════════════════════
# Inference functions
# ══════════════════════════════════════════════════════════════════════════════

def predict_face_xray(net, image_path):
    """Returns fake probability from Face X-Ray."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = net(tensor)
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=-1)
        return float(probs.view(-1)[-1].cpu())


def predict_gend(model, image_path):
    """Returns fake probability from GenD."""
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = model.feature_extractor.preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=-1)
            return float(probs.view(-1)[-1].cpu())
    except Exception as e:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# Collect test images
# ══════════════════════════════════════════════════════════════════════════════

def collect_test_images(max_per_class):
    print('\nCollecting test images...')
    with open(SPLITS_JSON) as f:
        pairs = json.load(f)
    test_ids = set()
    for pair in pairs:
        test_ids.add(pair[0])
        test_ids.add(pair[1])

    real_paths = []
    fake_paths_by_method = {ds: [] for ds in FAKE_DATASETS}

    # Real
    if os.path.exists(FF_REAL_IMAGES):
        for vid_dir in sorted(os.listdir(FF_REAL_IMAGES)):
            if vid_dir not in test_ids:
                continue
            vid_path = os.path.join(FF_REAL_IMAGES, vid_dir)
            if os.path.isdir(vid_path):
                for frame in sorted(os.listdir(vid_path)):
                    if frame.endswith('.png'):
                        real_paths.append(os.path.join(vid_path, frame))

    # Fake per method
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
                        fake_paths_by_method[ds].append(os.path.join(vid_path, frame))

    random.shuffle(real_paths)
    real_paths = real_paths[:max_per_class]

    all_fake = []
    for ds in FAKE_DATASETS:
        random.shuffle(fake_paths_by_method[ds])
        subset = fake_paths_by_method[ds][:max_per_class // len(FAKE_DATASETS)]
        all_fake.extend([(p, ds) for p in subset])

    print(f'  Real: {len(real_paths)}')
    for ds in FAKE_DATASETS:
        print(f'  {ds}: {len([x for x in all_fake if x[1]==ds])}')
    return real_paths, all_fake

# ══════════════════════════════════════════════════════════════════════════════
# Print results helper
# ══════════════════════════════════════════════════════════════════════════════

def print_results(name, labels, scores):
    auc = roc_auc_score(labels, scores) * 100
    preds = [1 if s > 0.5 else 0 for s in scores]
    acc = accuracy_score(labels, preds) * 100
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    tpr = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0  # sensitivity
    tnr = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0  # specificity

    print(f'\n{"─"*50}')
    print(f'  {name}')
    print(f'{"─"*50}')
    print(f'  AUC          : {auc:.3f}%')
    print(f'  Accuracy     : {acc:.2f}%')
    print(f'  Real correct : {tnr:.1f}%  ({tn}/{tn+fp})')
    print(f'  Fake correct : {tpr:.1f}%  ({tp}/{tp+fn})')
    return auc, acc

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Load models
    xray_net   = load_face_xray()
    gend_model = load_gend()

    # Collect test images
    real_paths, fake_paths = collect_test_images(MAX_SAMPLES)

    all_paths = [(p, 0) for p in real_paths] + [(p, ds) for p, ds in fake_paths]

    xray_scores  = []
    gend_scores  = []
    labels       = []
    method_labels = []
    skipped = 0

    print('\nRunning inference...')
    for item in tqdm(all_paths, desc='Testing'):
        if isinstance(item[1], str):  # fake with method name
            path, method = item
            label = 1
        else:
            path, label = item
            method = 'real'

        xp = predict_face_xray(xray_net, path)
        gp = predict_gend(gend_model, path)

        if xp is None or gp is None:
            skipped += 1
            continue

        xray_scores.append(xp)
        gend_scores.append(gp)
        labels.append(label)
        method_labels.append(method)

    print(f'\nTotal tested: {len(labels)}  (skipped: {skipped})')

    # ── Overall Results ────────────────────────────────────────────────────
    print(f'\n{"="*50}')
    print('INDIVIDUAL MODEL RESULTS — FF++ Test Set')
    print(f'{"="*50}')

    xray_auc, xray_acc = print_results('Face X-Ray (Baidu checkpoint)', labels, xray_scores)
    gend_auc,  gend_acc  = print_results('GenD DINOv3-L', labels, gend_scores)

    # ── Per-method breakdown ───────────────────────────────────────────────
    print(f'\n{"="*50}')
    print('PER-METHOD BREAKDOWN')
    print(f'{"="*50}')

    for ds in FAKE_DATASETS:
        ds_indices = [i for i, m in enumerate(method_labels) if m == ds]
        real_indices = [i for i, m in enumerate(method_labels) if m == 'real']
        combined = real_indices + ds_indices
        combined_labels = [labels[i] for i in combined]

        if len(ds_indices) == 0:
            continue

        xray_ds = [xray_scores[i] for i in combined]
        gend_ds  = [gend_scores[i]  for i in combined]

        xray_auc_ds = roc_auc_score(combined_labels, xray_ds) * 100
        gend_auc_ds  = roc_auc_score(combined_labels, gend_ds)  * 100

        print(f'\n  {ds}:')
        print(f'    Face X-Ray AUC : {xray_auc_ds:.2f}%')
        print(f'    GenD AUC       : {gend_auc_ds:.2f}%')

    # ── Summary ───────────────────────────────────────────────────────────
    print(f'\n{"="*50}')
    print('SUMMARY')
    print(f'{"="*50}')
    print(f'  {"Model":<30} {"AUC":>8}  {"Acc":>8}')
    print(f'  {"─"*48}')
    print(f'  {"Face X-Ray":<30} {xray_auc:>7.3f}%  {xray_acc:>7.2f}%')
    print(f'  {"GenD DINOv3":<30} {gend_auc:>7.3f}%  {gend_acc:>7.2f}%')
    print(f'  {"─"*48}')
    print(f'  {"Arch 2 Cross-Attention":<30} {"86.170":>7}%  {"~76":>7}%  (trained)')
    print(f'{"="*50}')
    print()
    print('NOTE: Cross-attention result is from training run.')
    print('Run fusion_crossattn_eval.py to evaluate saved checkpoint.')


if __name__ == '__main__':
    main()