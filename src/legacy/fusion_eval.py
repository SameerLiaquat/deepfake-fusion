"""
Feature Level Fusion — Face X-Ray + GenD
Tests the combined model on FF++ test set and prints AUC + accuracy.

Architecture 1: Projection + Concatenation
- Face X-Ray: 2048-dim → Linear → 512-dim → L2 norm
- GenD:       1024-dim → Linear → 512-dim → L2 norm
- Fused:      [512 + 512] = 1024-dim → Linear → 2 (real/fake)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Face X-Ray paths
FACE_XRAY_CKPT   = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\Face-X-Ray-master\Face-X-Ray-master\result\result_default\best_model.pth.tar"
FACE_XRAY_ROOT   = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\Face-X-Ray-master\Face-X-Ray-master"
HRNET_CONFIG = r"C:\Users\y5s86\OneDrive - University of Keele\Documents\KEELE\Face-X-Ray-master\Face-X-Ray-master\HRNet\hrnet_config\experiments\cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
# GenD
GEND_MODEL       = "yermandy/GenD_DINOv3_L"   # DINOv3 variant
HF_TOKEN         = None   # Set to your token string if needed e.g. "hf_xxxx"
                           # Or set env var HUGGINGFACE_TOKEN before running

# Test dataset — FF++ test split extracted frames
FF_REAL_IMAGES   = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\extract\original_sequences\youtube\c23\images"
FF_FAKE_IMAGES   = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\extract\manipulated_sequences\Deepfakes\c23\images"
SPLITS_JSON      = r"C:\Users\y5s86\Downloads\Dataset\dataset\FaceForensics++\splits\test.json"

# How many images to test (set None for all)
MAX_SAMPLES      = 500

# ──────────────────────────────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Using device: {DEVICE}')

# Add Face X-Ray to path
sys.path.insert(0, FACE_XRAY_ROOT)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load Face X-Ray backbone (HRNet) — feature extractor only
# ══════════════════════════════════════════════════════════════════════════════
def load_face_xray(ckpt_path, hrnet_config_path):
    print('\nLoading Face X-Ray...')
    from HRNet import get_net

    net = get_net(cfg_file=hrnet_config_path, devices=[torch.device('cuda:0')])
    # Replace 1000-class head with 2-class head
    net.classifier = nn.Linear(2048, 2)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    # Strip HRNet_layer. prefix if present
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('HRNet_layer.'):
            k = k[len('HRNet_layer.'):]
    new_sd[k] = v

    # Remove classifier keys from checkpoint - we replace with 2-class head
    new_sd = {k: v for k, v in new_sd.items() if not k.startswith('classifier')}
    net.load_state_dict(new_sd, strict=False)
    net = net.to(DEVICE)
    net.eval()

    best_auc = ckpt.get('best_auc', 'unknown')
    print(f'Face X-Ray loaded. Checkpoint AUC: {best_auc}')
    return net


def extract_xray_features(net, image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Run through HRNet_layer to get feature maps
        x = tensor
        # Use the HRNet_layer directly then global avg pool
        hrnet = net.HRNet_layer
        y_list = hrnet(x)
        # y_list is a list of feature maps from parallel streams
        # Use final_layer to get 2048-dim features
        y = hrnet.incre_modules[0](y_list[0])
        for i in range(len(hrnet.downsamp_modules)):
            y = hrnet.incre_modules[i+1](y_list[i+1]) + hrnet.downsamp_modules[i](y)
        y = hrnet.final_layer(y)
        # Global average pool
        feat = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        feat = feat.squeeze(0)  # [2048]

    return feat


# ══════════════════════════════════════════════════════════════════════════════
# 2. Load GenD (DINOv3) from HuggingFace
# ══════════════════════════════════════════════════════════════════════════════
def load_gend(model_name, hf_token=None):
    print(f'\nLoading GenD ({model_name}) from HuggingFace...')
    from src.hf.modeling_gend import GenD

    token = hf_token or os.environ.get('HUGGINGFACE_TOKEN', None)
    import torch
    torch.set_default_device('cpu')
    model = GenD.from_pretrained(model_name, token=token)
    torch.set_default_device(None)  # reset after loading
    model = model.to(DEVICE)
    model.eval()
    print('GenD loaded successfully.')
    return model


def extract_gend_features(model, image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = model.feature_extractor.preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Get features directly from feature_extractor
            feat = model.feature_extractor(tensor)  # [1, 1024]
            feat = F.normalize(feat, dim=-1)
            feat = feat.squeeze(0)  # [1024]

        return feat

    except Exception as e:
        print(f'  GenD feature extraction error: {e}')
        return None

# ══════════════════════════════════════════════════════════════════════════════
# 3. Fusion Model — Architecture 1: Projection + Concatenation
# ══════════════════════════════════════════════════════════════════════════════
class FusionModel(nn.Module):
    def __init__(self, xray_dim=2048, gend_dim=1024, proj_dim=512):
        super().__init__()
        self.xray_proj = nn.Linear(xray_dim, proj_dim)
        self.gend_proj  = nn.Linear(gend_dim, proj_dim)
        self.classifier = nn.Linear(proj_dim * 2, 2)

    def forward(self, xray_feat, gend_feat):
        # Project both to same dimension
        x = self.xray_proj(xray_feat)
        g = self.gend_proj(gend_feat)

        # L2 normalise so neither dominates
        x = F.normalize(x, dim=-1)
        g = F.normalize(g, dim=-1)

        # Concatenate
        fused = torch.cat([x, g], dim=-1)  # [1024]

        # Classify
        logits = self.classifier(fused)
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# 4. Collect test image paths
# ══════════════════════════════════════════════════════════════════════════════
def collect_test_images(real_base, fake_base, splits_json, max_samples=None):
    print('\nCollecting test images...')

    # Load test split IDs
    with open(splits_json) as f:
        pairs = json.load(f)
    test_ids = set()
    for pair in pairs:
        test_ids.add(pair[0])
        test_ids.add(pair[1])

    real_paths = []
    fake_paths = []

    # Real images
    if os.path.exists(real_base):
        for vid_dir in sorted(os.listdir(real_base)):
            if vid_dir not in test_ids:
                continue
            vid_path = os.path.join(real_base, vid_dir)
            if os.path.isdir(vid_path):
                for frame in sorted(os.listdir(vid_path)):
                    if frame.endswith('.png'):
                        real_paths.append(os.path.join(vid_path, frame))

    # Fake images
    if os.path.exists(fake_base):
        for vid_dir in sorted(os.listdir(fake_base)):
            if vid_dir[:3] not in test_ids:
                continue
            vid_path = os.path.join(fake_base, vid_dir)
            if os.path.isdir(vid_path):
                for frame in sorted(os.listdir(vid_path)):
                    if frame.endswith('.png'):
                        fake_paths.append(os.path.join(vid_path, frame))

    # Balance and limit
    import random
    random.shuffle(real_paths)
    random.shuffle(fake_paths)

    if max_samples:
        n = min(max_samples // 2, len(real_paths), len(fake_paths))
        real_paths = real_paths[:n]
        fake_paths = fake_paths[:n]

    print(f'Real images: {len(real_paths)}')
    print(f'Fake images: {len(fake_paths)}')
    return real_paths, fake_paths


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main evaluation
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Load models
    xray_net  = load_face_xray(FACE_XRAY_CKPT, HRNET_CONFIG)
    gend_model = load_gend(GEND_MODEL, HF_TOKEN)

    # Fusion model (randomly initialized — for testing feature quality)
    fusion = FusionModel(xray_dim=2048, gend_dim=1024, proj_dim=512).to(DEVICE)
    fusion.eval()

    # Collect test images
    real_paths, fake_paths = collect_test_images(
        FF_REAL_IMAGES, FF_FAKE_IMAGES, SPLITS_JSON, MAX_SAMPLES
    )

    all_labels = []
    all_scores = []
    all_xray_scores = []
    all_gend_scores  = []

    correct_real  = 0
    correct_fake  = 0
    skipped       = 0

    print('\nExtracting features and evaluating...\n')

    all_paths  = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]

    softmax = nn.Softmax(dim=-1)

    for image_path, label in tqdm(all_paths, desc='Evaluating'):

        # Extract features from both models
        xray_feat = extract_xray_features(xray_net, image_path)
        gend_feat  = extract_gend_features(gend_model, image_path)

        if xray_feat is None or gend_feat is None:
            skipped += 1
            continue

        # Debug — print first sample dimensions
        if len(all_labels) == 0:
            print(f'DEBUG xray_feat shape: {xray_feat.shape}')
            print(f'DEBUG gend_feat shape: {gend_feat.shape}')

        # Ensure right dimensions
        if xray_feat.shape[0] != 2048:
            skipped += 1
            continue
        if gend_feat.shape[0] != 1024:
            skipped += 1
            continue

        # Fused prediction
        with torch.no_grad():
            xray_feat_b = xray_feat.unsqueeze(0)
            gend_feat_b  = gend_feat.unsqueeze(0)
            logits = fusion(xray_feat_b, gend_feat_b)
            probs  = softmax(logits)[0]

        fake_prob = float(probs[1].cpu())
        all_scores.append(fake_prob)
        all_labels.append(label)

        # Individual model scores for comparison
        with torch.no_grad():
            xray_input = transforms.ToTensor()(
                cv2.cvtColor(cv2.resize(cv2.imread(image_path), (256, 256)), cv2.COLOR_BGR2RGB)
            ).unsqueeze(0).to(DEVICE)
            xray_out = xray_net(xray_input)
            if isinstance(xray_out, tuple):
                xray_out = xray_out[0]
            if len(all_labels) == 0:
                print(f'DEBUG xray_out shape: {xray_out.shape}')
            xray_out_s = softmax(xray_out)
            xray_prob = float(xray_out_s.view(-1)[-1].cpu())
            all_xray_scores.append(xray_prob)

            gend_img = Image.open(image_path).convert('RGB')
            gend_tensor = gend_model.feature_extractor.preprocess(gend_img).unsqueeze(0).to(DEVICE)
            gend_logits = gend_model(gend_tensor)
            if isinstance(gend_logits, tuple):
                gend_logits = gend_logits[0]
            if len(all_labels) == 0:
                print(f'DEBUG gend_logits shape: {gend_logits.shape}')
            gend_out_s = softmax(gend_logits)
            gend_prob = float(gend_out_s.view(-1)[-1].cpu())
            all_gend_scores.append(gend_prob)

        # Accuracy
        pred = 1 if fake_prob > 0.5 else 0
        if label == 0 and pred == 0:
            correct_real += 1
        elif label == 1 and pred == 1:
            correct_fake += 1

    # ── Results ───────────────────────────────────────────────────────────────
    total   = len(all_labels)
    correct = correct_real + correct_fake
    n_real  = sum(1 for l in all_labels if l == 0)
    n_fake  = sum(1 for l in all_labels if l == 1)

    print('\n' + '='*60)
    print('FEATURE LEVEL FUSION RESULTS')
    print('Architecture 1: Projection + Concatenation')
    print('='*60)
    print(f'Total tested      : {total}  (skipped: {skipped})')
    print(f'Real images       : {n_real}')
    print(f'Fake images       : {n_fake}')
    print()

    try:
        fusion_auc = roc_auc_score(all_labels, all_scores) * 100
        xray_auc   = roc_auc_score(all_labels, all_xray_scores) * 100
        gend_auc   = roc_auc_score(all_labels, all_gend_scores) * 100
    except Exception as e:
        print(f'AUC error: {e}')
        fusion_auc = xray_auc = gend_auc = 0

    acc = 100 * correct / total if total > 0 else 0

    print(f'{"Model":<30} {"AUC (%)":>10}')
    print('-'*42)
    print(f'{"Face X-Ray alone":<30} {xray_auc:>10.3f}')
    print(f'{"GenD alone":<30} {gend_auc:>10.3f}')
    print(f'{"Fusion (Concat)":<30} {fusion_auc:>10.3f}')
    print('-'*42)
    print(f'{"Fusion Accuracy":<30} {acc:>10.2f}%')
    print()
    print(f'Correct real : {correct_real}/{n_real} ({100*correct_real/n_real:.1f}%)')
    print(f'Correct fake : {correct_fake}/{n_fake} ({100*correct_fake/n_fake:.1f}%)')
    print('='*60)
    print()
    print('NOTE: Fusion model is randomly initialized.')
    print('To get true fusion benefit, train the fusion')
    print('layers on FF++ training set first.')
    print('This test shows feature quality comparison.')


if __name__ == '__main__':
    main()