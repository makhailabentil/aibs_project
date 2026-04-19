"""
OPA Per-Epoch Robustness Tracking
==================================
Evaluates OPA attack strength at baseline and after each epoch of PGD
adversarial training. Tracks whether PGD training incidentally hardens
(or weakens) the model against OPA — a cross-attack transferability
experiment.

Checkpoint format expected (from retraining_pgd_yichiao.py):
    {
        "backbone_state_dict": ...,
        "classifier_state_dict": ...,
        "class_to_idx": ...,
        "args": ...,
    }

Usage:
    # First re-run training with per-epoch saving:
    python scripts/retraining_pgd_yichiao.py \
        --data data/casia_webface_extracted \
        --epochs 5 --batch_size 8 --max_classes 28 \
        --eps 0.031373 --alpha 0.007843 --steps 5 \
        --save_name arcface_pgd_adv_train_v3.pt \
        --save_per_epoch

    # Then run this script:
    python scripts/run_opa_epoch_tracking.py
"""

import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from insightface.model_zoo import get_model
from src.attacks.opa import OnePixelAttack
from src.attacks.utils import cosine_similarity

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

SEED         = 42
MODE         = "impersonation"       # "dodging" or "impersonation"
THRESHOLD    = 0.4002          # recalibrated threshold for retrained v2

TARGET_IMAGE  = "data/casia_webface_extracted/0000099/001.jpg"
SOURCE_IMAGES = [
    "data/casia_webface_extracted/0000045/001.jpg",
]

OPA_N       = 60
OPA_D       = 15
OPA_EPOCHS  = 30
OPA_CR      = 0.9
PRINT_EVERY = 5

BASE_MODEL_PATH = "models/w600k_r50.onnx"

# List of (label, checkpoint_path).
# None = baseline (original ONNX weights, no fine-tuning).
# If you ran with --save_per_epoch you will have one file per epoch.
CHECKPOINTS = [
    ("baseline", None),
    ("epoch_1",  "results/arcface_pgd_adv_train_v3_epoch1.pt"),
    ("epoch_2",  "results/arcface_pgd_adv_train_v3_epoch2.pt"),
    ("epoch_3",  "results/arcface_pgd_adv_train_v3_epoch3.pt"),
    ("epoch_4",  "results/arcface_pgd_adv_train_v3_epoch4.pt"),
    ("epoch_5",  "results/arcface_pgd_adv_train_v3.pt"),   # final = epoch 5
]

OUTPUT_DIR = "results/opa_epoch_tracking"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("opa_images", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------------
# Model definitions — mirror retraining_pgd_yichiao.py exactly
# so state_dict keys align when loading checkpoints
# ------------------------------------------------------------------

class ArcFaceOnnxWrapper(nn.Module):
    """Baseline: wraps the original ONNX ArcFace model unchanged."""
    def __init__(self, rec):
        super().__init__()
        self.rec = rec

    def forward(self, x):
        img = x.squeeze().permute(1, 2, 0).numpy()
        img = (img * 255).astype("uint8")
        emb = self.rec.get_feat(img)
        return torch.tensor(emb).unsqueeze(0)


class FineTunedBackboneWrapper(nn.Module):
    """
    Wraps a fine-tuned PyTorch backbone loaded from a checkpoint.
    Exposes the same forward interface as ArcFaceOnnxWrapper so
    OnePixelAttack works identically for both model types.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        # x: [1, 3, 112, 112] float in [0, 1]
        emb = self.backbone(x)              # [1, 512]
        emb = F.normalize(emb, p=2, dim=1)
        return emb


def build_baseline_model() -> nn.Module:
    rec = get_model(BASE_MODEL_PATH)
    rec.prepare(ctx_id=-1, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    model = ArcFaceOnnxWrapper(rec)
    model.eval()
    return model


def build_finetuned_model(checkpoint_path: str) -> nn.Module:
    """
    Loads backbone_state_dict from a checkpoint saved by
    retraining_pgd_yichiao.py and returns a model ready for inference.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(f"  [ckpt] {checkpoint_path}")
    print(f"  [ckpt] training args: {ckpt.get('args', {})}")

    from src.models import load_arcface_model
    backbone = load_arcface_model(
        onnx_path=None,
        device=torch.device("cpu"),
        input_bgr=False,
    )
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.eval()

    model = FineTunedBackboneWrapper(backbone)
    model.eval()
    return model


def build_model(checkpoint_path) -> nn.Module:
    if checkpoint_path is None:
        return build_baseline_model()
    return build_finetuned_model(checkpoint_path)


# ------------------------------------------------------------------
# Image loading
# ------------------------------------------------------------------

def load_image(path: str) -> torch.Tensor:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.tensor(img / 255.0).permute(2, 0, 1).float()


# ------------------------------------------------------------------
# OPA evaluation loop
# ------------------------------------------------------------------

def run_opa_on_model(model: nn.Module, label: str) -> dict:
    target_img = load_image(TARGET_IMAGE) if MODE == "impersonation" else None
    results = []

    for i, path in enumerate(SOURCE_IMAGES):
        print(f"\n  [{label}] Image {i+1}/{len(SOURCE_IMAGES)}: {path}")
        img = load_image(path)

        with torch.no_grad():
            orig_emb = model(img.unsqueeze(0))

        attack = OnePixelAttack(
            model,
            img,
            label=None,
            n=OPA_N,
            threshold=THRESHOLD,
            cr=OPA_CR,
            mode=MODE,
            target_img=target_img,
        )

        adv_img, _ = attack.perturb_img(epochs=OPA_EPOCHS, d=OPA_D, print_every=PRINT_EVERY)

        img_np = adv_img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"opa_images/adv_{label}_{MODE}_{i}.jpg", img_np)

        with torch.no_grad():
            adv_emb = model(adv_img.unsqueeze(0))

        sim_orig = float(cosine_similarity(orig_emb, adv_emb))

        if MODE == "impersonation":
            with torch.no_grad():
                target_emb = model(target_img.unsqueeze(0))
            sim_target = float(cosine_similarity(target_emb, adv_emb))
            succeeded  = sim_target > THRESHOLD
            print(f"  sim(orig,   adv) = {sim_orig:.4f}")
            print(f"  sim(target, adv) = {sim_target:.4f}  (threshold={THRESHOLD})")
        else:
            sim_target = None
            succeeded  = sim_orig < THRESHOLD
            print(f"  sim(orig,   adv) = {sim_orig:.4f}  (threshold={THRESHOLD})")

        print(f"  {'SUCCESS ✓' if succeeded else 'FAILED  ✗'}")
        results.append({
            "image":          path,
            "sim_orig_adv":   sim_orig,
            "sim_target_adv": sim_target,
            "succeeded":      succeeded,
        })

    n_success    = sum(r["succeeded"] for r in results)
    success_rate = n_success / len(results) if results else 0.0
    avg_sim      = float(np.mean([r["sim_orig_adv"] for r in results]))

    print(f"\n  [{label}] {n_success}/{len(results)} succeeded "
          f"(rate={success_rate:.1%})  avg sim(orig,adv)={avg_sim:.4f}")

    return {
        "label":            label,
        "mode":             MODE,
        "threshold":        THRESHOLD,
        "n_images":         len(results),
        "n_success":        n_success,
        "success_rate":     success_rate,
        "avg_sim_orig_adv": avg_sim,
        "per_image":        results,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("OPA Per-Epoch Robustness Tracking")
    print(f"Mode: {MODE}  |  Threshold: {THRESHOLD}")
    print(f"OPA: n={OPA_N}, d={OPA_D}, epochs={OPA_EPOCHS}")
    print("=" * 60)

    all_results = []

    for label, ckpt_path in CHECKPOINTS:
        if ckpt_path is not None and not Path(ckpt_path).exists():
            print(f"\n[SKIP] {label}: {ckpt_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        model  = build_model(ckpt_path)
        result = run_opa_on_model(model, label)
        all_results.append(result)

        out_path = os.path.join(OUTPUT_DIR, f"opa_{MODE}_{label}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY — OPA robustness over PGD training epochs")
    print(f"Mode: {MODE}  |  Threshold: {THRESHOLD}")
    print("=" * 60)
    hdr = f"{'Checkpoint':<15} {'Rate':>8} {'Succ':>9} {'Avg sim(orig,adv)':>18}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        print(
            f"{r['label']:<15} "
            f"{r['success_rate']:>7.1%} "
            f"  {r['n_success']}/{r['n_images']}     "
            f"{r['avg_sim_orig_adv']:>18.4f}"
        )

    summary_path = os.path.join(OUTPUT_DIR, f"opa_{MODE}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary: {summary_path}")

    # Interpretation
    if len(all_results) >= 2:
        baseline = all_results[0]
        final    = all_results[-1]
        delta    = final["success_rate"] - baseline["success_rate"]
        print("\n--- Interpretation ---")
        if delta < -0.05:
            print(f"OPA success rate fell by {abs(delta):.1%} → "
                  "PGD training incidentally hardened the model against OPA.")
        elif delta > 0.05:
            print(f"OPA success rate rose by {delta:.1%} → "
                  "PGD training introduced new OPA vulnerability.")
        else:
            print("OPA success rate unchanged → "
                  "PGD training is attack-specific, no cross-attack transfer.")


if __name__ == "__main__":
    main()
