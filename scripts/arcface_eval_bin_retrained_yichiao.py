#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import load_arcface_model


def load_bin(bin_path: str):
    """Load InsightFace-style eval .bin.

    Format:
      bins: list of encoded image bytes
      issame_list: list of labels (1=same, 0=different)
    """
    with open(bin_path, "rb") as f:
        bins, issame_list = pickle.load(f, encoding="bytes")
    return bins, np.asarray(issame_list, dtype=np.int32)


def decode_image(img_bytes: bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img


def cosine_sim(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    x = x / (np.linalg.norm(x) + eps)
    y = y / (np.linalg.norm(y) + eps)
    return float(np.dot(x, y))


def eer_and_threshold(labels: np.ndarray, scores: np.ndarray):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)

    uniq = np.unique(labels)
    if uniq.size < 2:
        return float("nan"), float("nan")

    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    if fpr.size == 0 or tpr.size == 0 or thr.size == 0:
        return float("nan"), float("nan")

    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh = float(thr[idx])

    if eer == 0.0:
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        if pos_scores.size > 0 and neg_scores.size > 0:
            min_pos = float(np.min(pos_scores))
            max_neg = float(np.max(neg_scores))
            if max_neg < min_pos:
                thresh = (max_neg + min_pos) / 2.0

    return eer, thresh


def far_frr_at_threshold(labels: np.ndarray, scores: np.ndarray, thresh: float):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)

    if not np.isfinite(thresh):
        return float("nan"), float("nan")

    pred = (scores >= thresh).astype(np.int32)
    imp = (labels == 0)
    gen = (labels == 1)

    far = float(((pred == 1) & imp).sum() / max(1, int(imp.sum())))
    frr = float(((pred == 0) & gen).sum() / max(1, int(gen.sum())))
    return far, frr


def preprocess_bgr_image(img_bgr: np.ndarray, size: int = 112) -> torch.Tensor:
    """Convert OpenCV BGR image to torch tensor in [0,1], shape [1,3,H,W].

    We intentionally keep BGR order here, because load_arcface_model can be
    called with input_bgr=True so no channel flip is needed.
    """
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W], BGR
    return x


@torch.no_grad()
def get_embed_torch(model: torch.nn.Module, img_bgr: np.ndarray, device: torch.device):
    if img_bgr is None:
        return None
    x = preprocess_bgr_image(img_bgr, size=112).to(device)
    emb = model(x)
    emb = F.normalize(emb, p=2, dim=1)
    emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", required=True, help="Path to eval/*.bin, e.g. data/eval/lfw.bin")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to retrained checkpoint (.pt)")
    parser.add_argument("--onnx", type=str, default=None, help="Path to w600k_r50.onnx")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--max_pairs", type=int, default=0, help="If >0, evaluate only first N pairs")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    print("Loading ArcFace PyTorch wrapper...")
    model = load_arcface_model(
        onnx_path=args.onnx,
        device=device,
        input_bgr=True,   # we feed OpenCV BGR directly
    )

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = (ROOT / ckpt_path).resolve()

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        print(f"Loaded retrained backbone from: {ckpt_path}")
    else:
        print("Using baseline ArcFace backbone (no checkpoint override).")

    model.eval()

    bin_path = Path(args.bin)
    if not bin_path.is_absolute():
        bin_path = (ROOT / bin_path).resolve()

    bins, labels = load_bin(str(bin_path))
    n_pairs = len(labels)

    if args.max_pairs > 0:
        n_pairs = min(n_pairs, int(args.max_pairs))
        bins = bins[: 2 * n_pairs]
        labels = labels[:n_pairs]

    scores = []
    kept_labels = []
    fail = 0

    for i in tqdm(range(n_pairs), desc=f"Embedding pairs from {bin_path.name}"):
        img1 = decode_image(bins[2 * i])
        img2 = decode_image(bins[2 * i + 1])

        if img1 is None or img2 is None:
            fail += 1
            continue

        e1 = get_embed_torch(model, img1, device)
        e2 = get_embed_torch(model, img2, device)

        if e1 is None or e2 is None:
            fail += 1
            continue

        scores.append(cosine_sim(e1, e2))
        kept_labels.append(int(labels[i]))

    scores = np.asarray(scores, dtype=np.float32)
    kept_labels = np.asarray(kept_labels, dtype=np.int32)

    n_used = int(kept_labels.size)
    pos = int((kept_labels == 1).sum())
    neg = int((kept_labels == 0).sum())

    print("\n=== Summary ===")
    print(f"bin: {bin_path}")
    print(f"pairs_total={n_pairs}  pairs_used={n_used}  decode_fail={fail}  fail_rate={fail/max(1,n_pairs):.3f}")
    print(f"class_counts: pos(same)={pos}  neg(diff)={neg}")

    if n_used == 0:
        print("No usable pairs.")
        return

    eer, thr = eer_and_threshold(kept_labels, scores)
    far, frr = far_frr_at_threshold(kept_labels, scores, thr)

    print("\n=== Results ===")
    if np.isfinite(eer):
        print(f"EER={eer:.4f}  threshold@EER={thr:.4f}  FAR@thr={far:.4f}  FRR@thr={frr:.4f}")
    else:
        print("EER/threshold could not be computed.")
        print(f"FAR/FRR: FAR={far} FRR={frr}")


if __name__ == "__main__":
    main()