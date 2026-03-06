# eval_bin_arcface.py

import os

# --- Set env vars ---
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("ORT_DISABLE_CPU_ARENA", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")

import pickle
import numpy as np
from tqdm import tqdm

import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from sklearn.metrics import roc_curve


def load_bin(bin_path: str):
    """Load InsightFace-style eval .bin.

    Format: (bins, issame_list)
      - bins: list of encoded images (bytes)
      - issame_list: list[bool] labels for each pair
    """
    with open(bin_path, "rb") as f:
        bins, issame_list = pickle.load(f, encoding="bytes")
    return bins, np.asarray(issame_list, dtype=np.int32)


def cosine_sim(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity for embeddings.

    InsightFace models sometimes return shape (512,) or (1,512). We flatten to 1-D.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    x = x / (np.linalg.norm(x) + eps)
    y = y / (np.linalg.norm(y) + eps)
    return float(np.dot(x, y))


def decode_image(img_bytes: bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def get_embed(app: FaceAnalysis, img_bgr: np.ndarray):
    """Detection+alignment+embedding path (FaceAnalysis)."""
    faces = app.get(img_bgr)
    if not faces:
        return None
    # Choose the largest face
    faces = sorted(
        faces,
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        reverse=True,
    )
    emb = faces[0].embedding
    return emb.astype(np.float32)


def get_embed_rec_only(rec_model, img_bgr: np.ndarray):
    """Recognition-only path (skip detection).

    Many InsightFace eval .bin datasets contain already-cropped faces.
    We resize to 112x112 (model input) and directly extract embedding.

    NOTE: insightface ArcFaceONNX.get() expects both (img, face) because it can align using landmarks.
    For recognition-only on already-cropped faces, use get_feat(img) instead.
    """
    if img_bgr is None:
        return None
    img = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)

    # Preferred API for ArcFaceONNX
    if hasattr(rec_model, "get_feat"):
        emb = rec_model.get_feat(img)
    else:
        # Fallback: try calling get() with single arg; if it errors, we cannot proceed
        try:
            emb = rec_model.get(img)
        except TypeError:
            raise TypeError(
                "Recognition model API mismatch: expected get_feat(img) to exist, "
                "but it was not found and rec_model.get(img) requires extra args."
            )

    if emb is None:
        return None
    emb = np.asarray(emb)
    return emb.astype(np.float32)


def eer_and_threshold(labels: np.ndarray, scores: np.ndarray):
    """Compute EER and a corresponding threshold robustly.

    We avoid interpolating thresholds over FPR because roc_curve can contain repeated FPR values,
    which can make interpolation unstable on small subsets. Instead, we pick the ROC operating
    point that minimizes |FPR - FNR| and report:
      EER = (FPR + FNR)/2 at that point,
      threshold = thresholds[idx].

    Returns (eer, threshold). If EER cannot be computed (degenerate labels), returns (nan, nan).
    """
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

    # If we have perfect separation (common in tiny sanity subsets), choose a nicer threshold
    # between negative and positive score ranges when possible.
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


# Globals for limiting evaluation (set via CLI)
_MAX_PAIRS = 0
_SAMPLE = False
_SEED = 0
_BALANCED = False


def main(bin_path: str, use_gpu: bool = True, det_size: int = 640, rec_only: bool = False):
    ctx_id = 0 if use_gpu else -1

    app = None
    rec = None
    if rec_only:
        # Load only the recognition model (fast, robust if .bin is aligned)
        # insightface.model_zoo.get_model expects an existing path or a resolvable model name.
        # On many clusters, pretrained models are downloaded under ~/.insightface/models/...
        candidates = []
        # 1) Common default download location
        home = os.path.expanduser("~")
        candidates.append(os.path.join(home, ".insightface", "models", "buffalo_l", "w600k_r50.onnx"))
        # 2) Respect INSIGHTFACE_HOME if set
        if os.environ.get("INSIGHTFACE_HOME"):
            candidates.append(os.path.join(os.environ["INSIGHTFACE_HOME"], "models", "buffalo_l", "w600k_r50.onnx"))
            candidates.append(os.path.join(os.environ["INSIGHTFACE_HOME"], "buffalo_l", "w600k_r50.onnx"))
        # 3) If someone copied models into the working dir
        candidates.append(os.path.join(os.getcwd(), "buffalo_l", "w600k_r50.onnx"))

        model_path = None
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

        if model_path is None:
            FileNotFoundError(f"""Could not locate w600k_r50.onnx.

            Tried:
            """ + "\n".join([f"  - {p}" for p in candidates]) + f"""

            Tip:
            - Run once without --rec_only (FaceAnalysis(name='buffalo_l')) to trigger model download, or
            - Set INSIGHTFACE_HOME to the directory containing the downloaded models.
            """)

        rec = get_model(model_path)
        rec.prepare(ctx_id=ctx_id)
    else:
        # Full pipeline: detection + alignment + embedding
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    bins, issame = load_bin(bin_path)

    if len(bins) % 2 != 0:
        raise ValueError(f"Unexpected bins length (must be even): {len(bins)}")

    n_pairs = len(bins) // 2
    labels = issame[:n_pairs]

    # --- Optional: limit number of pairs for quick sanity checks ---
    global _MAX_PAIRS, _SAMPLE, _SEED, _BALANCED
    if _MAX_PAIRS and _MAX_PAIRS > 0:
        k = min(int(_MAX_PAIRS), n_pairs)
        rng = np.random.default_rng(int(_SEED))

        # Default to balanced when limiting pairs (avoids single-class subsets from LFW's grouped layout)
        use_balanced = _BALANCED or (not _SAMPLE)

        if use_balanced:
            # Balanced sampling to ensure both classes exist (LFW bins are often grouped by class)
            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]
            if pos_idx.size == 0 or neg_idx.size == 0:
                # Fallback to random sample if dataset itself is single-class
                sel = rng.choice(n_pairs, size=k, replace=False)
            else:
                k_pos = k // 2
                k_neg = k - k_pos
                k_pos = min(k_pos, pos_idx.size)
                k_neg = min(k_neg, neg_idx.size)
                sel_pos = rng.choice(pos_idx, size=k_pos, replace=False)
                sel_neg = rng.choice(neg_idx, size=k_neg, replace=False)
                sel = np.concatenate([sel_pos, sel_neg])
        elif _SAMPLE:
            sel = rng.choice(n_pairs, size=k, replace=False)
        else:
            sel = np.arange(k)

        sel = np.sort(sel)

        # materialize selected pairs into compact bins/labels
        new_bins = []
        for i in sel:
            new_bins.append(bins[2 * i])
            new_bins.append(bins[2 * i + 1])
        bins = new_bins
        labels = labels[sel]
        n_pairs = int(sel.size)

    scores = []
    kept_labels = []
    fail = 0

    for i in tqdm(range(n_pairs), desc=f"Embedding pairs from {os.path.basename(bin_path)}"):
        img1 = decode_image(bins[2 * i])
        img2 = decode_image(bins[2 * i + 1])
        if img1 is None or img2 is None:
            fail += 1
            continue

        if rec_only:
            e1 = get_embed_rec_only(rec, img1)
            e2 = get_embed_rec_only(rec, img2)
        else:
            e1 = get_embed(app, img1)
            e2 = get_embed(app, img2)
        if e1 is None or e2 is None:
            fail += 1
            continue

        scores.append(cosine_sim(e1, e2))
        kept_labels.append(int(labels[i]))

    scores = np.asarray(scores, dtype=np.float32)
    kept_labels = np.asarray(kept_labels, dtype=np.int32)

    # --- clear diagnostics before metrics ---
    n_used = int(kept_labels.size)
    pos = int((kept_labels == 1).sum())
    neg = int((kept_labels == 0).sum())

    print("\n=== Summary ===")
    print(f"bin: {bin_path}")
    print(f"pairs_total={n_pairs}  pairs_used={n_used}  detect_or_decode_fail={fail}  fail_rate={fail/max(1,n_pairs):.3f}")
    print(f"class_counts: pos(same)={pos}  neg(diff)={neg}")

    # If everything failed (common on misaligned data), avoid sklearn crash and exit gracefully
    if n_used == 0:
        print("No usable pairs (all failed decode/detection). Try: larger det_size, different model, or recognition-only (skip detection).")
        return

    eer, thr = eer_and_threshold(kept_labels, scores)
    far, frr = far_frr_at_threshold(kept_labels, scores, thr)

    print("\n=== Results ===")
    if np.isfinite(eer):
        print(f"EER={eer:.4f}  threshold@EER={thr:.4f}  FAR@thr={far:.4f}  FRR@thr={frr:.4f}")
    else:
        print("EER/threshold could not be computed (labels may be single-class after filtering).")
        print(f"FAR/FRR: FAR={far} FRR={frr}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True, help="Path to eval/*.bin (e.g., lfw.bin)")
    p.add_argument("--cpu", action="store_true", help="Force CPU (ctx_id=-1)")
    p.add_argument("--det_size", type=int, default=640)
    p.add_argument("--rec_only", action="store_true",
                   help="Skip face detection and run recognition backbone only (recommended for eval/*.bin).")
    p.add_argument("--max_pairs", type=int, default=0,
                   help="If >0, evaluate only N pairs (sanity check).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed used when --max_pairs > 0 and sampling is enabled.")
    p.add_argument("--sample", action="store_true",
                   help="If set with --max_pairs, randomly sample N pairs instead of taking the first N.")
    p.add_argument("--balanced", action="store_true",
                   help="If set with --max_pairs, sample roughly half positive and half negative pairs (recommended).")
    args = p.parse_args()

    # Configure pair limiting (module-level globals)
    _MAX_PAIRS = int(args.max_pairs)
    _SAMPLE = bool(args.sample)
    _SEED = int(args.seed)
    _BALANCED = bool(args.balanced)

    main(args.bin, use_gpu=(not args.cpu), det_size=args.det_size, rec_only=bool(args.rec_only))
