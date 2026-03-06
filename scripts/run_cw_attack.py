#!/usr/bin/env python3
"""Run Carlini-Wagner attack on face verification pairs.

Usage:
  python scripts/run_cw_attack.py --data data/casia_webface_extracted
  python scripts/run_cw_attack.py --data data/casia_webface_extracted --mode dodging
  python scripts/run_cw_attack.py --data data/casia_webface_extracted --steps 500

Uses the baseline ArcFace model (InsightFace w600k_r50) converted to PyTorch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.attacks import CarliniWagnerAttack, AttackMode


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_face_pair(
    data_dir: Path,
    batch: int,
    size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Load source and target face images from different identities. Returns (x, target_img) or (None, None)."""
    from PIL import Image
    from torchvision import transforms

    if not data_dir.exists():
        return None, None
    identities = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if len(identities) < 2:
        return None, None
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    paths = []
    for ident_dir in identities[:batch + 1]:  # need at least 2 different ids
        imgs = list(ident_dir.glob("*.jpg")) or list(ident_dir.glob("*.jpeg")) or list(ident_dir.glob("*.png"))
        if imgs:
            paths.append(imgs[0])
    if len(paths) < 2:
        return None, None
    source_path, target_path = paths[0], paths[1]
    img1 = Image.open(source_path).convert("RGB")
    img2 = Image.open(target_path).convert("RGB")
    x = transform(img1).unsqueeze(0).to(device)
    target_img = transform(img2).unsqueeze(0).to(device)
    if batch > 1:
        x = x.repeat(batch, 1, 1, 1)
        target_img = target_img.repeat(batch, 1, 1, 1)
    return x, target_img


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C&W attack on face verification")
    parser.add_argument("--mode", choices=["impersonation", "dodging"], default="impersonation")
    parser.add_argument("--steps", type=int, default=1000, help="Optimizer steps per c value")
    parser.add_argument("--c-steps", type=int, default=7, help="Binary search steps for c")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--size", type=int, default=112, help="Image size (H=W)")
    parser.add_argument("--onnx", type=str, default=None, help="Path to w600k_r50.onnx (default: auto-detect)")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to casia_webface_extracted (folder-per-identity face images)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = AttackMode.IMPERSONATION if args.mode == "impersonation" else AttackMode.DODGING
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (ROOT / data_path).resolve()

    _log("Loading ArcFace model...")
    try:
        from src.models import load_arcface_model
        model = load_arcface_model(
            onnx_path=args.onnx,
            device=device,
            input_bgr=False,  # Images loaded as RGB
        )
        _log("Using ArcFace (InsightFace w600k_r50) baseline model.")
    except FileNotFoundError as e:
        _log(f"ArcFace model not found: {e}")
        sys.exit(1)

    x, target_img = _load_face_pair(data_path, args.batch, args.size, device)
    if x is None:
        _log("Could not load face images. Need ≥2 identity folders for impersonation.")
        _log(f"Run: python scripts/extract_rec_to_folders.py --limit 100")
        sys.exit(1)
    _log("Using real face images from CASIA.")

    with torch.no_grad():
        source_emb = model(x)
        target_emb = model(target_img)

    attack = CarliniWagnerAttack(
        model,
        device=device,
        optimizer_steps=args.steps,
        c_steps=args.c_steps,
        lr=0.1,
        c_init=1e3,  # Higher c = more weight on attack goal (was 10; often too small for face models)
    )

    _log(f"Running C&W {args.mode} attack (batch={args.batch}, steps={args.steps})...")
    x_adv = attack(
        x,
        target_embedding=target_emb if mode == AttackMode.IMPERSONATION else None,
        source_embedding=source_emb,
        mode=mode,
    )

    l2 = (x_adv - x).pow(2).sum(dim=(1, 2, 3)).sqrt()
    _log(f"L2 perturbation: {l2.tolist()} (mean={l2.mean().item():.4f})")


if __name__ == "__main__":
    main()
