#!/usr/bin/env python3
"""Run Carlini-Wagner attack on face verification pairs.

Usage:
  python scripts/run_cw_attack.py --data data/casia_webface_extracted
  python scripts/run_cw_attack.py --data data/casia_webface_extracted --mode dodging
  python scripts/run_cw_attack.py --data data/casia_webface_extracted --steps 500
  python scripts/run_cw_attack.py --data data/casia_webface_extracted --out-dir results/cw
  python scripts/run_cw_attack.py --data data/casia_webface_extracted --source data/casia_webface_extracted/0000045/003.jpg --target data/casia_webface_extracted/0000099/074.jpg

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


def _save_visualization(
    x: torch.Tensor,
    x_adv: torch.Tensor,
    target_img: torch.Tensor | None,
    mode: AttackMode,
    l2: float,
    out_dir: Path,
) -> None:
    """Save a figure (Original | Adversarial | Target for impersonation, or 2-panel for dodging)."""
    import numpy as np
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    x_np = torch.clamp(x[0], 0, 1).cpu().numpy().transpose(1, 2, 0)
    adv_np = torch.clamp(x_adv[0], 0, 1).cpu().numpy().transpose(1, 2, 0)

    if mode == AttackMode.IMPERSONATION and target_img is not None:
        t_np = torch.clamp(target_img[0], 0, 1).cpu().numpy().transpose(1, 2, 0)
        delta = (x_adv[0] - x[0]).cpu().numpy().transpose(1, 2, 0)
        # Amplify perturbation for visibility: scale to [0,1]
        d_min, d_max = delta.min(), delta.max()
        pert_np = (delta - d_min) / (d_max - d_min + 1e-8)
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        axs[0].imshow(x_np)
        axs[0].set_title("Source")
        axs[0].axis("off")
        axs[1].imshow(adv_np)
        axs[1].set_title("Adversarial")
        axs[1].axis("off")
        axs[2].imshow(pert_np)
        axs[2].set_title("Perturbation")
        axs[2].axis("off")
        axs[3].imshow(t_np)
        axs[3].set_title("Target")
        axs[3].axis("off")
        fig.suptitle(f"C&W Impersonation (L2 = {l2:.4f})", fontsize=12)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(x_np)
        axs[0].set_title("Source")
        axs[0].axis("off")
        axs[1].imshow(adv_np)
        axs[1].set_title("Adversarial")
        axs[1].axis("off")
        fig.suptitle(f"C&W Dodging (L2 = {l2:.4f})", fontsize=12)

    plt.tight_layout()
    out_path = out_dir / f"cw_{mode.name.lower()}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    _log(f"Saved visualization to {out_path}")


def _save_staged_visualization(
    x: torch.Tensor,
    x_adv: torch.Tensor,
    target_img: torch.Tensor | None,
    mode: AttackMode,
    l2: float,
    out_dir: Path,
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> None:
    """Save a staged figure: Clean | interpolated stages | Adversarial (optionally Target)."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    x_0 = x[0:1].float()
    x_1 = x_adv[0:1].float()
    stages = []
    labels = []
    for a in alphas:
        staged = (1 - a) * x_0 + a * x_1
        stages.append(torch.clamp(staged[0], 0, 1).cpu().numpy().transpose(1, 2, 0))
        if a == 0.0:
            labels.append("Clean")
        elif a == 1.0:
            labels.append("Adversarial")
        else:
            labels.append(f"{int(a*100)}%")

    n_panels = len(stages)
    if mode == AttackMode.IMPERSONATION and target_img is not None:
        t_np = torch.clamp(target_img[0], 0, 1).cpu().numpy().transpose(1, 2, 0)
        stages.append(t_np)
        labels.append("Target")
        n_panels += 1

    fig, axs = plt.subplots(1, n_panels, figsize=(2 * n_panels, 2.5))
    if n_panels == 1:
        axs = [axs]
    for ax, img, lbl in zip(axs, stages, labels):
        ax.imshow(img)
        ax.set_title(lbl)
        ax.axis("off")
    fig.suptitle(f"C&W {mode.name.capitalize()} – Staged perturbation (L2 = {l2:.4f})", fontsize=12)
    plt.tight_layout()
    staged_path = out_dir / f"cw_{mode.name.lower()}_staged.png"
    plt.savefig(staged_path, bbox_inches="tight", dpi=150)
    plt.close()
    _log(f"Saved staged visualization to {staged_path}")


def _load_face_pair(
    data_dir: Path,
    batch: int,
    size: int,
    device: torch.device,
    source_path: Path | None = None,
    target_path: Path | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Load source and target face images. If source_path and target_path are given, use those; otherwise auto-select first images from first two identity folders. Returns (x, target_img) or (None, None)."""
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    if source_path is not None and target_path is not None:
        if not source_path.exists() or not target_path.exists():
            return None, None
        img1 = Image.open(source_path).convert("RGB")
        img2 = Image.open(target_path).convert("RGB")
        x = transform(img1).unsqueeze(0).to(device)
        target_img = transform(img2).unsqueeze(0).to(device)
        if batch > 1:
            x = x.repeat(batch, 1, 1, 1)
            target_img = target_img.repeat(batch, 1, 1, 1)
        return x, target_img

    if not data_dir.exists():
        return None, None
    identities = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if len(identities) < 2:
        return None, None
    paths = []
    for ident_dir in identities[:batch + 1]:
        imgs = list(ident_dir.glob("*.jpg")) or list(ident_dir.glob("*.jpeg")) or list(ident_dir.glob("*.png"))
        if imgs:
            paths.append(sorted(imgs)[0])
    if len(paths) < 2:
        return None, None
    src, tgt = paths[0], paths[1]
    img1 = Image.open(src).convert("RGB")
    img2 = Image.open(tgt).convert("RGB")
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
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--size", type=int, default=112, help="Image size (H=W)")
    parser.add_argument("--onnx", type=str, default=None, help="Path to w600k_r50.onnx (default: auto-detect)")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to casia_webface_extracted (folder-per-identity face images)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/cw",
        help="Directory to save visualization figure (default: results/cw)",
    )
    parser.add_argument("--source", type=str, default=None, help="Path to source image (e.g. .../0000045/003.jpg)")
    parser.add_argument("--target", type=str, default=None, help="Path to target image (e.g. .../0000099/074.jpg)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = AttackMode.IMPERSONATION if args.mode == "impersonation" else AttackMode.DODGING
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (ROOT / data_path).resolve()

    source_path = Path(args.source).resolve() if args.source else None
    target_path = Path(args.target).resolve() if args.target else None
    if (source_path is None) != (target_path is None):
        _log("Error: --source and --target must be provided together.")
        sys.exit(1)

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

    x, target_img = _load_face_pair(
        data_path, args.batch, args.size, device,
        source_path=source_path, target_path=target_path,
    )
    if x is None:
        _log("Could not load face images. Need --source and --target, or ≥2 identity folders.")
        _log("Example: --source data/casia_webface_extracted/0000045/003.jpg --target data/casia_webface_extracted/0000099/074.jpg")
        sys.exit(1)
    if source_path and target_path:
        _log(f"Source: {source_path}")
        _log(f"Target: {target_path}")
    else:
        _log("Using real face images from CASIA (auto-selected).")

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
    l2_mean = l2.mean().item()
    _log(f"L2 perturbation: {l2.tolist()} (mean={l2_mean:.4f})")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save tensors for later reuse (interpolation, etc.)
    tensors_path = out_dir / "cw_tensors.pt"
    save_dict = {"x": x.cpu(), "x_adv": x_adv.cpu(), "l2": l2_mean}
    if mode == AttackMode.IMPERSONATION and target_img is not None:
        save_dict["target_img"] = target_img.cpu()
    torch.save(save_dict, tensors_path)
    _log(f"Saved tensors to {tensors_path}")

    _save_visualization(
        x, x_adv,
        target_img if mode == AttackMode.IMPERSONATION else None,
        mode, l2_mean, out_dir,
    )
    _save_staged_visualization(
        x, x_adv,
        target_img if mode == AttackMode.IMPERSONATION else None,
        mode, l2_mean, out_dir,
    )


if __name__ == "__main__":
    main()
