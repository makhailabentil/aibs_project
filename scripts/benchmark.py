#!/usr/bin/env python3
"""Benchmark: compare clean ArcFace baseline vs adversarially trained checkpoint.

Runs PGD attacks (impersonation + dodging) on both models and prints a
side-by-side comparison table.

Usage:
  python scripts/benchmark_robustness.py \
      --data  data/casia_webface_extracted \
      --adv_ckpt checkpoints/arcface_adv_ep020.pt \
      --num_pairs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.attacks import PGDAttack, AttackMode


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    t = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return t(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def _get_identities(data_dir: Path) -> list[Path]:
    return [
        d for d in sorted(data_dir.iterdir())
        if d.is_dir() and len(list(d.glob("*.jpg")) + list(d.glob("*.png"))) >= 2
    ]


def _run_attack_suite(
    model: torch.nn.Module,
    identities: list[Path],
    num_pairs: int,
    size: int,
    eps: float,
    steps: int,
    threshold: float,
    device: torch.device,
) -> dict:
    """Run both impersonation and dodging attacks, return metrics dict."""
    pgd = PGDAttack(
        model, device=device,
        eps=eps, alpha=2 * eps / steps,
        steps=steps, norm="Linf", random_start=True,
    )

    imp_eligible = imp_success = 0
    dod_eligible = dod_success = 0

    for i in range(min(num_pairs, len(identities) - 1)):
        # ---------- Impersonation ----------
        src_dir, tgt_dir = identities[i], identities[i + 1]
        src_imgs = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        tgt_imgs = list(tgt_dir.glob("*.jpg")) + list(tgt_dir.glob("*.png"))
        if src_imgs and tgt_imgs:
            x_src = _load_image(src_imgs[0], size, device)
            x_tgt = _load_image(tgt_imgs[0], size, device)
            with torch.no_grad():
                e_src = F.normalize(model(x_src), p=2, dim=1)
                e_tgt = F.normalize(model(x_tgt), p=2, dim=1)
                clean_sim = (e_src * e_tgt).sum().item()
            if clean_sim < threshold:
                imp_eligible += 1
                x_adv = pgd(x_src, target_embedding=e_tgt, source_embedding=e_src,
                            mode=AttackMode.IMPERSONATION)
                with torch.no_grad():
                    e_adv = F.normalize(model(x_adv), p=2, dim=1)
                    adv_sim = (e_adv * e_tgt).sum().item()
                if adv_sim >= threshold:
                    imp_success += 1

        # ---------- Dodging ----------
        same_dir = identities[i]
        imgs = list(same_dir.glob("*.jpg")) + list(same_dir.glob("*.png"))
        if len(imgs) >= 2:
            x1 = _load_image(imgs[0], size, device)
            x2 = _load_image(imgs[1], size, device)
            with torch.no_grad():
                e1 = F.normalize(model(x1), p=2, dim=1)
                e2 = F.normalize(model(x2), p=2, dim=1)
                clean_sim = (e1 * e2).sum().item()
            if clean_sim >= threshold:
                dod_eligible += 1
                x_adv = pgd(x1, source_embedding=e2, mode=AttackMode.DODGING)
                with torch.no_grad():
                    e_adv = F.normalize(model(x_adv), p=2, dim=1)
                    adv_sim = (e_adv * e2).sum().item()
                if adv_sim < threshold:
                    dod_success += 1

    return {
        "imp_asr":     imp_success / imp_eligible if imp_eligible else 0.0,
        "imp_eligible": imp_eligible,
        "dod_asr":     dod_success / dod_eligible if dod_eligible else 0.0,
        "dod_eligible": dod_eligible,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness benchmark: baseline vs AT model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--adv_ckpt", required=True, help="Path to adversarially trained .pt checkpoint")
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--num_pairs", type=int, default=100)
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.1767)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (ROOT / data_path).resolve()

    identities = _get_identities(data_path)
    _log(f"Found {len(identities)} identities with ≥2 images.")

    # ---- Baseline model -------------------------------------------------------
    _log("\nLoading baseline (clean-trained) ArcFace model...")
    from src.models import load_arcface_model
    baseline = load_arcface_model(onnx_path=args.onnx, device=device, input_bgr=False)
    baseline.eval()

    _log("Running attack suite on BASELINE model...")
    base_metrics = _run_attack_suite(
        baseline, identities, args.num_pairs,
        args.size, args.eps, args.steps, args.threshold, device,
    )

    # ---- Adversarially trained model ------------------------------------------
    _log(f"\nLoading adversarially trained model from {args.adv_ckpt}...")
    adv_model = load_arcface_model(onnx_path=args.onnx, device=device, input_bgr=False)
    ckpt = torch.load(args.adv_ckpt, map_location=device)
    adv_model.load_state_dict(ckpt["backbone"])
    adv_model.eval()

    _log("Running attack suite on ADVERSARIALLY TRAINED model...")
    adv_metrics = _run_attack_suite(
        adv_model, identities, args.num_pairs,
        args.size, args.eps, args.steps, args.threshold, device,
    )

    # ---- Report ---------------------------------------------------------------
    _log(f"\n{'='*65}")
    _log(f"{'ROBUSTNESS BENCHMARK':^65}")
    _log(f"eps={args.eps*255:.1f}/255, PGD-{args.steps}, threshold={args.threshold}")
    _log(f"{'='*65}")
    _log(f"{'Metric':<30} {'Baseline':>15} {'After AT':>15}")
    _log(f"{'-'*65}")

    def _pct(v): return f"{v*100:.1f}%"

    _log(f"{'Impersonation ASR':<30} {_pct(base_metrics['imp_asr']):>15} {_pct(adv_metrics['imp_asr']):>15}")
    _log(f"{'  (eligible pairs)':<30} {base_metrics['imp_eligible']:>15} {adv_metrics['imp_eligible']:>15}")
    _log(f"{'Dodging ASR':<30} {_pct(base_metrics['dod_asr']):>15} {_pct(adv_metrics['dod_asr']):>15}")
    _log(f"{'  (eligible pairs)':<30} {base_metrics['dod_eligible']:>15} {adv_metrics['dod_eligible']:>15}")
    _log(f"{'='*65}")
    _log(f"  Lower ASR = more robust against the attack")
    _log(f"{'='*65}")


if __name__ == "__main__":
    main()