#!/usr/bin/env python3
"""
Benchmark: Baseline ArcFace vs Adversarially Trained Model
===========================================================
Runs the same PGD attack (impersonation + dodging) on exactly the same
10 image pairs for both models, then prints a side-by-side comparison.

Usage:
  python scripts/benchmark.py \
      --data  data/casia_webface_extracted \
      --ckpt  models/arcface_pgd_adv_train.pt

Optional flags (same defaults as run_pgd_attack.py):
  --num_pairs 10
  --eps       0.03137   (8/255)
  --steps     40
  --threshold 0.1767
  --mode      both      (impersonation | dodging | both)
  --onnx      path/to/w600k_r50.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.attacks import AttackMode
from src.attacks.pgd import PGDAttack
from src.models import load_arcface_model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)


def load_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    t = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return t(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def get_identities(data_dir: Path) -> list[Path]:
    """Return identity folders that have ≥ 2 images, sorted for reproducibility."""
    result = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
            if len(imgs) >= 2:
                result.append(d)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pair builders  (deterministic — same seed → same pairs for both models)
# ─────────────────────────────────────────────────────────────────────────────

def build_dodging_pairs(
    identities: list[Path], num_pairs: int, size: int, device: torch.device
) -> list[dict]:
    """Two images of the SAME person. Attack tries to make them look different."""
    pairs = []
    for ident_dir in identities:
        if len(pairs) >= num_pairs:
            break
        imgs = sorted(list(ident_dir.glob("*.jpg")) + list(ident_dir.glob("*.png")))
        if len(imgs) >= 2:
            pairs.append({
                "attack_img": load_image(imgs[0], size, device),
                "ref_img":    load_image(imgs[1], size, device),
                "identity":   ident_dir.name,
            })
    return pairs


def build_impersonation_pairs(
    identities: list[Path], num_pairs: int, size: int, device: torch.device
) -> list[dict]:
    """Images from DIFFERENT people. Attack tries to make them look the same."""
    pairs = []
    for i in range(min(num_pairs, len(identities) - 1)):
        src_dir = identities[i]
        tgt_dir = identities[i + 1]
        src_imgs = sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")))
        tgt_imgs = sorted(list(tgt_dir.glob("*.jpg")) + list(tgt_dir.glob("*.png")))
        if src_imgs and tgt_imgs:
            pairs.append({
                "source_img": load_image(src_imgs[0], size, device),
                "target_img": load_image(tgt_imgs[0], size, device),
                "source_id":  src_dir.name,
                "target_id":  tgt_dir.name,
            })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Per-model attack runner
# ─────────────────────────────────────────────────────────────────────────────

def run_attack_on_model(
    model: nn.Module,
    pairs: list[dict],
    mode: AttackMode,
    eps: float,
    alpha: float,
    steps: int,
    threshold: float,
    device: torch.device,
    model_label: str,
) -> dict:
    """
    Run PGD attack on every pair and return aggregate stats.
    Prints one line per pair so you can see what's happening.
    """
    model.eval()
    attack = PGDAttack(
        model, device=device,
        eps=eps, alpha=alpha, steps=steps,
        norm="Linf", random_start=True,
    )

    eligible = 0
    successes = 0
    clean_sims = []
    adv_sims   = []

    mode_label = "DODGING" if mode == AttackMode.DODGING else "IMPERSONATION"
    log(f"\n  [{model_label}] — {mode_label} ({len(pairs)} pairs)")
    log(f"  {'Pair':<6}  {'Identity / IDs':<30}  {'Clean sim':>9}  {'Adv sim':>9}  {'Eligible':>8}  {'Success':>7}")
    log(f"  {'-'*80}")

    for i, pair in enumerate(pairs):
        with torch.no_grad():
            if mode == AttackMode.DODGING:
                x   = pair["attack_img"]
                ref = pair["ref_img"]
                e_x   = F.normalize(model(x),   p=2, dim=1)
                e_ref = F.normalize(model(ref),  p=2, dim=1)
                clean_sim = (e_x * e_ref).sum().item()
                id_label  = pair["identity"]
                # eligible when the clean pair IS accepted (sim >= threshold)
                elig = clean_sim >= threshold
            else:
                x      = pair["source_img"]
                target = pair["target_img"]
                e_x    = F.normalize(model(x),      p=2, dim=1)
                e_tgt  = F.normalize(model(target), p=2, dim=1)
                clean_sim = (e_x * e_tgt).sum().item()
                id_label  = f"{pair['source_id']} → {pair['target_id']}"
                # eligible when the clean pair is REJECTED (sim < threshold)
                elig = clean_sim < threshold

        clean_sims.append(clean_sim)

        if not elig:
            log(f"  {i+1:<6}  {id_label:<30}  {clean_sim:>9.4f}  {'—':>9}  {'No':>8}  {'—':>7}")
            adv_sims.append(None)
            continue

        eligible += 1

        # Run the attack
        if mode == AttackMode.DODGING:
            x_adv = attack(x, source_embedding=e_ref, mode=mode)
            with torch.no_grad():
                e_adv    = F.normalize(model(x_adv), p=2, dim=1)
                adv_sim  = (e_adv * e_ref).sum().item()
            success = adv_sim < threshold          # dodging: pushed below threshold
        else:
            x_adv = attack(x, target_embedding=e_tgt, source_embedding=e_x, mode=mode)
            with torch.no_grad():
                e_adv    = F.normalize(model(x_adv), p=2, dim=1)
                adv_sim  = (e_adv * e_tgt).sum().item()
            success = adv_sim >= threshold         # impersonation: pushed above threshold

        adv_sims.append(adv_sim)
        if success:
            successes += 1

        log(f"  {i+1:<6}  {id_label:<30}  {clean_sim:>9.4f}  {adv_sim:>9.4f}  {'Yes':>8}  {'✓' if success else '✗':>7}")

    asr = successes / eligible if eligible > 0 else 0.0
    valid_adv = [s for s in adv_sims if s is not None]

    return {
        "eligible":      eligible,
        "successes":     successes,
        "asr":           asr,
        "avg_clean_sim": sum(clean_sims) / len(clean_sims) if clean_sims else 0.0,
        "avg_adv_sim":   sum(valid_adv)  / len(valid_adv)  if valid_adv  else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(
    mode_label: str,
    baseline: dict,
    adv: dict,
    threshold: float,
) -> None:
    W = 65
    log(f"\n{'='*W}")
    log(f"  {mode_label} ATTACK — BASELINE vs ADVERSARIALLY TRAINED")
    log(f"  threshold = {threshold}")
    log(f"{'='*W}")
    log(f"  {'Metric':<32} {'Baseline':>13} {'Adv-Trained':>13}")
    log(f"  {'-'*60}")
    log(f"  {'Eligible pairs':<32} {baseline['eligible']:>13} {adv['eligible']:>13}")
    log(f"  {'Successful attacks':<32} {baseline['successes']:>13} {adv['successes']:>13}")
    log(f"  {'Attack Success Rate (ASR)':<32} {baseline['asr']*100:>12.1f}% {adv['asr']*100:>12.1f}%")
    log(f"  {'Avg clean similarity':<32} {baseline['avg_clean_sim']:>13.4f} {adv['avg_clean_sim']:>13.4f}")
    log(f"  {'Avg adv  similarity':<32} {baseline['avg_adv_sim']:>13.4f} {adv['avg_adv_sim']:>13.4f}")

    # Interpretation hint
    delta_asr = adv['asr'] - baseline['asr']
    if delta_asr < -0.05:
        verdict = f"✓  ASR dropped by {abs(delta_asr)*100:.1f}pp — robustness improved"
    elif delta_asr > 0.05:
        verdict = f"✗  ASR rose by {delta_asr*100:.1f}pp — robustness decreased"
    else:
        verdict = "~  ASR roughly unchanged"
    log(f"\n  → {verdict}")
    log(f"{'='*W}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs adversarially trained ArcFace")
    parser.add_argument("--data",      required=True,  help="Path to casia_webface_extracted")
    parser.add_argument("--ckpt",      required=True,  help="Path to adversarially trained checkpoint (.pt)")
    parser.add_argument("--onnx",      default=None,   help="Path to w600k_r50.onnx (optional override)")
    parser.add_argument("--num_pairs", type=int,   default=10,      help="Pairs per attack mode")
    parser.add_argument("--eps",       type=float, default=8/255,   help="PGD epsilon")
    parser.add_argument("--alpha",     type=float, default=2/255,   help="PGD step size")
    parser.add_argument("--steps",     type=int,   default=40,      help="PGD iterations")
    parser.add_argument("--threshold", type=float, default=0.1767,  help="Verification threshold")
    parser.add_argument("--size",      type=int,   default=112,     help="Image size")
    parser.add_argument("--mode",      default="both",
                        choices=["impersonation", "dodging", "both"])
    args = parser.parse_args()

    # Device — MPS on MacBook, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log(f"Device: {device}")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (ROOT / data_path).resolve()

    # ── Load identities (same list for both models) ───────────────────────────
    log(f"\nScanning identity folders in {data_path.name}...")
    identities = get_identities(data_path)
    log(f"  Found {len(identities)} usable identities.")
    if len(identities) < args.num_pairs + 1:
        log(f"  WARNING: only {len(identities)} identities available, "
            f"reducing num_pairs to {len(identities) - 1}")
        args.num_pairs = len(identities) - 1

    # ── Build pairs ONCE — reused for both models ─────────────────────────────
    log(f"\nBuilding {args.num_pairs} pairs per attack mode...")
    dodging_pairs       = build_dodging_pairs      (identities, args.num_pairs, args.size, device)
    impersonation_pairs = build_impersonation_pairs(identities, args.num_pairs, args.size, device)
    log(f"  Dodging pairs:       {len(dodging_pairs)}")
    log(f"  Impersonation pairs: {len(impersonation_pairs)}")

    # ── Load BASELINE model ───────────────────────────────────────────────────
    log("\nLoading BASELINE model (clean ArcFace)...")
    baseline_model = load_arcface_model(onnx_path=args.onnx, device=device, input_bgr=False)
    baseline_model.eval()
    log("  Baseline model loaded.")

    # ── Load ADVERSARIALLY TRAINED model ─────────────────────────────────────
    log(f"\nLoading ADVERSARIALLY TRAINED model from: {args.ckpt}")
    adv_backbone = load_arcface_model(onnx_path=args.onnx, device=device, input_bgr=False)
    ckpt = torch.load(args.ckpt, map_location=device)

    # The retraining_teresa.py checkpoint saves key "backbone_state_dict"
    if "backbone_state_dict" in ckpt:
        adv_backbone.load_state_dict(ckpt["backbone_state_dict"])
        trained_epochs = ckpt.get("args", {}).get("epochs", "?")
        log(f"  Loaded backbone_state_dict (trained for {trained_epochs} epochs).")
    elif "backbone" in ckpt:
        adv_backbone.load_state_dict(ckpt["backbone"])
        log("  Loaded backbone state dict.")
    else:
        log(f"  ERROR: checkpoint keys are {list(ckpt.keys())}")
        log("  Expected 'backbone_state_dict' or 'backbone'.")
        sys.exit(1)

    adv_backbone.eval()

    # ── Run attacks ───────────────────────────────────────────────────────────
    run_dodging       = args.mode in ("dodging",       "both")
    run_impersonation = args.mode in ("impersonation", "both")

    alpha = args.alpha

    if run_dodging and dodging_pairs:
        log(f"\n{'─'*65}")
        log("DODGING ATTACK  (same-person pairs, attacker tries to evade)")
        log(f"{'─'*65}")

        log("\n  Running on BASELINE model...")
        base_dod = run_attack_on_model(
            baseline_model, dodging_pairs, AttackMode.DODGING,
            args.eps, alpha, args.steps, args.threshold, device, "BASELINE"
        )
        log("\n  Running on ADVERSARIALLY TRAINED model...")
        adv_dod = run_attack_on_model(
            adv_backbone, dodging_pairs, AttackMode.DODGING,
            args.eps, alpha, args.steps, args.threshold, device, "ADV-TRAINED"
        )
        print_comparison("DODGING", base_dod, adv_dod, args.threshold)

    if run_impersonation and impersonation_pairs:
        log(f"\n{'─'*65}")
        log("IMPERSONATION ATTACK  (different-person pairs, attacker tries to spoof)")
        log(f"{'─'*65}")

        log("\n  Running on BASELINE model...")
        base_imp = run_attack_on_model(
            baseline_model, impersonation_pairs, AttackMode.IMPERSONATION,
            args.eps, alpha, args.steps, args.threshold, device, "BASELINE"
        )
        log("\n  Running on ADVERSARIALLY TRAINED model...")
        adv_imp = run_attack_on_model(
            adv_backbone, impersonation_pairs, AttackMode.IMPERSONATION,
            args.eps, alpha, args.steps, args.threshold, device, "ADV-TRAINED"
        )
        print_comparison("IMPERSONATION", base_imp, adv_imp, args.threshold)

    log("\nDone.")


if __name__ == "__main__":
    main()