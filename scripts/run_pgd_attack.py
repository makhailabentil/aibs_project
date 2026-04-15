#!/usr/bin/env python3
"""Run PGD attack on face verification pairs.

Usage:
  python scripts/run_pgd_attack.py --data data/casia_webface_extracted
  python scripts/run_pgd_attack.py --data data/casia_webface_extracted --mode dodging
  python scripts/run_pgd_attack.py --data data/casia_webface_extracted --num_pairs 10 --eps 0.03
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.attacks import AttackMode
from src.attacks.pgd import PGDAttack


def _log(msg: str) -> None:
    print(msg, flush=True)


def _get_identities(data_dir: Path) -> list[Path]:
    """Get list of identity folders that have at least 2 images."""
    identities = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
            if len(imgs) >= 2:
                identities.append(d)
    return identities


def _load_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    """Load and preprocess a single image."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def _load_impersonation_pairs(
    data_dir: Path,
    num_pairs: int,
    size: int,
    device: torch.device,
) -> list[dict]:
    """Load pairs for impersonation attack: source and target from DIFFERENT identities."""
    identities = _get_identities(data_dir)
    if len(identities) < 2:
        return []
    
    pairs = []
    for i in range(min(num_pairs, len(identities) - 1)):
        source_dir = identities[i]
        target_dir = identities[i + 1]
        
        source_imgs = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
        target_imgs = list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.png"))
        
        if source_imgs and target_imgs:
            pairs.append({
                "source_img": _load_image(source_imgs[0], size, device),
                "target_img": _load_image(target_imgs[0], size, device),
                "source_id": source_dir.name,
                "target_id": target_dir.name,
            })
    
    return pairs


def _load_dodging_pairs(
    data_dir: Path,
    num_pairs: int,
    size: int,
    device: torch.device,
) -> list[dict]:
    """Load pairs for dodging attack: two images from the SAME identity."""
    identities = _get_identities(data_dir)
    
    pairs = []
    for i in range(min(num_pairs, len(identities))):
        ident_dir = identities[i]
        imgs = list(ident_dir.glob("*.jpg")) + list(ident_dir.glob("*.png"))
        
        if len(imgs) >= 2:
            pairs.append({
                "attack_img": _load_image(imgs[0], size, device),
                "ref_img": _load_image(imgs[1], size, device),
                "identity": ident_dir.name,
            })
    
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PGD attack on face verification")
    parser.add_argument("--mode", choices=["impersonation", "dodging"], default="impersonation")
    parser.add_argument("--steps", type=int, default=40, help="Number of PGD iterations")
    parser.add_argument("--eps", type=float, default=8/255, help="Maximum perturbation (epsilon)")
    parser.add_argument("--alpha", type=float, default=2/255, help="Step size per iteration")
    parser.add_argument("--norm", choices=["Linf", "L2"], default="Linf", help="Norm type")
    parser.add_argument("--num_pairs", type=int, default=10, help="Number of pairs to test")
    parser.add_argument("--size", type=int, default=112, help="Image size (H=W)")
    parser.add_argument("--threshold", type=float, default=0.1767, help="Verification threshold (from baseline EER)")
    parser.add_argument("--onnx", type=str, default=None, help="Path to w600k_r50.onnx")
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON")
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
            input_bgr=False,
        )
        model.eval() 
        _log("Using ArcFace (InsightFace w600k_r50) baseline model.")
    except FileNotFoundError as e:
        _log(f"ArcFace model not found: {e}")
        sys.exit(1)

    # Load appropriate pairs based on mode
    _log(f"\nLoading {args.num_pairs} pairs for {args.mode} attack...")
    
    if mode == AttackMode.IMPERSONATION:
        pairs = _load_impersonation_pairs(data_path, args.num_pairs, args.size, device)
    else:
        pairs = _load_dodging_pairs(data_path, args.num_pairs, args.size, device)
    
    if not pairs:
        _log("Could not load face pairs. Make sure you have extracted images.")
        _log("Run: python scripts/extract_rec_to_folders.py --limit 500")
        sys.exit(1)
    
    _log(f"Loaded {len(pairs)} pairs.")

    # Initialize attack
    attack = PGDAttack(
        model,
        device=device,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
        norm=args.norm,
        random_start=True,
    )

    # Run attack on each pair
    results = []
    eligible_count = 0
    eligible_success_count = 0
    
    _log(f"\n{'='*60}")
    _log(f"Running PGD {args.mode} attack")
    _log(f"Parameters: eps={args.eps:.4f}, alpha={args.alpha:.4f}, steps={args.steps}, norm={args.norm}")
    _log(f"Threshold: {args.threshold}")
    _log(f"{'='*60}\n")

    for i, pair in enumerate(pairs):
        _log(f"\n--- Pair {i+1}/{len(pairs)} ---")
        
        if mode == AttackMode.IMPERSONATION:
            x = pair["source_img"]
            target_img = pair["target_img"]
            
            with torch.no_grad():
                source_emb = model(x)
                target_emb = model(target_img)
                source_emb = F.normalize(source_emb, p=2, dim=1)
                target_emb = F.normalize(target_emb, p=2, dim=1)
                clean_sim = (source_emb * target_emb).sum(dim=1).item()
            
            _log(f"Source ID: {pair['source_id']}, Target ID: {pair['target_id']}")
            _log(f"Clean similarity to target: {clean_sim:.4f}")
            
            # Check eligibility: for impersonation, clean should be REJECTED (< threshold)
            eligible = clean_sim < args.threshold
            _log(f"Eligible (clean < threshold): {'✓' if eligible else '✗'}")
            
            if not eligible:
                _log("Skipping attack (pair already accepted without attack)")
                results.append({
                    "pair_id": i,
                    "mode": args.mode,
                    "source_id": pair["source_id"],
                    "target_id": pair["target_id"],
                    "clean_sim": clean_sim,
                    "adv_sim": None,
                    "eligible": False,
                    "success": False,
                    "linf": None,
                    "l2": None,
                })
                continue
            
            eligible_count += 1
            
            # Run attack
            x_adv = attack(x, target_embedding=target_emb, source_embedding=source_emb, mode=mode)
            
            with torch.no_grad():
                emb_adv = model(x_adv)
                emb_adv = F.normalize(emb_adv, p=2, dim=1)
                adv_sim = (emb_adv * target_emb).sum(dim=1).item()
            
            # Success: now accepted (adv_sim >= threshold)
            success = adv_sim >= args.threshold
            
            # Calculate perturbation metrics
            linf = (x_adv - x).abs().max().item()
            l2 = (x_adv - x).pow(2).sum().sqrt().item()
            
            _log(f"Adversarial similarity: {adv_sim:.4f}")
            _log(f"Linf: {linf:.6f}, L2: {l2:.4f}")
            _log(f"Attack success: {'✓' if success else '✗'}")
            
            if success:
                if eligible_success_count == 0: 
                    save_image(x, "results/pgd_impersonation_source.jpg")
                    save_image(target_img, "results/pgd_impersonation_target.jpg")
                    save_image(x_adv, "results/pgd_impersonation_adv.jpg")
                
                eligible_success_count += 1
            
            results.append({
                "pair_id": i,
                "mode": args.mode,
                "source_id": pair["source_id"],
                "target_id": pair["target_id"],
                "clean_sim": clean_sim,
                "adv_sim": adv_sim,
                "eligible": True,
                "success": success,
                "linf": linf,
                "l2": l2,
            })
            
        else:  # DODGING
            x = pair["attack_img"]
            ref_img = pair["ref_img"]
            
            with torch.no_grad():
                attack_emb = model(x)
                ref_emb = model(ref_img)
                attack_emb = F.normalize(attack_emb, p=2, dim=1)
                ref_emb = F.normalize(ref_emb, p=2, dim=1)
                clean_sim = (attack_emb * ref_emb).sum(dim=1).item()
            
            _log(f"Identity: {pair['identity']}")
            _log(f"Clean similarity (same person, different image): {clean_sim:.4f}")
            
            # Check eligibility: for dodging, clean should be ACCEPTED (>= threshold)
            eligible = clean_sim >= args.threshold
            _log(f"Eligible (clean >= threshold): {'✓' if eligible else '✗'}")
            
            if not eligible:
                _log("Skipping attack (pair already rejected without attack)")
                results.append({
                    "pair_id": i,
                    "mode": args.mode,
                    "identity": pair["identity"],
                    "clean_sim": clean_sim,
                    "adv_sim": None,
                    "eligible": False,
                    "success": False,
                    "linf": None,
                    "l2": None,
                })
                continue
            
            eligible_count += 1
            
            # Run attack
            x_adv = attack(x, source_embedding=ref_emb, mode=mode)
            
            with torch.no_grad():
                emb_adv = model(x_adv)
                emb_adv = F.normalize(emb_adv, p=2, dim=1)
                adv_sim = (emb_adv * ref_emb).sum(dim=1).item()
            
            # Success: now rejected (adv_sim < threshold)
            success = adv_sim < args.threshold
            
            # Calculate perturbation metrics
            linf = (x_adv - x).abs().max().item()
            l2 = (x_adv - x).pow(2).sum().sqrt().item()
            
            _log(f"Adversarial similarity: {adv_sim:.4f}")
            _log(f"Linf: {linf:.6f}, L2: {l2:.4f}")
            _log(f"Attack success: {'✓' if success else '✗'}")
            
            if success:
                if eligible_success_count == 0: 
                    save_image(x, "results/pgd_dodging_source.jpg")
                    save_image(x_adv, "results/pgd_dodging_adv.jpg")
                
                eligible_success_count += 1
            
            results.append({
                "pair_id": i,
                "mode": args.mode,
                "identity": pair["identity"],
                "clean_sim": clean_sim,
                "adv_sim": adv_sim,
                "eligible": True,
                "success": success,
                "linf": linf,
                "l2": l2,
            })

    # Calculate summary statistics (only for eligible pairs)
    eligible_results = [r for r in results if r["eligible"]]
    
    if eligible_results:
        eligible_success_rate = 100.0 * eligible_success_count / eligible_count
        avg_clean_sim = sum(r["clean_sim"] for r in eligible_results) / len(eligible_results)
        avg_adv_sim = sum(r["adv_sim"] for r in eligible_results) / len(eligible_results)
        avg_linf = sum(r["linf"] for r in eligible_results) / len(eligible_results)
        avg_l2 = sum(r["l2"] for r in eligible_results) / len(eligible_results)
    else:
        eligible_success_rate = 0.0
        avg_clean_sim = avg_adv_sim = avg_linf = avg_l2 = 0.0

    # Summary
    _log(f"\n{'='*60}")
    _log(f"SUMMARY: PGD {args.mode.upper()} Attack")
    _log(f"{'='*60}")
    _log(f"Total pairs tested:     {len(pairs)}")
    _log(f"Eligible pairs:         {eligible_count}")
    _log(f"Successful attacks:     {eligible_success_count}")
    _log(f"")
    _log(f"*** Eligible Success Rate: {eligible_success_rate:.1f}% ***")
    _log(f"")
    _log(f"Avg clean similarity:   {avg_clean_sim:.4f}")
    _log(f"Avg adv similarity:     {avg_adv_sim:.4f}")
    _log(f"Similarity change:      {avg_adv_sim - avg_clean_sim:+.4f}")
    _log(f"Avg Linf perturbation:  {avg_linf:.6f}")
    _log(f"Avg L2 perturbation:    {avg_l2:.4f}")
    _log(f"{'='*60}")

    # Save results
    if args.save_results:
        results_dir = ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pgd_{args.mode}_eps{args.eps:.4f}_{timestamp}.json"
        
        output = {
            "config": {
                "mode": args.mode,
                "eps": args.eps,
                "alpha": args.alpha,
                "steps": args.steps,
                "norm": args.norm,
                "threshold": args.threshold,
                "num_pairs": len(pairs),
            },
            "summary": {
                "total_pairs": len(pairs),
                "eligible_pairs": eligible_count,
                "eligible_successes": eligible_success_count,
                "eligible_success_rate": eligible_success_rate,
                "avg_clean_sim": avg_clean_sim,
                "avg_adv_sim": avg_adv_sim,
                "avg_linf": avg_linf,
                "avg_l2": avg_l2,
            },
            "results": results,
        }
        
        with open(results_dir / filename, "w") as f:
            json.dump(output, f, indent=2)
        _log(f"\nResults saved to: results/{filename}")


if __name__ == "__main__":
    main()