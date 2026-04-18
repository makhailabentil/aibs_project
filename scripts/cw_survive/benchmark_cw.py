#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.attacks import CarliniWagnerAttack, AttackMode
from src.models import load_arcface_model


def log(msg: str):
    print(msg, flush=True)


def load_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    t = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return t(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def get_identities(data_dir: Path) -> list[Path]:
    result = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
            if len(imgs) >= 2:
                result.append(d)
    return result


def build_dodging_pairs(identities, num_pairs, size, device):
    pairs = []
    for ident_dir in identities:
        if len(pairs) >= num_pairs:
            break
        imgs = sorted(list(ident_dir.glob("*.jpg")) + list(ident_dir.glob("*.png")))
        if len(imgs) >= 2:
            pairs.append({
                "attack_img": load_image(imgs[0], size, device),
                "ref_img": load_image(imgs[1], size, device),
                "identity": ident_dir.name,
            })
    return pairs


def build_impersonation_pairs(identities, num_pairs, size, device):
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
                "source_id": src_dir.name,
                "target_id": tgt_dir.name,
            })
    return pairs


def run_cw_on_model(model, pairs, mode, threshold, device, model_label,
                    c_init, lr, steps, c_steps):
    model.eval()
    attack = CarliniWagnerAttack(
        model, device=device,
        c_init=c_init, lr=lr,
        optimizer_steps=steps, c_steps=c_steps,
    )

    eligible = 0
    successes = 0
    clean_sims = []
    adv_sims = []
    l2_values = []

    mode_label = "DODGING" if mode == AttackMode.DODGING else "IMPERSONATION"
    log(f"\n  [{model_label}] {mode_label} ({len(pairs)} pairs)")
    log(f"  {'Pair':<6}  {'IDs':<30}  {'Clean':>9}  {'Adv':>9}  {'L2':>9}  {'Elig':>6}  {'OK':>5}")
    log(f"  {'-'*85}")

    for i, pair in enumerate(pairs):
        with torch.no_grad():
            if mode == AttackMode.DODGING:
                x = pair["attack_img"]
                ref = pair["ref_img"]
                e_x = F.normalize(model(x), p=2, dim=1)
                e_ref = F.normalize(model(ref), p=2, dim=1)
                clean_sim = (e_x * e_ref).sum().item()
                id_label = pair["identity"]
                elig = clean_sim >= threshold
            else:
                x = pair["source_img"]
                target = pair["target_img"]
                e_x = F.normalize(model(x), p=2, dim=1)
                e_tgt = F.normalize(model(target), p=2, dim=1)
                clean_sim = (e_x * e_tgt).sum().item()
                id_label = f"{pair['source_id']}->{pair['target_id']}"
                elig = clean_sim < threshold

        clean_sims.append(clean_sim)

        if not elig:
            log(f"  {i+1:<6}  {id_label:<30}  {clean_sim:>9.4f}  {'--':>9}  {'--':>9}  {'No':>6}  {'--':>5}")
            continue

        eligible += 1

        if mode == AttackMode.DODGING:
            x_adv = attack(x, source_embedding=e_ref, mode=mode)
            with torch.no_grad():
                e_adv = F.normalize(model(x_adv), p=2, dim=1)
                adv_sim = (e_adv * e_ref).sum().item()
            success = adv_sim < threshold
        else:
            x_adv = attack(x, target_embedding=e_tgt, source_embedding=e_x, mode=mode)
            with torch.no_grad():
                e_adv = F.normalize(model(x_adv), p=2, dim=1)
                adv_sim = (e_adv * e_tgt).sum().item()
            success = adv_sim >= threshold

        l2 = (x_adv - x).pow(2).sum().sqrt().item()
        adv_sims.append(adv_sim)
        l2_values.append(l2)
        if success:
            successes += 1

        mark = "Y" if success else "N"
        log(f"  {i+1:<6}  {id_label:<30}  {clean_sim:>9.4f}  {adv_sim:>9.4f}  {l2:>9.4f}  {'Yes':>6}  {mark:>5}")

    asr = successes / eligible if eligible > 0 else 0.0
    return {
        "eligible": eligible,
        "successes": successes,
        "asr": asr,
        "avg_clean_sim": sum(clean_sims) / len(clean_sims) if clean_sims else 0.0,
        "avg_adv_sim": sum(adv_sims) / len(adv_sims) if adv_sims else 0.0,
        "avg_l2": sum(l2_values) / len(l2_values) if l2_values else 0.0,
    }


def print_comparison(mode_label, baseline, adv, threshold):
    W = 70
    log(f"\n{'='*W}")
    log(f"  {mode_label} — BASELINE vs ADV-TRAINED (threshold={threshold})")
    log(f"{'='*W}")
    log(f"  {'Metric':<35} {'Baseline':>14} {'Adv-Trained':>14}")
    log(f"  {'-'*65}")
    log(f"  {'Eligible pairs':<35} {baseline['eligible']:>14} {adv['eligible']:>14}")
    log(f"  {'Successful attacks':<35} {baseline['successes']:>14} {adv['successes']:>14}")
    log(f"  {'Attack Success Rate (ASR)':<35} {baseline['asr']*100:>13.1f}% {adv['asr']*100:>13.1f}%")
    log(f"  {'Avg clean similarity':<35} {baseline['avg_clean_sim']:>14.4f} {adv['avg_clean_sim']:>14.4f}")
    log(f"  {'Avg adv similarity':<35} {baseline['avg_adv_sim']:>14.4f} {adv['avg_adv_sim']:>14.4f}")
    log(f"  {'Avg L2 perturbation':<35} {baseline['avg_l2']:>14.4f} {adv['avg_l2']:>14.4f}")

    delta_asr = adv["asr"] - baseline["asr"]
    if delta_asr < -0.05:
        verdict = f"ASR dropped by {abs(delta_asr)*100:.1f}pp"
    elif delta_asr > 0.05:
        verdict = f"ASR rose by {delta_asr*100:.1f}pp"
    else:
        verdict = "ASR roughly unchanged"
    log(f"\n  -> {verdict}")
    log(f"{'='*W}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--num-pairs", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.1767)
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--mode", default="both", choices=["impersonation", "dodging", "both"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--c-steps", type=int, default=7)
    parser.add_argument("--c-init", type=float, default=1e3)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

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

    identities = get_identities(data_path)
    log(f"Found {len(identities)} identities")

    dodging_pairs = build_dodging_pairs(identities, args.num_pairs, args.size, device)
    impersonation_pairs = build_impersonation_pairs(identities, args.num_pairs, args.size, device)

    log("\nLoading BASELINE model...")
    baseline_model = load_arcface_model(onnx_path=args.onnx, device=device, input_bgr=False)
    baseline_model.eval()

    log(f"Loading ADV-TRAINED model from {args.ckpt}...")
    adv_model = load_arcface_model(onnx_path=args.onnx, device=device, input_bgr=False)
    ckpt = torch.load(args.ckpt, map_location=device)
    if "backbone_state_dict" in ckpt:
        adv_model.load_state_dict(ckpt["backbone_state_dict"])
    elif "backbone" in ckpt:
        adv_model.load_state_dict(ckpt["backbone"])
    else:
        log(f"ERROR: checkpoint keys are {list(ckpt.keys())}")
        sys.exit(1)
    adv_model.eval()

    cw_kwargs = dict(c_init=args.c_init, lr=args.lr, steps=args.steps, c_steps=args.c_steps)

    run_dodging = args.mode in ("dodging", "both")
    run_imp = args.mode in ("impersonation", "both")

    if run_dodging and dodging_pairs:
        log(f"\n{'='*70}")
        log("DODGING")
        log(f"{'='*70}")
        base_res = run_cw_on_model(baseline_model, dodging_pairs, AttackMode.DODGING,
                                   args.threshold, device, "BASELINE", **cw_kwargs)
        adv_res = run_cw_on_model(adv_model, dodging_pairs, AttackMode.DODGING,
                                  args.threshold, device, "ADV-TRAINED", **cw_kwargs)
        print_comparison("DODGING", base_res, adv_res, args.threshold)

    if run_imp and impersonation_pairs:
        log(f"\n{'='*70}")
        log("IMPERSONATION")
        log(f"{'='*70}")
        base_res = run_cw_on_model(baseline_model, impersonation_pairs, AttackMode.IMPERSONATION,
                                   args.threshold, device, "BASELINE", **cw_kwargs)
        adv_res = run_cw_on_model(adv_model, impersonation_pairs, AttackMode.IMPERSONATION,
                                  args.threshold, device, "ADV-TRAINED", **cw_kwargs)
        print_comparison("IMPERSONATION", base_res, adv_res, args.threshold)

    log("\nDone.")


if __name__ == "__main__":
    main()
