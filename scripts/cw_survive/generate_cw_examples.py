#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.attacks import CarliniWagnerAttack, AttackMode
from src.models import load_arcface_model


def load_image_tensor(path: Path, size: int) -> torch.Tensor:
    t = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return t(Image.open(path).convert("RGB"))


def get_identities(data_dir: Path) -> list[Path]:
    result = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
            if len(imgs) >= 2:
                result.append(d)
    return result


def build_pairs(identities: list[Path], num_pairs: int, mode: str, size: int):
    """Build pairs as CPU tensors (no device yet, for batching)."""
    pairs = []
    if mode == "dodging":
        for ident_dir in identities:
            if len(pairs) >= num_pairs:
                break
            imgs = sorted(list(ident_dir.glob("*.jpg")) + list(ident_dir.glob("*.png")))
            if len(imgs) >= 2:
                pairs.append({
                    "source": load_image_tensor(imgs[0], size),
                    "target": load_image_tensor(imgs[1], size),
                    "identity": ident_dir.name,
                })
    else:
        for i in range(min(num_pairs, len(identities) - 1)):
            src_dir = identities[i]
            tgt_dir = identities[i + 1]
            src_imgs = sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")))
            tgt_imgs = sorted(list(tgt_dir.glob("*.jpg")) + list(tgt_dir.glob("*.png")))
            if src_imgs and tgt_imgs:
                pairs.append({
                    "source": load_image_tensor(src_imgs[0], size),
                    "target": load_image_tensor(tgt_imgs[0], size),
                    "source_id": src_dir.name,
                    "target_id": tgt_dir.name,
                })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--num-pairs", type=int, default=20)
    parser.add_argument("--mode", choices=["impersonation", "dodging", "both"], default="both")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--c-steps", type=int, default=7)
    parser.add_argument("--c-init", type=float, default=1e3)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--out", type=str, default="data/cw_adversarial/cw_examples.pt")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of pairs to attack simultaneously (GPU parallelism, default=4)")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (ROOT / data_path).resolve()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if out_path.exists():
        existing = torch.load(out_path, map_location="cpu")
        print(f"Resuming: {len(existing)} examples already generated")

    model = load_arcface_model(onnx_path=args.onnx, device=device, input_bgr=False)

    attack = CarliniWagnerAttack(
        model, device=device,
        c_init=args.c_init, lr=args.lr,
        optimizer_steps=args.steps, c_steps=args.c_steps,
    )

    identities = get_identities(data_path)
    print(f"Found {len(identities)} identities")
    print(f"Batch size: {args.batch_size} pairs per attack call")

    modes = []
    if args.mode in ("impersonation", "both"):
        modes.append("impersonation")
    if args.mode in ("dodging", "both"):
        modes.append("dodging")

    results = list(existing)

    for mode in modes:
        attack_mode = AttackMode.IMPERSONATION if mode == "impersonation" else AttackMode.DODGING
        pairs = build_pairs(identities, args.num_pairs, mode, args.size)
        print(f"\n{mode.upper()}: {len(pairs)} pairs  (batch_size={args.batch_size})")

        bs = args.batch_size
        for batch_start in range(0, len(pairs), bs):
            batch = pairs[batch_start: batch_start + bs]
            actual_bs = len(batch)
            print(f"\n[{mode} pairs {batch_start+1}–{batch_start+actual_bs}/{len(pairs)}]")

            x_batch = torch.stack([p["source"] for p in batch]).to(device)
            t_batch = torch.stack([p["target"] for p in batch]).to(device)

            with torch.no_grad():
                source_emb = model(x_batch)
                target_emb = model(t_batch)

            if mode == "impersonation":
                x_adv_batch = attack(
                    x_batch,
                    target_embedding=target_emb,
                    source_embedding=source_emb,
                    mode=attack_mode,
                )
            else:
                x_adv_batch = attack(
                    x_batch,
                    source_embedding=source_emb,
                    mode=attack_mode,
                )

            for j, pair in enumerate(batch):
                global_idx = batch_start + j
                x_j    = x_batch[j].unsqueeze(0)
                x_adv_j = x_adv_batch[j].unsqueeze(0)
                l2 = (x_adv_j - x_j).pow(2).sum().sqrt().item()
                identity = pair.get("identity", pair.get("source_id", str(global_idx)))
                print(f"  [{global_idx+1}] identity={identity}  L2={l2:.4f}")

                results.append({
                    "clean": x_j.squeeze(0).cpu(),
                    "adv": x_adv_j.squeeze(0).cpu(),
                    "label": global_idx,
                    "identity": identity,
                    "l2": l2,
                    "mode": mode,
                })

            torch.save(results, out_path)
            print(f"  Saved ({len(results)} total)")

    print(f"\nDone. {len(results)} examples saved to {out_path}")


if __name__ == "__main__":
    main()
