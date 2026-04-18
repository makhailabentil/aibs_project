#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.models import load_arcface_model


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class CasiaTripletDataset(Dataset):
    """Returns (anchor, positive, negative) triplets for metric learning.

    Positive = different image of same identity.
    Negative = random image from a different identity.
    Only keeps classes with >= 2 images so positive pairs always exist.
    """

    def __init__(self, data_dir, image_size=112, max_classes=None, max_imgs_per_class=None):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        class_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
        if max_classes is not None:
            class_dirs = class_dirs[:max_classes]

        self.class_imgs: dict[str, list[Path]] = {}
        for d in class_dirs:
            imgs = sorted(
                list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
            )
            if max_imgs_per_class is not None:
                imgs = imgs[:max_imgs_per_class]
            if len(imgs) >= 2:
                self.class_imgs[d.name] = imgs

        self.classes = list(self.class_imgs.keys())

        # One entry per image (anchor); positive/negative sampled on the fly
        self.samples: list[tuple[Path, str]] = []
        for cls, imgs in self.class_imgs.items():
            for img in imgs:
                self.samples.append((img, cls))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_path, anchor_cls = self.samples[idx]

        # Positive: different image of same class
        pos_candidates = [p for p in self.class_imgs[anchor_cls] if p != anchor_path]
        pos_path = random.choice(pos_candidates)

        # Negative: random image from a different class
        neg_cls = random.choice([c for c in self.classes if c != anchor_cls])
        neg_path = random.choice(self.class_imgs[neg_cls])

        return self._load(anchor_path), self._load(pos_path), self._load(neg_path)

    def _load(self, path: Path) -> torch.Tensor:
        return self.transform(Image.open(path).convert("RGB"))


class CWAdversarialDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path, map_location="cpu")
        self.items = data

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return item["clean"], item["adv"], item["label"]


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def triplet_loss_fn(anchor: torch.Tensor,
                    positive: torch.Tensor,
                    negative: torch.Tensor,
                    margin: float = 0.3) -> torch.Tensor:
    """Cosine triplet loss: max(0, cos_sim(a,n) - cos_sim(a,p) + margin)."""
    sim_ap = (anchor * positive).sum(dim=1)
    sim_an = (anchor * negative).sum(dim=1)
    return F.relu(sim_an - sim_ap + margin).mean()


# ─────────────────────────────────────────────────────────────────────────────
# PGD for adversarial training (triplet / dodging direction)
# ─────────────────────────────────────────────────────────────────────────────

def pgd_attack_triplet(model: nn.Module,
                       anchor: torch.Tensor,
                       pos_emb_detached: torch.Tensor,
                       eps: float = 4 / 255,
                       alpha: float = 1 / 255,
                       steps: int = 3) -> torch.Tensor:
    """PGD dodging: perturb anchor so its embedding drifts away from positive."""
    was_training = model.training
    model.eval()

    x_orig = anchor.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        emb = F.normalize(model(x_adv), p=2, dim=1)
        # Minimise cos_sim(adv, positive) → push adv embedding away from positive
        loss = (emb * pos_emb_detached).sum(dim=1).sum()
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv - alpha * grad.sign()
            delta = torch.clamp(x_adv - x_orig, -eps, eps)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    if was_training:
        model.train()
    return x_adv.detach()


# ─────────────────────────────────────────────────────────────────────────────
# PGD for adversarial training (impersonation direction)
# ─────────────────────────────────────────────────────────────────────────────

def pgd_attack_impersonation(model: nn.Module,
                              anchor: torch.Tensor,
                              neg_emb_detached: torch.Tensor,
                              eps: float = 4 / 255,
                              alpha: float = 1 / 255,
                              steps: int = 3) -> torch.Tensor:
    """PGD impersonation: perturb anchor so its embedding drifts toward negative."""
    was_training = model.training
    model.eval()

    x_orig = anchor.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        emb = F.normalize(model(x_adv), p=2, dim=1)
        # Maximise cos_sim(adv, negative) → push adv embedding toward negative
        loss = -(emb * neg_emb_detached).sum(dim=1).sum()
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv - alpha * grad.sign()
            delta = torch.clamp(x_adv - x_orig, -eps, eps)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    if was_training:
        model.train()
    return x_adv.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, cw_loader, optimizer, device,
                    eps, alpha, pgd_steps, w_clean, w_pgd, w_imp, w_cw, w_trades,
                    margin, epoch, total_epochs):
    model.train()
    cw_iter = itertools.cycle(cw_loader) if cw_loader is not None else None
    total_loss = total_clean = total_pgd = total_imp = total_cw = total_trades = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}",
                unit="batch", dynamic_ncols=True, leave=True)

    for anchor, positive, negative in pbar:
        anchor   = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        # Detached embeddings for PGD target directions
        with torch.no_grad():
            model.eval()
            pos_emb_for_pgd = F.normalize(model(positive), p=2, dim=1)
            neg_emb_for_pgd = F.normalize(model(negative), p=2, dim=1)

        # Dodging PGD: push anchor away from positive
        anchor_dodge_adv = pgd_attack_triplet(model, anchor, pos_emb_for_pgd,
                                              eps, alpha, pgd_steps)
        # Impersonation PGD: push anchor toward negative
        anchor_imp_adv = pgd_attack_impersonation(model, anchor, neg_emb_for_pgd,
                                                   eps, alpha, pgd_steps)

        # All embeddings in train mode for loss
        model.train()
        anc_emb       = F.normalize(model(anchor),           p=2, dim=1)
        anc_dodge_emb = F.normalize(model(anchor_dodge_adv), p=2, dim=1)
        anc_imp_emb   = F.normalize(model(anchor_imp_adv),   p=2, dim=1)
        pos_emb       = F.normalize(model(positive),         p=2, dim=1)
        neg_emb       = F.normalize(model(negative),         p=2, dim=1)

        loss_clean = triplet_loss_fn(anc_emb,       pos_emb, neg_emb, margin)
        loss_pgd   = triplet_loss_fn(anc_dodge_emb, pos_emb, neg_emb, margin)
        loss_imp   = triplet_loss_fn(anc_imp_emb,   pos_emb, neg_emb, margin)

        # TRADES consistency: adversarial embeddings should stay close to clean
        anc_clean_det = anc_emb.detach()
        loss_trades = (
            (1 - (anc_clean_det * anc_dodge_emb).sum(dim=1)).mean() +
            (1 - (anc_clean_det * anc_imp_emb).sum(dim=1)).mean()
        ) * 0.5

        loss = w_clean * loss_clean + w_pgd * loss_pgd + w_imp * loss_imp + w_trades * loss_trades

        # CW consistency loss: (clean_emb, adv_emb) should stay close
        if cw_iter is not None:
            x_clean_cw, x_adv_cw, _ = next(cw_iter)
            x_clean_cw = x_clean_cw.to(device)
            x_adv_cw   = x_adv_cw.to(device)
            emb_clean_cw = F.normalize(model(x_clean_cw), p=2, dim=1)
            emb_adv_cw   = F.normalize(model(x_adv_cw),   p=2, dim=1)
            loss_cw = (1 - (emb_clean_cw * emb_adv_cw).sum(dim=1)).mean()
            loss = loss + w_cw * loss_cw
            total_cw += loss_cw.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        total_clean  += loss_clean.item()
        total_pgd    += loss_pgd.item()
        total_imp    += loss_imp.item()
        total_trades += loss_trades.item()

        n = max(1, pbar.n)
        pbar.set_postfix(
            loss=f"{total_loss/n:.4f}",
            clean=f"{total_clean/n:.4f}",
            dodge=f"{total_pgd/n:.4f}",
            imp=f"{total_imp/n:.4f}",
            trades=f"{total_trades/n:.4f}",
            cw=f"{total_cw/n:.4f}",
        )

    n = max(1, len(loader))
    return total_loss / n, total_clean / n, total_pgd / n, total_imp / n, total_trades / n, total_cw / n


@torch.no_grad()
def quick_eval_metric(model, loader, device, max_batches: int = 20):
    """Evaluate mean pos/neg cosine similarity (verification proxy)."""
    model.eval()
    pos_sims, neg_sims = [], []

    pbar = tqdm(enumerate(loader), total=min(max_batches, len(loader)),
                desc="  eval", unit="batch", leave=False, dynamic_ncols=True)
    for i, (anchor, positive, negative) in pbar:
        if i >= max_batches:
            break
        anchor   = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        a_emb = F.normalize(model(anchor),   p=2, dim=1)
        p_emb = F.normalize(model(positive), p=2, dim=1)
        n_emb = F.normalize(model(negative), p=2, dim=1)
        pos_sims.append((a_emb * p_emb).sum(dim=1).mean().item())
        neg_sims.append((a_emb * n_emb).sum(dim=1).mean().item())
        pbar.set_postfix(pos_sim=f"{sum(pos_sims)/len(pos_sims):.4f}",
                         neg_sim=f"{sum(neg_sims)/len(neg_sims):.4f}")

    mean_pos = sum(pos_sims) / max(1, len(pos_sims))
    mean_neg = sum(neg_sims) / max(1, len(neg_sims))
    return mean_pos, mean_neg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial fine-tuning with triplet loss + CW consistency"
    )
    parser.add_argument("--data",          type=str, required=True)
    parser.add_argument("--cw-examples",   type=str, required=True)
    parser.add_argument("--onnx",          type=str, default=None)
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch-size",    type=int,   default=8)
    parser.add_argument("--cw-batch-size", type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--image-size",    type=int,   default=112)
    parser.add_argument("--eps",           type=float, default=4 / 255)
    parser.add_argument("--alpha",         type=float, default=1 / 255)
    parser.add_argument("--pgd-steps",     type=int,   default=3)
    parser.add_argument("--margin",        type=float, default=0.4,
                        help="Triplet loss margin (default 0.4)")
    parser.add_argument("--w-clean",       type=float, default=0.2,
                        help="Weight for clean triplet loss")
    parser.add_argument("--w-pgd",         type=float, default=0.2,
                        help="Weight for dodging adversarial triplet loss")
    parser.add_argument("--w-imp",         type=float, default=0.2,
                        help="Weight for impersonation adversarial triplet loss")
    parser.add_argument("--w-trades",      type=float, default=0.3,
                        help="Weight for TRADES consistency loss")
    parser.add_argument("--w-cw",          type=float, default=0.4,
                        help="Weight for CW consistency loss")
    parser.add_argument("--max-classes",         type=int, default=200)
    parser.add_argument("--max-imgs-per-class",  type=int, default=30)
    parser.add_argument("--num-workers",   type=int, default=2)
    parser.add_argument("--save-name",     type=str, default="arcface_cw_adv_train.pt")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = CasiaTripletDataset(
        data_dir=args.data,
        image_size=args.image_size,
        max_classes=args.max_classes,
        max_imgs_per_class=args.max_imgs_per_class,
    )
    print(f"Classes: {len(dataset.classes)}  Triplets: {len(dataset)}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # ── CW adversarial examples ───────────────────────────────────────────────
    cw_path = Path(args.cw_examples)
    if not cw_path.is_absolute():
        cw_path = (ROOT / cw_path).resolve()
    cw_dataset = CWAdversarialDataset(cw_path)
    print(f"CW examples: {len(cw_dataset)}")
    cw_loader = DataLoader(
        cw_dataset, batch_size=args.cw_batch_size,
        shuffle=True, drop_last=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_arcface_model(onnx_path=args.onnx, device=device, input_bgr=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nTraining: {args.epochs} epochs | "
          f"batch={args.batch_size} | margin={args.margin} | "
          f"w_clean={args.w_clean} w_pgd={args.w_pgd} w_imp={args.w_imp} "
          f"w_trades={args.w_trades} w_cw={args.w_cw}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        loss, clean, pgd, imp, trades, cw = train_one_epoch(
            model, loader, cw_loader, optimizer, device,
            args.eps, args.alpha, args.pgd_steps,
            args.w_clean, args.w_pgd, args.w_imp, args.w_cw, args.w_trades,
            args.margin, epoch + 1, args.epochs,
        )
        pos_sim, neg_sim = quick_eval_metric(model, loader, device)
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss={loss:.4f}  clean={clean:.4f}  dodge={pgd:.4f}  imp={imp:.4f}  "
            f"trades={trades:.4f}  cw={cw:.4f} | "
            f"pos_sim={pos_sim:.4f}  neg_sim={neg_sim:.4f}  "
            f"gap={pos_sim - neg_sim:.4f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / args.save_name
    torch.save({
        "backbone_state_dict": model.state_dict(),
        "args": vars(args),
    }, save_path)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
