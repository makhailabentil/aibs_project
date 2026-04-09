#!/usr/bin/env python3
"""Adversarial training to make ArcFace robust against PGD attacks.

This implements AT-FaceNet-style adversarial training where, on each mini-batch,
a fraction of images are replaced with their PGD adversarial counterparts before
the standard ArcFace / CosFace classification loss is computed.

Usage:
  # Fine-tune the existing ONNX/PyTorch backbone:
  python scripts/train_adversarial.py --data data/casia_webface_extracted

  # Full training run with custom settings:
  python scripts/retraining_teresa.py \
      --data data/casia_webface_extracted \
      --epochs 20 \
      --adv_fraction 0.3 \
      --eps 8 \
      --pgd_steps 5 \
      --lr 1e-4 \
      --save_every 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from src.attacks import PGDAttack, AttackMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FaceDataset(Dataset):
    """Folder-per-identity face image dataset.

    Structure expected:
        root/
            identity_001/
                img1.jpg
                img2.jpg
            identity_002/
                ...
    """

    EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(self, root: Path, size: int = 112):
        self.root = root
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
        ])
        self.samples: list[tuple[Path, int]] = []
        self.class_names: list[str] = []
        self._build_index()

    def _build_index(self) -> None:
        identity_dirs = sorted(
            d for d in self.root.iterdir()
            if d.is_dir()
        )
        for label, ident_dir in enumerate(identity_dirs):
            self.class_names.append(ident_dir.name)
            for img_path in ident_dir.iterdir():
                if img_path.suffix.lower() in self.EXTS:
                    self.samples.append((img_path, label))

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ---------------------------------------------------------------------------
# ArcFace / CosFace classification head
# ---------------------------------------------------------------------------

class ArcFaceHead(nn.Module):
    """Additive Angular Margin (ArcFace) classification head.

    Args:
        in_features: Embedding dimension (e.g. 512).
        num_classes: Number of identities.
        s: Feature scale (logit scale). Default 64.
        m: Additive angular margin in radians. Default 0.5.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.5,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        import math
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight_norm)           # (B, C)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m        # cos(θ + m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


# ---------------------------------------------------------------------------
# PGD attack used *during* training (fast inner loop)
# ---------------------------------------------------------------------------

class TrainingPGD:
    """Lightweight PGD wrapper optimised for training speed.

    Uses a fixed 7-step PGD (TRADES-style) with random start.
    Operates in *no_grad* except for the loss gradient w.r.t. x.
    """

    def __init__(
        self,
        model: nn.Module,
        head: ArcFaceHead,
        eps: float,
        alpha: float,
        steps: int,
        norm: str = "Linf",
    ):
        self.model = model
        self.head = head
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.norm = norm

    @torch.enable_grad()
    def perturb(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return adversarial examples that maximise cross-entropy loss."""
        x = x.detach()

        # Random initialisation within epsilon ball
        if self.norm == "Linf":
            delta = torch.empty_like(x).uniform_(-self.eps, self.eps)
        else:
            delta = torch.zeros_like(x)
            delta.normal_()
            delta = delta * self.eps / (delta.view(len(x), -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-12)

        delta = delta.clamp(-self.eps, self.eps)
        x_adv = (x + delta).clamp(0, 1)

        for _ in range(self.steps):
            x_adv = x_adv.detach().requires_grad_(True)
            emb = self.model(x_adv)
            logits = self.head(emb, labels)
            loss = F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, x_adv)[0]

            with torch.no_grad():
                if self.norm == "Linf":
                    x_adv = x_adv + self.alpha * grad.sign()
                    delta = (x_adv - x).clamp(-self.eps, self.eps)
                    x_adv = (x + delta).clamp(0, 1)
                else:  # L2
                    grad_norm = grad.view(len(x), -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-12
                    x_adv = x_adv + self.alpha * grad / grad_norm
                    delta = x_adv - x
                    delta_norm = delta.view(len(x), -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-12
                    delta = delta * torch.min(
                        torch.ones_like(delta_norm),
                        self.eps / delta_norm,
                    )
                    x_adv = (x + delta).clamp(0, 1)

        return x_adv.detach()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _build_optimizer(
    backbone: nn.Module,
    head: ArcFaceHead,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    params = [
        {"params": backbone.parameters(), "lr": lr},
        {"params": head.parameters(), "lr": lr * 10},  # head learns faster
    ]
    return torch.optim.SGD(params, momentum=0.9, weight_decay=weight_decay)


def _cosine_schedule(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int = 2,
) -> None:
    """In-place LR update with linear warmup + cosine decay."""
    import math
    for pg in optimizer.param_groups:
        base_lr = pg.get("initial_lr", pg["lr"])
        if epoch < warmup_epochs:
            scale = (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        pg["lr"] = base_lr * scale


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    backbone: nn.Module,
    head: ArcFaceHead,
    pgd: TrainingPGD,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    adv_fraction: float,
    epoch: int,
) -> dict:
    backbone.train()
    head.train()

    total_loss = 0.0
    clean_correct = 0
    adv_correct = 0
    total_samples = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x, labels = x.to(device), labels.to(device)
        B = len(x)

        # ---- Build mixed batch (clean + adversarial) ----------------------
        num_adv = max(1, int(B * adv_fraction))
        adv_indices = torch.randperm(B, device=device)[:num_adv]

        # Generate adversarial examples for selected indices
        x_adv_partial = pgd.perturb(x[adv_indices], labels[adv_indices])

        x_mixed = x.clone()
        x_mixed[adv_indices] = x_adv_partial

        # ---- Forward (mixed batch) ----------------------------------------
        optimizer.zero_grad()
        emb_mixed = backbone(x_mixed)
        logits_mixed = head(emb_mixed, labels)
        loss = F.cross_entropy(logits_mixed, labels)

        # ---- Backward --------------------------------------------------------
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(head.parameters()), max_norm=5.0
        )
        optimizer.step()

        # ---- Metrics (clean accuracy on full batch) -----------------------
        with torch.no_grad():
            emb_clean = backbone(x)
            logits_clean = head(emb_clean, labels)
            clean_pred = logits_clean.argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()

            # Adv accuracy on the adversarial subset
            emb_adv = backbone(x_adv_partial)
            logits_adv = head(emb_adv, labels[adv_indices])
            adv_pred = logits_adv.argmax(dim=1)
            adv_correct += (adv_pred == labels[adv_indices]).sum().item()

        total_loss += loss.item() * B
        total_samples += B

        if (batch_idx + 1) % 20 == 0:
            _log(
                f"  [Epoch {epoch}] step {batch_idx+1}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"clean_acc={clean_correct/total_samples*100:.1f}% "
                f"adv_acc_partial={adv_correct/total_samples*100:.1f}%"
            )

    return {
        "loss": total_loss / total_samples,
        "clean_acc": clean_correct / total_samples,
        "adv_acc": adv_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    backbone: nn.Module,
    eval_pairs: list[dict],
    threshold: float,
    device: torch.device,
    pgd_eval: Optional[PGDAttack] = None,
) -> dict:
    """Evaluate verification performance on held-out pairs.

    Returns clean TAR@FAR, adversarial TAR@FAR (if pgd_eval provided),
    and ASR (Attack Success Rate under adversarial conditions).
    """
    backbone.eval()

    clean_correct = 0
    adv_correct = 0  # correct means attack FAILED to fool the model
    n_eligible_attack = 0

    for pair in eval_pairs:
        x1 = pair["img1"].to(device)
        x2 = pair["img2"].to(device)
        same = pair["same"]

        e1 = F.normalize(backbone(x1), p=2, dim=1)
        e2 = F.normalize(backbone(x2), p=2, dim=1)
        sim = (e1 * e2).sum(dim=1).item()
        pred_same = sim >= threshold

        if pred_same == same:
            clean_correct += 1

        # Adversarial evaluation: try to fool the model on the *source* image
        if pgd_eval is not None:
            if same:
                # Dodging: try to push sim below threshold
                eligible = sim >= threshold
                if eligible:
                    n_eligible_attack += 1
                    x1_adv = pgd_eval(x1, source_embedding=e2, mode=AttackMode.DODGING)
                    e1_adv = F.normalize(backbone(x1_adv), p=2, dim=1)
                    adv_sim = (e1_adv * e2).sum(dim=1).item()
                    if adv_sim >= threshold:  # attack failed → correct
                        adv_correct += 1
            else:
                # Impersonation: try to push sim above threshold
                eligible = sim < threshold
                if eligible:
                    n_eligible_attack += 1
                    x1_adv = pgd_eval(x1, target_embedding=e2, source_embedding=e1, mode=AttackMode.IMPERSONATION)
                    e1_adv = F.normalize(backbone(x1_adv), p=2, dim=1)
                    adv_sim = (e1_adv * e2).sum(dim=1).item()
                    if adv_sim < threshold:  # attack failed → correct
                        adv_correct += 1

    n = len(eval_pairs)
    result = {
        "clean_acc": clean_correct / n if n > 0 else 0.0,
        "adv_robust_acc": adv_correct / n_eligible_attack if n_eligible_attack > 0 else None,
        "asr": 1.0 - adv_correct / n_eligible_attack if n_eligible_attack > 0 else None,
        "n_pairs": n,
        "n_eligible_attack": n_eligible_attack,
    }
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial training for ArcFace robustness")

    # Data
    parser.add_argument("--data", type=str, required=True, help="Path to casia_webface_extracted")
    parser.add_argument("--size", type=int, default=112, help="Input image size")
    parser.add_argument("--val_split", type=float, default=0.05, help="Fraction of identities held out for eval")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--adv_fraction", type=float, default=0.5,
                        help="Fraction of each mini-batch that is adversarial (0=standard, 1=full AT)")

    # Inner PGD (training)
    parser.add_argument("--pgd_steps", type=int, default=7, help="PGD steps during training (fast, keep ≤10)")
    parser.add_argument("--eps", type=float, default=8.0, help="Max perturbation in [0,255] scale")
    parser.add_argument("--alpha_train", type=float, default=None,
                        help="PGD step size during training. Defaults to 2*eps/steps")
    parser.add_argument("--norm", choices=["Linf", "L2"], default="Linf")

    # Evaluation PGD (stronger, used at end of each epoch)
    parser.add_argument("--eval_pgd_steps", type=int, default=40, help="PGD steps for evaluation")
    parser.add_argument("--num_eval_pairs", type=int, default=200, help="Number of pairs for epoch evaluation")

    # ArcFace head
    parser.add_argument("--emb_dim", type=int, default=512, help="Backbone embedding dimension")
    parser.add_argument("--arc_s", type=float, default=64.0, help="ArcFace scale s")
    parser.add_argument("--arc_m", type=float, default=0.5, help="ArcFace margin m")

    # Model
    parser.add_argument("--onnx", type=str, default=None, help="Path to w600k_r50.onnx (optional override)")
    parser.add_argument("--threshold", type=float, default=0.1767, help="Verification threshold")

    # Saving
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # ---- Setup ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    eps_norm = args.eps / 255.0
    alpha_train = args.alpha_train / 255.0 if args.alpha_train else 2 * eps_norm / args.pgd_steps

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (ROOT / data_path).resolve()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load backbone --------------------------------------------------------
    _log("Loading ArcFace backbone...")
    try:
        from src.models import load_arcface_model
        backbone = load_arcface_model(onnx_path=args.onnx, device=device, input_bgr=False)
    except FileNotFoundError as e:
        _log(f"Model not found: {e}")
        sys.exit(1)

    # ---- Dataset --------------------------------------------------------------
    _log("Building dataset...")
    full_dataset = FaceDataset(data_path, size=args.size)
    num_classes = full_dataset.num_classes
    _log(f"  {len(full_dataset)} images, {num_classes} identities")

    # Split train/val by identity
    num_val_ids = max(1, int(num_classes * args.val_split))
    val_id_set = set(range(num_classes - num_val_ids, num_classes))
    train_samples = [(p, l) for p, l in full_dataset.samples if l not in val_id_set]
    val_samples   = [(p, l) for p, l in full_dataset.samples if l in val_id_set]

    # Re-index validation labels to be contiguous (not strictly needed but tidy)
    class FaceSubset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            path, label = self.samples[i]
            img = Image.open(path).convert("RGB")
            return self.transform(img), label

    train_ds = FaceSubset(train_samples, full_dataset.transform)
    val_ds   = FaceSubset(val_samples, transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
    ]))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    _log(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ---- Classification head --------------------------------------------------
    head = ArcFaceHead(
        in_features=args.emb_dim,
        num_classes=num_classes - num_val_ids,
        s=args.arc_s,
        m=args.arc_m,
    ).to(device)

    # ---- Optimiser -----------------------------------------------------------
    optimizer = _build_optimizer(backbone, head, args.lr, args.weight_decay)

    # Store initial LRs for scheduler
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    # ---- Inner attack (training) ---------------------------------------------
    pgd_train = TrainingPGD(
        backbone, head,
        eps=eps_norm,
        alpha=alpha_train,
        steps=args.pgd_steps,
        norm=args.norm,
    )

    # ---- Eval attack (stronger, for post-epoch evaluation) -------------------
    pgd_eval = PGDAttack(
        backbone,
        device=device,
        eps=eps_norm,
        alpha=2 * eps_norm / args.eval_pgd_steps,
        steps=args.eval_pgd_steps,
        norm=args.norm,
        random_start=True,
    )

    # ---- Build evaluation pairs from val set --------------------------------
    _log("Building evaluation pairs...")
    eval_pairs = _build_eval_pairs(val_ds, args.num_eval_pairs, args.size, device)
    _log(f"  {len(eval_pairs)} evaluation pairs built.")

    # ---- Resume --------------------------------------------------------------
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        _log(f"Resumed from epoch {ckpt['epoch']}: {args.resume}")

    # ---- Training loop -------------------------------------------------------
    history = []
    _log(f"\n{'='*60}")
    _log(f"Adversarial Training — ArcFace on {data_path.name}")
    _log(f"epochs={args.epochs}, adv_fraction={args.adv_fraction}, "
         f"eps={args.eps}/255, pgd_steps={args.pgd_steps}")
    _log(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        _cosine_schedule(optimizer, epoch - 1, args.epochs)
        cur_lr = optimizer.param_groups[0]["lr"]
        _log(f"\n[Epoch {epoch}/{args.epochs}] lr={cur_lr:.2e}")

        train_stats = train_one_epoch(
            backbone, head, pgd_train, train_loader,
            optimizer, device, args.adv_fraction, epoch,
        )

        # Evaluate
        eval_stats = evaluate(backbone, eval_pairs, args.threshold, device, pgd_eval)

        _log(
            f"  Train  loss={train_stats['loss']:.4f}  "
            f"clean_acc={train_stats['clean_acc']*100:.1f}%  "
            f"adv_acc={train_stats['adv_acc']*100:.1f}%"
        )
        _log(
            f"  Eval   clean_acc={eval_stats['clean_acc']*100:.1f}%  "
            + (f"robust_acc={eval_stats['adv_robust_acc']*100:.1f}%  "
               f"ASR={eval_stats['asr']*100:.1f}%"
               if eval_stats["adv_robust_acc"] is not None else "(no eligible attack pairs)")
        )

        record = {"epoch": epoch, **train_stats, **{f"eval_{k}": v for k, v in eval_stats.items()}}
        history.append(record)

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = output_dir / f"arcface_adv_ep{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "history": history,
            }, ckpt_path)
            _log(f"  ✓ Checkpoint saved: {ckpt_path}")

    # ---- Final evaluation ----------------------------------------------------
    _log(f"\n{'='*60}")
    _log("FINAL EVALUATION (strong PGD-40)")
    _log(f"{'='*60}")
    final = evaluate(backbone, eval_pairs, args.threshold, device, pgd_eval)
    _log(f"  clean_acc={final['clean_acc']*100:.1f}%")
    if final["adv_robust_acc"] is not None:
        _log(f"  robust_acc={final['adv_robust_acc']*100:.1f}%  ASR={final['asr']*100:.1f}%")

    # Save training log
    log_path = output_dir / f"adv_training_log_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(log_path, "w") as f:
        json.dump({"args": vars(args), "history": history, "final": final}, f, indent=2)
    _log(f"\nTraining log saved: {log_path}")


# ---------------------------------------------------------------------------
# Eval pair builder
# ---------------------------------------------------------------------------

def _build_eval_pairs(
    val_ds,
    num_pairs: int,
    size: int,
    device: torch.device,
) -> list[dict]:
    """Build genuine (same) and impostor (different) pairs from val dataset."""
    from collections import defaultdict

    by_label: dict[int, list] = defaultdict(list)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    # Collect samples per label
    for path, label in val_ds.samples:
        by_label[label].append(path)

    labels = list(by_label.keys())
    pairs = []
    half = num_pairs // 2

    # Genuine pairs (same identity)
    for label in labels:
        if len(pairs) >= half:
            break
        imgs = by_label[label]
        if len(imgs) >= 2:
            def load(p):
                img = Image.open(p).convert("RGB")
                return transform(img).unsqueeze(0).to(device)
            pairs.append({"img1": load(imgs[0]), "img2": load(imgs[1]), "same": True})

    # Impostor pairs (different identities)
    random.shuffle(labels)
    for i in range(len(labels) - 1):
        if len(pairs) >= num_pairs:
            break
        imgs_a = by_label[labels[i]]
        imgs_b = by_label[labels[i + 1]]
        if imgs_a and imgs_b:
            def load(p):
                img = Image.open(p).convert("RGB")
                return transform(img).unsqueeze(0).to(device)
            pairs.append({"img1": load(imgs_a[0]), "img2": load(imgs_b[0]), "same": False})

    return pairs


if __name__ == "__main__":
    main()