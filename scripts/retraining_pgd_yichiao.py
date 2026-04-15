#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import load_arcface_model


class CasiaFolderDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 112,
        max_classes: int | None = None,
        max_imgs_per_class: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        if max_classes is not None:
            class_dirs = class_dirs[:max_classes]

        self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        self.samples: list[tuple[Path, int]] = []

        for d in class_dirs:
            imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png")))
            if max_imgs_per_class is not None:
                imgs = imgs[:max_imgs_per_class]
            for img_path in imgs:
                self.samples.append((img_path, self.class_to_idx[d.name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class ArcFaceClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        emb = self.backbone(x)              # [B, 512]
        emb = F.normalize(emb, p=2, dim=1) # same style as verification
        logits = self.classifier(emb)
        return logits, emb


def pgd_attack_for_training(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 4 / 255,
    alpha: float = 1 / 255,
    steps: int = 3,
):
    """
    PGD for adversarial training using CE loss.
    This is different from your existing verification-style PGD attack.
    """
    was_training = model.training
    model.eval()

    x_orig = x.detach()
    x_adv = x_orig.clone()

    # random start
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)

        logits_adv, _ = model(x_adv)
        loss = F.cross_entropy(logits_adv, y)

        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    if was_training:
        model.train()

    return x_adv.detach()


def train_one_epoch(model, loader, optimizer, device, eps, alpha, steps, adv_weight):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_adv = pgd_attack_for_training(
            model=model,
            x=x,
            y=y,
            eps=eps,
            alpha=alpha,
            steps=steps,
        )

        logits_clean, _ = model(x)
        logits_adv, _ = model(x_adv)

        loss_clean = F.cross_entropy(logits_clean, y)
        loss_adv = F.cross_entropy(logits_adv, y)
        loss = (1.0 - adv_weight) * loss_clean + adv_weight * loss_adv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def quick_train_acc(model, loader, device, max_batches: int = 20):
    model.eval()
    total = 0
    correct = 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        pred = logits.argmax(dim=1)

        total += y.numel()
        correct += (pred == y).sum().item()

    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to data/casia_webface_extracted")
    parser.add_argument("--onnx", type=str, default=None, help="Path to w600k_r50.onnx")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--eps", type=float, default=4/255)
    parser.add_argument("--alpha", type=float, default=1/255)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--adv_weight", type=float, default=0.5)
    parser.add_argument("--max_classes", type=int, default=100)
    parser.add_argument("--max_imgs_per_class", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_name", type=str, default="arcface_pgd_adv_train.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CasiaFolderDataset(
        data_dir=args.data,
        image_size=args.image_size,
        max_classes=args.max_classes,
        max_imgs_per_class=args.max_imgs_per_class,
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check --data path.")

    print(f"Num classes: {len(dataset.class_to_idx)}")
    print(f"Num samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    backbone = load_arcface_model(
        onnx_path=args.onnx,
        device=device,
        input_bgr=False,
    )

    model = ArcFaceClassifier(
        backbone=backbone,
        num_classes=len(dataset.class_to_idx),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            adv_weight=args.adv_weight,
        )

        acc = quick_train_acc(model, loader, device=device)
        print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | quick_train_acc={acc:.4f}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / args.save_name
    torch.save(
        {
            "backbone_state_dict": model.backbone.state_dict(),
            "classifier_state_dict": model.classifier.state_dict(),
            "class_to_idx": dataset.class_to_idx,
            "args": vars(args),
        },
        save_path,
    )

    print(f"Saved checkpoint to: {save_path}")


if __name__ == "__main__":
    main()