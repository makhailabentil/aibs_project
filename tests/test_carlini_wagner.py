"""Unit tests for Carlini-Wagner attack.

Uses CASIA-WebFace extracted data (data/casia_webface_extracted/) and a placeholder
model to verify the attack moves embeddings in the correct direction. Skips if
extracted data or dependencies (torch, torchvision, PIL) are not available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    from src.attacks import CarliniWagnerAttack, AttackMode
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    CarliniWagnerAttack = None  # type: ignore[misc, assignment]
    AttackMode = None  # type: ignore[misc, assignment]
    nn = None  # type: ignore[assignment]

REQUIRES_TORCH = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch, torchvision, Pillow required")


def _cos_sim(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Cosine similarity (a, b normalized)."""
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    return (a * b).sum(dim=1)


@REQUIRES_TORCH
def test_cw_impersonation_linear_model() -> None:
    """Impersonation on a linear model: attack must succeed (deterministic, no data needed)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    dim_in = 3 * 16 * 16  # smaller for faster optimization
    dim_emb = 32
    W = torch.randn(dim_emb, dim_in, device=device) * 0.5

    class LinearModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x.view(x.size(0), -1), W)

    model = LinearModel().to(device)
    # Source and target with clear separation: x near 0, target near 1
    x = torch.full((1, 3, 16, 16), 0.1, device=device)
    target_img = torch.full((1, 3, 16, 16), 0.9, device=device)
    with torch.no_grad():
        source_emb = model(x)
        target_emb = model(target_img)

    attack = CarliniWagnerAttack(
        model, device=device, optimizer_steps=2000, c_steps=9, lr=0.5, c_init=100.0
    )
    x_adv = attack(
        x, target_embedding=target_emb, source_embedding=source_emb, mode=AttackMode.IMPERSONATION
    )

    assert x_adv.shape == x.shape
    assert (x_adv >= 0).all() and (x_adv <= 1).all()
    l2 = (x_adv - x).pow(2).sum().sqrt().item()
    assert l2 > 1e-10, "Attack must produce non-zero perturbation"
    with torch.no_grad():
        adv_emb = model(x_adv)
        sim_to_target = _cos_sim(adv_emb, target_emb).item()
        sim_to_source = _cos_sim(adv_emb, source_emb).item()
    assert sim_to_target > sim_to_source, (
        f"Impersonation failed: sim_to_target={sim_to_target:.4f}, sim_to_source={sim_to_source:.4f}"
    )


@REQUIRES_TORCH
def test_cw_dodging_linear_model() -> None:
    """Dodging on a linear model: attack must succeed (deterministic, no data needed)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(43)
    dim_in = 3 * 32 * 32
    dim_emb = 64
    W = torch.randn(dim_emb, dim_in, device=device) * 0.1

    class LinearModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x.view(x.size(0), -1), W)

    model = LinearModel().to(device)
    x = torch.rand(1, 3, 32, 32, device=device)
    with torch.no_grad():
        source_emb = model(x)

    attack = CarliniWagnerAttack(
        model, device=device, optimizer_steps=1000, c_steps=7, lr=0.2, c_init=1.0, dodge_threshold=0.5
    )
    x_adv = attack(x, source_embedding=source_emb, mode=AttackMode.DODGING)

    assert x_adv.shape == x.shape
    assert (x_adv >= 0).all() and (x_adv <= 1).all()
    l2 = (x_adv - x).pow(2).sum().sqrt().item()
    assert l2 > 0, "Attack must produce non-zero perturbation"
    with torch.no_grad():
        adv_emb = model(x_adv)
        sim_orig = _cos_sim(model(x), source_emb).item()
        sim_adv = _cos_sim(adv_emb, source_emb).item()
    assert sim_adv < sim_orig, (
        f"Dodging failed: sim_adv={sim_adv:.4f}, sim_orig={sim_orig:.4f}"
    )


# Path to extracted CASIA-WebFace (folder-per-identity .jpg)
ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = ROOT / "data" / "casia_webface_extracted"
REQUIRES_EXTRACTED = pytest.mark.skipif(
    not EXTRACTED_DIR.exists(),
    reason="CASIA-WebFace extracted data not found. Run: python scripts/extract_rec_to_folders.py --limit 1000",
)

if TORCH_AVAILABLE:

    class PlaceholderEmbeddingModel(nn.Module):
        """Placeholder model for testing before baseline exists."""

        def __init__(self, embed_dim: int = 512, in_channels: int = 3):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.conv(x))
            h = self.pool(h)
            h = h.view(h.size(0), -1)
            return self.fc(h)

    def load_image(path: Path, size: int = 112) -> torch.Tensor:
        """Load image, resize, normalize to [0,1], return (1, C, H, W)."""
        img = Image.open(path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        x = transform(img)
        return x.unsqueeze(0)

    def sample_images(n_identities: int = 2, images_per_identity: int = 1) -> list[Path]:
        """Sample images from different identity folders."""
        if not EXTRACTED_DIR.exists():
            return []
        identities = sorted([d for d in EXTRACTED_DIR.iterdir() if d.is_dir()])[:n_identities]
        paths = []
        for ident_dir in identities:
            imgs = list(ident_dir.glob("*.jpg"))[:images_per_identity]
            if not imgs:
                imgs = list(ident_dir.glob("*.jpeg"))[:images_per_identity]
            if not imgs:
                imgs = list(ident_dir.glob("*.png"))[:images_per_identity]
            paths.extend(imgs[:images_per_identity])
        return paths

    def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _cos_sim(a, b)


@REQUIRES_TORCH
@REQUIRES_EXTRACTED
def test_cw_impersonation_success() -> None:
    """Impersonation: x_adv should be closer to target than source."""
    paths = sample_images(n_identities=2, images_per_identity=1)
    if len(paths) < 2:
        pytest.skip("Need at least 2 images from different identities")
    source_path, target_path = paths[0], paths[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    x = load_image(source_path).to(device)
    target_img = load_image(target_path).to(device)

    model = PlaceholderEmbeddingModel(embed_dim=512).to(device)
    model.eval()

    with torch.no_grad():
        source_emb = model(x)
        target_emb = model(target_img)

    attack = CarliniWagnerAttack(
        model, device=device, optimizer_steps=600, c_steps=6, c_init=1e3, lr=0.1
    )
    x_adv = attack(
        x,
        target_embedding=target_emb,
        source_embedding=source_emb,
        mode=AttackMode.IMPERSONATION,
    )

    # Output validity
    assert x_adv.shape == x.shape
    assert (x_adv >= 0).all() and (x_adv <= 1).all()
    l2 = (x_adv - x).pow(2).sum().sqrt()
    assert l2.item() >= 0 and torch.isfinite(l2)

    # Success: cos_sim(embed(x_adv), target) > cos_sim(embed(x_adv), source)
    with torch.no_grad():
        adv_emb = model(x_adv)
        sim_to_target = cos_sim(adv_emb, target_emb).item()
        sim_to_source = cos_sim(adv_emb, source_emb).item()
    assert sim_to_target > sim_to_source, (
        f"Impersonation failed: sim_to_target={sim_to_target:.4f}, sim_to_source={sim_to_source:.4f}"
    )


@REQUIRES_TORCH
@REQUIRES_EXTRACTED
def test_cw_dodging_success() -> None:
    """Dodging: x_adv should be farther from source (lower cos_sim)."""
    paths = sample_images(n_identities=1, images_per_identity=1)
    if len(paths) < 1:
        pytest.skip("Need at least 1 image")
    source_path = paths[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(43)
    x = load_image(source_path).to(device)

    model = PlaceholderEmbeddingModel(embed_dim=512).to(device)
    model.eval()

    with torch.no_grad():
        source_emb = model(x)

    attack = CarliniWagnerAttack(
        model, device=device, optimizer_steps=500, c_steps=5, c_init=1e3, dodge_threshold=0.7
    )
    x_adv = attack(x, source_embedding=source_emb, mode=AttackMode.DODGING)

    assert x_adv.shape == x.shape
    assert (x_adv >= 0).all() and (x_adv <= 1).all()

    with torch.no_grad():
        adv_emb = model(x_adv)
        sim_orig = cos_sim(model(x), source_emb).item()
        sim_adv = cos_sim(adv_emb, source_emb).item()
    assert sim_adv < sim_orig, (
        f"Dodging failed: sim_adv={sim_adv:.4f}, sim_orig={sim_orig:.4f}"
    )
