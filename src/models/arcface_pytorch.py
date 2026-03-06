"""Differentiable ArcFace model from InsightFace ONNX for C&W attack.

Converts w600k_r50.onnx to PyTorch so we can backprop through it.
Same weights as the baseline (arcface_eval_bin), enabling white-box attack
on the real verification model.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


def _find_onnx_path() -> str:
    """Locate w600k_r50.onnx (same logic as arcface_eval_bin)."""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".insightface", "models", "buffalo_l", "w600k_r50.onnx"),
        os.path.join(os.getcwd(), "buffalo_l", "w600k_r50.onnx"),
    ]
    if os.environ.get("INSIGHTFACE_HOME"):
        base = os.environ["INSIGHTFACE_HOME"]
        candidates.extend([
            os.path.join(base, "models", "buffalo_l", "w600k_r50.onnx"),
            os.path.join(base, "buffalo_l", "w600k_r50.onnx"),
        ])
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find w600k_r50.onnx. "
        "Run arcface_eval_bin.py with FaceAnalysis(name='buffalo_l') once to download, "
        "or set INSIGHTFACE_HOME to the model directory."
    )


class ArcFacePyTorch(nn.Module):
    """Differentiable ArcFace embedding model from InsightFace ONNX.

    Accepts images in [0, 1], BGR channel order (matches cv2/OpenCV).
    For RGB input (e.g. from PIL), set input_bgr=False to flip channels internally.
    """

    def __init__(
        self,
        onnx_path: str | Path | None = None,
        input_bgr: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        from onnx2torch import convert

        onnx_path = onnx_path or _find_onnx_path()
        self.input_bgr = input_bgr
        import onnx
        onnx_model = onnx.load(onnx_path)
        self._backbone = convert(onnx_model)
        if device is not None:
            self._backbone = self._backbone.to(device)
        self.eval()

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [0,1] input to InsightFace normalization: (x*255 - 127.5) / 127.5."""
        if not self.input_bgr:
            x = x.flip(1)  # RGB -> BGR
        x = x * 255.0
        x = (x - 127.5) / 127.5  # Match w600k_r50 (mean=127.5, std=127.5)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        out = self._backbone(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    @property
    def embed_dim(self) -> int:
        return 512


def load_arcface_model(
    onnx_path: str | Path | None = None,
    device: torch.device | None = None,
    input_bgr: bool = True,
) -> nn.Module:
    """Load differentiable ArcFace model (same weights as InsightFace baseline).

    Returns an nn.Module suitable for CarliniWagnerAttack.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArcFacePyTorch(onnx_path=onnx_path, input_bgr=input_bgr)
    model = model.to(device)
    model.eval()
    return model
