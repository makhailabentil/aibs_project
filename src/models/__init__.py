"""Face recognition models for verification and attack evaluation."""

from .arcface_pytorch import ArcFacePyTorch, load_arcface_model

__all__ = ["ArcFacePyTorch", "load_arcface_model"]
