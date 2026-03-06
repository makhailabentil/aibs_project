"""Base interface for adversarial attacks on face verification."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import torch


class AttackMode(str, Enum):
    """Attack mode for face verification."""

    IMPERSONATION = "impersonation"  # FAR: make face A verified as identity B
    DODGING = "dodging"  # FRR: make face A NOT verified as identity A


class AdversarialAttack(ABC):
    """Abstract base class for adversarial attacks on face verification models."""

    def __init__(self, model: torch.nn.Module, device: torch.device | None = None):
        """
        Args:
            model: Face embedding model. Must expose forward(x) or get_embedding(x)
                   returning a fixed-size embedding vector.
            device: Device to run on. Defaults to cuda if available else cpu.
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding from model. Handles both forward and get_embedding interfaces."""
        with torch.no_grad():
            if hasattr(self.model, "get_embedding"):
                return self.model.get_embedding(x)
            return self.model(x)

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        target_embedding: torch.Tensor | None = None,
        source_embedding: torch.Tensor | None = None,
        mode: AttackMode = AttackMode.IMPERSONATION,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate adversarial example.

        Args:
            x: Input image tensor (N, C, H, W), values in [0, 1].
            target_embedding: Embedding of target identity (for impersonation).
            source_embedding: Embedding of source identity (for dodging).
            mode: Impersonation or dodging.
            **kwargs: Attack-specific parameters.

        Returns:
            Adversarial image tensor same shape as x, values in [0, 1].
        """
        raise NotImplementedError
