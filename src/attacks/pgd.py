"""Projected Gradient Descent (PGD) adversarial attack for face verification.

Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"
arXiv:1706.06083

Adapted from classification to face verification with impersonation (FAR) and
dodging (FRR) attack modes using embedding-space objectives.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdversarialAttack, AttackMode


class PGDAttack(AdversarialAttack):
    """Projected Gradient Descent attack for face verification.

    Iteratively perturbs input within an Lp-norm ball to maximize attack objective.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        *,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 40,
        norm: str = "Linf",
        random_start: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            model: Face embedding model.
            device: Device to run on.
            eps: Maximum perturbation (epsilon ball radius).
            alpha: Step size per iteration.
            steps: Number of PGD iterations.
            norm: Norm type ('Linf' or 'L2').
            random_start: Whether to start from random point within eps-ball.
        """
        super().__init__(model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.norm = norm
        self.random_start = random_start

    def __call__(
        self,
        x: torch.Tensor,
        target_embedding: torch.Tensor | None = None,
        source_embedding: torch.Tensor | None = None,
        mode: AttackMode = AttackMode.IMPERSONATION,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate PGD adversarial example."""
        if mode == AttackMode.IMPERSONATION and target_embedding is None:
            raise ValueError("target_embedding required for impersonation")
        if mode == AttackMode.DODGING and source_embedding is None:
            raise ValueError("source_embedding required for dodging")

        x = x.to(self.device)
        if target_embedding is not None:
            target_embedding = target_embedding.to(self.device)
        if source_embedding is not None:
            source_embedding = source_embedding.to(self.device)

        # Initialize adversarial example
        x_adv = x.clone().detach()

        if self.random_start:
            # Start from random point within eps-ball
            if self.norm == "Linf":
                x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            else:  # L2
                random_noise = torch.randn_like(x_adv)
                random_noise = random_noise / random_noise.norm(p=2, dim=(1, 2, 3), keepdim=True)
                x_adv = x_adv + random_noise * self.eps * torch.rand(x.size(0), 1, 1, 1, device=self.device)
            x_adv = torch.clamp(x_adv, 0, 1)

        x_adv.requires_grad_(True)

        from tqdm import tqdm

        for step in tqdm(range(self.steps), desc="PGD", unit="step"):
            # Zero gradients
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            
            # Forward pass
            emb_adv = self.model(x_adv)
            emb_adv = F.normalize(emb_adv, p=2, dim=1)

            # Compute loss based on attack mode
            if mode == AttackMode.IMPERSONATION:
                target_emb_norm = F.normalize(target_embedding, p=2, dim=1)
                sim_to_target = (emb_adv * target_emb_norm).sum(dim=1)
                # Maximize similarity to target: gradient ascent on similarity
                loss = -sim_to_target.sum()  # Negative because we'll do gradient descent
            else:  # DODGING
                source_emb_norm = F.normalize(source_embedding, p=2, dim=1)
                sim_to_source = (emb_adv * source_emb_norm).sum(dim=1)
                # Minimize similarity to source: gradient descent on similarity
                loss = sim_to_source.sum()

            # Backward pass
            loss.backward()

            with torch.no_grad():
                # Get gradient direction
                grad = x_adv.grad

                if self.norm == "Linf":
                    # Linf PGD step (gradient descent direction)
                    x_adv = x_adv - self.alpha * grad.sign()
                    # Project back to eps-ball
                    delta = torch.clamp(x_adv - x, -self.eps, self.eps)
                    x_adv = x + delta
                else:  # L2
                    # L2 PGD step
                    grad_norm = grad.norm(p=2, dim=(1, 2, 3), keepdim=True)
                    grad_normalized = grad / (grad_norm + 1e-8)
                    x_adv = x_adv - self.alpha * grad_normalized
                    # Project back to eps-ball
                    delta = x_adv - x
                    delta_norm = delta.norm(p=2, dim=(1, 2, 3), keepdim=True)
                    factor = self.eps / (delta_norm + 1e-8)
                    factor = torch.min(factor, torch.ones_like(factor))
                    x_adv = x + delta * factor

                # Clamp to valid image range [0, 1]
                x_adv = torch.clamp(x_adv, 0, 1)

            x_adv = x_adv.detach().requires_grad_(True)

        return x_adv.detach()
