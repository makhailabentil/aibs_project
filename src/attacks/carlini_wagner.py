"""Carlini-Wagner L2 adversarial attack adapted for face verification.

Reference: Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks"
arXiv:1608.04644

Adapted from classification to face verification with impersonation (FAR) and
dodging (FRR) attack modes using embedding-space objectives.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdversarialAttack, AttackMode


class CarliniWagnerAttack(AdversarialAttack):
    """Carlini-Wagner L2 attack for face verification.

    Optimizes: min ||x - x'||_2^2 + c * f(x')
    with x' = 0.5 * (tanh(w) + 1) to keep pixels in [0, 1].
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        *,
        c_init: float = 0.1,
        c_upper: float = 1e10,
        c_steps: int = 9,
        optimizer_steps: int = 1000,
        lr: float = 0.05,
        kappa: float = 0.0,
        dodge_threshold: float = 0.0,
            **kwargs: Any,
    ):
        """
        Args:
            model: Face embedding model.
            device: Device to run on.
            c_init: Initial value of c for binary search.
            c_upper: Upper bound for c binary search.
            c_steps: Number of binary search steps for c.
            optimizer_steps: Adam steps per c value.
            lr: Learning rate for Adam.
            kappa: Confidence margin (for impersonation: require target to be
                   kappa better than source).
            dodge_threshold: For dodging, success when cos_sim(emb, source) <= this
                   (default 0 = orthogonal; use 0.5–0.7 for easier success).
        """
        super().__init__(model, device)
        self.c_init = c_init
        self.c_upper = c_upper
        self.c_steps = c_steps
        self.optimizer_steps = optimizer_steps
        self.lr = lr
        self.kappa = kappa
        self.dodge_threshold = dodge_threshold

    def __call__(
        self,
        x: torch.Tensor,
        target_embedding: torch.Tensor | None = None,
        source_embedding: torch.Tensor | None = None,
        mode: AttackMode = AttackMode.IMPERSONATION,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate C&W adversarial example."""
        if mode == AttackMode.IMPERSONATION and target_embedding is None:
            raise ValueError("target_embedding required for impersonation")
        if mode == AttackMode.DODGING and source_embedding is None:
            raise ValueError("source_embedding required for dodging")

        x = x.to(self.device)
        if target_embedding is not None:
            target_embedding = target_embedding.to(self.device)
        if source_embedding is not None:
            source_embedding = source_embedding.to(self.device)

        # Inverse tanh to initialize w from x in [0,1]
        # x = 0.5*(tanh(w)+1) => tanh(w) = 2*x - 1, w = arctanh(2*x - 1)
        val = (2 * x - 1).clamp(-1 + 1e-7, 1 - 1e-7)
        w = torch.atanh(val)
        w = w.detach().requires_grad_(True)

        c_lower = 0.0
        c = self.c_init
        best_adv = x.clone()
        best_l2 = torch.full((x.size(0),), float("inf"), device=self.device)
        # Best-effort fallback when goal never achieved: track lowest f_val
        best_effort_adv = x.clone()
        best_effort_f = torch.full((x.size(0),), float("inf"), device=self.device)

        from tqdm import tqdm
        total_steps = self.c_steps * self.optimizer_steps
        pbar = tqdm(total=total_steps, desc="C&W", unit="step")
        for c_step in range(self.c_steps):
            opt = torch.optim.Adam([w], lr=self.lr)
            for step in range(self.optimizer_steps):
                opt.zero_grad()
                x_adv = 0.5 * (torch.tanh(w) + 1)
                l2 = ((x_adv - x) ** 2).sum(dim=(1, 2, 3))
                f_val = self._attack_loss(x_adv, target_embedding, source_embedding, mode)
                loss = (l2 + c * f_val).sum()
                loss.backward()
                opt.step()
                pbar.update(1)

            with torch.no_grad():
                x_adv = 0.5 * (torch.tanh(w) + 1)
                l2_adv = (x_adv - x).pow(2).sum(dim=(1, 2, 3))
                improved = self._is_goal_achieved(
                    x_adv, x, target_embedding, source_embedding, mode
                ) & (l2_adv < best_l2)
                if improved.any():
                    best_l2 = torch.where(improved, l2_adv, best_l2)
                    mask = improved.view(-1, 1, 1, 1).expand_as(best_adv)
                    best_adv = torch.where(mask, x_adv, best_adv)

            f_val = self._attack_loss(x_adv, target_embedding, source_embedding, mode)
            success = self._is_success(f_val, mode)
            # Track best effort (lowest f_val) when goal not achieved
            better_effort = (f_val < best_effort_f) & ~success
            if better_effort.any():
                best_effort_f = torch.where(better_effort, f_val, best_effort_f)
                mask = better_effort.view(-1, 1, 1, 1).expand_as(best_effort_adv)
                best_effort_adv = torch.where(mask, x_adv, best_effort_adv)
            if success.all():
                c_upper = c
                c = (c_lower + c_upper) / 2
            else:
                c_lower = c
                c = min(c * 2, self.c_upper)
        pbar.close()
        # If we never achieved the goal, return best effort (closest attempt)
        if (best_l2 == float("inf")).all():
            return best_effort_adv.detach()
        return best_adv.detach()

    def _attack_loss(
        self,
        x_adv: torch.Tensor,
        target_embedding: torch.Tensor | None,
        source_embedding: torch.Tensor | None,
        mode: AttackMode,
    ) -> torch.Tensor:
        """Compute f(x') for the C&W objective."""
        emb = self.model(x_adv)
        emb = F.normalize(emb, p=2, dim=1)

        if mode == AttackMode.IMPERSONATION:
            target_embedding = F.normalize(target_embedding, p=2, dim=1)
            sim_to_target = (emb * target_embedding).sum(dim=1)
            if source_embedding is not None:
                source_embedding = F.normalize(source_embedding, p=2, dim=1)
                sim_to_source = (emb * source_embedding).sum(dim=1)
                # Ranking: want sim_to_target > sim_to_source
                # f = max(0, sim_to_source - sim_to_target + kappa)
                return torch.clamp(sim_to_source - sim_to_target + self.kappa, min=0)
            # Fallback: just maximize cos_sim to target
            return torch.clamp(self.kappa - sim_to_target, min=0)

        else:  # DODGING
            source_embedding = F.normalize(source_embedding, p=2, dim=1)
            # Minimize cos_sim(emb, source); success when cos_sim <= dodge_threshold
            cos_sim = (emb * source_embedding).sum(dim=1)
            dodge_threshold = self.dodge_threshold
            return torch.clamp(cos_sim - dodge_threshold, min=0)

    def _is_success(self, f_val: torch.Tensor, mode: AttackMode) -> torch.Tensor:
        """True where attack succeeded (f <= 0)."""
        return f_val <= 0

    def _is_goal_achieved(
        self,
        x_adv: torch.Tensor,
        x: torch.Tensor,
        target_embedding: torch.Tensor | None,
        source_embedding: torch.Tensor | None,
        mode: AttackMode,
    ) -> torch.Tensor:
        """True where the practical attack goal is achieved (used for best_adv updates)."""
        emb_adv = self.model(x_adv)
        emb_adv = F.normalize(emb_adv, p=2, dim=1)

        if mode == AttackMode.IMPERSONATION:
            if target_embedding is None:
                return torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            t_emb = F.normalize(target_embedding, p=2, dim=1)
            s_emb = F.normalize(self.model(x), p=2, dim=1) if source_embedding is None else F.normalize(source_embedding, p=2, dim=1)
            sim_to_target = (emb_adv * t_emb).sum(dim=1)
            sim_to_source = (emb_adv * s_emb).sum(dim=1)
            return sim_to_target > sim_to_source
        else:
            if source_embedding is None:
                return torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            s_emb = F.normalize(source_embedding, p=2, dim=1)
            sim_adv_to_source = (emb_adv * s_emb).sum(dim=1)
            emb_orig = F.normalize(self.model(x), p=2, dim=1)
            sim_orig_to_source = (emb_orig * s_emb).sum(dim=1)
            return sim_adv_to_source < sim_orig_to_source
