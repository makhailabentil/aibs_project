import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import List, Literal
from src.attacks.utils import visualize_perturbations


class OnePixelAttack:
    """
    Optimized One Pixel Attack (OPA) using differential evolution.

    Supports two attack modes:
    - 'dodging'       : Push the embedding away from the original identity (false reject).
    - 'impersonation' : Pull the embedding toward a target identity (false accept).
    """

    def __init__(
        self,
        model: nn.Module,
        img: torch.Tensor,
        label: int = None,
        n: int = 100,
        threshold: float = 0.4,
        F_scale: float = 0.8,
        cr: float = 0.9,
        mode: Literal["dodging", "impersonation"] = "dodging",
        target_img: torch.Tensor = None,
    ) -> None:
        """
        Parameters
        ----------
        model      : Black-box model. Must accept (1, C, H, W) and return embeddings.
        img        : Original image tensor. Shape: [C, H, W], values in [0, 1].
        label      : Unused (kept for API compatibility).
        n          : Population size per pixel.
        threshold  : Cosine similarity target for early stopping.
                     Dodging:       stop when sim(orig, adv)   < threshold.
                     Impersonation: stop when sim(target, adv) > threshold.
        F_scale    : Differential evolution scale factor (mutation strength).
        cr         : Crossover probability for DE.
        mode       : 'dodging' or 'impersonation'.
        target_img : Required when mode='impersonation'. The identity to impersonate.
        """
        if mode == "impersonation" and target_img is None:
            raise ValueError("target_img must be provided when mode='impersonation'.")

        self.n = n
        self.F = F_scale
        self.cr = cr
        self.threshold = threshold
        self.img = img
        self.label = label
        self.model = model
        self.mode = mode
        self.target_img = target_img

        _, self.H, self.W = img.shape

        # Cache original embedding (used in dodging + for cosine reporting)
        with torch.no_grad():
            orig_emb = self.model(img.unsqueeze(0))
        self.orig_embedding = F.normalize(orig_emb.reshape(-1), dim=0)

        # Reference embedding used in fitness depends on mode
        if mode == "impersonation":
            with torch.no_grad():
                target_emb = self.model(target_img.unsqueeze(0))
            self.ref_embedding = F.normalize(target_emb.reshape(-1), dim=0)
        else:
            self.ref_embedding = self.orig_embedding

        self.perturbed_img = img.clone()
        self.historical_fitness = []

        self.population = torch.rand(n, 5)
        self.fitness = self._evaluate_population_bulk(self.population, [])


    def _apply_perturbations(
        self, base: torch.Tensor, perturbations: List[torch.Tensor]
    ) -> torch.Tensor:
        p_img = base.clone()
        for per in perturbations:
            xi = int(np.floor(per[0].item() * self.W))
            yi = int(np.floor(per[1].item() * self.H))
            xi = min(xi, self.W - 1)
            yi = min(yi, self.H - 1)
            p_img[0, yi, xi] = per[2].item()
            p_img[1, yi, xi] = per[3].item()
            p_img[2, yi, xi] = per[4].item()
        return p_img

    def _fitness(self, perturbed_img: torch.Tensor) -> float:
        with torch.no_grad():
            emb = self.model(perturbed_img.unsqueeze(0))

        emb_vec = emb.reshape(-1)
        if emb_vec.norm() < 1e-8:
            return 1e6

        emb_vec = F.normalize(emb_vec, dim=0)
        sim = torch.sum(self.ref_embedding * emb_vec)  # in [-1, 1], no clamp

        if self.mode == "dodging":
            # Push sim toward -1: minimise sim directly
            loss = sim  # lower sim = lower loss = better
        else:
            # Pull sim toward 1: minimise (1 - sim)
            loss = 1.0 - sim  # lower = more similar to target

        val = loss.item()
        if not np.isfinite(val):
            return 1e6
        return val

    def _current_sim(self, candidate: torch.Tensor) -> float:
        """Returns the sim value used for early-stopping, depending on mode."""
        with torch.no_grad():
            emb = self.model(candidate.unsqueeze(0))
        emb = F.normalize(emb.reshape(-1), dim=0)
        return torch.sum(self.ref_embedding * emb).item()

    def _should_stop(self, sim: float) -> bool:
        if self.mode == "dodging":
            return sim < self.threshold
        else:
            return sim > self.threshold

    def _evaluate_population_bulk(
        self,
        population: torch.Tensor,
        fixed_perturbations: List[torch.Tensor],
    ) -> torch.Tensor:
        fitness = torch.zeros(self.n, 1)
        for i, per in enumerate(population):
            p_img = self._apply_perturbations(self.img, fixed_perturbations + [per])
            fitness[i, 0] = self._fitness(p_img)
        return fitness

    def _evolve(
        self,
        fixed_perturbations: List[torch.Tensor],
        epochs: int,
        print_every: int | None,
    ) -> None:
        for epoch in range(epochs):
            for i in range(self.n):
                idxs = np.random.choice(self.n, 3, replace=False)
                while i in idxs:
                    idxs = np.random.choice(self.n, 3, replace=False)
                r1, r2, r3 = (self.population[j] for j in idxs)

                f_jitter = self.F + 0.1 * (np.random.rand() - 0.5)
                mutant = (r1 + f_jitter * (r2 - r3)).clamp(0.0, 1.0)

                cross_mask = torch.rand(5) < self.cr
                if not cross_mask.any():
                    cross_mask[np.random.randint(5)] = True
                trial = torch.where(cross_mask, mutant, self.population[i])

                trial_img = self._apply_perturbations(self.img, fixed_perturbations + [trial])
                trial_fitness = self._fitness(trial_img)

                if trial_fitness < self.fitness[i, 0].item():
                    self.population[i] = trial
                    self.fitness[i, 0] = trial_fitness

            best = torch.min(self.fitness).item()
            self.historical_fitness.append(best)

            if print_every is not None and epoch % print_every == 0:
                print(f"  Epoch {epoch:3d} | best fitness: {best:.4f}")

    def _get_perturbations(
        self,
        epochs: int,
        d: int,
        print_every: int | None,
    ) -> List[torch.Tensor]:
        d_best: List[torch.Tensor] = []

        for pixel_idx in range(d):
            if print_every is not None:
                print(f"\n[Pixel {pixel_idx + 1}/{d}]")

            self.population = torch.rand(self.n, 5)
            self.fitness = self._evaluate_population_bulk(self.population, d_best)
            self._evolve(d_best, epochs, print_every)

            best_idx = int(torch.argmin(self.fitness).item())
            d_best.append(self.population[best_idx].clone())

            candidate = self._apply_perturbations(self.img, d_best)
            sim = self._current_sim(candidate)

            if print_every is not None:
                label = "sim(orig, adv)" if self.mode == "dodging" else "sim(target, adv)"
                print(f"  → {label} after {pixel_idx + 1} pixel(s): {sim:.4f}")

            if self._should_stop(sim):
                if print_every is not None:
                    print(f"  ✓ Early stop: threshold reached ({sim:.4f})")
                break

        return d_best

    def perturb_img(
        self,
        epochs: int = 50,
        d: int = 1,
        print_every: int | None = None,
        show: bool = True,
        **kwargs_visualize,
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        perturbations = self._get_perturbations(epochs, d, print_every)
        self.perturbed_img = self._apply_perturbations(self.img, perturbations)

        if show:
            visualize_perturbations(
                self.perturbed_img,
                self.img,
                self.model,
                mode=self.mode,
                target_img=self.target_img if self.mode == "impersonation" else None,
                **kwargs_visualize,
            )

        return self.perturbed_img, perturbations