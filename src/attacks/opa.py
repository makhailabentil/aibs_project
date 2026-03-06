import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import List


class OnePixelAttack:
    """
    Optimized One Pixel Attack (OPA) using differential evolution.
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
    ) -> None:
        """
        Parameters
        ----------
        model     : Black-box model. Must accept (1, C, H, W) and return embeddings.
        img       : Original image tensor. Shape: [C, H, W], values in [0, 1].
        label     : Unused (kept for API compatibility).
        n         : Population size per pixel.
        threshold : Cosine similarity target — stop early when sim < threshold.
        F_scale   : Differential evolution scale factor (mutation strength).
        cr        : Crossover probability for DE.
        """
        self.n = n
        self.F = F_scale
        self.cr = cr
        self.threshold = threshold
        self.img = img
        self.label = label
        self.model = model

        _, self.H, self.W = img.shape

        with torch.no_grad():
            emb = self.model(img.unsqueeze(0))

        # Pre-normalize and cache the original embedding once
        self.orig_embedding = F.normalize(emb.reshape(-1), dim=0)
        self.perturbed_img = img.clone()
        self.historical_fitness = []

        # Initial population and fitness (lazy: populated per pixel in _get_perturbations)
        self.population = torch.rand(n, 5)
        self.fitness = self._evaluate_population_bulk(self.population, [])


    def _apply_perturbations(
        self, base: torch.Tensor, perturbations: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply a list of perturbations to *base* image in one shot.
        Perturbation vector: [x_norm, y_norm, r, g, b] all in [0, 1].

        Returns a new tensor (does not modify base).
        """
        p_img = base.clone()
        for per in perturbations:
            xi = int(np.floor(per[0].item() * self.W))
            yi = int(np.floor(per[1].item() * self.H))
            xi = min(xi, self.W - 1)
            yi = min(yi, self.H - 1)
            # Continuous RGB — no binarization, allows subtle perturbations
            r = per[2].item()
            g = per[3].item()
            b = per[4].item()
            p_img[0, yi, xi] = r
            p_img[1, yi, xi] = g
            p_img[2, yi, xi] = b
        return p_img

    def _fitness(self, perturbed_img: torch.Tensor) -> float:
        """
        Loss = -log(1 - cosine_sim + eps).
        Higher cosine sim → higher loss → worse fitness.
        We minimise fitness, so minimum = most dissimilar embedding.
        """
        with torch.no_grad():
            emb = self.model(perturbed_img.unsqueeze(0))
        emb = F.normalize(emb.reshape(-1), dim=0)
        sim = torch.sum(self.orig_embedding * emb).clamp(-1 + 1e-6, 1 - 1e-6)
        loss = -torch.log(1.0 - sim + 1e-6)
        return loss.item()

    def _evaluate_population_bulk(
        self,
        population: torch.Tensor,
        fixed_perturbations: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Evaluate fitness for the whole population given already-fixed perturbations.
        """
        fitness = torch.zeros(self.n, 1)
        for i, per in enumerate(population):
            perturbs = fixed_perturbations + [per]
            p_img = self._apply_perturbations(self.img, perturbs)
            fitness[i, 0] = self._fitness(p_img)
        return fitness
    
    def _evolve(
        self,
        fixed_perturbations: List[torch.Tensor],
        epochs: int,
        print_every: int | None,
    ) -> None:
        """
        Run DE for one pixel slot, modifying self.population and self.fitness in-place.
        Uses binomial crossover in addition to mutation for better exploration.
        """
        for epoch in range(epochs):
            for i in range(self.n):
                # --- Mutation ---
                idxs = np.random.choice(self.n, 3, replace=False)
                while i in idxs:
                    idxs = np.random.choice(self.n, 3, replace=False)
                r1, r2, r3 = (self.population[j] for j in idxs)

                # Adaptive F with small per-individual jitter
                f_jitter = self.F + 0.1 * (np.random.rand() - 0.5)
                mutant = r1 + f_jitter * (r2 - r3)
                mutant = mutant.clamp(0.0, 1.0)

                # --- Binomial crossover ---
                cross_mask = torch.rand(5) < self.cr
                if not cross_mask.any():
                    cross_mask[np.random.randint(5)] = True
                trial = torch.where(cross_mask, mutant, self.population[i])

                # --- Selection ---
                trial_perturbs = fixed_perturbations + [trial]
                trial_img = self._apply_perturbations(self.img, trial_perturbs)
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
        """
        Sequentially optimise d pixel perturbations.
        Each pixel is optimised while holding previous pixels fixed (greedy).
        Includes early stopping if cosine similarity already below threshold.
        """
        d_best: List[torch.Tensor] = []

        for pixel_idx in range(d):
            if print_every is not None:
                print(f"\n[Pixel {pixel_idx + 1}/{d}]")

            # Fresh population for this pixel slot
            self.population = torch.rand(self.n, 5)
            self.fitness = self._evaluate_population_bulk(self.population, d_best)

            self._evolve(d_best, epochs, print_every)

            best_idx = int(torch.argmin(self.fitness).item())
            best_per = self.population[best_idx].clone()
            d_best.append(best_per)

            # Early stopping: check current cosine similarity
            candidate = self._apply_perturbations(self.img, d_best)
            with torch.no_grad():
                emb = self.model(candidate.unsqueeze(0))
            emb = F.normalize(emb.reshape(-1), dim=0)
            sim = torch.sum(self.orig_embedding * emb).item()

            if print_every is not None:
                print(f"  → cosine sim after {pixel_idx + 1} pixel(s): {sim:.4f}")

            if sim < self.threshold:
                if print_every is not None:
                    print(f"  ✓ Early stop: sim {sim:.4f} < threshold {self.threshold}")
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
        """
        Run the attack and return the perturbed image.

        Parameters
        ----------
        epochs      : DE epochs per pixel.
        d           : Max number of pixels to perturb.
        print_every : Print progress every N epochs (None = silent).
        show        : If True, save comparison figure via visualize_perturbations.

        Returns
        -------
        perturbed_img : Adversarial image. Shape: [C, H, W].
        perturbations : List of best perturbation vectors found.
        """
        perturbations = self._get_perturbations(epochs, d, print_every)
        self.perturbed_img = self._apply_perturbations(self.img, perturbations)

        if show:
            from attacks.utils import visualize_perturbations
            visualize_perturbations(
                self.perturbed_img, self.img, self.model, **kwargs_visualize
            )

        return self.perturbed_img, perturbations