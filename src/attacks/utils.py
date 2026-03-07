import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


def cosine_similarity(e1: torch.Tensor, e2: torch.Tensor) -> float:
    e1 = F.normalize(e1.reshape(1, -1), dim=1)
    e2 = F.normalize(e2.reshape(1, -1), dim=1)
    return torch.sum(e1 * e2).item()


def visualize_perturbations(
    perturbed_img: torch.Tensor,
    img: torch.Tensor,
    model,
    title: str | None = None,
    mode: str = "dodging",
    target_img: torch.Tensor | None = None,
):
    """
    Visualizes attack results and saves the figure.

    Dodging      : 2-panel figure (original | adversarial).
                   Reports sim(orig, adv) — should be low.

    Impersonation: 3-panel figure (original | adversarial | target).
                   Reports sim(orig, adv) and sim(target, adv) — second should be high.

    Parameters
    ----------
    perturbed_img : Adversarial image tensor. Shape: [C, H, W].
    img           : Original image tensor.   Shape: [C, H, W].
    model         : Face recognition model.
    title         : Filename stem for saving (default: 'opa_attack').
    mode          : 'dodging' or 'impersonation'.
    target_img    : Target identity image. Required when mode='impersonation'.
    """
    if mode == "impersonation" and target_img is None:
        raise ValueError("target_img must be provided when mode='impersonation'.")

    os.makedirs("results", exist_ok=True)

    def to_np(t):
        return np.transpose(torch.clamp(t, 0, 1).cpu().numpy(), (1, 2, 0))

    with torch.no_grad():
        emb_orig = model(img.unsqueeze(0))
        emb_adv  = model(perturbed_img.unsqueeze(0))

    sim_orig_adv = cosine_similarity(emb_orig, emb_adv)

    if mode == "impersonation":
        with torch.no_grad():
            emb_target = model(target_img.unsqueeze(0))
        sim_target_adv = cosine_similarity(emb_target, emb_adv)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"OPA — Impersonation\n"
            f"sim(orig, adv) = {sim_orig_adv:.4f}    "
            f"sim(target, adv) = {sim_target_adv:.4f}",
            fontsize=13,
            fontfamily="monospace",
        )
        axs[0].imshow(to_np(img))
        axs[0].set_title("Original")

        axs[1].imshow(to_np(perturbed_img))
        axs[1].set_title("Adversarial")

        axs[2].imshow(to_np(target_img))
        axs[2].set_title("Target")

    else:  # dodging
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(
            f"OPA — Dodging\n"
            f"sim(orig, adv) = {sim_orig_adv:.4f}",
            fontsize=13,
            fontfamily="monospace",
        )
        axs[0].imshow(to_np(img))
        axs[0].set_title("Original")

        axs[1].imshow(to_np(perturbed_img))
        axs[1].set_title("Adversarial")

    for ax in axs:
        ax.axis("off")

    if title is None:
        title = f"opa_{mode}"

    save_path = f"results/{title}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Figure saved → {save_path}")