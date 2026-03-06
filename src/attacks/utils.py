import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


def cosine_similarity(e1: torch.Tensor, e2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two embedding tensors.
    Handles any shape by flattening to 1-D before comparing.
    """
    e1 = F.normalize(e1.reshape(1, -1), dim=1)
    e2 = F.normalize(e2.reshape(1, -1), dim=1)
    # sum over the embedding dim → scalar
    return torch.sum(e1 * e2).item()


def visualize_perturbations(
    perturbed_img: torch.Tensor,
    img: torch.Tensor,
    model,
    title: str | None = None,
):
    """
    Visualizes original vs adversarial image and reports cosine similarity.
    """

    os.makedirs("results", exist_ok=True)

    with torch.no_grad():
        emb_orig = model(img.unsqueeze(0))
        emb_adv = model(perturbed_img.unsqueeze(0))

    sim = cosine_similarity(emb_orig, emb_adv)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(
        f"ArcFace attack\nCosine similarity: {sim:.4f}",
        fontsize=14,
        fontfamily="monospace",
    )

    axs[0].imshow(
        np.transpose(torch.clamp(img, 0, 1).cpu().numpy(), (1, 2, 0))
    )
    axs[0].set_title("Original")

    axs[1].imshow(
        np.transpose(torch.clamp(perturbed_img, 0, 1).cpu().numpy(), (1, 2, 0))
    )
    axs[1].set_title("Adversarial")

    for ax in axs:
        ax.axis("off")

    if title is None:
        title = "opa_attack"

    plt.savefig(f"images/{title}.png", bbox_inches="tight")
    plt.close()