import cv2
import torch
import random
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from insightface.model_zoo import get_model
from src.attacks.opa import OnePixelAttack
from src.attacks.utils import cosine_similarity
import os


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATASET       = "data/casia_webface_extracted"
N_IMAGES      = 1
THRESHOLD     = 0.4
SEED          = 42
MODE          = "impersonation"       # "dodging" or "impersonation"

# Only used when MODE = "impersonation": path to the target identity image
TARGET_IMAGE  = "data/casia_webface_extracted/0000099/001.jpg"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs("results", exist_ok=True)


def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img / 255.).permute(2, 0, 1).float()
    return img



def main():

    # Load ArcFace model
    rec = get_model("models/w600k_r50.onnx")
    rec.prepare(ctx_id=-1, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])

    class ArcFaceWrapper(torch.nn.Module):
        def __init__(self, rec):
            super().__init__()
            self.rec = rec

        def forward(self, x):
            img = x.squeeze().permute(1, 2, 0).numpy()
            img = (img * 255).astype("uint8")
            emb = self.rec.get_feat(img)
            return torch.tensor(emb).unsqueeze(0)

    model = ArcFaceWrapper(rec)

    # Load target image for impersonation (ignored in dodging mode)
    target_img = load_image(TARGET_IMAGE) if MODE == "impersonation" else None

    if MODE == "impersonation":
        print(f"Mode: IMPERSONATION → target: {TARGET_IMAGE}")
        print(f"Success criterion: sim(target, adv) > {THRESHOLD}")
    else:
        print(f"Mode: DODGING")
        print(f"Success criterion: sim(orig, adv) < {THRESHOLD}")

    images = ["data/casia_webface_extracted/0000045/001.jpg"]
    success = 0

    for i, path in enumerate(images):
        print(f"\nImage {i + 1}: {path}")

        img = load_image(path)

        with torch.no_grad():
            orig_emb = model(img.unsqueeze(0))

        attack = OnePixelAttack(
            model,
            img,
            label=None,
            n=60,
            threshold=THRESHOLD,
            cr=0.9,
            mode=MODE,
            target_img=target_img,
        )

        adv_img, _ = attack.perturb_img(epochs=30, d=15, print_every=5)

        # Save adversarial image
        img_np = adv_img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"opa_images/adv_{MODE}_{i}.jpg", img_np)

        with torch.no_grad():
            adv_emb = model(adv_img.unsqueeze(0))

        # Report both similarities regardless of mode
        sim_orig   = cosine_similarity(orig_emb, adv_emb)
        print(f"sim(orig,   adv): {sim_orig:.4f}")

        if MODE == "impersonation":
            target_emb = model(target_img.unsqueeze(0))
            sim_target = cosine_similarity(target_emb, adv_emb)
            print(f"sim(target, adv): {sim_target:.4f}")
            attack_succeeded = sim_target > THRESHOLD
        else:
            attack_succeeded = sim_orig < THRESHOLD

        if attack_succeeded:
            print("Attack SUCCESS ✓")
            success += 1
        else:
            print("Attack FAILED ✗")

    print(f"\nAttack success rate: {success} / {N_IMAGES}")


if __name__ == "__main__":
    main()