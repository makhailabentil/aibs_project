import cv2
import torch
import random
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from insightface.model_zoo import get_model
from opa import OnePixelAttack
import os


DATASET = "data/casia_webface_extracted"
N_IMAGES = 1
THRESHOLD = 0.4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs("opa_images", exist_ok=True)


def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (112,112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = torch.tensor(img / 255.).permute(2,0,1).float()
    return img


def cosine(e1,e2):
    e1 = e1.reshape(-1)
    e2 = e2.reshape(-1)
    e1 = F.normalize(e1,dim=0)
    e2 = F.normalize(e2,dim=0)
    return torch.sum(e1*e2).item()


def main():

    # Load ArcFace model
    rec = get_model("models/w600k_r50.onnx")
    rec.prepare(ctx_id=-1)

    class ArcFaceWrapper(torch.nn.Module):

        def __init__(self, rec):
            super().__init__()
            self.rec = rec

        def forward(self,x):

            img = x.squeeze().permute(1,2,0).numpy()
            img = (img*255).astype("uint8")

            emb = self.rec.get_feat(img)

            return torch.tensor(emb).unsqueeze(0)

    model = ArcFaceWrapper(rec)

    # collect random images
    images = ["data/casia_webface_extracted/0000045/001.jpg"]
   #sample = random.sample(images, N_IMAGES)

    success = 0

    for i,path in enumerate(images):

        print(f"\nImage {i+1}: {path}")

        img = load_image(path)

        with torch.no_grad():
            orig_emb = model(img.unsqueeze(0))

        attack = OnePixelAttack(model, img, label=None, n=60, threshold=0.4, cr=0.9)
        adv_img, _ = attack.perturb_img(epochs=30, d=15, print_every=5)
        
        diff = torch.abs(adv_img - img).mean().item()
        print("Mean perturbation:", diff)

        # SAVE THE IMAGE
        img_np = adv_img.permute(1,2,0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        save_path = f"opa_images/adv_{i}.jpg"
        cv2.imwrite(save_path, img_np)

        with torch.no_grad():
            adv_emb = model(adv_img.unsqueeze(0))

        sim = cosine(orig_emb,adv_emb)

        print("Cosine similarity:",sim)

        if sim < THRESHOLD:
            print("Attack SUCCESS")
            success+=1
        else:
            print("Attack FAILED")

    print("\nAttack success rate:",success,"/",N_IMAGES)


if __name__ == "__main__":
    main()