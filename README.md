# Design and Evaluation of Face Recognition Systems Under Adversarial Attacks

**Authors:** Maria Teresa Franco, Luqi Sun, MaKhaila Bentil, YiChiao Wang, Xiutian Zhao  
**Institution:** Johns Hopkins University  
**Domain:** AI & Biometrics Security

---

## Motivation

Face verification (Face ID) is widely used for biometric authentication but is vulnerable to **adversarial example attacks**, which threaten its security and reliability.

**Project goal:** Set up a Face ID system, study its vulnerabilities to adversarial attacks, improve its security, anticipate possible attacks, and explore retraining the model with adversarial examples to enhance robustness.

**Hypothesis:** Adversarial training improves robustness under attack.

---

## Dataset and Task

- **Task:** Face verification (Face ID): identity labels and multiple images per identity.
- **Dataset:** [CASIA-WebFace](docs/DATASET.md) [3]
  - ~10,575 identities, ~494,414 face images
  - One folder per person, multiple images per identity
  - Collected from the Internet; wide variation in pose, expression, illumination, and resolution

### Workspace setup (get the dataset)

The dataset is not in the repo. Each team member should run the setup script from the **project root**:

1. Install the helper once: `pip install kagglehub`
2. Run: `python scripts/get_dataset.py --kaggle`

This downloads from Kaggle, then writes into:
- **`data/casia_webface/`** – training data (record format: `train.rec`, `train.idx`, `train.lst`; use an MXNet/RecordIO loader in code)
- **`data/eval/`** – evaluation bins (e.g. `lfw.bin`, `agedb_30.bin`)

**If you want folder-per-identity .jpg** (e.g. `0000045/001.jpg`) instead of the record file, run:  
`python scripts/extract_rec_to_folders.py`  
Output: `data/casia_webface_extracted/`. Full extraction takes ~30–60 min and needs **about 2.5–3 GB** extra disk space; use `--limit 5000` to test. See [docs/DATASET.md](docs/DATASET.md).

To save disk space when downloading, add `--symlink` to the get_dataset command. For other options (local archive, etc.), see [docs/DATASET.md](docs/DATASET.md).

---


## Adversarial Attacks

- **C&W** (Carlini–Wagner) — implemented
- **One Pixel** — implemented
- **PGD** (Projected Gradient Descent) — implemented

---

## Project Timeline & Checklist

### Mid-Semester Report: **Mar 8, 11:59pm**
- [ ] Submit mid-semester report
- **Target for report:** Tasks 1–3 complete; baseline pipeline, clean benchmark (FAR, FRR, EER), and attack implementations (PGD, One Pixel, C&W) done. Plan the rest of the semester (Task 4) in the report.

---

### Task 1: Baseline Verification Pipeline  
**Deadline:** _[Before mid-semester report, e.g., Feb 28]_  
**Owner:** Luqi Sun, Xiutian Zhao

- [x] Implement ArcFace (InsightFace) baseline model
- [x] Implement initial Face ID verification system
- [ ] Calibrate verification thresholds
- [x] Document pipeline (config, scripts, usage)
- [x] Update README with model details once finalized

---

### Task 2: Clean Benchmark Performance  
**Deadline:** _[Before mid-semester report, e.g., March 8]_  
**Owner:** Luqi Sun, Xiutian Zhao

- [ ] Define verification pairs from held-out split (consistent with baseline model)
- [ ] Calibrate decision threshold on validation set
- [ ] Report **FAR**, **FRR**, **EER** on clean data
- [ ] Save benchmarks and plots for comparison

---

### Task 3: Adversarial Attacks & Metrics  
**Deadline:** _[Before mid-semester report, Mar 8]_  
**Owners:** PGD (YiChiao Wang), One Pixel (Maria Teresa Franco), C&W (MaKhaila Bentil)

- [x] Document baseline model interface (embedding API) for attack integration
- [x] Implement **PGD**
- [x] Implement **One Pixel**
- [x] Implement **C&W**
- [ ] Report attack success rates per attack
- [ ] Report robust verification metrics under attack

---

### Task 4: Adversarial Training & Re-evaluation  
**Deadline:** _[Update: e.g., Week 9 / Apr 5]_  
**Owner:** _[Assign; after Tasks 1–3]_

- [ ] Implement adversarial training (e.g. PGD-based)
- [ ] Re-evaluate on clean metrics (FAR, FRR, EER)
- [ ] Re-evaluate on robust metrics under PGD, One Pixel, C&W
- [ ] Compare baseline vs adversarially trained model; document findings

---

## Repository Structure

```
aibs_project/
├── README.md              # This file
├── docs/
│   └── DATASET.md         # How to get the dataset and run the setup script
├── scripts/
│   ├── get_dataset.py     # Download/set up CASIA-WebFace (data/casia_webface/, data/eval/)
│   ├── extract_rec_to_folders.py   # Optional: unpack train.rec to folder-per-identity .jpg (data/casia_webface_extracted/)
│   ├── arcface_eval_bin.py        # Evaluate ArcFace on .bin verification pairs (LFW, AgeDB, etc.)
│   ├── run_cw_attack.py           # Run C&W attack on ArcFace with CASIA face images
│   ├── run_pgd_attack.py          # Run PGD attack on ArcFace with CASIA face images
│   └── run_opa.py                 # Run One Pixel attack (CLI entry)
├── src/                   # Code: verification, attacks (e.g. carlini_wagner, pgd, opa), training
├── tests/                 # Unit tests (e.g. test_carlini_wagner.py)
├── data/                  # casia_webface/, eval/; optional casia_webface_extracted/ (gitignored)
├── results/               # Logs, metrics, figures
└── requirements.txt       # Python dependencies
```

---

## References

### Face recognition & architecture

[1] M. Jha, A. Tiwari, M. Himansh, and V. M. Manikandan, "Face recognition: recent advancements and research challenges," in 2022 13th International Conference on Computing Communication and Networking Technologies (ICCCNT). IEEE, 2022, pp. 1–6.

[2] S. Kilany and A. Mahfouz, "A comprehensive survey of deep face verification systems adversarial attacks and defense strategies," Scientific Reports, vol. 15, p. 30861, 2025.

[3] D. Yi, Z. Lei, S. Liao, and S. Z. Li, "Learning face representation from scratch," arXiv preprint arXiv:1411.7923, 2014.

[4] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "Arcface: Additive angular margin loss for deep face recognition," Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4690–4699, 2019.

[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

[6] Y. Taigman, M. Yang, M. Ranzato, and L. Wolf, "DeepFace: Closing the gap to human-level performance in face verification," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.

[7] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

[8] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the inception architecture for computer vision," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

### Adversarial attacks & robustness

[9] I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in International Conference on Learning Representations (ICLR), 2015.

[10] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in Advances in Neural Information Processing Systems (NeurIPS), 2014.

[11] A. Kurakin, I. Goodfellow, and S. Bengio, "Adversarial examples in the physical world," in ICLR Workshop, 2017.

[12] N. Carlini and D. Wagner, "Adversarial examples are not easily detected: Bypassing ten detection methods," in Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security (AISEC), 2017.

[13] N. Carlini and D. Wagner, "Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples," in International Conference on Machine Learning (ICML), 2018.

[14] J. Cohen, E. Rosenfeld, and Z. Kolter, "Certified adversarial robustness via randomized smoothing," in International Conference on Machine Learning (ICML), 2019.

[15] J. Gilmer, L. Metz, F. R. Schroff, I. Goodfellow, D. Sussillo, and J. Snoek, "Perceptual adversarial examples," in International Conference on Learning Representations (ICLR), 2019.

[16] Z. Li, B. Li, J. Bi, X. Jia, Z. Liu, and J. Yan, "Attack that network: A two-stage adversarial attack against deep face recognition," in IEEE Transactions on Information Forensics and Security, 2020.

### Adversarial attack references (Medium)

[17] A. G. Zachariah, "Adversarial attacks with Carlini & Wagner approach," Medium, 2023. https://medium.com/@zachariaharungeorge/adversarial-attacks-with-carlini-wagner-approach-8307daa9a503

[18] A. G. Zachariah, "Unveiling the power of projected gradient descent in adversarial attacks," Medium, 2023. https://medium.com/@zachariaharungeorge/unveiling-the-power-of-projected-gradient-descent-in-adversarial-attacks-2f92509dde3c

[19] "One pixel attack: Breaking deep learning models with minimal perturbation," Medium, 2019. https://hiya31.medium.com/one-pixel-attack-breaking-deep-learning-models-with-minimal-perturbation-766d8df397e8

---

## Getting Started

1. Clone the repo, then run the dataset script (see **Workspace setup** above or [docs/DATASET.md](docs/DATASET.md)).
2. Extract folder-per-identity images (for C&W attack): `python scripts/extract_rec_to_folders.py --limit 100`
3. Run tests: `pytest tests/` (some tests require extracted data)
4. Run C&W attack: `python scripts/run_cw_attack.py --data data/casia_webface_extracted`  
   (Requires ArcFace model w600k_r50.onnx; run `arcface_eval_bin.py` once with FaceAnalysis to download.)
5. Run One Pixel attack: `python scripts/run_opa.py` (uses data/casia_webface_extracted; see script help for output paths)
6. Use the checklist above to assign tasks and set deadlines; update this README as you go.
