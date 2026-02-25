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

## Architecture Options

| Option | Approach | Details |
|--------|----------|---------|
| **I** | Feature extractor | ArcFace [4] embedding model via InsightFace; design verification logic, attacks, and adversarial training |
| **II** | Train from scratch | CNN (e.g. ResNet-18 [5]) for laptop-scale training; same focus on verification, attacks, and adversarial training |

---

## Adversarial Attacks (Planned)

- **Setting:** Digital adversarial example attacks [6]. Evaluate using **impersonation (FAR)** and **dodging (FRR)** under verification.
- **White-box:** PGD (Projected Gradient Descent), C&W (Carlini–Wagner).
- **Black-box:** Transfer attack (craft on surrogate model, test on target).

---

## Project Timeline & Checklist

### Mid-Semester Report: **Mar 8, 11:59pm**
- [ ] Submit mid-semester report
- **Target for report:** Tasks 1–2 complete; baseline pipeline and clean benchmark (FAR, FRR, EER) done. Optionally include progress on Task 3 (attacks). Plan the rest of the semester (Tasks 3–4) in the report.

---

### Task 1: Baseline Verification Pipeline  
**Deadline:** _[Before mid-semester report, e.g., Feb 28]_  
**Owner:** _[Assign]_

- [ ] Implement initial Face ID verification system (Option I or II)
- [ ] Calibrate verification thresholds
- [ ] Document pipeline (config, scripts, usage)

---

### Task 2: Clean Benchmark Performance  
**Deadline:** _[Before mid-semester report, e.g., March 8]_  
**Owner:** _[Assign]_

- [ ] Generate verification pairs from held-out split
- [ ] Calibrate decision threshold on validation set
- [ ] Report **FAR**, **FRR**, **EER** on clean data
- [ ] Save benchmarks and plots for comparison

---

### Task 3: Adversarial Attacks & Metrics  
**Deadline:** _[Update: e.g., Week 7 / Mar 22]_  
**Owner:** _[Assign]_ (can split: one person PGD, one CW, one transfer)

- [ ] Implement **PGD** (white-box)
- [ ] Implement **C&W** (white-box)
- [ ] Implement **transfer attack** (black-box)
- [ ] Report attack success rates per attack
- [ ] Report robust verification metrics (FAR/FRR under attack)

---

### Task 4: Adversarial Training & Re-evaluation  
**Deadline:** _[Update: e.g., Week 9 / Apr 5]_  
**Owner:** _[Assign]_

- [ ] Implement adversarial training (e.g. PGD-based)
- [ ] Re-evaluate on **clean** metrics (FAR, FRR, EER)
- [ ] Re-evaluate on **robust** metrics under same attacks
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
│   └── extract_rec_to_folders.py   # Optional: unpack train.rec to folder-per-identity .jpg (data/casia_webface_extracted/)
├── data/                  # casia_webface/, eval/; optional casia_webface_extracted/ (gitignored)
├── src/                   # Code: verification, attacks, training
├── experiments/           # Scripts and configs for runs
├── results/               # Logs, metrics, figures
└── requirements.txt       # Python dependencies (add as you go)
```

---

## References

[1] M. Jha, A. Tiwari, M. Himansh, and V. M. Manikandan, "Face recognition: recent advancements and research challenges," in 2022 13th International Conference on Computing Communication and Networking Technologies (ICCCNT). IEEE, 2022, pp. 1–6.

[2] S. Kilany and A. Mahfouz, "A comprehensive survey of deep face verification systems adversarial attacks and defense strategies," Scientific Reports, vol. 15, p. 30861, 2025.

[3] D. Yi, Z. Lei, S. Liao, and S. Z. Li, "Learning face representation from scratch," arXiv preprint arXiv:1411.7923, 2014.

[4] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "Arcface: Additive angular margin loss for deep face recognition," Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4690–4699, 2019.

[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

[6] I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in International Conference on Learning Representations (ICLR), 2015.

---

## Getting Started

1. Clone the repo, then run the dataset script (see **Workspace setup** above or [docs/DATASET.md](docs/DATASET.md)).
2. Pick architecture (Option I or II) and implement the baseline (Task 1).
3. Use the checklist above to assign tasks and set deadlines; update this README as you go.
