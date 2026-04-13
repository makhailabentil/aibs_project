# PGD Adversarial Training Experiment Log (YiChiao)

## 1. Goal

The goal of this stage is to complete **Task 4: Adversarial Training and Re-evaluation** for our face verification project.

Our baseline face verification system uses the ArcFace model as an embedding extractor. Face verification is performed using cosine similarity between two embeddings, and a threshold is used to determine whether two images belong to the same identity.

From our earlier clean evaluation on LFW, the baseline threshold is:

- `tau_EER = 0.1767`

After implementing PGD attack and observing that the baseline model is highly vulnerable, the next step is to apply **PGD-based adversarial training** and evaluate whether the retrained model becomes more robust.

---

## 2. Overall Workflow

In this stage, I completed the following process:

1. Evaluated the baseline ArcFace model under PGD attack.
2. Implemented PGD-based adversarial training.
3. Trained a first retrained model (**v1**).
4. Re-ran PGD attack on the retrained v1 model.
5. Observed that v1 showed only partial improvement.
6. Strengthened the adversarial training setting and trained a second retrained model (**v2**).
7. Re-ran PGD attack on the retrained v2 model.
8. Compared the results across:
   - baseline
   - retrained v1
   - retrained v2

The comparison focused on two attack modes:

- **Impersonation attack**
- **Dodging attack**

---

## 3. PGD Evaluation Setting

For the PGD attack evaluation in this experiment, I used:

- Attack: PGD
- Norm: Linf
- `eps = 8/255 = 0.031373`
- `alpha = 2/255 = 0.007843`
- `steps = 40`
- Threshold: `0.1767`

The same threshold was used for baseline, retrained v1, and retrained v2 in this comparison.

### Important note

This comparison is still **preliminary**, because the threshold `0.1767` was originally calibrated for the baseline model on clean LFW evaluation.

The retrained models have **not yet been recalibrated with a new clean threshold**, so the current comparison is useful for identifying the overall trend, but it is not yet the final cleanest evaluation.

---

## 4. Baseline PGD Attack Results

### 4.1 Baseline Impersonation

- Total pairs: 10
- Eligible pairs: 10
- Successful attacks: 10
- Success rate: 100.0%
- Average clean similarity: -0.0049
- Average adversarial similarity: 0.8344
- Similarity change: +0.8392
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 4.4562

### Interpretation

The baseline model is highly vulnerable to PGD impersonation attack.

All eligible pairs were successfully attacked, and the average similarity increased from a value near zero to a clearly accepted region above the verification threshold.

---

### 4.2 Baseline Dodging

- Total pairs: 10
- Eligible pairs: 5
- Successful attacks: 5
- Success rate: 100.0%
- Average clean similarity: 0.4615
- Average adversarial similarity: -0.6884
- Similarity change: -1.1499
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 4.5636

### Interpretation

The baseline model is also highly vulnerable to PGD dodging attack.

For all eligible same-identity pairs, PGD successfully reduced the cosine similarity below the threshold, causing false rejection.

---

## 5. Adversarial Training v1

### 5.1 v1 Training Command

```bash
python scripts/retraining_pgd_yichiao.py --data data/casia_webface_extracted --epochs 2 --batch_size 8 --max_classes 100 --max_imgs_per_class 20 --steps 3 --eps 0.015686 --alpha 0.003922 --save_name arcface_pgd_adv_train_v2.pt
```

### 5.2 v1 Training Output

- Device: CPU
- Number of classes used: 28
- Number of samples used: 555

Epoch results:
- Epoch 1/2: loss = 3.1829, quick_train_acc = 0.8562
- Epoch 2/2: loss = 2.9540, quick_train_acc = 0.9187

Saved checkpoint:
- `results/arcface_pgd_adv_train_v2.pt`

### 5.3 v1 Interpretation

This means the adversarial training pipeline ran successfully.

The script was able to:

- load the data
- generate adversarial examples during training
- fine-tune the ArcFace backbone
- save a retrained checkpoint for later evaluation

The decreasing loss and increasing quick training accuracy suggest that the training process was working normally on the selected subset.

---

## 6. PGD Attack Results on Retrained v1

### 6.1 Retrained v1 Impersonation

- Total pairs: 10
- Eligible pairs: 8
- Successful attacks: 8
- Success rate: 100.0%
- Average clean similarity: 0.0094
- Average adversarial similarity: 0.7154
- Similarity change: +0.7060
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 5.4622

### Interpretation

The retrained v1 model is still highly vulnerable to PGD impersonation attack.

Although the average adversarial similarity is lower than the baseline case (`0.7154` vs `0.8344`), all eligible impersonation attacks still succeeded. This suggests that the v1 adversarial training setup did **not** provide sufficient robustness against strong PGD impersonation attacks.

---

### 6.2 Retrained v1 Dodging

- Total pairs: 10
- Eligible pairs: 9
- Successful attacks: 7
- Success rate: 77.8%
- Average clean similarity: 0.7948
- Average adversarial similarity: -0.1079
- Similarity change: -0.9027
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 5.5808

### Interpretation

Compared with the baseline result, the retrained v1 model shows **partial improvement** against PGD dodging attack.

The attack success rate decreased from **100.0%** to **77.8%**, which suggests that the v1 adversarial training setup may provide some robustness to dodging attacks. However, the retrained model is still vulnerable, since most eligible attacks still succeed.

---

## 7. Overall Interpretation of v1

The main findings of v1 are:

1. **PGD adversarial training v1 did not improve robustness against impersonation attack.**
   - The success rate remained at **100.0%** on eligible pairs.

2. **PGD adversarial training v1 provided only partial robustness against dodging attack.**
   - The success rate decreased from **100.0%** in the baseline model to **77.8%** in the retrained model.

3. **v1 was a successful first defense experiment, but not yet strong enough.**
   - The retrained model was evaluated using the same baseline threshold `0.1767`.
   - A stronger training setup was needed.

---

## 8. Adversarial Training v2

After observing that v1 was not strong enough, I trained a stronger v2 model with:

- more epochs
- stronger PGD epsilon and alpha during training
- more PGD steps during training

### 8.1 v2 Training Command

```bash
python scripts/retraining_pgd_yichiao.py --data data/casia_webface_extracted --epochs 5 --batch_size 8 --max_classes 28 --max_imgs_per_class 20 --steps 5 --eps 0.031373 --alpha 0.007843 --save_name arcface_pgd_adv_train_v3.pt
```

### 8.2 v2 Training Output

- Device: CPU
- Number of classes used: 28
- Number of samples used: 555

Epoch results:
- Epoch 1/5: loss = 3.2635, quick_train_acc = 0.8438
- Epoch 2/5: loss = 3.0629, quick_train_acc = 0.8688
- Epoch 3/5: loss = 2.9413, quick_train_acc = 0.9313
- Epoch 4/5: loss = 2.8444, quick_train_acc = 0.9812
- Epoch 5/5: loss = 2.7665, quick_train_acc = 0.9938

Saved checkpoint:
- `results/arcface_pgd_adv_train_v3.pt`

### 8.3 v2 Interpretation

Compared with v1, this version used stronger adversarial training settings.

The training loss kept decreasing and the quick training accuracy increased to a very high value, indicating that the stronger training setup was stable and successfully converged on the selected subset.

---

## 9. PGD Attack Results on Retrained v2

### 9.1 Retrained v2 Impersonation

- Total pairs: 10
- Eligible pairs: 7
- Successful attacks: 4
- Success rate: 57.1%
- Average clean similarity: -0.0553
- Average adversarial similarity: 0.2350
- Similarity change: +0.2904
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 5.9023

### Interpretation

Compared with both the baseline model and retrained v1, the stronger adversarial training setup in v2 substantially improved robustness against PGD impersonation attack.

The eligible success rate dropped from **100.0%** to **57.1%**, indicating that it became significantly harder for PGD to push different-identity pairs across the verification threshold.

The defense is still not perfect, but the improvement is clear.

---

### 9.2 Retrained v2 Dodging

- Total pairs: 10
- Eligible pairs: 10
- Successful attacks: 1
- Success rate: 10.0%
- Average clean similarity: 0.8660
- Average adversarial similarity: 0.6614
- Similarity change: -0.2046
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 5.9066

### Interpretation

The stronger adversarial training setup in v2 dramatically improved robustness against PGD dodging attack.

Compared with:

- baseline dodging success rate = **100.0%**
- retrained v1 dodging success rate = **77.8%**
- retrained v2 dodging success rate = **10.0%**

the v2 model is much more resistant to PGD-based false rejection attacks.

This is the strongest result obtained so far.

---

## 10. Full Results Comparison

| Model | Attack Mode | Total Pairs | Eligible Pairs | Successful Attacks | Success Rate |
|---|---:|---:|---:|---:|---:|
| Baseline | Impersonation | 10 | 10 | 10 | 100.0% |
| Retrained v1 | Impersonation | 10 | 8 | 8 | 100.0% |
| Retrained v2 | Impersonation | 10 | 7 | 4 | 57.1% |
| Baseline | Dodging | 10 | 5 | 5 | 100.0% |
| Retrained v1 | Dodging | 10 | 9 | 7 | 77.8% |
| Retrained v2 | Dodging | 10 | 10 | 1 | 10.0% |

---

## 11. Final Interpretation of v1 and v2

The progression from baseline to v1 to v2 shows a clear trend:

### For impersonation attack
- Baseline: **100.0%**
- Retrained v1: **100.0%**
- Retrained v2: **57.1%**

This suggests that v1 was not strong enough, but the stronger adversarial training setup in v2 significantly improved resistance to PGD impersonation.

### For dodging attack
- Baseline: **100.0%**
- Retrained v1: **77.8%**
- Retrained v2: **10.0%**

This shows that robustness against PGD dodging improved step by step, and v2 achieved a very large reduction in attack success rate.

### Overall conclusion
The stronger adversarial training setup in v2 is clearly more effective than v1.

The current results suggest that:

- adversarial training can improve robustness
- stronger adversarial training settings are important
- the improvement is especially strong for PGD dodging
- impersonation robustness also improved, although the model is still not fully robust

---

## 12. Current Conclusion

The first adversarial training experiment (v1) successfully produced a retrained checkpoint and allowed direct comparison between the baseline and retrained models under PGD attack.

However, v1 only showed partial improvement.

After strengthening the adversarial training setting, the v2 model showed **clear robustness improvement**:

- PGD impersonation success rate dropped from **100.0%** to **57.1%**
- PGD dodging success rate dropped from **100.0%** to **10.0%**

Therefore, the current defense direction is effective, and the stronger v2 setup is the best result obtained so far.

---

## 13. Remaining Limitation

This comparison still uses the baseline threshold `0.1767` for all models.

A more complete next step would be:

1. run clean evaluation again on the retrained v2 model
2. recalibrate a new threshold for the retrained model
3. re-run robust evaluation using that new threshold

This would provide a cleaner and more complete final evaluation.

---

## 14. Files Produced

### Checkpoints
- `results/arcface_pgd_adv_train_v2.pt`  → retrained v1
- `results/arcface_pgd_adv_train_v3.pt`  → retrained v2

### Attack result JSON files
- baseline impersonation result JSON
- baseline dodging result JSON
- retrained v1 impersonation result JSON
- retrained v1 dodging result JSON
- retrained v2 impersonation result JSON
- retrained v2 dodging result JSON

### Scripts
- `scripts/retraining_pgd_yichiao.py`
- `scripts/run_pgd_attack.py`
- `scripts/run_pgd_attack_retrained_yichiao_v1.py`

---

## 15. Short Summary

Baseline ArcFace was highly vulnerable to PGD attack in both impersonation and dodging settings.

Adversarial training v1 provided only limited improvement.

After strengthening the training setup, adversarial training v2 significantly improved robustness:
- impersonation success rate decreased to **57.1%**
- dodging success rate decreased to **10.0%**

This suggests that PGD-based adversarial training is effective, especially when the training setting is strong enough.