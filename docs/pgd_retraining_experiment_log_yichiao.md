# PGD Adversarial Training Experiment Log (YiChiao)

## 1. Goal

The goal of this stage is to complete the **PGD-related part of Task 4: Adversarial Training and Re-evaluation** for our face verification project.

Our baseline face verification system uses the ArcFace model as an embedding extractor. Face verification is performed using cosine similarity between two embeddings, and a threshold is used to determine whether two images belong to the same identity.

Initially, our earlier clean evaluation used the baseline threshold:

- `tau_EER = 0.1767`

After implementing PGD attack and confirming that the baseline model is highly vulnerable, the next step was to apply **PGD-based adversarial training** and evaluate whether the retrained model becomes more robust.

This log records the complete PGD workflow, including:

- baseline PGD attack
- adversarial training v1
- adversarial training v2
- clean re-evaluation
- recalibrated-threshold PGD re-evaluation
- final interpretation

---

## 2. Scope of My Work

For the project task split, I focused on the **PGD part** of the adversarial attack / defense pipeline.

This includes:

1. PGD attack evaluation on the baseline model
2. PGD-based adversarial training
3. PGD attack evaluation on retrained models
4. clean re-evaluation of the retrained model
5. threshold recalibration and re-evaluation under PGD

Other attacks such as **OPA** and **C&W** are handled by other group members.

---

## 3. Overall Workflow

In this stage, I completed the following process:

1. Evaluated the baseline ArcFace model under PGD attack.
2. Implemented PGD-based adversarial training.
3. Trained a first retrained model (**v1**).
4. Re-ran PGD attack on the retrained v1 model.
5. Observed that v1 showed only partial improvement.
6. Strengthened the adversarial training setting and trained a second retrained model (**v2**).
7. Re-ran PGD attack on the retrained v2 model.
8. Re-evaluated clean verification performance on LFW using the retrained v2 model.
9. Recalibrated a new threshold for the retrained v2 model.
10. Re-ran PGD attack using the recalibrated threshold.
11. Compared results across:
    - baseline
    - retrained v1
    - retrained v2

The comparison focused on two attack modes:

- **Impersonation attack**
- **Dodging attack**

---

## 4. Scripts Used

The following scripts were used in this PGD workflow:

### Core scripts
- `scripts/retraining_pgd_yichiao.py`
- `scripts/run_pgd_attack.py`
- `scripts/run_pgd_attack_retrained_yichiao.py`
- `scripts/arcface_eval_bin_retrained_yichiao.py`

### Notes
- I did **not** create separate v1/v2 Python files for retraining.
- v1 and v2 were distinguished by:
  - different training commands
  - different training settings
  - different saved checkpoints

This kept the workflow cleaner and made it easier to compare versions.

---

## 5. PGD Attack Setting

For PGD attack evaluation, I used:

- Attack: PGD
- Norm: Linf
- `eps = 8/255 = 0.031373`
- `alpha = 2/255 = 0.007843`
- `steps = 40`

The two attack modes are:

- **Impersonation**
  - attack succeeds if the similarity is pushed **above** the threshold
- **Dodging**
  - attack succeeds if the similarity is pushed **below** the threshold

---

## 6. Initial Baseline PGD Evaluation

### 6.1 Commands

#### Baseline impersonation
```bash
python scripts/run_pgd_attack.py --data data/casia_webface_extracted --mode impersonation --num_pairs 10 --eps 0.031373 --alpha 0.007843 --steps 40 --save_results
```

#### Baseline dodging
```bash
python scripts/run_pgd_attack.py --data data/casia_webface_extracted --mode dodging --num_pairs 10 --eps 0.031373 --alpha 0.007843 --steps 40 --save_results
```

### 6.2 Threshold used
At this stage, the threshold used was the earlier baseline threshold:

- `0.1767`

### 6.3 Results

#### Baseline impersonation
- Total pairs: 10
- Eligible pairs: 10
- Successful attacks: 10
- Success rate: 100.0%
- Average clean similarity: -0.0049
- Average adversarial similarity: 0.8344
- Similarity change: +0.8392
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 4.4562

#### Baseline dodging
- Total pairs: 10
- Eligible pairs: 5
- Successful attacks: 5
- Success rate: 100.0%
- Average clean similarity: 0.4615
- Average adversarial similarity: -0.6884
- Similarity change: -1.1499
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 4.5636

### 6.4 Interpretation

The baseline ArcFace model is highly vulnerable to PGD attack in both attack modes.

- In impersonation, PGD can successfully push different-identity pairs into the accepted region.
- In dodging, PGD can successfully pull same-identity pairs below the threshold and cause false rejection.

This establishes a strong motivation for adversarial training.

---

## 7. Adversarial Training v1

### 7.1 Training Command

```bash
python scripts/retraining_pgd_yichiao.py --data data/casia_webface_extracted --epochs 2 --batch_size 8 --max_classes 100 --max_imgs_per_class 20 --steps 3 --eps 0.015686 --alpha 0.003922 --save_name arcface_pgd_adv_train_v2.pt
```

### 7.2 Training Output

- Device: CPU
- Number of classes used: 28
- Number of samples used: 555

Epoch results:
- Epoch 1/2: loss = 3.1829, quick_train_acc = 0.8562
- Epoch 2/2: loss = 2.9540, quick_train_acc = 0.9187

Saved checkpoint:
- `results/arcface_pgd_adv_train_v2.pt`

### 7.3 Interpretation

This confirms that the PGD adversarial training pipeline ran successfully.

The script was able to:

- load the selected training subset
- generate adversarial examples during training
- fine-tune the ArcFace backbone
- save a retrained checkpoint for later evaluation

The training loss decreased and the quick training accuracy increased, indicating that the training process was stable on the selected subset.

---

## 8. PGD Evaluation on Retrained v1

### 8.1 Commands

#### Retrained v1 impersonation
```bash
python scripts/run_pgd_attack_retrained_yichiao.py --data data/casia_webface_extracted --mode impersonation --num_pairs 10 --eps 0.031373 --alpha 0.007843 --steps 40 --ckpt results/arcface_pgd_adv_train_v2.pt --save_results
```

#### Retrained v1 dodging
```bash
python scripts/run_pgd_attack_retrained_yichiao.py --data data/casia_webface_extracted --mode dodging --num_pairs 10 --eps 0.031373 --alpha 0.007843 --steps 40 --ckpt results/arcface_pgd_adv_train_v2.pt --save_results
```

### 8.2 Results

#### Retrained v1 impersonation
- Total pairs: 10
- Eligible pairs: 8
- Successful attacks: 8
- Success rate: 100.0%
- Average clean similarity: 0.0094
- Average adversarial similarity: 0.7154
- Similarity change: +0.7060
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 5.4622

#### Retrained v1 dodging
- Total pairs: 10
- Eligible pairs: 9
- Successful attacks: 7
- Success rate: 77.8%
- Average clean similarity: 0.7948
- Average adversarial similarity: -0.1079
- Similarity change: -0.9027
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 5.5808

### 8.3 Interpretation

The v1 adversarial training setup was not strong enough.

- For impersonation, robustness did **not** improve.
- For dodging, there was **partial improvement**.

This suggests that the general direction was correct, but the training setting needed to be strengthened.

---

## 9. Adversarial Training v2

After observing that v1 was not strong enough, I trained a stronger v2 model with:

- more epochs
- stronger PGD epsilon and alpha during training
- more PGD steps during training

### 9.1 Training Command

```bash
python scripts/retraining_pgd_yichiao.py --data data/casia_webface_extracted --epochs 5 --batch_size 8 --max_classes 28 --max_imgs_per_class 20 --steps 5 --eps 0.031373 --alpha 0.007843 --save_name arcface_pgd_adv_train_v3.pt
```

### 9.2 Training Output

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

### 9.3 Interpretation

Compared with v1, this version used a stronger adversarial training setting.

The training loss kept decreasing and the quick training accuracy increased to a very high value, indicating that the stronger setup converged successfully on the selected subset.

---

## 10. PGD Evaluation on Retrained v2 (Before Clean Recalibration)

### 10.1 Commands

#### Retrained v2 impersonation
```bash
python scripts/run_pgd_attack_retrained_yichiao.py --data data/casia_webface_extracted --mode impersonation --num_pairs 10 --eps 0.031373 --alpha 0.007843 --steps 40 --ckpt results/arcface_pgd_adv_train_v3.pt --save_results
```

#### Retrained v2 dodging
```bash
python scripts/run_pgd_attack_retrained_yichiao.py --data data/casia_webface_extracted --mode dodging --num_pairs 10 --eps 0.031373 --alpha 0.007843 --steps 40 --ckpt results/arcface_pgd_adv_train_v3.pt --save_results
```

### 10.2 Threshold used
At this stage, the threshold was still the old baseline threshold:

- `0.1767`

### 10.3 Results

#### Retrained v2 impersonation
- Total pairs: 10
- Eligible pairs: 7
- Successful attacks: 4
- Success rate: 57.1%
- Average clean similarity: -0.0553
- Average adversarial similarity: 0.2350
- Similarity change: +0.2904
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 5.9023

#### Retrained v2 dodging
- Total pairs: 10
- Eligible pairs: 10
- Successful attacks: 1
- Success rate: 10.0%
- Average clean similarity: 0.8660
- Average adversarial similarity: 0.6614
- Similarity change: -0.2046
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 5.9066

### 10.4 Interpretation

Compared with both baseline and v1, the v2 model substantially improved robustness against PGD attack.

- Impersonation success rate dropped from 100.0% to 57.1%
- Dodging success rate dropped from 100.0% to 10.0%

This showed that the stronger adversarial training direction was effective.

However, at this point the comparison was still based on the original baseline threshold, so a clean re-evaluation was needed.

---

## 11. Clean Re-evaluation

To make the evaluation more complete, I re-evaluated clean verification performance on `lfw.bin` using the retrained v2 model.

### 11.1 Clean Evaluation Script

```bash
python scripts/arcface_eval_bin_retrained_yichiao.py --bin data/eval/lfw.bin --cpu
python scripts/arcface_eval_bin_retrained_yichiao.py --bin data/eval/lfw.bin --cpu --ckpt results/arcface_pgd_adv_train_v3.pt
```

### 11.2 Baseline Clean Re-evaluation

- EER: 0.0023
- threshold@EER: 0.2213
- FAR@thr: 0.0013
- FRR@thr: 0.0033

### 11.3 Retrained v2 Clean Re-evaluation

- EER: 0.1335
- threshold@EER: 0.4002
- FAR@thr: 0.1333
- FRR@thr: 0.1337

### 11.4 Interpretation

The retrained v2 model became much more robust to PGD attack, but its clean verification performance dropped substantially.

This indicates a clear **robustness–accuracy trade-off**:

- robustness improved
- clean verification performance worsened

This is an important final conclusion of the PGD defense experiment.

---

## 12. Recalibrated PGD Evaluation

After re-evaluating clean performance, I recalibrated thresholds for each model:

- Baseline threshold: `0.2213`
- Retrained v2 threshold: `0.4002`

Then I re-ran PGD attack using these recalibrated thresholds.

---

## 13. Baseline PGD with Recalibrated Threshold

### 13.1 Commands

#### Baseline impersonation
```bash
python scripts/run_pgd_attack.py --data data/casia_webface_extracted --mode impersonation --num_pairs 10 --eps 0.031373 --alpha 0.007843 --steps 40 --threshold 0.2213 --save_results
```

#### Baseline dodging
```bash
python scripts/run_pgd_attack.py --data data/casia_webface_extracted --mode dodging --num_pairs 10 --eps 0.031373 --alpha 0.007843 --steps 40 --threshold 0.2213 --save_results
```

### 13.2 Results

#### Baseline impersonation
- Total pairs: 10
- Eligible pairs: 10
- Successful attacks: 10
- Success rate: 100.0%
- Average clean similarity: -0.0049
- Average adversarial similarity: 0.8403
- Similarity change: +0.8452
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 4.4711

#### Baseline dodging
- Total pairs: 10
- Eligible pairs: 5
- Successful attacks: 5
- Success rate: 100.0%
- Average clean similarity: 0.4615
- Average adversarial similarity: -0.6809
- Similarity change: -1.1424
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 4.5784

### 13.3 Interpretation

Even after using the recalibrated clean threshold, the baseline model remained fully vulnerable to PGD attack.

This confirms that the baseline model is inherently weak against PGD in both attack modes.

---

## 14. Retrained v2 PGD with Recalibrated Threshold

### 14.1 Commands

#### Retrained v2 impersonation
```bash
python scripts/run_pgd_attack_retrained_yichiao.py --data data/casia_webface_extracted --mode impersonation --num_pairs 10 --eps 0.031373 --alpha 0.007843 --steps 40 --threshold 0.4002 --ckpt results/arcface_pgd_adv_train_v3.pt --save_results
```

#### Retrained v2 dodging
```bash
python scripts/run_pgd_attack_retrained_yichiao.py --data data/casia_webface_extracted --mode dodging --num_pairs 10 --eps 0.031373 --alpha 0.007843 --steps 40 --threshold 0.4002 --ckpt results/arcface_pgd_adv_train_v3.pt --save_results
```

### 14.2 Results

#### Retrained v2 impersonation
- Total pairs: 10
- Eligible pairs: 9
- Successful attacks: 4
- Success rate: 44.4%
- Average clean similarity: 0.0126
- Average adversarial similarity: 0.2886
- Similarity change: +0.2759
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 5.8906

#### Retrained v2 dodging
- Total pairs: 10
- Eligible pairs: 9
- Successful attacks: 1
- Success rate: 11.1%
- Average clean similarity: 0.9308
- Average adversarial similarity: 0.7326
- Similarity change: -0.1983
- Average Linf perturbation: 0.031373
- Average L2 perturbation: 5.8993

### 14.3 Interpretation

Using the recalibrated threshold, the retrained v2 model still shows strong robustness improvement:

- impersonation success rate = 44.4%
- dodging success rate = 11.1%

This confirms that the robustness gain of v2 is real and does not disappear after threshold recalibration.

---

## 15. Full Results Comparison

### 15.1 Initial comparison (using threshold 0.1767)

| Model | Attack Mode | Total Pairs | Eligible Pairs | Successful Attacks | Success Rate |
|---|---:|---:|---:|---:|---:|
| Baseline | Impersonation | 10 | 10 | 10 | 100.0% |
| Retrained v1 | Impersonation | 10 | 8 | 8 | 100.0% |
| Retrained v2 | Impersonation | 10 | 7 | 4 | 57.1% |
| Baseline | Dodging | 10 | 5 | 5 | 100.0% |
| Retrained v1 | Dodging | 10 | 9 | 7 | 77.8% |
| Retrained v2 | Dodging | 10 | 10 | 1 | 10.0% |

### 15.2 Clean re-evaluation

| Model | EER | Threshold@EER | FAR@thr | FRR@thr |
|---|---:|---:|---:|---:|
| Baseline | 0.0023 | 0.2213 | 0.0013 | 0.0033 |
| Retrained v2 | 0.1335 | 0.4002 | 0.1333 | 0.1337 |

### 15.3 Final recalibrated PGD comparison

| Model | Attack Mode | Threshold | Total Pairs | Eligible Pairs | Successful Attacks | Success Rate |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | Impersonation | 0.2213 | 10 | 10 | 10 | 100.0% |
| Retrained v2 | Impersonation | 0.4002 | 10 | 9 | 4 | 44.4% |
| Baseline | Dodging | 0.2213 | 10 | 5 | 5 | 100.0% |
| Retrained v2 | Dodging | 0.4002 | 10 | 9 | 1 | 11.1% |

---

## 16. Final Interpretation

The overall trend is clear:

### Robustness
The stronger adversarial training setup in v2 substantially improved robustness against PGD attack.

Compared with the baseline model:

- impersonation success rate dropped from **100.0%** to **44.4%**
- dodging success rate dropped from **100.0%** to **11.1%**

### Clean performance
However, the clean verification performance of the retrained v2 model became much worse:

- baseline EER = **0.0023**
- retrained v2 EER = **0.1335**

This shows a strong robustness–accuracy trade-off.

### Main conclusion
PGD-based adversarial training is effective in improving robustness, especially when the training setting is strong enough.

However, in the current setup, the improvement in PGD robustness comes at a large cost in clean verification performance.

---

## 17. Practical Conclusion for the PGD Part

For the PGD part of the project, the current best checkpoint is:

- `results/arcface_pgd_adv_train_v3.pt`

This model provides the strongest PGD robustness result obtained so far, especially for dodging attack.

Therefore:

- v1 should be treated as an initial defense attempt
- v2 should be treated as the main PGD defense result
- the final interpretation should emphasize the robustness–accuracy trade-off

---

## 18. Handoff Note

This markdown file records the current PGD work in a complete and reproducible way.

A teammate who wants to continue optimizing the PGD defense can directly use:

- `scripts/retraining_pgd_yichiao.py`
- `scripts/run_pgd_attack_retrained_yichiao.py`
- `scripts/arcface_eval_bin_retrained_yichiao.py`
- `results/arcface_pgd_adv_train_v3.pt`

Recommended next directions for future improvement (if needed):
- improve clean performance while keeping robustness
- try larger training subsets
- try alternative loss balancing
- try more careful adversarial training schedules

---

## 19. Files Produced

### Checkpoints
- `results/arcface_pgd_adv_train_v2.pt` → retrained v1
- `results/arcface_pgd_adv_train_v3.pt` → retrained v2

### Result JSON files
- baseline impersonation result JSON
- baseline dodging result JSON
- retrained v1 impersonation result JSON
- retrained v1 dodging result JSON
- retrained v2 impersonation result JSON
- retrained v2 dodging result JSON
- recalibrated baseline impersonation result JSON
- recalibrated baseline dodging result JSON
- recalibrated retrained v2 impersonation result JSON
- recalibrated retrained v2 dodging result JSON

### Scripts
- `scripts/retraining_pgd_yichiao.py`
- `scripts/run_pgd_attack.py`
- `scripts/run_pgd_attack_retrained_yichiao.py`
- `scripts/arcface_eval_bin_retrained_yichiao.py`

---

## 20. Short Summary

Baseline ArcFace was highly vulnerable to PGD attack in both impersonation and dodging settings.

Adversarial training v1 provided only limited improvement.

A stronger adversarial training setup in v2 significantly improved robustness:

- recalibrated impersonation success rate = **44.4%**
- recalibrated dodging success rate = **11.1%**

However, clean verification performance dropped substantially:

- baseline EER = **0.0023**
- retrained v2 EER = **0.1335**

Therefore, the final PGD result is a clear robustness–accuracy trade-off.