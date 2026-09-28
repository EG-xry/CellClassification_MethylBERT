# Results

Records of all 45 training runs, copied from the original run folders, plus read-only audits of the saved train/test splits. Model weights are not included.

## What each run folder holds

- `final_metrics.txt`: accuracy, macro F1, per-class accuracy, and a classification report, computed from the lowest eval loss checkpoint.
- `eval.csv`: eval loss and accuracy at every eval step.
- `training_curves.png`, `confusion_matrix*.png`.
- `model_config.json`: the saved model config (encoder layers, loss, class count).
- `train_param.txt`: training parameters, where recorded.
  - 12 runs had a stale template in place of real parameters (the same file every time: 39 classes, 100 steps). Those copies were removed.
  - For those runs, use `model_config.json` and `eval.csv`.
- `confidence_analysis.txt`: the 15 least and most confident reads per class.
  - Confidence values are compressed toward uniform by a double softmax. See the known issues in [`../Training/src/methylbert/MODIFICATIONS.md`](../Training/src/methylbert/MODIFICATIONS.md).
- `training_output_excerpt.log`: first and last 150 lines of the run log, lines clipped to 300 characters. Full logs were 9 to 160 MB.
- `deconvolution/`: results of classifying every read from one pure sample, where they exist.

## Audits

- `audit_splits.py` recomputes three counts on the saved splits. No model is involved.
- Output is in `audit_results.txt`.

| Audit | Split | Result |
| --- | --- | --- |
| Region ID alone predicts cell type | First flat 39-class split (`res_1M`) | 100% of covered test reads; every one of 37,918 regions maps to a single cell type |
| Test reads with an identical training read | Excitable (`Excitable_1`) | 49.0% |
| Test reads whose identical training read has a different label | Excitable (`Excitable_1`) | 35.4% |
| Region ID alone predicts cell type | Excitable (`Excitable_1`) | 0.386 (chance 0.333) |
| Reads inside a marker region of their own cell type | Raw preprocessing output, one skeletal muscle sample | 1.7% |

## Class sets

| Set | Cell types |
| --- | --- |
| Blood (6) | `Blood-B`, `Blood-Granul`, `Blood-Mono+Macro`, `Blood-NK`, `Blood-T`, `Eryth-prog` |
| Blood (5) | Blood (6) without `Eryth-prog` |
| Structural | `Adipocytes`, `Colon-Fibro`, `Endothel`, `Heart-Fibro`, `Skeletal-Musc`, `Smooth-Musc` |
| Surface | `Bladder-Ep`, `Breast-Basal-Ep`, `Breast-Luminal-Ep`, `Head-Neck-Ep`, `Lung-Ep-Alveo`, `Prostate-Ep` |
| Metabolic | `Liver-Hep`, `Pancreas-Acinar`, `Pancreas-Alpha`, `Pancreas-Beta`, `Pancreas-Delta`, `Pancreas-Duct` |
| Digestive | `Colon-Ep`, `Gastric-Ep`, `Lung-Ep-Alveo`, `Small-Int-Ep` |
| Reproductive | `Fallopian-Ep`, `Kidney-Ep`, `Ovary-Ep`, `Thyroid-Ep` |
| Excitable | `Heart-Cardio`, `Neuron`, `Oligodend` |
| HLE (early subset) | `Breast-Basal-Ep`, `Breast-Luminal-Ep`, `Fallopian-Ep`, `Ovary-Ep`, `Prostate-Ep`, `Thyroid-Ep` |

## Run index

- Dates are 2025, from the saved model config.
- "Final" is the lowest eval loss checkpoint; "best" is the highest accuracy at any eval step.
- Chance is 1 / (number of classes). Where the test set is unbalanced, macro F1 is the fairer number.

### 1. Flat 39-class, first split (leaky)

| Run | Date | Layers | Loss | Final acc | Macro F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `res_1M` | 07-19 | 2 | cross entropy | 0.991 | 0.969 | Region ID alone gives the answer (see audits). Split: 80/20 random by read (`Training/Toolbox/split_data_simple.py`). |
| `res_pre` | 07-27 | 2 | cross entropy | 1.000 | 1.000 | Same data construction; data file not recorded. |

### 2. Flat 39-class, region balanced

| Run | Date | Layers | Loss | Final acc | Macro F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `res_39Class` | 08-07 | 6 | cross entropy | 0.520 | 0.511 | Each sample was balanced to 50% on-target reads (`Data/ctype_dmr_agreement_analysis.txt`). Per-class recall is 0.46 to 0.53 for 36 of 39 types, which is what a region-only rule would score. |

### 3. Early subsets

| Run | Date | Classes | Layers | Loss | Final acc | Macro F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `res_2Class_6Enc` | 08-08 | Acinar vs Duct | 6 | cross entropy | 0.830 | 0.821 | Pancreas acinar vs duct. Split details not saved. |
| `res_2Class_2Enc` | 08-10 | Acinar vs Duct | 6 (config) | cross entropy | 0.833 | 0.825 | Folder name says 2 layers; saved config says 6. |
| `res_6Class_6Enc` | 08-10 | HLE | 6 | cross entropy | 0.523 | 0.502 | Eval loss rose from 1.29 to 2.32 (overfitting). Data construction not saved; may share the region shortcut of `res_39Class`. |
| `W_res_6Class_2Enc` | 08-11 | HLE | 2 | cross entropy | 0.530 | 0.514 | As above. |

### 4. Blood sweeps (22 runs)

Loss, depth, learning rate, sequence length, marker set (250 vs 1000 per type), batch balancing, more data, and split variants.

| Run | Date | Classes | Layers | Loss | Final acc | Macro F1 | Change tested |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `res_Focal_5Class_2Enc` | 08-13 | 5 | 4 | focal | 0.401 | 0.376 | Focal loss |
| `res_Decoup_5Class_2Enc` | 08-18 | 5 | 2 | decoupled focal | 0.372 | 0.366 | Decoupled focal, auto alphas |
| `res_Decoup_6Class` | 08-18 | 6 | 2 | decoupled focal | 0.233 | 0.221 | Added `Eryth-prog` |
| `res_Entropy_6Class_2Enc_1000` | 08-19 | 6 | 2 | cross entropy | 0.198 | 0.189 | Cross entropy, 1000 markers; skewed onto `Blood-Mono+Macro` |
| `res_Entropy_6Class_2Enc_250` | 08-19 | 6 | 2 | cross entropy | 0.284 | 0.282 | 250 markers |
| `res_BiasDecoup_6Class_2Enc_250` | 08-20 | 6 | 2 | decoupled focal | 0.301 | 0.300 | Larger positive alphas (1.4 to 1.9) |
| `res_Decoup_6Class_2Enc_250` | 08-20 | 6 | 2 | decoupled focal | 0.323 | 0.318 | Balanced batches |
| `res_Decoup_6Class_4Enc_250` | 08-20 | 6 | 4 | decoupled focal | 0.323 | 0.307 | 4 layers |
| `NoOverlap_4Enc_1000_5` | 08-23 | 5 | 4 | focal | 0.377 | 0.358 | New split ("NoOverlap" is not defined in any saved file; likely no identical reads on both sides) |
| `NoOverlap_4Enc_1000_6` | 08-24 | 6 | 4 | focal | 0.340 | 0.323 | 6 classes |
| `NoOverlap_4Enc_1000_6_Focal` | 08-24 | 6 | 4 | focal | 0.341 | 0.329 | Manual class weights (param file only) |
| `NoOverlap_6Enc_1000_6_Focal` | 08-25 | 6 | 6 | focal | 0.336 | 0.335 | 6 layers |
| `NoOverlap_6Enc_1000_6_Focal_Gamma` | 08-25 | 6 | 6 | focal | 0.336 | 0.336 | Gamma 1.5 |
| `NoOverlap_6Enc_250_6` | 08-26 | 6 | 6 | focal | 0.351 | 0.346 | 250 markers; eval loss rose from 1.29 to 3.53 |
| `NoOverlap_NoBalance` | 08-27 | 6 | 6 | focal | 0.430 | 0.346 | No batch balancing, unbalanced test set (largest class 33%) |
| `NoOverlap_BalanceBatch` | 08-27 | 6 | 6 | focal | 0.370 | 0.325 | Balanced batches |
| `NoOverlap_MoreData` | 08-28 | 6 | 6 | focal | 0.381 | 0.348 | 12% more training reads |
| `NoOverlap_LearningRate` | 08-30 | 6 | 6 | focal | 0.318 | 0.308 | Learning rate 4e-5 |
| `NoOverlap_SplitData` | 08-31 | 6 | 6 | focal | 0.303 | 0.294 | Data file `train_Blood_split.csv` (likely long reads cut in two by `convert_data_format_flexible.py`; not recorded) |
| `NoOverlap_SplitData_EqualWeight` | 09-01 | 6 | 4 | focal | 0.330 | 0.313 | Equal class weights |
| `Blood_1` | 09-03 | 6 | 6 | focal | 0.296 | 0.283 | Data file `train_Blood_cross.csv` (likely the cross-sample deduplication mode of `split_data_flexible.py`; not recorded) |
| `DataLeakage` | 09-04 | 6 | 6 | focal | 0.276 | 0.261 | Data file `train_1000_cross.csv`; purpose and log not saved |

### 5. Families

| Run | Date | Family | Classes | Layers | Loss | Final acc | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Structural_1` | 09-07 | Structural | 6 | 6 | focal | 0.220 | 0.175 |
| `Structural_2` | 09-07 | Structural | 6 | 6 | focal, label smoothing 0.2 | 0.233 | 0.206 |
| `Structural_3` | 09-08 | Structural | 6 | 2 | focal (gamma 0) | 0.246 | 0.222 |
| `Structural_4` | 09-10 | Structural | 6 | 2 | focal (gamma 0) | 0.217 | 0.207 |
| `Structural_5` | 09-10 | Structural | 6 | 4 | focal (gamma 0) | 0.243 | 0.221 |
| `Structural_6` | 09-11 | Structural | 6 | 4 | focal | 0.238 | 0.228 |
| `Excitable_1` | 09-08 | Excitable | 3 | 2 | focal (gamma 0) | 0.548 | 0.542 |
| `Metabolic_1` | 09-08 | Metabolic | 6 | 2 | focal (gamma 0) | 0.273 | 0.261 |
| `Metabolic_2` | 09-11 | Metabolic | 6 | 6 | focal (gamma 0) | 0.244 | 0.239 |
| `Metabolic_3` | 09-12 | Metabolic | 6 | 4 | focal (gamma 0) | 0.273 | 0.270 |
| `Reproductive_1` | 09-08 | Reproductive | 4 | 2 | focal (gamma 0) | 0.394 | 0.382 |
| `Reproductive_2` | 09-08 | Reproductive | 4 | 2 | focal (gamma 0) | 0.378 | 0.372 |
| `Digestive_1` | 09-09 | Digestive | 4 | 2 | focal (gamma 0) | 0.395 | 0.386 |
| `Surface_1` | 09-09 | Surface | 6 | 2 | focal (gamma 0) | 0.276 | 0.271 |
| `Surface_2` | 09-09 | Surface | 6 | 2 | focal | 0.277 | 0.267 |
| `Surface_3` | 09-09 | Surface | 6 | 4 | focal (gamma 0) | 0.277 | 0.273 |

Notes:
- `Structural_2` used a learning rate of 5e-4 and collapsed: from step 7000 on, most evals predict a single class.
- `Structural_5`, `Structural_6`, and `Metabolic_3` were evaluated on regenerated test files with the same names, so their test sets differ slightly from earlier runs in the same family.

## Pure-sample deconvolution

Every read from a single pure sample was classified, and the class shares counted. A working model would put most reads in the true class. The two 5-file runs also scored the mixed `test_1000_6.csv`; that file is left out of the counts below.

| Model | Samples | Outcome |
| --- | --- | --- |
| `NoOverlap_MoreData` (Blood) | 12 | Every sample dominated by `Blood-B` (42% to 60%), including T cells, NK cells, monocytes, and erythrocyte progenitors |
| `NoOverlap_BalanceBatch` (Blood) | 4 | Every sample dominated by `Blood-B` (37% to 48%), including NK cells, erythrocyte progenitors, and macrophages |
| `NoOverlap_6Enc_1000_6_Focal_Gamma` (Blood) | 4 | Same pattern as the row above (`Blood-B` 37% to 48%) |
| `Blood_1` (Blood) | 14 | Near uniform; top class 21% to 36%. Granulocytes, monocytes, NK cells, and all three macrophage samples put `Blood-T` on top |
| `Structural_6` (Structural) | 5 | Near uniform (top class 20% to 30% against 16.7% for uniform); true class on top in 1 of 5 |

The deconvolution samples were drawn from the same atlas, and the saved files do not record whether each one was held out from training.
