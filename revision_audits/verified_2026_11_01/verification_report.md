# Verified statistical audit — 2026-11-01

Audit inputs: author's Kaggle `GlioMIL_statistical_data.zip` containing `revA_delong.csv`, `revA_oof.csv`, `phase5_oof_predictions.csv`, and `phase5_fold_splits.json`; 619 IDH-labelled patients, 36 pairwise comparisons, 5 cached patient folds.

## Verified Holm-adjusted p-values (36-comparison expanded IDH family)

| Pair | Raw p | Holm (36) |
|---|---:|---:|
| All-209 logistic vs fusion | 2.677529e-10 | 9.371351e-09 |
| All-209 logistic vs structured-only NN | 1.022592e-06 | 2.760999e-05 |
| All-209 logistic vs age only | 5.660732e-05 | 0.001358576 |
| Fusion vs structured-only NN | 0.008969806 | 0.1704263 |

The ablation-wide 28-comparison family was **not** available in the supplied raw CSV; do not mark it verified.

## Real-data diagnostic meta-level cross-fitting

| Prediction | IDH AUC |
|---|---:|
| Original structured-only NN | 0.881297 |
| Cross-fitted structured-score recalibration | 0.875577 |
| Cross-fitted structured + image-only multi-task scores | 0.879502 |

Paired delta stacked vs recalibrated structured: **+0.003925**; exploratory 2,000 patient-bootstrap 95% CI: **[-0.006138, +0.014093]**.

**Warning:** The diagnostic reuses base-model OOF predictions from models trained on overlapping UTSW training folds. It is not fully nested or leakage-isolated validation of the stacking pipeline. The old 0.881→0.886 AUC and likelihood-ratio results are exploratory; do not assert verified incremental utility.

This folder records an actual audit but **does not contain a new model-training experiment or a UCSF external validation**. Existing v1.2 release remains unchanged.

The full patient-level diagnostic CSV and reviewer manuscript drafts were produced locally in the ChatGPT conversation but have not been committed in this folder.
