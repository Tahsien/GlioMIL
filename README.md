# GlioMIL

Code for **"Frozen-backbone multiple-instance learning versus structured clinical and radiomic features for glioma molecular profiling: a controlled evaluation on the UTSW-Glioma cohort"** (revised manuscript under review at *BMC Medical Imaging*; originally titled "GlioMIL: A Frozen-Backbone Multiple-Instance Learning Framework for Calibrated and Explainable Glioma Radiogenomics").

The study compares a 2.5D multiple-instance-learning (MIL) pipeline (15 tumour-guided axial slices x 4 MRI contrasts, frozen ImageNet-pretrained EfficientNetV2-S, attention pooling, multi-task heads, optional fusion with 209 structured clinical and radiomic features) against structured-feature models and an end-to-end 3D CNN reference, for IDH status, grade, MGMT methylation and 1p/19q codeletion, under one shared patient-level 5-fold split (seed 42).

**Main finding in this cohort:** age alone gives an IDH AUC of 0.878, and an L2-regularised logistic regression on all 209 structured features gives 0.932, above the fusion model (0.847). The pooled frozen image embedding adds nothing measurable to age. Attention weights did not concentrate on tumour-rich slices. See the paper for all numbers and limitations (single cohort, internal cross-validation only).

## Data

The UTSW-Glioma dataset is public on The Cancer Imaging Archive and is **not** included here.

- TCIA collection: https://doi.org/10.7937/DFAE-1B86
- Data descriptor: Reddy et al., *Scientific Data* 13, 934 (2026), https://doi.org/10.1038/s41597-026-07274-4

Download the imaging package and the metadata TSV (`UTSW_Glioma_Metadata-2-1.tsv`).

## Repository layout

```
notebook/GlioMIL_full_pipeline.ipynb    full pipeline as run on Kaggle (outputs cleared)
scripts/                                the same code split by phase, for reading
  01_data_foundation.py                 metadata, cohort building, file checks
  02_slice_extraction.py                normalisation, 15 tumour-guided slices per patient
  03_frozen_backbone_features.py        frozen backbone embeddings (+ t-SNE)
  04a_install_pyradiomics.py            pyradiomics 3.0.1 install patch
  04b_radiomics_extraction.py           radiomic features
  05_mil_training.py                    attention-MIL multi-task training
  06_ablation_patched.py                ablation, OOF predictions, bootstrap CIs
  07_label_efficiency.py                10/25/50/100% label fraction (earlier configuration, exploratory)
  08a_mc_dropout_uncertainty.py         MC dropout (earlier configuration, exploratory)
  08b_calibration.py                    ECE / reliability analysis
  09_explainability.py                  attention vs tumour-fraction analysis
  10_summary.py                         summary tables/figures
  11_revision_A_baselines.py            (v1.1) age / clinical / radiomics / all-209 logistic and embedding baselines, DeLong tests
  12_revision_B_attention.py            (v1.1) per-patient permutation null, slice-position analysis
  13_revision_C_label_efficiency_mc.py  (v1.1) repeated-subsample label efficiency and MC dropout for the corrected pipeline (provided; not run for the paper)
structured_features_209.csv             (v1.1) the 209 structured features (5 clinical/tumour-summary + 204 PyRadiomics)
```

## How to run

The pipeline was developed and run in a Kaggle notebook (GPU); the notebook is the runnable version.

1. Upload the dataset to Kaggle (or edit the two input paths below).
2. Run the notebook cells in order. Phases share state through notebook variables and cached files in `/kaggle/working/` (`master_dataset_fets.csv`, `slices/`, `features/`, `radiomics_features.csv`, `phase5_fold_splits.json`, ...). The fold file is generated in the notebook with a fixed seed (42), so it is recreated deterministically on a new run.
3. Final ablation results come from the patched ablation cell (`06_ablation_patched.py`), which adds the structured-only condition, saves per-patient out-of-fold predictions and fits feature scaling per training fold. Scripts 11-13 are run in the same session after the training cells (see the docstring of each).
4. Model training is not seeded, so repeated runs give slightly different numbers (for example the IDH AUC of the fusion model varied by about 0.02 between runs, and the attention statistics varied more); the paper reports this.

Input paths to edit if your data lives elsewhere (search for `/kaggle/input`):
- `.../glioma/UTSW_Glioma_Metadata-2-1.tsv`
- `.../glioma/PKG - UTSW-Glioma/PKG - UTSW-Glioma/UTSW-Glioma`

## Environment

See `requirements.txt`. pyradiomics 3.0.1 needs the small build patch in `04a_install_pyradiomics.py` on recent Python versions.

## Citation

If you use this code, please cite the paper (reference to be added on publication), the archived release on Zenodo, and the UTSW-Glioma dataset papers above.

## Licence

MIT, see `LICENSE`.
