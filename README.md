# GlioMIL

Code for **"GlioMIL: A Frozen-Backbone Multiple-Instance Learning Framework for Calibrated and Explainable Glioma Radiogenomics"** (submitted to *Discover Computing*).

GlioMIL predicts IDH mutation status, tumor grade, MGMT promoter methylation and 1p/19q codeletion from multi-contrast MRI. Each tumor is a bag of 15 tumor-guided axial slices, embedded by a **frozen ImageNet-pretrained backbone** and aggregated by **attention-based MIL pooling** with multi-task heads. The repository also contains the five-condition ablation (3D CNN baseline, single-task MIL, multi-task MIL, MIL + clinical/radiomic fusion, structured-features-only), label-efficiency, MC-dropout/calibration and attention-tumor explainability analyses.

## Data

The UTSW-Glioma dataset is public on The Cancer Imaging Archive and is **not** included here.

- TCIA collection: https://doi.org/10.7937/DFAE-1B86
- Data descriptor: Reddy et al., *Scientific Data* 13, 934 (2026), https://doi.org/10.1038/s41597-026-07274-4

Download the imaging package and the metadata TSV (`UTSW_Glioma_Metadata-2-1.tsv`).

## Repository layout

```
notebook/GlioMIL_full_pipeline.ipynb   full pipeline as run on Kaggle (outputs cleared)
scripts/                               the same code split by phase, for reading
  01_data_foundation.py                metadata, cohort building, file checks
  02_slice_extraction.py               z-score normalisation, 15 tumor-guided slices per patient
  03_frozen_backbone_features.py       frozen backbone embeddings (+ t-SNE)
  04a_install_pyradiomics.py           pyradiomics 3.0.1 install patch
  04b_radiomics_extraction.py          radiomic features
  05_mil_training.py                   attention-MIL multi-task training
  06_ablation_patched.py               five-condition ablation, OOF predictions, bootstrap CIs
  07_label_efficiency.py               10/25/50/100% label fractions
  08a_mc_dropout_uncertainty.py        MC dropout
  08b_calibration.py                   ECE / reliability analysis
  09_explainability.py                 attention vs tumor-fraction analysis
  10_summary.py                        summary tables/figures
```

## How to run

The pipeline was developed and run in a Kaggle notebook (GPU), and the notebook is the runnable version.

1. Upload the dataset to Kaggle (or edit the two input paths, see below).
2. Run the notebook cells in order. Phases share state through notebook variables and cached files in `/kaggle/working/` (`master_dataset_fets.csv`, `slices/`, `features/`, `radiomics_features.csv`, `phase5_fold_splits.json`, ...).
3. **Final ablation results come from the patched Phase 5 cell** (`06_ablation_patched.py`), which adds the structured-features-only condition, saves per-patient out-of-fold predictions and fits feature scaling per training fold. The earlier unpatched Phase 5 cells are kept for transparency.

Input paths to edit if your data lives elsewhere (search for `/kaggle/input`):
- `.../glioma/UTSW_Glioma_Metadata-2-1.tsv`
- `.../glioma/PKG - UTSW-Glioma/PKG - UTSW-Glioma/UTSW-Glioma`

Key settings: 5-fold patient-level stratified CV (seed 42), 15 slices of 224x224, 300 bootstrap resamples for AUC confidence intervals.

## Environment

See `requirements.txt`. pyradiomics 3.0.1 needs the small build patch in `04a_install_pyradiomics.py` on recent Python versions.

## Citation

If you use this code, please cite the paper (reference to be added on publication) and the UTSW-Glioma dataset papers above.

## License

MIT, see `LICENSE`.
