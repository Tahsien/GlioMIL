# GlioMIL independent external validation: UCSF-PDGM

**Status: design and analysis scripts only. No external AUC has been generated.**

## Verified source

TCIA UCSF-PDGM version 5: https://doi.org/10.7937/TCIA.BDGF-8V37 — 495 subjects, 501 examinations, approximately 142 GB of NIfTI MRI and annotations, clinical/molecular CSV. Obtain the clinical CSV and scans from TCIA and document version and downloads. Remove follow-up scans using a prespecified index-exam rule, not molecular outcomes.

## Important blocker identified in v1.2 code

`scripts/06_ablation_patched.py` trains models per UTSW fold and restores `best_state` only in memory. It **does not export trained fold checkpoints or their scaling statistics**. Existing UTSW OOF predictions are insufficient to test UCSF. You must retrain on UTSW, saving each fold's `model.state_dict()`, model configuration, training-feature column order, training-fold means/stds/imputation values, and model weights. Do not train on UCSF.

## Correct evaluation

1. **Primary endpoint:** IDH mutation AUC on independent UCSF patients, with 2,000 patient-bootstrap resamples; sensitivity, specificity and accuracy at the *pre-specified* 0.5 threshold; Brier score, calibration intercept/slope and observed-vs-predicted prevalence.
2. **Comparison models:** frozen MIL/fusion/structured-only network and all-209 L2 logistic model, if each is fitted only using UTSW. Compare against UCSF age-only performance as a transparent *transported UTSW-trained age model*, not a refit on UCSF.
3. **No label-guided adaptation:** Do not pick among the five fold checkpoints using UCSF performance. Either average their probabilities according to a prespecified ensemble method or retrain one final model on all UTSW training cases using UTSW-only validation, then apply unchanged. This will be a new external-validation model, *not the existing OOF estimate*; distinguish the two.
4. **Preprocessing:** Match four contrasts T1, T1CE, T2 and FLAIR; handle orientation/voxel geometry, brain masking, segmentation labels, 15 tumour-guided slices and 224x224 resizing exactly. UTSW radiomics uses a whole-tumour mask (nonzero labels), T1CE and FLAIR, PyRadiomics 3.0.1 with binCount 32. Before predicting, validate feature names/order (209), shape and finite values. Do not silently zero-fill 204 missing features. Keep training-fold statistics frozen when transforming UCSF. Segmentation and MRI domain shift must be documented. The different image acquisition and processing do not make matching trivial.
5. **Population:** Report UCSF flow: 495 subjects -> one index examination per patient -> IDH-label eligibility -> required-contrast and segmentation availability -> analysis N. Report exclusions with counts, missingness and IDH class counts; do not presume all 495 qualify.
6. **Leakage checks:** Prevent overlap with UTSW or pretrained task-specific datasets where applicable, and ensure no UCSF images, labels or sample statistics are used for fitting any classifier, tuning or calibration. Report the possible pretraining overlap separately.
7. **Statistics:** Descriptive external CIs; clearly label any posthoc comparisons exploratory. Never copy UTSW OOF AUCs into the UCSF validation column.

## Run scripts

- `python 01_audit_ucsf_metadata.py /path/to/clinical.csv` identifies candidate IDH and subject-ID fields, reports missingness and produces `ucsf_metadata_audit.json`. A human must verify coding from official metadata glossary.
- After genuine UCSF inference from **frozen UTSW-trained models**, create a CSV with columns `PatientID,true_idh,prob_mutant`, exactly one row per patient, and run `python 02_evaluate_frozen_external_predictions.py ucsf_predictions.csv`. This outputs `ucsf_external_metrics.json`.

## What to upload next

1. UCSF clinical CSV and metadata glossary, preferably first (small).
2. UTSW training assets or saved trained weights, exact fitted feature scalers and preprocessing code.
3. UCSF MRI image package or a Kaggle dataset/notebook with access to all relevant MRI sequences and segmentations.

The scripts deliberately do **not** manufacture results or claim that external testing has run.
