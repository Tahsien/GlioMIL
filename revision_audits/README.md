# Manuscript revision / statistical audit work

This folder contains source-code diagnostics produced during the November 2026 GlioMIL revision:

- `01_audit_holm.py`: recompute Holm-adjusted p-values **only** from original raw `revA_delong.csv` and other genuine significance outputs. Verify the family definition (within-task ablation versus expanded IDH 36-pair tests) before use.
- `02_check_stacking.py`: check coverage of the base-model OOF predictions and warn about second-stage stacking leakage. It cannot validate stacking coefficients or likelihood-ratio results on its own.

Run from Kaggle after the relevant OOF/result files exist. **Neither audit has been run on the author's raw Kaggle outputs within this repository.**

## Statistical interpretation

Base-model OOF predictions do not by themselves make a logistic stacking model independently validated. AUC 0.881 to 0.886 and the associated likelihood-ratio p-values should be treated as exploratory until meta-level cross-fitting/nested validation is demonstrated.

## Other prepared files

Separately generated manuscript revisions and DOCX/PDF reviewer letters were produced in the conversation but **have not been committed into this repository**. Avoid assuming that the released v1.2 manuscript has been replaced by those working versions. The v1.2 Zenodo release and DOI remain unchanged.

The UCSF-PDGM external validation starter scripts were already committed under `external_validation/ucsf_pdgm/`; they are **not external validation results**.

## Still necessary before claiming complete resolution

1. Original raw `revA_delong.csv` and ablation comparison outputs.
2. Exact second-stage stacking fitting and testing provenance, or a proper outer-fold rerun.
3. Actual corrected-pipeline GPU experiments for seeds, matched architectures, stronger backbone, label efficiency, MC-dropout and attribution.
4. Full Overleaf compilation and rendered manuscript inspection.
