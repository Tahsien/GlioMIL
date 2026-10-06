import os
import pandas as pd
from collections import Counter

# ============================================================
# PHASE 1 — DATA FOUNDATION
# Loads metadata, drops missing-IDH patients, verifies file
# completeness, builds FeTS and manual-segmentation cohorts,
# and audits the final dataset.
# ============================================================

metadata_path = "/kaggle/input/datasets/tano1974/glioma/UTSW_Glioma_Metadata-2-1.tsv"
base_path = "/kaggle/input/datasets/tano1974/glioma/PKG - UTSW-Glioma/PKG - UTSW-Glioma/UTSW-Glioma"

# ------------------------------------------------------------
# Step 1: Load metadata, drop missing IDH
# ------------------------------------------------------------
df = pd.read_csv(metadata_path, sep="\t")
print("Original metadata shape:", df.shape)

df = df.dropna(subset=["IDH"]).copy()
df["IDH_Label"] = df["IDH"].map({
    "wild type": 0,
    "mutated": 1
})

print("\nAfter removing missing IDH:", df.shape)
print("\nIDH Distribution:")
print(df["IDH_Label"].value_counts())

# ------------------------------------------------------------
# Step 2: Verify file completeness per patient (FeTS cohort)
# ------------------------------------------------------------
required_files = {
    "brain_t1_ants.nii",
    "brain_t1ce_ants.nii",
    "brain_t2_ants.nii",
    "brain_fl_ants.nii",
    "tumorseg_FeTS.nii"
}

complete_patients = []
missing_patients = []

for patient in os.listdir(base_path):
    patient_path = os.path.join(base_path, patient)
    if not os.path.isdir(patient_path):
        continue
    files = set(os.listdir(patient_path))
    if required_files.issubset(files):
        complete_patients.append(patient)
    else:
        missing_patients.append(patient)

print("\nPatients with complete MRI data (FeTS):", len(complete_patients))
print("Patients with missing data:", len(missing_patients))

# ------------------------------------------------------------
# Step 3: Build the FeTS master dataset
# ------------------------------------------------------------
master_df = df[df["Subject ID"].isin(complete_patients)].copy()

master_df = master_df[[
    "Subject ID",
    "IDH",
    "IDH_Label",
    "Age at Imaging",
    "Sex at birth",
    "Tumor Grade",
    "Tumor Type",
    "Scanner Make",
    "Scanner Model",
    "Scanner Strength"
]]

master_df.rename(columns={
    "Subject ID": "PatientID",
    "Age at Imaging": "Age",
    "Sex at birth": "Sex",
    "Tumor Grade": "Grade"
}, inplace=True)

master_df["T1_Path"] = master_df["PatientID"].apply(
    lambda x: os.path.join(base_path, x, "brain_t1_ants.nii")
)
master_df["T1CE_Path"] = master_df["PatientID"].apply(
    lambda x: os.path.join(base_path, x, "brain_t1ce_ants.nii")
)
master_df["T2_Path"] = master_df["PatientID"].apply(
    lambda x: os.path.join(base_path, x, "brain_t2_ants.nii")
)
master_df["FLAIR_Path"] = master_df["PatientID"].apply(
    lambda x: os.path.join(base_path, x, "brain_fl_ants.nii")
)
master_df["Mask_Path"] = master_df["PatientID"].apply(
    lambda x: os.path.join(base_path, x, "tumorseg_FeTS.nii")
)

path_columns = ["T1_Path", "T1CE_Path", "T2_Path", "FLAIR_Path", "Mask_Path"]

for col in path_columns:
    master_df[col + "_Exists"] = master_df[col].apply(os.path.exists)

print("\nPath Verification")
for col in path_columns:
    print(col, ":", master_df[col + "_Exists"].sum(), "/", len(master_df))

exist_cols = [col + "_Exists" for col in path_columns]
master_df = master_df[master_df[exist_cols].all(axis=1)].copy()
master_df.drop(columns=exist_cols, inplace=True)

print("\nFinal FeTS Master Dataset Shape:", master_df.shape)
print("\nFinal IDH Distribution:")
print(master_df["IDH_Label"].value_counts())
print("\nFinal IDH Percentage:")
print(master_df["IDH_Label"].value_counts(normalize=True) * 100)

master_df.to_csv("/kaggle/working/master_dataset_fets.csv", index=False)
print("\nSaved: /kaggle/working/master_dataset_fets.csv")

# ------------------------------------------------------------
# Step 4: Build the manual-segmentation subset (higher quality)
# ------------------------------------------------------------
required_manual = {
    "brain_t1_ants.nii",
    "brain_t1ce_ants.nii",
    "brain_t2_ants.nii",
    "brain_fl_ants.nii",
    "tumorseg_manual_correction.nii"
}

manual_patients = []

for patient in os.listdir(base_path):
    patient_path = os.path.join(base_path, patient)
    if not os.path.isdir(patient_path):
        continue
    files = set(os.listdir(patient_path))
    if required_manual.issubset(files):
        manual_patients.append(patient)

manual_df = df[df["Subject ID"].isin(manual_patients)].copy()

manual_df = manual_df[[
    "Subject ID",
    "IDH",
    "IDH_Label",
    "Age at Imaging",
    "Sex at birth",
    "Tumor Grade"
]]

manual_df.rename(columns={
    "Subject ID": "PatientID",
    "Age at Imaging": "Age",
    "Sex at birth": "Sex",
    "Tumor Grade": "Grade"
}, inplace=True)

manual_df.to_csv("/kaggle/working/master_dataset_manual_subset.csv", index=False)

print("\nManual Subset Shape:", manual_df.shape)
print("\nManual IDH Distribution:")
print(manual_df["IDH_Label"].value_counts())
print("\nSaved: /kaggle/working/master_dataset_manual_subset.csv")

# ------------------------------------------------------------
# Step 5: Audit the final FeTS cohort
# ------------------------------------------------------------
audit_df = pd.read_csv("/kaggle/working/master_dataset_fets.csv")

print("\n" + "="*60)
print("FINAL DATASET AUDIT (FeTS cohort)")
print("="*60)

print("\nDataset shape:", audit_df.shape)
print("\nIDH Distribution:")
print(audit_df["IDH_Label"].value_counts())
print("\nSex Distribution:")
print(audit_df["Sex"].value_counts())
print("\nGrade Distribution:")
print(audit_df["Grade"].value_counts())
print("\nAge Statistics:")
print(audit_df["Age"].describe())
print("\nTumour Type Distribution:")
print(audit_df["Tumor Type"].value_counts())
print("\nMissing Values:")
print(audit_df.isnull().sum())

print("\n" + "="*60)
print("PHASE 1 COMPLETE")
print("="*60)