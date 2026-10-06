import os
import gc
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import logging

logging.getLogger("radiomics").setLevel(logging.ERROR)

MASTER_PATH = "/kaggle/working/master_dataset_fets_with_slices.csv"
OUT_PATH = "/kaggle/working/radiomics_features.csv"
SEQUENCES = {"T1CE": "T1CE_Path", "FLAIR": "FLAIR_Path"}

master_df = pd.read_csv(MASTER_PATH)
master_df = master_df[master_df["has_slices"] == True].reset_index(drop=True)

if os.path.exists(OUT_PATH):
    existing_df = pd.read_csv(OUT_PATH)
    done_ids = set(existing_df["PatientID"])
    print(f"Resuming: {len(done_ids)} patients already extracted, skipping them.")
else:
    done_ids = set()

# Fixed: binCount instead of binWidth. binWidth=64 caused a ~12,000-bin
# explosion on patients with extreme outlier intensity values (e.g.
# BT0020's max of 765,553), since MRI intensities aren't standardized
# across scanners the way CT Hounsfield units are. binCount fixes the
# number of bins regardless of intensity range, avoiding this entirely.
settings = {"interpolator": None, "resampledPixelSpacing": None, "binCount": 32}
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.disableAllFeatures()
for fc in ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm"]:
    extractor.enableFeatureClassByName(fc)

remaining_df = master_df[~master_df["PatientID"].isin(done_ids)].reset_index(drop=True)
print(f"Extracting radiomics for {len(remaining_df)} remaining patients...")

failed = []

for i, row in remaining_df.iterrows():
    pid = row["PatientID"]
    print(f"[{i+1}/{len(remaining_df)}] Starting {pid}...", flush=True)

    patient_features = {"PatientID": pid}

    try:
        mask_img = sitk.ReadImage(row["Mask_Path"])
        mask_arr = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)
        mask_img_binary = sitk.GetImageFromArray(mask_arr)
        mask_img_binary.CopyInformation(mask_img)

        for seq_name, path_col in SEQUENCES.items():
            image = sitk.ReadImage(row[path_col])
            result = extractor.execute(image, mask_img_binary, label=1)
            for key, value in result.items():
                if key.startswith("diagnostics"):
                    continue
                patient_features[f"{seq_name}_{key}"] = float(value)
            del image, result

        row_df = pd.DataFrame([patient_features])
        if os.path.exists(OUT_PATH):
            row_df.to_csv(OUT_PATH, mode="a", header=False, index=False)
        else:
            row_df.to_csv(OUT_PATH, index=False)

        print(f"[{i+1}/{len(remaining_df)}] Done {pid} (saved)", flush=True)

    except Exception as e:
        print(f"FAILED for {pid}: {e}", flush=True)
        failed.append(pid)

    del mask_img, mask_arr, mask_img_binary
    gc.collect()

    if (i + 1) % 50 == 0:
        print(f"--- Progress: {i+1}/{len(remaining_df)} ---", flush=True)

final_df = pd.read_csv(OUT_PATH)
print(f"\nDone. Total patients in {OUT_PATH}: {len(final_df)}")
print(f"Feature columns: {final_df.shape[1] - 1}")
if failed:
    print(f"Failed this run: {len(failed)} --", failed[:20])