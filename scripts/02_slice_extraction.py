import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom

NUM_SLICES = 15
SLICE_SIZE = (224, 224)
MARGIN = 2
EPS = 1e-6

OUT_DIR = "/kaggle/working/slices"
os.makedirs(OUT_DIR, exist_ok=True)

master_df = pd.read_csv("/kaggle/working/master_dataset_fets.csv")


def load_nii(path):
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)


def zscore(vol, mask):
    voxels = vol[mask]
    if voxels.size == 0:
        return vol
    mean = voxels.mean()
    std = voxels.std()
    return np.clip((vol - mean) / (std + EPS), -5, 5)


def tumour_z_range(seg, margin):
    present = np.any(seg > 0, axis=(0, 1))
    idx = np.nonzero(present)[0]
    if idx.size == 0:
        return None
    z_min = max(0, int(idx.min()) - margin)
    z_max = min(seg.shape[2] - 1, int(idx.max()) + margin)
    return z_min, z_max


def resize2d(img, size):
    factors = (size[0] / img.shape[0], size[1] / img.shape[1])
    return zoom(img, factors, order=1).astype(np.float32)


def extract_patient(row):
    t1 = load_nii(row["T1_Path"])
    t1ce = load_nii(row["T1CE_Path"])
    t2 = load_nii(row["T2_Path"])
    flair = load_nii(row["FLAIR_Path"])
    seg = load_nii(row["Mask_Path"])

    z_range = tumour_z_range(seg, MARGIN)
    if z_range is None:
        return None
    z_min, z_max = z_range

    brain_mask = (t1 > 0) | (t1ce > 0) | (t2 > 0) | (flair > 0)
    t1 = zscore(t1, brain_mask)
    t1ce = zscore(t1ce, brain_mask)
    t2 = zscore(t2, brain_mask)
    flair = zscore(flair, brain_mask)

    if z_max == z_min:
        z_indices = [z_min] * NUM_SLICES
    else:
        z_indices = np.round(np.linspace(z_min, z_max, NUM_SLICES)).astype(int).tolist()

    bag = np.zeros((NUM_SLICES, 4, *SLICE_SIZE), dtype=np.float32)
    for i, z in enumerate(z_indices):
        bag[i, 0] = resize2d(t1[:, :, z], SLICE_SIZE)
        bag[i, 1] = resize2d(t1ce[:, :, z], SLICE_SIZE)
        bag[i, 2] = resize2d(t2[:, :, z], SLICE_SIZE)
        bag[i, 3] = resize2d(flair[:, :, z], SLICE_SIZE)

    return bag


processed = []
skipped = []

for _, row in master_df.iterrows():
    pid = row["PatientID"]
    bag = extract_patient(row)
    if bag is None:
        skipped.append(pid)
        continue
    np.savez_compressed(os.path.join(OUT_DIR, f"{pid}.npz"), slices=bag)
    processed.append(pid)

    if len(processed) % 50 == 0:
        print(f"{len(processed)} patients done")

print(f"\nDone. Processed: {len(processed)}, skipped: {len(skipped)}")
if skipped:
    print("Skipped (empty mask):", skipped[:20])

master_df["has_slices"] = master_df["PatientID"].isin(processed)
master_df.to_csv("/kaggle/working/master_dataset_fets_with_slices.csv", index=False)
print("Saved: /kaggle/working/master_dataset_fets_with_slices.csv")