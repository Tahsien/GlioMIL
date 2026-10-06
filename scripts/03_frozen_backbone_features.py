import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

SLICES_DIR = "/kaggle/working/slices"
OUT_DIR = "/kaggle/working/features"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

master_df = pd.read_csv("/kaggle/working/master_dataset_fets_with_slices.csv")
master_df = master_df[master_df["has_slices"] == True]

# ---- Build backbone, adapt first conv layer for 4-channel MRI input ----
backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

old_conv = backbone.features[0][0]
new_conv = nn.Conv2d(
    in_channels=4,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None,
)

with torch.no_grad():
    new_conv.weight[:, :3] = old_conv.weight
    new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)

backbone.features[0][0] = new_conv
backbone.classifier = nn.Identity()

for p in backbone.parameters():
    p.requires_grad = False

backbone.eval()
backbone.to(device)

EMBED_DIM = 1280

processed = []

with torch.no_grad():
    for _, row in master_df.iterrows():
        pid = row["PatientID"]
        bag = np.load(os.path.join(SLICES_DIR, f"{pid}.npz"))["slices"]
        bag_t = torch.from_numpy(bag).float().to(device)

        embeddings = []
        for i in range(0, bag_t.shape[0], BATCH_SIZE):
            batch = bag_t[i:i + BATCH_SIZE]
            out = backbone(batch)
            embeddings.append(out.cpu().numpy())

        patient_embeddings = np.concatenate(embeddings, axis=0)
        np.save(os.path.join(OUT_DIR, f"{pid}.npy"), patient_embeddings)
        processed.append(pid)

        if len(processed) % 50 == 0:
            print(f"{len(processed)} patients done")

print(f"\nDone. Extracted features for {len(processed)} patients.")
print(f"Each patient: (num_slices, {EMBED_DIM}) embedding array saved to {OUT_DIR}")

master_df["has_features"] = master_df["PatientID"].isin(processed)
master_df.to_csv("/kaggle/working/master_dataset_fets_with_features.csv", index=False)
print("Saved: /kaggle/working/master_dataset_fets_with_features.csv")