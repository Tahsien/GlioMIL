import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix, average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight

FEATURES_DIR = "/kaggle/working/features"
METADATA_PATH = "/kaggle/input/datasets/tano1974/glioma/UTSW_Glioma_Metadata-2-1.tsv"
MASTER_PATH = "/kaggle/working/master_dataset_fets_with_slices.csv"
RADIOMICS_PATH = "/kaggle/working/radiomics_features.csv"
CACHED_CLINICAL_PATH = "/kaggle/working/clinical_features_cache.csv"

EMBED_DIM = 1280
ATTN_HIDDEN = 128
DROPOUT_P = 0.4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 40       # reduced from 60
PATIENCE = 5          # reduced from 8
BATCH_SIZE = 16
LR = 1e-3
N_FOLDS = 5
N_BOOTSTRAP = 300      # reduced from 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
if device.type == "cpu":
    print("WARNING: GPU not detected -- training will be much slower. "
          "Check Settings -> Accelerator in Kaggle.")

GRADE_MAP = {2.0: 0, 3.0: 1, 4.0: 2}
BINARY_MAP = {
    "MGMT": {"unmethylated": 0, "methylated": 1},
    "1p19Q CODEL": {"non co-deleted": 0, "co-deleted": 1},
}
TASKS = {
    "idh": {"num_classes": 2}, "grade": {"num_classes": 3},
    "mgmt": {"num_classes": 2}, "codel": {"num_classes": 2},
}

# ============================================================
# STEP 2: Clinical features (CACHED -- only computed once ever)
# ============================================================
print("\n" + "=" * 70 + "\nSTEP 2: Clinical + Radiomics Merge\n" + "=" * 70)


def load_nii(path):
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)


master_df = pd.read_csv(MASTER_PATH)
master_df = master_df[master_df["has_slices"] == True].reset_index(drop=True)

if os.path.exists(CACHED_CLINICAL_PATH):
    print(f"Loading cached clinical features from {CACHED_CLINICAL_PATH} (skipping recomputation)")
    cache_df = pd.read_csv(CACHED_CLINICAL_PATH)
    master_df = master_df.merge(cache_df, on="PatientID", how="left")
else:
    print("No cache found -- computing clinical features (one-time cost)...")
    volumes, t1ce_means, flair_means = [], [], []
    for i, row in master_df.iterrows():
        seg = load_nii(row["Mask_Path"])
        mask = seg > 0
        volumes.append(int(mask.sum()))
        t1ce = load_nii(row["T1CE_Path"])
        flair = load_nii(row["FLAIR_Path"])
        t1ce_means.append(float(t1ce[mask].mean()) if mask.sum() > 0 else 0.0)
        flair_means.append(float(flair[mask].mean()) if mask.sum() > 0 else 0.0)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(master_df)} patients processed")

    master_df["TumourVolumeVoxels"] = volumes
    master_df["T1CEMeanIntensity"] = t1ce_means
    master_df["FLAIRMeanIntensity"] = flair_means

    cache_out = master_df[["PatientID", "TumourVolumeVoxels", "T1CEMeanIntensity", "FLAIRMeanIntensity"]]
    cache_out.to_csv(CACHED_CLINICAL_PATH, index=False)
    print(f"Cached to {CACHED_CLINICAL_PATH} -- future runs will skip this step")

master_df["SexEncoded"] = master_df["Sex"].map({"M": 0, "F": 1})

base_clinical_cols = ["AgeNorm", "SexEncoded", "VolumeNorm", "T1CENorm", "FLAIRNorm"]
for col, out_col in [("Age", "AgeNorm"), ("TumourVolumeVoxels", "VolumeNorm"),
                      ("T1CEMeanIntensity", "T1CENorm"), ("FLAIRMeanIntensity", "FLAIRNorm")]:
    master_df[out_col] = (master_df[col] - master_df[col].mean()) / (master_df[col].std() + 1e-8)

radiomics_cols = []
if os.path.exists(RADIOMICS_PATH):
    radiomics_df = pd.read_csv(RADIOMICS_PATH)
    master_df = master_df.merge(radiomics_df, on="PatientID", how="left")
    radiomics_cols = [c for c in radiomics_df.columns if c != "PatientID"]
    for col in radiomics_cols:
        mean, std = master_df[col].mean(), master_df[col].std()
        master_df[col] = (master_df[col] - mean) / (std + 1e-8)
        master_df[col] = master_df[col].fillna(0.0)
    print(f"Merged {len(radiomics_cols)} full radiomics features")
else:
    print("No radiomics file -- using lightweight clinical features only")

CLINICAL_COLS = base_clinical_cols + radiomics_cols
CLINICAL_DIM = len(CLINICAL_COLS)
print(f"Total clinical/radiomic feature dimension: {CLINICAL_DIM}")

# ============================================================
# STEP 3: Merge labels
# ============================================================
raw_meta = pd.read_csv(METADATA_PATH, sep="\t")
raw_meta = raw_meta.rename(columns={"Subject ID": "PatientID"})
if "MGMT" not in master_df.columns:
    master_df = master_df.merge(raw_meta[["PatientID", "MGMT", "1p19Q CODEL"]], on="PatientID", how="left")


def encode_labels(row):
    out = {}
    out["idh_label"] = int(row["IDH_Label"])
    out["idh_valid"] = True
    grade_val = row.get("Grade", np.nan)
    if grade_val in GRADE_MAP:
        out["grade_label"] = GRADE_MAP[grade_val]
        out["grade_valid"] = True
    else:
        out["grade_label"] = 0
        out["grade_valid"] = False
    mgmt_val = row.get("MGMT", None)
    if isinstance(mgmt_val, str) and mgmt_val.strip() in BINARY_MAP["MGMT"]:
        out["mgmt_label"] = BINARY_MAP["MGMT"][mgmt_val.strip()]
        out["mgmt_valid"] = True
    else:
        out["mgmt_label"] = 0
        out["mgmt_valid"] = False
    codel_val = row.get("1p19Q CODEL", None)
    is_idh_mutant = int(row["IDH_Label"]) == 1
    if is_idh_mutant and isinstance(codel_val, str) and codel_val.strip() in BINARY_MAP["1p19Q CODEL"]:
        out["codel_label"] = BINARY_MAP["1p19Q CODEL"][codel_val.strip()]
        out["codel_valid"] = True
    else:
        out["codel_label"] = 0
        out["codel_valid"] = False
    return pd.Series(out)


labels_df = master_df.apply(encode_labels, axis=1)
master_df = pd.concat([master_df, labels_df], axis=1)

print("\nTask label availability:")
for task in TASKS:
    print(f"  {task}: {master_df[f'{task}_valid'].sum()} / {len(master_df)} valid")


# ============================================================
# STEP 4: Dataset + Model
# ============================================================
class GliomaFeatureDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["PatientID"]
        embeddings = np.load(os.path.join(FEATURES_DIR, f"{pid}.npy"))
        clinical = row[CLINICAL_COLS].values.astype(np.float32)
        sample = {"embeddings": torch.from_numpy(embeddings).float(),
                  "clinical": torch.from_numpy(clinical).float(), "PatientID": pid}
        for task in TASKS:
            sample[f"{task}_label"] = torch.tensor(row[f"{task}_label"], dtype=torch.long)
            sample[f"{task}_valid"] = torch.tensor(bool(row[f"{task}_valid"]), dtype=torch.bool)
        return sample


class AttentionMIL(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_p):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Tanh(),
                                        nn.Dropout(dropout_p), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        w = torch.softmax(self.attention(x), dim=1)
        return (x * w).sum(dim=1), w.squeeze(-1)


class MultiTaskMILModel(nn.Module):
    def __init__(self, embed_dim, clinical_dim, attn_hidden, tasks, dropout_p):
        super().__init__()
        self.pooling = AttentionMIL(embed_dim, attn_hidden, dropout_p)
        self.fusion_dropout = nn.Dropout(dropout_p)
        fused_dim = embed_dim + clinical_dim
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(fused_dim, 64), nn.ReLU(),
                              nn.Dropout(dropout_p), nn.Linear(64, c["num_classes"]))
            for t, c in tasks.items()
        })

    def forward(self, x, clinical):
        pooled, attn = self.pooling(x)
        fused = self.fusion_dropout(torch.cat([pooled, clinical], dim=1))
        return {t: h(fused) for t, h in self.heads.items()}, attn


def masked_multitask_loss(logits, batch, tasks, class_weights):
    total = 0.0
    for t in tasks:
        valid = batch[f"{t}_valid"]
        if valid.sum() == 0:
            continue
        loss = nn.functional.cross_entropy(logits[t][valid], batch[f"{t}_label"][valid],
                                            weight=class_weights.get(t))
        total = total + loss
    return total


def compute_class_weights(train_df, tasks):
    cw = {}
    for task, cfg in tasks.items():
        valid_mask = train_df[f"{task}_valid"]
        if valid_mask.sum() < 2:
            continue
        labels = train_df.loc[valid_mask, f"{task}_label"].values
        present = np.unique(labels)
        if len(present) < 2:
            continue
        w = compute_class_weight("balanced", classes=present, y=labels)
        full_w = np.ones(cfg["num_classes"], dtype=np.float32)
        for cls, ww in zip(present, w):
            full_w[int(cls)] = ww
        cw[task] = torch.tensor(full_w, dtype=torch.float32).to(device)
    return cw


# ============================================================
# STEP 5: 5-fold cross-validation
# ============================================================
print("\n" + "=" * 70 + "\nSTEP 5: 5-Fold Cross-Validation\n" + "=" * 70)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_logits = {t: {} for t in TASKS}
oof_labels = {}
oof_valid = {t: {} for t in TASKS}
fold_idh_aucs = []

for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(master_df, master_df["idh_label"])):
    print(f"\n{'='*70}\nFOLD {fold_idx+1}/{N_FOLDS}\n{'='*70}")
    train_val_df = master_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = master_df.iloc[test_idx].reset_index(drop=True)
    train_df, val_df = train_test_split(train_val_df, test_size=0.15,
                                          stratify=train_val_df["idh_label"], random_state=fold_idx)

    class_weights = compute_class_weights(train_df, TASKS)
    train_loader = DataLoader(GliomaFeatureDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(GliomaFeatureDataset(val_df), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(GliomaFeatureDataset(test_df), batch_size=BATCH_SIZE, shuffle=False)

    model = MultiTaskMILModel(EMBED_DIM, CLINICAL_DIM, ATTN_HIDDEN, TASKS, DROPOUT_P).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            emb = batch["embeddings"].to(device)
            clin = batch["clinical"].to(device)
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            logits, _ = model(emb, clin)
            loss = masked_multitask_loss(logits, b, TASKS, class_weights)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                emb = batch["embeddings"].to(device)
                clin = batch["clinical"].to(device)
                b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                logits, _ = model(emb, clin)
                val_loss += masked_multitask_loss(logits, b, TASKS, class_weights).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"/kaggle/working/mil_model_fold{fold_idx}.pt")
    print(f"Fold {fold_idx+1} best val_loss: {best_val_loss:.4f} (model saved)")

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            emb = batch["embeddings"].to(device)
            clin = batch["clinical"].to(device)
            logits, _ = model(emb, clin)
            pids = batch["PatientID"]
            for i, pid in enumerate(pids):
                oof_labels[pid] = {t: batch[f"{t}_label"][i].item() for t in TASKS}
                for t in TASKS:
                    oof_logits[t][pid] = logits[t][i].cpu().numpy()
                    oof_valid[t][pid] = batch[f"{t}_valid"][i].item()

    idh_probs, idh_true = [], []
    for pid in test_df["PatientID"]:
        if oof_valid["idh"][pid]:
            p = torch.softmax(torch.tensor(oof_logits["idh"][pid]), dim=0).numpy()
            idh_probs.append(p[1])
            idh_true.append(oof_labels[pid]["idh"])
    if len(set(idh_true)) > 1:
        fold_auc = roc_auc_score(idh_true, idh_probs)
        fold_idh_aucs.append(fold_auc)
        print(f"Fold {fold_idx+1} IDH AUC: {fold_auc:.3f}")

print(f"\n{'='*70}")
print(f"PER-FOLD IDH AUC: {[f'{a:.3f}' for a in fold_idh_aucs]}")
print(f"Mean ± SD: {np.mean(fold_idh_aucs):.3f} ± {np.std(fold_idh_aucs):.3f}")


# ============================================================
# STEP 6: Pooled out-of-fold metrics (HONEST RESULT)
# ============================================================
def bootstrap_auc_ci(y_true, probs, num_classes, n_boot=N_BOOTSTRAP, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        pp = probs[idx]
        try:
            if num_classes == 2:
                boot_aucs.append(roc_auc_score(yt, pp[:, 1]))
            else:
                boot_aucs.append(roc_auc_score(yt, pp, multi_class="ovr"))
        except ValueError:
            continue
    if len(boot_aucs) == 0:
        return float("nan"), float("nan"), float("nan")
    return (np.mean(boot_aucs), np.percentile(boot_aucs, (100-ci)/2), np.percentile(boot_aucs, 100-(100-ci)/2))


print(f"\n{'='*70}\nPOOLED OUT-OF-FOLD PERFORMANCE (all {len(master_df)} patients) -- HONEST RESULT\n{'='*70}")

for task in TASKS:
    num_classes = TASKS[task]["num_classes"]
    pids = [p for p in master_df["PatientID"] if oof_valid[task].get(p, False)]
    print(f"\n{'-'*70}\n{task.upper()}  (n={len(pids)})\n{'-'*70}")
    if len(pids) < 2:
        print("  Insufficient valid samples.")
        continue
    yv = np.array([oof_labels[p][task] for p in pids])
    probs = np.stack([torch.softmax(torch.tensor(oof_logits[task][p]), dim=0).numpy() for p in pids])
    preds = probs.argmax(axis=1)
    if len(np.unique(yv)) < 2:
        print("  Only one class present, skipping.")
        continue

    acc = accuracy_score(yv, preds)
    bal_acc = balanced_accuracy_score(yv, preds)
    f1 = f1_score(yv, preds, average="macro")
    precision = precision_score(yv, preds, average="macro", zero_division=0)
    recall = recall_score(yv, preds, average="macro", zero_division=0)
    cm = confusion_matrix(yv, preds, labels=list(range(num_classes)))

    try:
        if num_classes == 2:
            auc = roc_auc_score(yv, probs[:, 1])
            pr_auc = average_precision_score(yv, probs[:, 1])
        else:
            auc = roc_auc_score(yv, probs, multi_class="ovr")
            pr_auc = float("nan")
    except ValueError:
        auc = float("nan")
        pr_auc = float("nan")

    mean_auc, ci_lo, ci_hi = bootstrap_auc_ci(yv, probs, num_classes)

    print(f"  Accuracy:            {acc:.3f}")
    print(f"  Balanced accuracy:   {bal_acc:.3f}")
    print(f"  Macro F1:            {f1:.3f}")
    print(f"  Macro precision:     {precision:.3f}")
    print(f"  Macro recall:        {recall:.3f}")
    print(f"  AUC-ROC:             {auc:.3f}  (bootstrap mean={mean_auc:.3f}, 95% CI: {ci_lo:.3f}-{ci_hi:.3f})")
    if num_classes == 2:
        print(f"  PR-AUC:              {pr_auc:.3f}")
    print(f"  Confusion matrix:\n  {cm}")

    for cls in range(num_classes):
        tp = cm[cls, cls]
        fn = cm[cls, :].sum() - tp
        fp = cm[:, cls].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        print(f"    Class {cls}: sensitivity={sens:.3f}, specificity={spec:.3f}")

print(f"\n{'='*70}\nDONE\n{'='*70}")