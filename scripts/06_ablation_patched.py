"""
======================================================================
PHASE 5 (PATCHED): Baseline and Ablation Comparisons -- REAL PIPELINE
======================================================================
Phase 5 script with three targeted patches applied:

  PATCH A -- New condition (v): "Radiomics + clinical only" -- a plain
             classifier trained on the 209-dim clinical/radiomic vector
             ALONE (no image embeddings, no attention pooling), under
             the identical cached 5-fold split. This lets the paper
             report how much of the fusion gain is coming from the
             image branch vs. the tabular features alone.

  PATCH B -- Per-patient out-of-fold (OOF) predictions are now written
             to disk for EVERY condition/task, not just the pooled
             summary metrics. Saved to phase5_oof_predictions.csv with
             columns: PatientID, condition, task, num_classes,
             true_label, pred_label, prob_0, prob_1, prob_2 (if 3-class).
             This is what a companion DeLong/permutation significance
             test needs (see phase5_significance_test.py).

  PATCH C -- Fixed feature-scaling leakage: z-score normalization of
             the four base clinical scalars (Age/Volume/T1CE/FLAIR) and
             all radiomics columns is now computed PER TRAINING FOLD
             (fit on train_df only, applied to val_df/test_df) instead
             of globally over the whole cohort before fold-splitting.
             This only affects conditions that actually use
             clinical_cols: (iv) fusion and the new (v) structured-only
             condition. Conditions (i)-(iii) are untouched because they
             never see clinical_cols.

Everything else -- file paths, hyperparameters, model architectures,
the 5-fold split cache, bootstrap CI, metric definitions -- is
UNCHANGED from the original Phase 5 script, so conditions (i)-(iv) should
reproduce numbers close to the earlier run (condition (iv)'s AUC
may shift slightly because of Patch C's normalization fix removing a
small amount of leakage).

Usage: run as-is.
It reuses the existing cached artifacts (features/, radiomics,
clinical cache, fold-split cache) -- delete phase5_fold_splits.json
only if you want a fresh split.
======================================================================
"""

import os
import json
import time
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix, average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------------------------------------------------
# CONFIG -- unchanged from the Step-5 run
# ----------------------------------------------------------------------
FEATURES_DIR = "/kaggle/working/features"
SLICES_DIR = "/kaggle/working/slices"
METADATA_PATH = "/kaggle/input/datasets/tano1974/glioma/UTSW_Glioma_Metadata-2-1.tsv"
MASTER_PATH = "/kaggle/working/master_dataset_fets_with_slices.csv"
RADIOMICS_PATH = "/kaggle/working/radiomics_features.csv"
CACHED_CLINICAL_PATH = "/kaggle/working/clinical_features_cache.csv"
SPLIT_CACHE = "/kaggle/working/phase5_fold_splits.json"
RESULTS_CSV = "/kaggle/working/phase5_ablation_results.csv"
OOF_CSV = "/kaggle/working/phase5_oof_predictions.csv"          # NEW (Patch B)

EMBED_DIM = 1280
ATTN_HIDDEN = 128
DROPOUT_P = 0.4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 40
PATIENCE = 5
BATCH_SIZE = 16
LR = 1e-3
N_FOLDS = 5
N_BOOTSTRAP = 300
SEED = 42

# Baseline 3D CNN -- smaller/faster since it trains end-to-end from raw volumes
BASELINE_EPOCHS = 25
BASELINE_PATIENCE = 5
BASELINE_LR = 1e-4
BASELINE_BATCH = 2
VOLUME_SHAPE = (64, 128, 128)  # (D, H, W) after resampling

# Structured-only (radiomics+clinical) classifier -- Patch A
STRUCT_EPOCHS = 60
STRUCT_PATIENCE = 8
STRUCT_LR = 1e-3
STRUCT_HIDDEN = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

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

# Global accumulator for per-patient OOF predictions across ALL conditions
# (Patch B) -- flushed to OOF_CSV at the end of main().
ALL_OOF_ROWS = []


# ======================================================================
# STEP 2+3: Clinical + radiomics merge, label encoding
# ----------------------------------------------------------------------
# PATCH C: this now returns the RAW (un-normalized) clinical/radiomic
# columns plus the list of column names that need z-score normalization.
# Normalization itself is done per-training-fold inside the condition
# runners below, not here.
# ======================================================================
def load_nii(path):
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)


def build_master_df():
    master_df = pd.read_csv(MASTER_PATH)
    master_df = master_df[master_df["has_slices"] == True].reset_index(drop=True)

    if os.path.exists(CACHED_CLINICAL_PATH):
        cache_df = pd.read_csv(CACHED_CLINICAL_PATH)
        master_df = master_df.merge(cache_df, on="PatientID", how="left")
    else:
        raise FileNotFoundError(
            f"{CACHED_CLINICAL_PATH} not found -- run Phase 4's Step 2 cell first "
            f"(this file should already exist from your prior run)."
        )

    master_df["SexEncoded"] = master_df["Sex"].map({"M": 0, "F": 1})

    # PATCH C: keep these RAW (no global normalization). The output
    # column names are unchanged (AgeNorm etc.) purely so the rest of
    # the pipeline (column lists, cached artifacts) doesn't need to
    # change -- but the values here are the raw quantities, not
    # z-scored yet. Per-fold normalization happens later.
    base_clinical_cols = ["AgeNorm", "SexEncoded", "VolumeNorm", "T1CENorm", "FLAIRNorm"]
    for col, out_col in [("Age", "AgeNorm"), ("TumourVolumeVoxels", "VolumeNorm"),
                          ("T1CEMeanIntensity", "T1CENorm"), ("FLAIRMeanIntensity", "FLAIRNorm")]:
        master_df[out_col] = master_df[col].astype(np.float32)

    radiomics_cols = []
    if os.path.exists(RADIOMICS_PATH):
        radiomics_df = pd.read_csv(RADIOMICS_PATH)
        master_df = master_df.merge(radiomics_df, on="PatientID", how="left")
        radiomics_cols = [c for c in radiomics_df.columns if c != "PatientID"]
        # NOTE (Patch C): no normalization and no fillna(0.0) here anymore --
        # both are now done per-fold (fit on train, applied to val/test) in
        # normalize_clinical_per_fold() below, so NaNs are filled with the
        # TRAINING fold's mean rather than a fixed 0.0 computed globally.
    else:
        print("No radiomics file -- using lightweight clinical features only")

    # Columns that actually need z-score normalization (excludes the
    # already-binary SexEncoded column).
    cols_to_normalize = ["AgeNorm", "VolumeNorm", "T1CENorm", "FLAIRNorm"] + radiomics_cols
    clinical_cols = base_clinical_cols + radiomics_cols
    print(f"Total clinical/radiomic feature dimension: {len(clinical_cols)}")

    raw_meta = pd.read_csv(METADATA_PATH, sep="\t")
    raw_meta = raw_meta.rename(columns={"Subject ID": "PatientID"})
    if "MGMT" not in master_df.columns:
        master_df = master_df.merge(raw_meta[["PatientID", "MGMT", "1p19Q CODEL"]],
                                     on="PatientID", how="left")

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

    return master_df, clinical_cols, cols_to_normalize


# ======================================================================
# PATCH C: per-training-fold normalization helper
# ======================================================================
def normalize_clinical_per_fold(train_df, val_df, test_df, cols_to_normalize):
    """Fits mean/std (and NaN-fill value) on train_df ONLY, applies the
    same statistics to val_df/test_df. Returns new copies -- does not
    mutate the inputs (so the same master_df can be reused across folds
    and conditions).
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    for col in cols_to_normalize:
        train_mean = train_df[col].mean()
        train_std = train_df[col].std()
        fill_value = train_mean if not np.isnan(train_mean) else 0.0
        for d in (train_df, val_df, test_df):
            d[col] = d[col].fillna(fill_value)
            d[col] = (d[col] - train_mean) / (train_std + 1e-8)
    return train_df, val_df, test_df


# ======================================================================
# Fixed 5-fold split, stratified on IDH -- shared across ALL conditions
# ======================================================================
def get_or_create_folds(master_df):
    if os.path.exists(SPLIT_CACHE):
        print(f"Loading cached fold splits from {SPLIT_CACHE}")
        with open(SPLIT_CACHE, "r") as f:
            folds = json.load(f)
        return folds

    print("No cached splits found -- creating fixed 5-fold split (stratified on IDH)...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for train_val_idx, test_idx in skf.split(master_df, master_df["idh_label"]):
        folds.append({
            "train_val": master_df.iloc[train_val_idx]["PatientID"].tolist(),
            "test": master_df.iloc[test_idx]["PatientID"].tolist(),
        })
    with open(SPLIT_CACHE, "w") as f:
        json.dump(folds, f)
    print(f"Cached fold splits to {SPLIT_CACHE} -- all conditions will reuse these")
    return folds


# ======================================================================
# Shared model pieces (unchanged)
# ======================================================================
class AttentionMIL(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_p):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Tanh(),
                                        nn.Dropout(dropout_p), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        w = torch.softmax(self.attention(x), dim=1)
        return (x * w).sum(dim=1), w.squeeze(-1)


class MILModel(nn.Module):
    """Generalized version of your MultiTaskMILModel -- clinical_dim=0 gives
    the image-only variant used by conditions (ii) and (iii)."""
    def __init__(self, embed_dim, tasks, attn_hidden=ATTN_HIDDEN, dropout_p=DROPOUT_P, clinical_dim=0):
        super().__init__()
        self.pooling = AttentionMIL(embed_dim, attn_hidden, dropout_p)
        self.use_fusion = clinical_dim > 0
        self.fusion_dropout = nn.Dropout(dropout_p)
        fused_dim = embed_dim + clinical_dim
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(fused_dim, 64), nn.ReLU(),
                              nn.Dropout(dropout_p), nn.Linear(64, c["num_classes"]))
            for t, c in tasks.items()
        })

    def forward(self, x, clinical=None):
        pooled, attn = self.pooling(x)
        if self.use_fusion:
            fused = self.fusion_dropout(torch.cat([pooled, clinical], dim=1))
        else:
            fused = self.fusion_dropout(pooled)
        return {t: h(fused) for t, h in self.heads.items()}, attn


class GliomaFeatureDataset(Dataset):
    """Reuses your cached embeddings -- no backbone forward pass needed."""
    def __init__(self, df, clinical_cols, tasks):
        self.df = df.reset_index(drop=True)
        self.clinical_cols = clinical_cols
        self.tasks = tasks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["PatientID"]
        embeddings = np.load(os.path.join(FEATURES_DIR, f"{pid}.npy"))
        clinical = row[self.clinical_cols].values.astype(np.float32)
        sample = {"embeddings": torch.from_numpy(embeddings).float(),
                  "clinical": torch.from_numpy(clinical).float(), "PatientID": pid}
        for task in self.tasks:
            sample[f"{task}_label"] = torch.tensor(row[f"{task}_label"], dtype=torch.long)
            sample[f"{task}_valid"] = torch.tensor(bool(row[f"{task}_valid"]), dtype=torch.bool)
        return sample


# ------------------------------------------------------------------
# PATCH A: dataset for the structured-only (no-image) condition
# ------------------------------------------------------------------
class ClinicalOnlyDataset(Dataset):
    """Same label/validity plumbing as GliomaFeatureDataset, but no
    image embeddings at all -- just the clinical/radiomic vector."""
    def __init__(self, df, clinical_cols, tasks):
        self.df = df.reset_index(drop=True)
        self.clinical_cols = clinical_cols
        self.tasks = tasks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["PatientID"]
        clinical = row[self.clinical_cols].values.astype(np.float32)
        sample = {"clinical": torch.from_numpy(clinical).float(), "PatientID": pid}
        for task in self.tasks:
            sample[f"{task}_label"] = torch.tensor(row[f"{task}_label"], dtype=torch.long)
            sample[f"{task}_valid"] = torch.tensor(bool(row[f"{task}_valid"]), dtype=torch.bool)
        return sample


class StructuredOnlyModel(nn.Module):
    """Plain MLP over the clinical/radiomic vector, one head per task --
    architecturally matched to MILModel's heads (Linear->ReLU->Dropout->
    Linear) so any AUC gap vs. condition (iv) is attributable to the
    image branch, not to a bigger/smaller classifier."""
    def __init__(self, clinical_dim, tasks, hidden=STRUCT_HIDDEN, dropout_p=DROPOUT_P):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(clinical_dim, hidden), nn.ReLU(), nn.Dropout(dropout_p))
        self.heads = nn.ModuleDict({
            t: nn.Linear(hidden, c["num_classes"]) for t, c in tasks.items()
        })

    def forward(self, clinical):
        h = self.trunk(clinical)
        return {t: head(h) for t, head in self.heads.items()}


def masked_multitask_loss(logits, batch, tasks, class_weights):
    # BUGFIX: the original version initialized `total = 0.0` (a plain Python
    # float) and only ever reassigned it to a tensor if at least one task had
    # a valid label in this batch. For single-task conditions on sparsely
    # labeled tasks (mgmt: 277/619 valid, codel: 140/619 valid), a shuffled
    # batch of size BATCH_SIZE can easily contain ZERO patients with a valid
    # label for that one task -- so `total` stayed a float, and the caller's
    # `loss.backward()` / `.item()` then crashed with
    # "AttributeError: 'float' object has no attribute 'backward'".
    # Fix: always return a tensor. When no task contributes a real loss,
    # return a zero-valued tensor still attached to the graph (via the
    # model's own output), so backward()/step()/.item() are harmless no-ops
    # for that batch instead of crashing.
    total = None
    for t in tasks:
        valid = batch[f"{t}_valid"]
        if valid.sum() == 0:
            continue
        loss = nn.functional.cross_entropy(logits[t][valid], batch[f"{t}_label"][valid],
                                            weight=class_weights.get(t))
        total = loss if total is None else total + loss
    if total is None:
        any_logits = next(iter(logits.values()))
        total = any_logits.sum() * 0.0
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
    return (np.mean(boot_aucs), np.percentile(boot_aucs, (100 - ci) / 2),
            np.percentile(boot_aucs, 100 - (100 - ci) / 2))


def summarize_task(task, num_classes, oof_labels, oof_logits, oof_valid, all_pids):
    pids = [p for p in all_pids if oof_valid[task].get(p, False)]
    if len(pids) < 2:
        return None
    yv = np.array([oof_labels[p][task] for p in pids])
    probs = np.stack([torch.softmax(torch.tensor(oof_logits[task][p]), dim=0).numpy() for p in pids])
    preds = probs.argmax(axis=1)
    if len(np.unique(yv)) < 2:
        return None

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
        auc, pr_auc = float("nan"), float("nan")

    mean_auc, ci_lo, ci_hi = bootstrap_auc_ci(yv, probs, num_classes)

    return {
        "task": task, "n": len(pids), "accuracy": acc, "balanced_accuracy": bal_acc,
        "macro_f1": f1, "macro_precision": precision, "macro_recall": recall,
        "auc": auc, "pr_auc": pr_auc, "auc_ci_low": ci_lo, "auc_ci_high": ci_hi,
        "confusion_matrix": cm.tolist(),
    }


# ------------------------------------------------------------------
# PATCH B: turn a condition's raw OOF dicts into per-patient CSV rows
# ------------------------------------------------------------------
def record_oof_predictions(condition_name, task, num_classes, oof_labels, oof_logits, oof_valid, all_pids):
    pids = [p for p in all_pids if oof_valid[task].get(p, False)]
    for pid in pids:
        probs = torch.softmax(torch.tensor(oof_logits[task][pid]), dim=0).numpy()
        row = {
            "PatientID": pid,
            "condition": condition_name,
            "task": task,
            "num_classes": num_classes,
            "true_label": int(oof_labels[pid][task]),
            "pred_label": int(np.argmax(probs)),
        }
        for c in range(num_classes):
            row[f"prob_{c}"] = float(probs[c])
        ALL_OOF_ROWS.append(row)


# ======================================================================
# Conditions (ii), (iii), (iv) -- MIL variants, all reuse cached embeddings
# ======================================================================
def run_mil_condition(master_df, clinical_cols, cols_to_normalize, folds, tasks, use_fusion, condition_name):
    print(f"\n{'=' * 70}\nCONDITION: {condition_name}\n{'=' * 70}")
    oof_logits = {t: {} for t in tasks}
    oof_labels = {}
    oof_valid = {t: {} for t in tasks}
    clinical_dim = len(clinical_cols) if use_fusion else 0

    for fold_idx, fold in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        train_val_df = master_df[master_df["PatientID"].isin(fold["train_val"])].reset_index(drop=True)
        test_df = master_df[master_df["PatientID"].isin(fold["test"])].reset_index(drop=True)
        train_df, val_df = train_test_split(train_val_df, test_size=0.15,
                                             stratify=train_val_df["idh_label"], random_state=fold_idx)

        # PATCH C: fit clinical normalization on train_df only, apply to val/test.
        # Only matters when use_fusion=True (otherwise clinical_cols aren't consumed),
        # but it's harmless/cheap to always do it.
        if use_fusion:
            train_df, val_df, test_df = normalize_clinical_per_fold(
                train_df, val_df, test_df, cols_to_normalize)

        class_weights = compute_class_weights(train_df, tasks)
        train_loader = DataLoader(GliomaFeatureDataset(train_df, clinical_cols, tasks),
                                   batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(GliomaFeatureDataset(val_df, clinical_cols, tasks),
                                 batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(GliomaFeatureDataset(test_df, clinical_cols, tasks),
                                  batch_size=BATCH_SIZE, shuffle=False)

        model = MILModel(EMBED_DIM, tasks, clinical_dim=clinical_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(NUM_EPOCHS):
            model.train()
            for batch in train_loader:
                emb = batch["embeddings"].to(device)
                clin = batch["clinical"].to(device) if use_fusion else None
                b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                optimizer.zero_grad()
                logits, _ = model(emb, clin)
                loss = masked_multitask_loss(logits, b, tasks, class_weights)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    emb = batch["embeddings"].to(device)
                    clin = batch["clinical"].to(device) if use_fusion else None
                    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    logits, _ = model(emb, clin)
                    val_loss += masked_multitask_loss(logits, b, tasks, class_weights).item()
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
        print(f"Fold {fold_idx + 1} best val_loss: {best_val_loss:.4f} (model restored)")

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                emb = batch["embeddings"].to(device)
                clin = batch["clinical"].to(device) if use_fusion else None
                logits, _ = model(emb, clin)
                pids = batch["PatientID"]
                for i, pid in enumerate(pids):
                    oof_labels[pid] = {t: batch[f"{t}_label"][i].item() for t in tasks}
                    for t in tasks:
                        oof_logits[t][pid] = logits[t][i].cpu().numpy()
                        oof_valid[t][pid] = batch[f"{t}_valid"][i].item()

    results = []
    all_pids = master_df["PatientID"].tolist()
    for t in tasks:
        r = summarize_task(t, tasks[t]["num_classes"], oof_labels, oof_logits, oof_valid, all_pids)
        # Patch B: always record predictions, even if summarize_task can't
        # compute a summary (e.g. too few valid patients) -- the raw
        # predictions are still useful downstream.
        record_oof_predictions(condition_name, t, tasks[t]["num_classes"],
                                oof_labels, oof_logits, oof_valid, all_pids)
        if r is None:
            continue
        r["condition"] = condition_name
        results.append(r)
        print(f"  {t.upper():8s} n={r['n']:4d}  AUC={r['auc']:.3f}  "
              f"Acc={r['accuracy']:.3f}  BalAcc={r['balanced_accuracy']:.3f}")
    return results


# ======================================================================
# PATCH A: Condition (v) -- Radiomics + clinical only (no images)
# ======================================================================
def run_structured_only_condition(master_df, clinical_cols, cols_to_normalize, folds, tasks,
                                   condition_name="(v) Radiomics+clinical only"):
    print(f"\n{'=' * 70}\nCONDITION: {condition_name}\n{'=' * 70}")
    oof_logits = {t: {} for t in tasks}
    oof_labels = {}
    oof_valid = {t: {} for t in tasks}
    clinical_dim = len(clinical_cols)

    for fold_idx, fold in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        train_val_df = master_df[master_df["PatientID"].isin(fold["train_val"])].reset_index(drop=True)
        test_df = master_df[master_df["PatientID"].isin(fold["test"])].reset_index(drop=True)
        train_df, val_df = train_test_split(train_val_df, test_size=0.15,
                                             stratify=train_val_df["idh_label"], random_state=fold_idx)

        # Patch C: per-fold normalization, same as the fusion condition.
        train_df, val_df, test_df = normalize_clinical_per_fold(
            train_df, val_df, test_df, cols_to_normalize)

        class_weights = compute_class_weights(train_df, tasks)
        train_loader = DataLoader(ClinicalOnlyDataset(train_df, clinical_cols, tasks),
                                   batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(ClinicalOnlyDataset(val_df, clinical_cols, tasks),
                                 batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(ClinicalOnlyDataset(test_df, clinical_cols, tasks),
                                  batch_size=BATCH_SIZE, shuffle=False)

        model = StructuredOnlyModel(clinical_dim, tasks).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=STRUCT_LR, weight_decay=WEIGHT_DECAY)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(STRUCT_EPOCHS):
            model.train()
            for batch in train_loader:
                clin = batch["clinical"].to(device)
                b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                optimizer.zero_grad()
                logits = model(clin)
                loss = masked_multitask_loss(logits, b, tasks, class_weights)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    clin = batch["clinical"].to(device)
                    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    logits = model(clin)
                    val_loss += masked_multitask_loss(logits, b, tasks, class_weights).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= STRUCT_PATIENCE:
                break

        model.load_state_dict(best_state)
        print(f"Fold {fold_idx + 1} best val_loss: {best_val_loss:.4f} (model restored)")

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                clin = batch["clinical"].to(device)
                logits = model(clin)
                pids = batch["PatientID"]
                for i, pid in enumerate(pids):
                    oof_labels[pid] = {t: batch[f"{t}_label"][i].item() for t in tasks}
                    for t in tasks:
                        oof_logits[t][pid] = logits[t][i].cpu().numpy()
                        oof_valid[t][pid] = batch[f"{t}_valid"][i].item()

    results = []
    all_pids = master_df["PatientID"].tolist()
    for t in tasks:
        r = summarize_task(t, tasks[t]["num_classes"], oof_labels, oof_logits, oof_valid, all_pids)
        record_oof_predictions(condition_name, t, tasks[t]["num_classes"],
                                oof_labels, oof_logits, oof_valid, all_pids)
        if r is None:
            continue
        r["condition"] = condition_name
        results.append(r)
        print(f"  {t.upper():8s} n={r['n']:4d}  AUC={r['auc']:.3f}  "
              f"Acc={r['accuracy']:.3f}  BalAcc={r['balanced_accuracy']:.3f}")
    return results


# ======================================================================
# Condition (i) -- Baseline 3D CNN, fully-supervised, IDH only (unchanged)
# ======================================================================
def zscore(vol, mask):
    voxels = vol[mask]
    if voxels.size == 0:
        return vol
    mean, std = voxels.mean(), voxels.std()
    return np.clip((vol - mean) / (std + 1e-6), -5, 5)


def load_and_resample_volume(row, target_shape=VOLUME_SHAPE):
    t1 = load_nii(row["T1_Path"])
    t1ce = load_nii(row["T1CE_Path"])
    t2 = load_nii(row["T2_Path"])
    flair = load_nii(row["FLAIR_Path"])
    brain_mask = (t1 > 0) | (t1ce > 0) | (t2 > 0) | (flair > 0)

    channels = []
    for vol in [t1, t1ce, t2, flair]:
        vol = zscore(vol, brain_mask)
        vol = np.transpose(vol, (2, 0, 1))  # (H,W,D) -> (D,H,W)
        factors = [t / s for t, s in zip(target_shape, vol.shape)]
        channels.append(zoom(vol, factors, order=1).astype(np.float32))
    return np.stack(channels, axis=0)  # (4, D, H, W)


class VolumeDataset(Dataset):
    def __init__(self, df, cache_dir="/kaggle/working/baseline_volumes"):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["PatientID"]
        cache_path = os.path.join(self.cache_dir, f"{pid}.npy")
        if os.path.exists(cache_path):
            vol = np.load(cache_path)
        else:
            vol = load_and_resample_volume(row)
            np.save(cache_path, vol)
        return {"volume": torch.from_numpy(vol).float(),
                "label": torch.tensor(int(row["idh_label"]), dtype=torch.long),
                "PatientID": pid}


class Baseline3DCNN(nn.Module):
    def __init__(self, n_classes=2, in_channels=4):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1), nn.BatchNorm3d(cout), nn.ReLU(),
                nn.MaxPool3d(2)
            )
        self.features = nn.Sequential(block(in_channels, 32), block(32, 64),
                                       block(64, 128), block(128, 256))
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
                                   nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.gap(self.features(x)))


def run_baseline_condition(master_df, folds, condition_name="(i) Baseline 3D CNN (fully-supervised, IDH)"):
    print(f"\n{'=' * 70}\nCONDITION: {condition_name}\n{'=' * 70}")
    oof_logits, oof_labels = {}, {}

    for fold_idx, fold in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        train_val_df = master_df[master_df["PatientID"].isin(fold["train_val"])].reset_index(drop=True)
        test_df = master_df[master_df["PatientID"].isin(fold["test"])].reset_index(drop=True)
        train_df, val_df = train_test_split(train_val_df, test_size=0.15,
                                             stratify=train_val_df["idh_label"], random_state=fold_idx)

        train_loader = DataLoader(VolumeDataset(train_df), batch_size=BASELINE_BATCH, shuffle=True)
        val_loader = DataLoader(VolumeDataset(val_df), batch_size=BASELINE_BATCH, shuffle=False)
        test_loader = DataLoader(VolumeDataset(test_df), batch_size=BASELINE_BATCH, shuffle=False)

        model = Baseline3DCNN(n_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=BASELINE_LR, weight_decay=WEIGHT_DECAY)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        fold_start = time.time()

        for epoch in range(BASELINE_EPOCHS):
            epoch_start = time.time()
            model.train()
            n_batches = 0
            for batch in train_loader:
                vol = batch["volume"].to(device)
                label = batch["label"].to(device)
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(vol), label)
                loss.backward()
                optimizer.step()
                n_batches += 1

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    vol = batch["volume"].to(device)
                    label = batch["label"].to(device)
                    val_loss += nn.functional.cross_entropy(model(vol), label).item()
            val_loss /= max(len(val_loader), 1)

            epoch_time = time.time() - epoch_start
            sec_per_batch = epoch_time / max(n_batches, 1)
            elapsed = time.time() - fold_start
            remaining_epochs_this_fold = BASELINE_EPOCHS - (epoch + 1)
            eta_this_fold = remaining_epochs_this_fold * epoch_time
            folds_remaining = N_FOLDS - (fold_idx + 1)
            eta_total = eta_this_fold + folds_remaining * BASELINE_EPOCHS * epoch_time
            print(f"    epoch {epoch + 1}/{BASELINE_EPOCHS}  "
                  f"val_loss={val_loss:.4f}  "
                  f"{epoch_time:.1f}s/epoch ({sec_per_batch:.2f}s/batch)  "
                  f"elapsed={elapsed / 60:.1f}min  "
                  f"ETA this fold={eta_this_fold / 60:.1f}min  "
                  f"ETA all folds={eta_total / 60:.1f}min")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= BASELINE_PATIENCE:
                print(f"    early stopping at epoch {epoch + 1} (no improvement for {BASELINE_PATIENCE} epochs)")
                break

        model.load_state_dict(best_state)
        print(f"Fold {fold_idx + 1} best val_loss: {best_val_loss:.4f} (model restored)")

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                vol = batch["volume"].to(device)
                logits = model(vol).cpu()
                for i, pid in enumerate(batch["PatientID"]):
                    oof_logits[pid] = logits[i].numpy()
                    oof_labels[pid] = {"idh": batch["label"][i].item()}

    all_pids = list(oof_labels.keys())
    oof_valid = {"idh": {p: True for p in all_pids}}
    r = summarize_task("idh", 2, oof_labels, {"idh": oof_logits}, oof_valid, all_pids)
    record_oof_predictions(condition_name, "idh", 2, oof_labels, {"idh": oof_logits}, oof_valid, all_pids)
    if r is None:
        print("  Insufficient valid samples.")
        return []
    r["condition"] = condition_name
    print(f"  IDH      n={r['n']:4d}  AUC={r['auc']:.3f}  "
          f"Acc={r['accuracy']:.3f}  BalAcc={r['balanced_accuracy']:.3f}")
    return [r]


# ======================================================================
# MASTER RUN
# ======================================================================
def main():
    print("\n" + "=" * 70 + "\nPHASE 5 (PATCHED): Baseline and Ablation Comparisons\n" + "=" * 70)

    master_df, clinical_cols, cols_to_normalize = build_master_df()
    folds = get_or_create_folds(master_df)
    print(f"\nUsing {N_FOLDS} fixed folds across all conditions "
          f"({len(master_df)} patients total)")

    all_results = []
    all_results += run_baseline_condition(master_df, folds)

    for t in TASKS:
        all_results += run_mil_condition(master_df, clinical_cols, cols_to_normalize, folds,
                                          tasks={t: TASKS[t]}, use_fusion=False,
                                          condition_name=f"(ii) 2.5D MIL single-task [{t}]")

    all_results += run_mil_condition(master_df, clinical_cols, cols_to_normalize, folds, tasks=TASKS,
                                      use_fusion=False, condition_name="(iii) 2.5D MIL multi-task")

    all_results += run_mil_condition(master_df, clinical_cols, cols_to_normalize, folds, tasks=TASKS,
                                      use_fusion=True,
                                      condition_name="(iv) 2.5D MIL multi-task + fusion")

    # PATCH A: new radiomics/clinical-only condition, same folds, same tasks.
    all_results += run_structured_only_condition(master_df, clinical_cols, cols_to_normalize,
                                                  folds, tasks=TASKS)

    df = pd.DataFrame(all_results)
    df_display = df[["condition", "task", "n", "accuracy", "balanced_accuracy",
                      "macro_f1", "auc", "auc_ci_low", "auc_ci_high"]]
    df_display.to_csv(RESULTS_CSV, index=False)

    # PATCH B: flush all per-patient OOF predictions collected across every
    # condition/task to disk. This is the file phase5_significance_test.py
    # (companion script) consumes to run DeLong / permutation tests.
    oof_df = pd.DataFrame(ALL_OOF_ROWS)
    oof_df.to_csv(OOF_CSV, index=False)
    print(f"\nSaved {len(oof_df)} per-patient OOF prediction rows to {OOF_CSV}")

    print("\n" + "=" * 70 + "\nPHASE 5 SUMMARY -- pooled out-of-fold, all conditions\n" + "=" * 70)
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(df_display.round(3).to_string(index=False))
    print(f"\nSaved full results (incl. confusion matrices) to {RESULTS_CSV}")
    print("\nDONE")
    return df


if __name__ == "__main__":
    main()