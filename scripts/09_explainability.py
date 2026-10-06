import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold, train_test_split

print("=" * 70)
print("PHASE 8: Explainability -- MIL Attention Analysis")
print("=" * 70)

SLICE_SIZE = (224, 224)
NUM_SLICES = 15
MARGIN = 2


def make_dataset(df):
    # GliomaFeatureDataset's signature can vary depending on which phase
    # last redefined it in this session (some versions take just `df`,
    # others take `df, clinical_cols, tasks`). Inspect and call correctly.
    import inspect
    n_params = len(inspect.signature(GliomaFeatureDataset.__init__).parameters) - 1  # exclude self
    if n_params >= 3:
        return GliomaFeatureDataset(df, CLINICAL_COLS, TASKS)
    return GliomaFeatureDataset(df)


def masked_multitask_loss_safe(logits, batch, tasks, class_weights):
    total = None
    for t in tasks:
        valid = batch[f"{t}_valid"]
        if valid.sum() == 0:
            continue
        loss = torch.nn.functional.cross_entropy(
            logits[t][valid], batch[f"{t}_label"][valid], weight=class_weights.get(t)
        )
        total = loss if total is None else total + loss
    if total is None:
        any_task = next(iter(logits))
        total = logits[any_task].sum() * 0.0
    return total


# ------------------------------------------------------------
# Step 1: Train the multi-task + fusion model with 5-fold CV
# (same architecture, same fold seed as Step 5) and collect
# out-of-fold attention weights per patient.
# ------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

phase8_oof_attn = {}
phase8_oof_pred = {}
phase8_oof_prob = {}
phase8_oof_true = {}

for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(master_df, master_df["idh_label"])):
    print(f"\nFold {fold_idx + 1}/{N_FOLDS}")
    train_val_df = master_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = master_df.iloc[test_idx].reset_index(drop=True)
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.15, stratify=train_val_df["idh_label"], random_state=fold_idx
    )

    class_weights = compute_class_weights(train_df, TASKS)
    train_loader = DataLoader(make_dataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(make_dataset(val_df), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(make_dataset(test_df), batch_size=BATCH_SIZE, shuffle=False)

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
            loss = masked_multitask_loss_safe(logits, b, TASKS, class_weights)
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
                val_loss += masked_multitask_loss_safe(logits, b, TASKS, class_weights).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            emb = batch["embeddings"].to(device)
            clin = batch["clinical"].to(device)
            logits, attn = model(emb, clin)
            probs = torch.softmax(logits["idh"], dim=1)[:, 1].cpu().numpy()
            preds = logits["idh"].argmax(dim=1).cpu().numpy()
            attn_np = attn.cpu().numpy()
            true_np = batch["idh_label"].numpy()
            pids = batch["PatientID"]
            for i, pid in enumerate(pids):
                phase8_oof_attn[pid] = attn_np[i]
                phase8_oof_pred[pid] = int(preds[i])
                phase8_oof_prob[pid] = float(probs[i])
                phase8_oof_true[pid] = int(true_np[i])

    print(f"  Fold {fold_idx + 1} test patients: {len(test_df)}, "
          f"total collected so far: {len(phase8_oof_attn)}")

print(f"\nCollected OOF attention weights for {len(phase8_oof_attn)} / {len(master_df)} patients")


# ------------------------------------------------------------
# Step 2: Quantitative overlap -- attention weight vs. tumour
# content, recomputed on the same tumour-guided z-slices used
# in Phase 2 slice extraction.
# ------------------------------------------------------------
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


def slice_tumor_fractions(mask_path):
    seg = np.asarray(nib.load(mask_path).get_fdata(), dtype=np.float32)
    z_range = tumour_z_range(seg, MARGIN)
    if z_range is None:
        return None, None
    z_min, z_max = z_range
    if z_max == z_min:
        z_indices = [z_min] * NUM_SLICES
    else:
        z_indices = np.round(np.linspace(z_min, z_max, NUM_SLICES)).astype(int).tolist()
    fracs = []
    for z in z_indices:
        sl = (seg[:, :, z] > 0).astype(np.float32)
        sl_resized = resize2d(sl, SLICE_SIZE)
        fracs.append(float((sl_resized > 0.5).mean()))
    return np.array(fracs), z_indices


pid_to_row = master_df.set_index("PatientID")

correlations = []
top3_vs_all_ratio = []
attn_mass_on_high_tumor = []
per_patient_records = []

for pid, attn in phase8_oof_attn.items():
    row = pid_to_row.loc[pid]
    fracs, z_indices = slice_tumor_fractions(row["Mask_Path"])
    if fracs is None or fracs.std() < 1e-8:
        continue

    r, _ = pearsonr(attn, fracs)
    correlations.append(r)

    top3_idx = np.argsort(attn)[-3:]
    top3_mean_frac = fracs[top3_idx].mean()
    all_mean_frac = fracs.mean()
    ratio = top3_mean_frac / (all_mean_frac + 1e-8)
    top3_vs_all_ratio.append(ratio)

    median_frac = np.median(fracs)
    high_tumor_mask = fracs >= median_frac
    mass = float(attn[high_tumor_mask].sum())
    attn_mass_on_high_tumor.append(mass)

    per_patient_records.append({
        "PatientID": pid,
        "attn_tumor_corr": r,
        "top3_vs_all_tumor_frac_ratio": ratio,
        "attn_mass_on_high_tumor_slices": mass,
        "idh_true": phase8_oof_true[pid],
        "idh_pred": phase8_oof_pred[pid],
        "idh_prob": phase8_oof_prob[pid],
    })

correlations = np.array(correlations)
top3_vs_all_ratio = np.array(top3_vs_all_ratio)
attn_mass_on_high_tumor = np.array(attn_mass_on_high_tumor)

print("\n" + "=" * 70)
print("Phase 8: Quantitative attention-tumour overlap")
print("=" * 70)
print(f"Patients analyzed: {len(correlations)}")
print(f"Mean per-patient attention~tumour-fraction correlation: "
      f"{correlations.mean():.3f} (median {np.median(correlations):.3f})")
print(f"Fraction of patients with positive correlation: {(correlations > 0).mean():.3f}")
print(f"Mean tumour-fraction ratio, top-3 attended slices vs. all 15 slices: "
      f"{top3_vs_all_ratio.mean():.3f}x  (>1.0 means attention concentrates on more tumour-rich slices)")
print(f"Mean attention mass placed on above-median-tumour slices: "
      f"{attn_mass_on_high_tumor.mean():.3f}  (chance level = 0.500)")

results_df = pd.DataFrame(per_patient_records)
results_df.to_csv("/kaggle/working/phase8_attention_tumor_overlap.csv", index=False)
print("\nSaved: /kaggle/working/phase8_attention_tumor_overlap.csv")


# ------------------------------------------------------------
# Step 3: Qualitative -- attention-weighted slice grids for
# correctly and incorrectly classified patients.
# ------------------------------------------------------------
def plot_patient_attention(pid, save_suffix):
    bag = np.load(f"/kaggle/working/slices/{pid}.npz")["slices"]  # (15, 4, 224, 224)
    attn = phase8_oof_attn[pid]
    row = pid_to_row.loc[pid]
    fracs, _ = slice_tumor_fractions(row["Mask_Path"])

    n = bag.shape[0]
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4))
    axes = np.array(axes).reshape(-1)
    for i in range(n):
        ax = axes[i]
        flair_img = bag[i, 3]  # FLAIR channel
        ax.imshow(flair_img, cmap="gray")
        frac_txt = f"{fracs[i]:.2f}" if fracs is not None else "n/a"
        ax.set_title(f"w={attn[i]:.2f}\ntumour={frac_txt}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, len(axes)):
        axes[j].axis("off")

    true_lbl = phase8_oof_true[pid]
    pred_lbl = phase8_oof_pred[pid]
    prob = phase8_oof_prob[pid]
    correct = "CORRECT" if true_lbl == pred_lbl else "INCORRECT"
    fig.suptitle(
        f"Patient {pid} -- IDH true={true_lbl} pred={pred_lbl} (p={prob:.2f}) [{correct}]",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = f"/kaggle/working/phase8_attention_{save_suffix}_{pid}.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_png}")


correct_df = results_df[results_df["idh_true"] == results_df["idh_pred"]].sort_values(
    "idh_prob", ascending=False
)
incorrect_df = results_df[results_df["idh_true"] != results_df["idh_pred"]]

example_correct = correct_df.head(2)["PatientID"].tolist()
example_incorrect = incorrect_df.head(2)["PatientID"].tolist()

print(f"\nPlotting attention maps for correct examples: {example_correct}")
for pid in example_correct:
    plot_patient_attention(pid, "correct")

print(f"\nPlotting attention maps for incorrect examples: {example_incorrect}")
for pid in example_incorrect:
    plot_patient_attention(pid, "incorrect")

print("\n" + "=" * 70)
print("Phase 8 complete.")
print("=" * 70)
