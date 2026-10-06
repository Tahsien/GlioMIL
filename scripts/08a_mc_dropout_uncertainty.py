import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

MC_PASSES = 20  # stochastic forward passes per patient for MC dropout

phase7_oof_probs = {t: {} for t in TASKS}   # pid -> mean prob array across MC passes
phase7_oof_std = {t: {} for t in TASKS}     # pid -> std across MC passes (uncertainty)
phase7_oof_true = {t: {} for t in TASKS}

for fold_idx, fold in enumerate(folds):
    print(f"\n--- Fold {fold_idx+1}/{N_FOLDS} ---")
    train_val_df = master_df[master_df["PatientID"].isin(fold["train_val"])].reset_index(drop=True)
    test_df = master_df[master_df["PatientID"].isin(fold["test"])].reset_index(drop=True)
    train_df, val_df = train_test_split(train_val_df, test_size=0.15,
                                         stratify=train_val_df["idh_label"], random_state=fold_idx)

    class_weights = compute_class_weights(train_df, TASKS)
    train_loader = DataLoader(GliomaFeatureDataset(train_df, clinical_cols, TASKS),
                               batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(GliomaFeatureDataset(val_df, clinical_cols, TASKS),
                             batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(GliomaFeatureDataset(test_df, clinical_cols, TASKS),
                              batch_size=BATCH_SIZE, shuffle=False)

    model = MILModel(EMBED_DIM, TASKS, clinical_dim=len(clinical_cols)).to(device)
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
    print(f"  best val_loss: {best_val_loss:.4f}")

    # MC Dropout inference: keep dropout active (model.train()) but no grad, no weight updates
    model.train()
    pid_order = []
    accum_probs = {t: [] for t in TASKS}
    true_labels = {t: [] for t in TASKS}

    with torch.no_grad():
        for batch in test_loader:
            emb = batch["embeddings"].to(device)
            clin = batch["clinical"].to(device)
            pids = batch["PatientID"]
            pid_order.extend(pids)
            for t in TASKS:
                true_labels[t].extend(batch[f"{t}_label"].numpy())

            pass_probs = {t: [] for t in TASKS}
            for _ in range(MC_PASSES):
                logits, _ = model(emb, clin)
                for t in TASKS:
                    probs = F.softmax(logits[t], dim=-1).cpu().numpy()
                    pass_probs[t].append(probs)
            for t in TASKS:
                stacked = np.stack(pass_probs[t], axis=0)  # (MC_PASSES, batch, n_classes)
                accum_probs[t].append(stacked)

    for t in TASKS:
        all_stacked = np.concatenate(accum_probs[t], axis=1)  # (MC_PASSES, n_test, n_classes)
        mean_probs = all_stacked.mean(axis=0)
        std_probs = all_stacked.std(axis=0)
        for i, pid in enumerate(pid_order):
            row = test_df[test_df["PatientID"] == pid].iloc[0]
            if not bool(row[f"{t}_valid"]):
                continue
            phase7_oof_probs[t][pid] = mean_probs[i]
            phase7_oof_std[t][pid] = std_probs[i]
            phase7_oof_true[t][pid] = int(row[f"{t}_label"])

print("\nPhase 7 MC-Dropout inference complete.")