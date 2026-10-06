import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Self-contained safe loss (avoids the earlier stale-globals issue)
def masked_multitask_loss_safe(logits, batch, tasks, class_weights):
    total = None
    for t in tasks:
        valid = batch[f"{t}_valid"]
        if valid.sum() == 0:
            continue
        loss = nn.functional.cross_entropy(logits[t][valid], batch[f"{t}_label"][valid],
                                            weight=class_weights.get(t))
        total = loss if total is None else total + loss
    if total is None:
        any_task = next(iter(logits))
        total = logits[any_task].sum() * 0.0
    return total

LABEL_FRACTIONS = [0.10, 0.25, 0.50, 1.00]
label_efficiency_results = []

for frac in LABEL_FRACTIONS:
    condition_name = f"Phase6 label-fraction={frac:.2f}"
    print(f"\n{'='*70}\n{condition_name}\n{'='*70}")

    oof_logits = {t: {} for t in TASKS}
    oof_labels = {}
    oof_valid = {t: {} for t in TASKS}

    for fold_idx, fold in enumerate(folds):
        train_val_df = master_df[master_df["PatientID"].isin(fold["train_val"])].reset_index(drop=True)
        test_df = master_df[master_df["PatientID"].isin(fold["test"])].reset_index(drop=True)
        train_df, val_df = train_test_split(train_val_df, test_size=0.15,
                                             stratify=train_val_df["idh_label"], random_state=fold_idx)

        if frac < 1.0:
            train_df, _ = train_test_split(train_df, train_size=frac,
                                            stratify=train_df["idh_label"], random_state=fold_idx)

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
        print(f"  Fold {fold_idx+1} (n_train={len(train_df)}) best val_loss: {best_val_loss:.4f}")

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

    all_pids = master_df["PatientID"].tolist()
    for t in TASKS:
        r = summarize_task(t, TASKS[t]["num_classes"], oof_labels, oof_logits, oof_valid, all_pids)
        if r is None:
            continue
        r["label_fraction"] = frac
        r["condition"] = condition_name
        label_efficiency_results.append(r)
        print(f"  {t.upper():8s} AUC={r['auc']:.3f}  Acc={r['accuracy']:.3f}  BalAcc={r['balanced_accuracy']:.3f}")

print("\nPhase 6 complete.")