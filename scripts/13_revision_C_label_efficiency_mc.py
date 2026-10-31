"""
GlioMIL revision -- Part C: label-efficiency (repeated subsamples, per-fold normalization, structured-only
logistic comparator) and MC-dropout reliability, re-run for the corrected pipeline.
Run in the notebook session AFTER the Phase 4 cell (needs master_df, folds, TASKS, clinical_cols, MILModel,
GliomaFeatureDataset, compute_class_weights, summarize_task, device, BATCH_SIZE, NUM_EPOCHS, PATIENCE, LR,
WEIGHT_DECAY, EMBED_DIM, N_FOLDS). GPU. Writes /kaggle/working/revC_*.csv. REPS=3 -> roughly 3x the Phase 6 time.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

FRACS = [0.10, 0.25, 0.50, 1.00]; REPS = 3; MC_PASSES = 20; NBINS = 10

def loss_fn(logits, batch, tasks, cw):
    total = None
    for t in tasks:
        valid = batch[f"{t}_valid"]
        if valid.sum() == 0: continue
        l = nn.functional.cross_entropy(logits[t][valid], batch[f"{t}_label"][valid], weight=cw.get(t))
        total = l if total is None else total + l
    if total is None:
        total = logits[next(iter(logits))].sum() * 0.0
    return total

def to_dev(batch):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

def train_fusion(train_df, val_df, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cw = compute_class_weights(train_df, TASKS)
    tl = DataLoader(GliomaFeatureDataset(train_df, clinical_cols, TASKS), batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(GliomaFeatureDataset(val_df, clinical_cols, TASKS), batch_size=BATCH_SIZE, shuffle=False)
    model = MILModel(EMBED_DIM, TASKS, clinical_dim=len(clinical_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, state, bad = float("inf"), None, 0
    for _ in range(NUM_EPOCHS):
        model.train()
        for batch in tl:
            b = to_dev(batch); opt.zero_grad()
            logits, _ = model(b["embeddings"], b["clinical"])
            loss_fn(logits, b, TASKS, cw).backward(); opt.step()
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for batch in vl:
                b = to_dev(batch); logits, _ = model(b["embeddings"], b["clinical"])
                vloss += loss_fn(logits, b, TASKS, cw).item()
        vloss /= len(vl)
        if vloss < best - 1e-4: best, state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(state); return model

def standardize(train_df, *others):
    mu = train_df[clinical_cols].mean(); sd = train_df[clinical_cols].std() + 1e-8
    out = []
    for d in (train_df,) + others:
        d = d.copy(); d[clinical_cols] = (d[clinical_cols] - mu) / sd; out.append(d)
    return out

rows, lr_rows, mc = [], [], []
for frac in FRACS:
    for rep in range(REPS):
        print(f"frac={frac:.2f} rep={rep+1}/{REPS}")
        oof_logits = {t: {} for t in TASKS}; oof_labels = {}; oof_valid = {t: {} for t in TASKS}
        lr_prob = {}; lr_true = {}
        for fi, fold in enumerate(folds):
            tv = master_df[master_df["PatientID"].isin(fold["train_val"])].reset_index(drop=True)
            te = master_df[master_df["PatientID"].isin(fold["test"])].reset_index(drop=True)
            tr, va = train_test_split(tv, test_size=0.15, stratify=tv["idh_label"], random_state=fi)
            if frac < 1.0:
                tr, _ = train_test_split(tr, train_size=frac, stratify=tr["idh_label"], random_state=1000 * (rep + 1) + fi)
            tr, va, te = standardize(tr.reset_index(drop=True), va.reset_index(drop=True), te)
            model = train_fusion(tr, va, seed=10000 * rep + 100 * fi + int(frac * 100))
            tl = DataLoader(GliomaFeatureDataset(te, clinical_cols, TASKS), batch_size=BATCH_SIZE, shuffle=False)
            model.eval()
            with torch.no_grad():
                for batch in tl:
                    b = to_dev(batch); logits, _ = model(b["embeddings"], b["clinical"])
                    for i, pid in enumerate(batch["PatientID"]):
                        oof_labels[pid] = {t: batch[f"{t}_label"][i].item() for t in TASKS}
                        for t in TASKS:
                            oof_logits[t][pid] = logits[t][i].cpu().numpy(); oof_valid[t][pid] = batch[f"{t}_valid"][i].item()
            # structured-only L2 logistic regression on the same training subset (penalty chosen inside the training subset)
            Xtr, ytr = tr[clinical_cols].values, tr["idh_label"].values
            cv = max(2, min(5, int(np.bincount(ytr).min())))
            lr = LogisticRegressionCV(Cs=6, cv=cv, penalty="l2", scoring="roc_auc", max_iter=5000, class_weight="balanced").fit(Xtr, ytr)
            for pid, p in zip(te["PatientID"], lr.predict_proba(te[clinical_cols].values)[:, 1]):
                lr_prob[pid] = p; lr_true[pid] = int(te.loc[te["PatientID"] == pid, "idh_label"].iloc[0])
            # MC dropout (fusion, full training data only)
            if frac == 1.0:
                model.train(); pid_order, passes = [], []
                with torch.no_grad():
                    for batch in tl:
                        b = to_dev(batch); pid_order.extend(batch["PatientID"])
                        passes.append(np.stack([F.softmax(model(b["embeddings"], b["clinical"])[0]["idh"], -1).cpu().numpy()
                                                for _ in range(MC_PASSES)]))
                P = np.concatenate(passes, axis=1)  # (passes, n, 2)
                for i, pid in enumerate(pid_order):
                    mc.append(dict(rep=rep, PatientID=pid, p_mean=P[:, i, 1].mean(), p_std=P[:, i, 1].std(),
                                   y=int(te.loc[te["PatientID"] == pid, "idh_label"].iloc[0])))
        all_pids = master_df["PatientID"].tolist()
        for t in TASKS:
            r = summarize_task(t, TASKS[t]["num_classes"], oof_labels, oof_logits, oof_valid, all_pids)
            if r is not None: rows.append(dict(frac=frac, rep=rep, model="fusion", task=t, auc=r["auc"]))
        pids = list(lr_prob)
        lr_rows.append(dict(frac=frac, rep=rep, model="structured L2 logistic", task="idh",
                            auc=roc_auc_score([lr_true[p] for p in pids], [lr_prob[p] for p in pids])))

res = pd.DataFrame(rows + lr_rows); res.to_csv("/kaggle/working/revC_label_efficiency.csv", index=False)
summ = res.groupby(["model", "task", "frac"]).auc.agg(["mean", "std", "min", "max"]).round(3)
summ.to_csv("/kaggle/working/revC_label_efficiency_summary.csv")
print(summ.to_string())

# ---- MC-dropout reliability (IDH, fusion, 100% of training data; pooled over repeats) ----
m = pd.DataFrame(mc); m.to_csv("/kaggle/working/revC_mc_dropout_idh.csv", index=False)
out = []
for rep, g in m.groupby("rep"):
    pred = (g.p_mean >= 0.5).astype(int); conf = np.where(pred == 1, g.p_mean, 1 - g.p_mean); correct = (pred == g.y).values
    bins = np.minimum((conf * NBINS).astype(int), NBINS - 1); ece = 0.0
    for b in range(NBINS):
        s = bins == b
        if s.any(): ece += s.mean() * abs(correct[s].mean() - conf[s].mean())
    hi = conf >= np.median(conf)
    out.append(dict(rep=rep, n=len(g), accuracy=correct.mean(), majority=max(g.y.mean(), 1 - g.y.mean()), ECE=ece,
                    acc_confident_half=correct[hi].mean(), acc_uncertain_half=correct[~hi].mean(),
                    auc_mc_mean=roc_auc_score(g.y, g.p_mean)))
print(pd.DataFrame(out).round(3).to_string()); print(pd.DataFrame(out).drop(columns="rep").mean().round(3).to_string())
