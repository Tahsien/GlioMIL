import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print("=" * 70)
print("Results Consolidation -- combined summary figure + master table")
print("=" * 70)

WORK = "/kaggle/working"


def try_read_csv(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\nLoaded {path}  (shape={df.shape})")
        print("Columns:", df.columns.tolist())
        return df
    print(f"\nMISSING: {path}")
    return None


phase5_df = try_read_csv(f"{WORK}/phase5_ablation_results.csv")
phase6_df = try_read_csv(f"{WORK}/phase6_label_efficiency_results.csv")
phase7_summary_df = try_read_csv(f"{WORK}/phase7_summary.csv")
phase7_decile_df = try_read_csv(f"{WORK}/phase7_confidence_deciles.csv")
phase8_df = try_read_csv(f"{WORK}/phase8_attention_tumor_overlap.csv")

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# ------------------------------------------------------------
# Panel 1: Phase 5 ablation -- IDH AUC by condition
# ------------------------------------------------------------
ax = axes[0, 0]
try:
    df = phase5_df.copy()
    idh_col = None
    cond_col = None
    for c in df.columns:
        lc = c.lower()
        if "condition" in lc or lc == "task" and False:
            pass
    cond_candidates = [c for c in df.columns if "condition" in c.lower() or c.lower() == "model"]
    cond_col = cond_candidates[0] if cond_candidates else df.columns[0]

    if "task" in [c.lower() for c in df.columns]:
        task_col = [c for c in df.columns if c.lower() == "task"][0]
        auc_col = [c for c in df.columns if "auc" in c.lower()][0]
        sub = df[df[task_col].astype(str).str.lower() == "idh"]
        labels = sub[cond_col].astype(str).tolist()
        values = sub[auc_col].astype(float).tolist()
    else:
        auc_cols = [c for c in df.columns if "idh" in c.lower() and "auc" in c.lower()]
        auc_col = auc_cols[0] if auc_cols else [c for c in df.columns if "auc" in c.lower()][0]
        labels = df[cond_col].astype(str).tolist()
        values = df[auc_col].astype(float).tolist()

    bars = ax.bar(range(len(labels)), values, color="#4C72B0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("IDH AUC")
    ax.set_title("Phase 5 -- Ablation: IDH AUC by condition")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
except Exception as e:
    ax.text(0.5, 0.5, f"Could not auto-plot Phase 5.\nCheck columns above.\n({e})",
            ha="center", va="center", wrap=True, fontsize=9)
    ax.set_title("Phase 5 -- Ablation (see console for columns)")

# ------------------------------------------------------------
# Panel 2: Phase 6 label-efficiency curve
# ------------------------------------------------------------
ax = axes[0, 1]
try:
    img_path = f"{WORK}/label_efficiency_curve.png"
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Phase 6 -- Label-Efficiency Curve")
    else:
        df = phase6_df.copy()
        frac_col = [c for c in df.columns if "frac" in c.lower()][0]
        auc_cols = [c for c in df.columns if "idh" in c.lower() and "auc" in c.lower()]
        auc_col = auc_cols[0] if auc_cols else [c for c in df.columns if "auc" in c.lower()][0]
        sub = df.sort_values(frac_col)
        ax.plot(sub[frac_col], sub[auc_col], marker="o", color="#DD8452")
        ax.set_xlabel("Fraction of labelled training data")
        ax.set_ylabel("IDH AUC")
        ax.set_title("Phase 6 -- Label-Efficiency Curve")
        ax.set_ylim(0, 1.0)
except Exception as e:
    ax.text(0.5, 0.5, f"Could not load Phase 6 panel.\n({e})", ha="center", va="center", wrap=True, fontsize=9)
    ax.set_title("Phase 6 -- Label-Efficiency (see console)")

# ------------------------------------------------------------
# Panel 3: Phase 7 reliability diagram (reuse saved image)
# ------------------------------------------------------------
ax = axes[1, 0]
try:
    img_path = f"{WORK}/idh_reliability_diagram.png"
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Phase 7 -- Calibration (MC-Dropout)")
except Exception as e:
    ax.text(0.5, 0.5, f"Could not load reliability diagram.\n({e})", ha="center", va="center", wrap=True, fontsize=9)
    ax.set_title("Phase 7 -- Calibration (see console)")

# ------------------------------------------------------------
# Panel 4: Phase 8 -- attention vs tumour-fraction summary
# ------------------------------------------------------------
ax = axes[1, 1]
try:
    df = phase8_df.copy()
    corr_col = [c for c in df.columns if "corr" in c.lower()][0]
    ax.hist(df[corr_col].dropna(), bins=30, color="#55A868", edgecolor="white")
    mean_corr = df[corr_col].mean()
    ax.axvline(mean_corr, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Per-patient attention~tumour-fraction correlation")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Phase 8 -- Attention/Tumour Overlap\n(mean r = {mean_corr:.3f}, n={len(df)})")
except Exception as e:
    ax.text(0.5, 0.5, f"Could not plot Phase 8 panel.\n({e})", ha="center", va="center", wrap=True, fontsize=9)
    ax.set_title("Phase 8 -- Explainability (see console)")

fig.suptitle("Results Overview -- Phases 5-8", fontsize=15, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{WORK}/results_overview_figure.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{WORK}/results_overview_figure.pdf", bbox_inches="tight")
plt.show()
print(f"\nSaved: {WORK}/results_overview_figure.png")
print(f"Saved: {WORK}/results_overview_figure.pdf")

# ------------------------------------------------------------
# Master results table (best-effort, robust to column-name drift)
# ------------------------------------------------------------
rows = []
try:
    rows.append({"Phase": "5 - Ablation", "Metric": "See phase5_ablation_results.csv for full table",
                 "Value": ""})
except Exception:
    pass

if phase6_df is not None:
    try:
        frac_col = [c for c in phase6_df.columns if "frac" in c.lower()][0]
        auc_cols = [c for c in phase6_df.columns if "idh" in c.lower() and "auc" in c.lower()]
        auc_col = auc_cols[0] if auc_cols else [c for c in phase6_df.columns if "auc" in c.lower()][0]
        for _, r in phase6_df.sort_values(frac_col).iterrows():
            rows.append({"Phase": "6 - Label efficiency",
                          "Metric": f"IDH AUC @ {r[frac_col]*100:.0f}% labels" if r[frac_col] <= 1
                          else f"IDH AUC @ {r[frac_col]:.0f}% labels",
                          "Value": round(float(r[auc_col]), 4)})
    except Exception as e:
        print(f"Could not summarize Phase 6 into master table: {e}")

if phase7_summary_df is not None:
    try:
        for _, r in phase7_summary_df.iterrows():
            rows.append({"Phase": "7 - Calibration", "Metric": str(r.to_dict()), "Value": ""})
    except Exception as e:
        print(f"Could not summarize Phase 7 into master table: {e}")

if phase8_df is not None:
    try:
        corr_col = [c for c in phase8_df.columns if "corr" in c.lower()][0]
        rows.append({"Phase": "8 - Explainability", "Metric": "Mean attention~tumour correlation",
                      "Value": round(float(phase8_df[corr_col].mean()), 4)})
        rows.append({"Phase": "8 - Explainability", "Metric": "Fraction of patients with positive correlation",
                      "Value": round(float((phase8_df[corr_col] > 0).mean()), 4)})
    except Exception as e:
        print(f"Could not summarize Phase 8 into master table: {e}")

master_df = pd.DataFrame(rows)
master_df.to_csv(f"{WORK}/master_results_table.csv", index=False)
print(f"\nSaved: {WORK}/master_results_table.csv")
print("\nMaster table preview:")
print(master_df.to_string(index=False))

print("\n" + "=" * 70)
print("Consolidation complete.")
print("=" * 70)
