import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 300})

def expected_calibration_error(confidences, correct, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_stats.append((lo, hi, np.nan, np.nan, 0))
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        bin_stats.append((lo, hi, bin_conf, bin_acc, mask.sum()))
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece, bin_stats


phase7_summary = []
print(f"\n{'='*70}\nPhase 7: Point-estimate sanity check (mean of MC passes)\n{'='*70}")
for t in TASKS:
    pids = [p for p in master_df["PatientID"] if p in phase7_oof_true[t]]
    y_true = np.array([phase7_oof_true[t][p] for p in pids])
    probs = np.stack([phase7_oof_probs[t][p] for p in pids])
    preds = probs.argmax(axis=1)
    acc = (preds == y_true).mean()
    print(f"  {t.upper():8s} n={len(pids):4d}  Acc={acc:.3f}")
    phase7_summary.append({"task": t, "n": len(pids), "accuracy": acc})

# --- Calibration + confidence-stratified accuracy, using IDH (binary, primary endpoint) ---
task = "idh"
pids = [p for p in master_df["PatientID"] if p in phase7_oof_true[task]]
y_true = np.array([phase7_oof_true[task][p] for p in pids])
probs = np.stack([phase7_oof_probs[task][p] for p in pids])
preds = probs.argmax(axis=1)
correct = (preds == y_true).astype(float)
confidence = probs.max(axis=1)  # predicted-class confidence

ece, bin_stats = expected_calibration_error(confidence, correct, n_bins=10)
print(f"\n{'='*70}\nIDH Calibration -- Expected Calibration Error (ECE) = {ece:.4f}\n{'='*70}")
for lo, hi, bc, ba, n in bin_stats:
    if n > 0:
        print(f"  [{lo:.1f}-{hi:.1f}]  n={n:4d}  mean_confidence={bc:.3f}  accuracy={ba:.3f}")

# Reliability diagram
fig, ax = plt.subplots(figsize=(4.5, 4.5))
bin_confs = [b[2] for b in bin_stats if b[4] > 0]
bin_accs = [b[3] for b in bin_stats if b[4] > 0]
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
ax.plot(bin_confs, bin_accs, marker="o", color="#1f6f8b", label=f"Model (ECE={ece:.3f})")
ax.set_xlabel("Mean predicted confidence")
ax.set_ylabel("Observed accuracy")
ax.set_title("IDH Reliability Diagram (MC-Dropout)", fontsize=9.5)
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("/kaggle/working/idh_reliability_diagram.png", dpi=300, bbox_inches="tight")
fig.savefig("/kaggle/working/idh_reliability_diagram.pdf", bbox_inches="tight")
plt.show()

# --- Confidence-stratified accuracy: does higher confidence = higher accuracy? ---
order = np.argsort(-confidence)  # most confident first
n = len(order)
deciles = np.array_split(order, 10)
print(f"\n{'='*70}\nIDH Accuracy by Confidence Decile (1=most confident, 10=least)\n{'='*70}")
decile_rows = []
for i, idx in enumerate(deciles):
    dec_acc = correct[idx].mean()
    dec_conf = confidence[idx].mean()
    print(f"  Decile {i+1:2d}  n={len(idx):3d}  mean_confidence={dec_conf:.3f}  accuracy={dec_acc:.3f}")
    decile_rows.append({"decile": i + 1, "n": len(idx), "mean_confidence": dec_conf, "accuracy": dec_acc})

top50 = order[:n // 2]
bottom50 = order[n // 2:]
print(f"\nTop 50% most confident  -- n={len(top50):3d}  accuracy={correct[top50].mean():.3f}")
print(f"Bottom 50% least confident -- n={len(bottom50):3d}  accuracy={correct[bottom50].mean():.3f}")
print(f"Overall accuracy -- n={n:3d}  accuracy={correct.mean():.3f}")

pd.DataFrame(decile_rows).to_csv("/kaggle/working/phase7_confidence_deciles.csv", index=False)
pd.DataFrame(phase7_summary).to_csv("/kaggle/working/phase7_summary.csv", index=False)
print("\nSaved: idh_reliability_diagram.png/.pdf, phase7_confidence_deciles.csv, phase7_summary.csv")