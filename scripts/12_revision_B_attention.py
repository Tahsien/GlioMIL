"""
GlioMIL revision -- Part B: attention analysis with per-patient permutation nulls (CPU).
Run in the notebook session AFTER Phase 8. Reads /kaggle/working/phase8_attn_fracs.pkl, written by the
Phase 8 + pickle cell: {"attn": {PatientID: 15 attention weights}, "fracs": {PatientID: 15 slice tumour fractions}}.
Writes /kaggle/working/revB_attention_per_patient.csv.
"""
import numpy as np, pandas as pd, pickle
from scipy import stats
D = pickle.load(open("/kaggle/working/phase8_attn_fracs.pkl","rb"))
ATTN, FR = D["attn"], D["fracs"]
pids = [p for p in ATTN if FR.get(p) is not None]
A = {p: np.asarray(ATTN[p], float) for p in pids}
F = {p: np.asarray(FR[p], float) for p in pids}
rng = np.random.default_rng(42); NPERM = 1000

def corr_rows(a, M):                      # correlation of a with each row of M
    am = a - a.mean(); Mm = M - M.mean(1, keepdims=True)
    den = np.sqrt((am**2).sum() * (Mm**2).sum(1))
    return np.where(den > 0, (Mm @ am) / np.where(den > 0, den, 1), 0.0)

def stat_all(a, f_rows):                  # f_rows: (n_rows, 15)
    r = corr_rows(a, f_rows)
    order = np.argsort(-a)[:3]
    top3 = f_rows[:, order].mean(1) / (f_rows.mean(1) + 1e-12)
    med = np.median(f_rows, axis=1, keepdims=True)
    mass = ((f_rows > med) * a).sum(1)    # strictly above median; ties are not counted
    return r, top3, mass

r_obs, t3_obs, m_obs, pos_r, nr, n3, nm = [], [], [], [], [], [], []
for p in pids:
    a, f = A[p], F[p]
    r, t3, m = stat_all(a, f[None, :]); r_obs.append(r[0]); t3_obs.append(t3[0]); m_obs.append(m[0])
    pos_r.append(np.corrcoef(a, np.arange(len(a)))[0, 1] if a.std() > 0 else 0.0)
    P = np.stack([rng.permutation(f) for _ in range(NPERM)])
    r_, t_, m_ = stat_all(a, P); nr.append(r_.mean()); n3.append(t_.mean()); nm.append(m_.mean())
r_obs, t3_obs, m_obs, pos_r, nr, n3, nm = map(np.array, (r_obs, t3_obs, m_obs, pos_r, nr, n3, nm))

out = {}
for name, obs, nul in [("Pearson r", r_obs, nr), ("mass above median", m_obs, nm), ("top-3 tumour ratio", t3_obs, n3)]:
    d = obs - nul
    out[name] = dict(observed_mean=obs.mean(), perm_null_mean=nul.mean(), mean_diff=d.mean(),
                     t_p=stats.ttest_1samp(d, 0).pvalue, wilcoxon_p=stats.wilcoxon(d).pvalue)
print(pd.DataFrame(out).T.round(4).to_string())

sd_null = []
for _ in range(200):
    rr = []
    for p in pids[:150]:
        f = rng.permutation(F[p]); rr.append(np.corrcoef(A[p], f)[0, 1] if f.std() > 0 else 0)
    sd_null.append(np.std(rr))
print(f"SD of per-patient r: observed {r_obs.std():.3f}; under shuffled slice tumour fractions {np.mean(sd_null):.3f}")
print(f"share of patients with |r|>0.5: observed {(abs(r_obs)>0.5).mean():.3f}")
print(f"attention vs slice position: mean r={pos_r.mean():.3f}, SD={pos_r.std():.3f}; patients with r>0: {(pos_r>0).mean():.3f}; one-sample t p={stats.ttest_1samp(pos_r,0).pvalue:.3g}")
sdv = np.array([F[p].std() for p in pids]); mv = np.array([F[p].mean() for p in pids])
print(f"tumour fraction across bag slices: mean-of-patient-means={mv.mean():.3f}, median within-patient SD={np.median(sdv):.3f}, share of slices with zero tumour={np.mean([(F[p]==0).mean() for p in pids]):.3f}")
pd.DataFrame({"PatientID": pids, "r": r_obs, "mass_above_median": m_obs, "top3_ratio": t3_obs,
              "r_vs_slice_position": pos_r, "null_r": nr, "null_mass": nm}).to_csv("/kaggle/working/revB_attention_per_patient.csv", index=False)
print("saved revB_attention_per_patient.csv")
