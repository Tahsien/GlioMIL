"""Evaluate truly external, already generated patient-level predictions.
Input CSV MUST have PatientID, true_idh (0 wildtype, 1 mutant), prob_mutant.
Predictions must come from UTSW-only-fitted, frozen preprocessing and model.
Usage: python 02_evaluate_frozen_external_predictions.py ucsf_predictions.csv
"""
import argparse, json, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix
from scipy.special import logit
p=argparse.ArgumentParser(); p.add_argument('predictions');p.add_argument('--out',default='ucsf_external_metrics.json');args=p.parse_args()
df=pd.read_csv(args.predictions); required=['PatientID','true_idh','prob_mutant'];missing=[x for x in required if x not in df]
if missing: raise ValueError('Missing columns: '+str(missing))
if df.PatientID.duplicated().any(): raise ValueError('Duplicate patient IDs; reduce to one pre-specified MRI per patient before evaluating.')
if df[required].isna().any().any(): raise ValueError('Missing identifiers/labels/predictions: fix and document exclusions')
y=df.true_idh.to_numpy();q=df.prob_mutant.to_numpy(dtype=float)
if set(np.unique(y))!={0,1}: raise ValueError('Need both 0 and 1 IDH labels. Confirm label orientation, do not infer it.')
if not np.all((0<=q)&(q<=1)): raise ValueError('Probabilities outside 0..1')
pred=(q>=0.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
rng=np.random.default_rng(42); samples=[]
for _ in range(2000):
  ix=rng.integers(0,len(y),len(y))
  if len(np.unique(y[ix]))==2: samples.append(roc_auc_score(y[ix],q[ix]))
from sklearn.linear_model import LogisticRegression
slope=None;intercept=None
try:
  model=LogisticRegression(penalty=None,max_iter=1000).fit(logit(np.clip(q,1e-6,1-1e-6)).reshape(-1,1),y)
  slope=float(model.coef_[0,0]);intercept=float(model.intercept_[0])
except Exception: pass
metrics=dict(n=int(len(y)),n_mutant=int(y.sum()),prevalence=float(y.mean()),auc=float(roc_auc_score(y,q)),auc_ci95=[float(x) for x in np.percentile(samples,[2.5,97.5])],accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),brier=float(brier_score_loss(y,q)),mean_predicted_mutation=float(q.mean()),calibration_slope=slope,calibration_intercept=intercept,decision_threshold=0.5)
pathlib.Path(args.out).write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
