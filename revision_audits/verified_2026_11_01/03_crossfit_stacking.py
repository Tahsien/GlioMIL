"""Exploratory fivefold meta-level stacking diagnostic on EXISTING base-model OOF predictions.
This is NOT nested validation: base models have overlapping training datasets.

Usage: python 03_crossfit_stacking.py phase5_oof_predictions.csv phase5_fold_splits.json
Creates crossfit_stacking_oof.csv and crossfit_stacking_summary.json.
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

p=argparse.ArgumentParser()
p.add_argument("oof")
p.add_argument("splits")
p.add_argument("--seed",type=int,default=42)
a=p.parse_args()
d=pd.read_csv(a.oof).query("task == 'idh'")
splits=json.load(open(a.splits))
print("Conditions:", sorted(d.condition.unique().tolist()))
# Require explicit condition identification instead of silently guessing a different baseline.
def find(part):
  hits=[x for x in d.condition.unique() if part.lower() in x.lower()]
  if len(hits)!=1: raise ValueError(f"Ambiguous condition {part}: {hits}")
  return hits[0]
st=find("(v)"); im=find("(iii)")
one=d[d.condition==st][["PatientID","true_label","prob_1"]].rename(columns={"prob_1":"structured"})
two=d[d.condition==im][["PatientID","true_label","prob_1"]].rename(columns={"prob_1":"image","true_label":"image_label"})
m=one.merge(two,on="PatientID",validate="one_to_one")
if len(m)!=619 or m.isna().any().any():raise ValueError("Expected 619 complete patient-paired predictions")
if not (m.true_label==m.image_label).all():raise ValueError("Label mismatch")
m=m.drop(columns="image_label")
eps=1e-6; Xs=logit(m[["structured"]].clip(eps,1-eps).values); Xi=logit(m[["structured","image"]].clip(eps,1-eps).values)
y=m.true_label.astype(int).values
s=np.full(len(m),np.nan);z=np.full(len(m),np.nan); seen=set()
for i,f in enumerate(splits):
  test=set(map(str,f["test"])); tr=set(map(str,f["train_val"]))
  if seen&test:raise ValueError("Duplicate test patient in split file")
  seen|=test
  it=m.PatientID.astype(str).isin(test).values
  itr=m.PatientID.astype(str).isin(tr).values
  if not it.any() or not itr.any() or it.sum()+itr.sum()!=len(m):raise ValueError("Incompatible split")
  for features,out in [(Xs,s),(Xi,z)]:
    model=LogisticRegression(penalty=None,max_iter=2000).fit(features[itr],y[itr])
    out[it]=model.predict_proba(features[it])[:,1]
if np.isnan(s).any() or np.isnan(z).any():raise ValueError("Incomplete external meta predictions")
results={"n":len(y),"original_structured_auc":float(roc_auc_score(y,m.structured)),"crossfit_recalibrated_structured_auc":float(roc_auc_score(y,s)),"crossfit_stacked_auc":float(roc_auc_score(y,z))}
results["delta_crossfit"]=results["crossfit_stacked_auc"]-results["crossfit_recalibrated_structured_auc"]
rng=np.random.default_rng(a.seed); boot=[]
for _ in range(2000):
  idx=rng.integers(0,len(y),len(y))
  if len(set(y[idx]))<2:continue
  boot.append(roc_auc_score(y[idx],z[idx])-roc_auc_score(y[idx],s[idx]))
results["delta_bootstrap_ci95"]=list(map(float,np.percentile(boot,[2.5,97.5])))
m["meta_structured"]=s;m["meta_stacked"]=z
m.to_csv("crossfit_stacking_oof.csv",index=False)
with open("crossfit_stacking_summary.json","w") as f:json.dump(results,f,indent=2)
print(json.dumps(results,indent=2))
print("WARNING: exploratory only, not fully nested or leakage-isolated.")
