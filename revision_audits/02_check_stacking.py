"""Diagnostic of base OOF prediction coverage; NOT proof of leakage-free stacking."""
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
p=pd.read_csv(sys.argv[1])
required={"PatientID","condition","task","true_label","prob_1"}
if not required.issubset(p):raise ValueError("Missing "+str(required-set(p)))
p=p[p.task=="idh"]
for name,g in p.groupby("condition"):
 if g.PatientID.duplicated().any():raise ValueError("Duplicate PatientID in "+name)
 if g.prob_1.isna().any():raise ValueError("Missing probabilities in "+name)
 print(name,"N",len(g),"AUC",roc_auc_score(g.true_label,g.prob_1))
if p.groupby("PatientID").true_label.nunique().max()!=1:raise ValueError("Inconsistent IDH labels")
print("WARNING: Base OOF predictions alone do not establish leakage-free stacking.")
print("If meta-level fit/test used the same outcomes, coefficients, likelihood-ratio p-values, and AUC are in-sample.")
print("Confirmatory stacking requires cross-fitted or nested meta-level evaluation; re-train base models within outer folds.")
