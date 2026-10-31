"""Recompute Holm adjustment from raw comparison CSVs (not from reported prose).
Usage: python 01_audit_holm.py revA_delong.csv --out holm_audit.csv
"""
import argparse
import pandas as pd
from statsmodels.stats.multitest import multipletests
p=argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--out",default="holm_audit.csv")
p.add_argument("--family",choices=["global","by-task"],default="global")
args=p.parse_args()
df=pd.read_csv(args.csv)
if "p" not in df: raise ValueError("Raw p column required")
if df.p.isna().any() or (df.p<0).any() or (df.p>1).any(): raise ValueError("Invalid raw p-values")
if {"a","b"}.issubset(df.columns):
 pairs=df.apply(lambda r:tuple(sorted((str(r.a),str(r.b)))),axis=1)
 if pairs.duplicated().any():raise ValueError("Duplicated pair")
if args.family=="by-task":
 if "task" not in df:raise ValueError("task column required")
 df["holm_adj"]=df.groupby("task")["p"].transform(lambda x:multipletests(x,method="holm")[1])
 df["family_size"]=df.groupby("task")["p"].transform("size")
else:
 df["holm_adj"]=multipletests(df["p"],method="holm")[1]
 df["family_size"]=len(df)
df.to_csv(args.out,index=False)
print("Pairs:",len(df),"Family sizes:",df.family_size.unique().tolist())
print(df.sort_values("p").head(20).to_string(index=False))
print("Saved:",args.out)
