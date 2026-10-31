"""
GlioMIL revision -- Part A: structured-feature decomposition (CPU only, ~5-10 min).
Run in the SAME Kaggle session after the notebook's Step 2 (so /kaggle/working/ cache files exist).
Reads : master_dataset_fets_with_slices.csv, clinical_features_cache.csv, radiomics_features.csv,
        phase5_fold_splits.json, phase5_oof_predictions.csv (optional, for DeLong vs fusion/structured-only),
        UTSW_Glioma_Metadata-2-1.tsv (labels), /kaggle/working/features/{PatientID}.npy (optional, image+age model)
Writes: /kaggle/working/revA_oof.csv, revA_summary.csv, revA_delong.csv
Normalisation statistics are fit on each fold's training patients only (no leakage).
"""
import os, json, itertools, numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
W="/kaggle/working/"
META="/kaggle/input/datasets/tano1974/glioma/UTSW_Glioma_Metadata-2-1.tsv"
rng=np.random.default_rng(42)

df=pd.read_csv(W+"master_dataset_fets_with_slices.csv")
df=df[df["has_slices"]==True].reset_index(drop=True)
df=df.merge(pd.read_csv(W+"clinical_features_cache.csv"),on="PatientID",how="left")
rad=pd.read_csv(W+"radiomics_features.csv"); rad_cols=[c for c in rad.columns if c!="PatientID"]
df=df.merge(rad,on="PatientID",how="left")
df["SexEncoded"]=df["Sex"].map({"M":0,"F":1})
folds=json.load(open(W+"phase5_fold_splits.json"))
y_all=df["IDH_Label"].astype(int).values
clin5=["Age","SexEncoded","TumourVolumeVoxels","T1CEMeanIntensity","FLAIRMeanIntensity"]
SETS={
 "age only":["Age"],
 "age+sex":["Age","SexEncoded"],
 "age+sex+volume+mean intensities (5 clinical)":clin5,
 "204 radiomics only":rad_cols,
 "all 209 structured (L2 logistic)":clin5+rad_cols,
}
def fit_predict(cols,tr,te,C=None):
    Xtr=df.loc[tr,cols].astype(float); Xte=df.loc[te,cols].astype(float)
    mu,sd=Xtr.mean(),Xtr.std()+1e-8
    Xtr=((Xtr-mu)/sd).fillna(0).values; Xte=((Xte-mu)/sd).fillna(0).values
    if len(cols)<=5:
        m=LogisticRegression(penalty=None,max_iter=2000,class_weight="balanced")
    else:
        m=LogisticRegressionCV(Cs=10,cv=5,penalty="l2",scoring="roc_auc",max_iter=5000,class_weight="balanced")
    m.fit(Xtr,y_all[tr]); return m.predict_proba(Xte)[:,1]
oof={}
for name,cols in SETS.items():
    p=np.full(len(df),np.nan)
    for f in folds:
        te=df.index[df.PatientID.isin(f["test"])].values; tr=df.index[df.PatientID.isin(f["train_val"])].values
        p[te]=fit_predict(cols,tr,te)
    oof[name]=p; print(name,"AUC",round(roc_auc_score(y_all,p),3))
# optional: mean-pooled frozen embedding (+ age) -- logistic stand-in for "image-only + age"
if os.path.exists(W+"features/"+df.PatientID[0]+".npy"):
    E=np.stack([np.load(W+f"features/{p}.npy").mean(0) for p in df.PatientID])
    for name,use_age in [("frozen embedding (mean-pooled), L2 logistic",False),("frozen embedding (mean-pooled) + age",True)]:
        p=np.full(len(df),np.nan)
        for f in folds:
            te=df.index[df.PatientID.isin(f["test"])].values; tr=df.index[df.PatientID.isin(f["train_val"])].values
            X=E if not use_age else np.c_[E,df["Age"].values]
            mu,sd=X[tr].mean(0),X[tr].std(0)+1e-8
            m=LogisticRegressionCV(Cs=6,cv=5,penalty="l2",scoring="roc_auc",max_iter=3000,class_weight="balanced").fit((X[tr]-mu)/sd,y_all[tr])
            p[te]=m.predict_proba((X[te]-mu)/sd)[:,1]
        oof[name]=p; print(name,"AUC",round(roc_auc_score(y_all,p),3))
pd.DataFrame({"PatientID":df.PatientID,"IDH":y_all,**oof}).to_csv(W+"revA_oof.csv",index=False)
# DeLong
def _mr(x):
    J=np.argsort(x);Z=x[J];N=len(x);T=np.zeros(N);i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1;i=j
    R=np.empty(N);R[J]=T;return R
def delong(y,a,b):
    pos=y==1;m=pos.sum();n=(~pos).sum();S=np.vstack([a,b])
    tx=np.empty((2,m));ty=np.empty((2,n));tz=np.empty((2,m+n))
    for r in range(2):
        px,nx=S[r][pos],S[r][~pos];tx[r]=_mr(px);ty[r]=_mr(nx);tz[r]=_mr(np.r_[px,nx])
    au=tz[:,:m].sum(1)/m/n-(m+1.)/2./n
    v01=(tz[:,:m]-tx)/n;v10=1.-(tz[:,m:]-ty)/m
    cov=np.cov(v01)/m+np.cov(v10)/n;d=au[0]-au[1]
    return d,2*stats.norm.sf(abs(d/np.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1])))
ref={}
if os.path.exists(W+"phase5_oof_predictions.csv"):
    o=pd.read_csv(W+"phase5_oof_predictions.csv").query("task=='idh'").set_index("PatientID")
    for k,lab in [("(iv) 2.5D MIL multi-task + fusion","fusion (iv)"),("(v) Radiomics+clinical only","structured-only MLP (v)")]:
        s=o[o.condition==k].prob_1.reindex(df.PatientID).values; ref[lab]=s; oof[lab]=s
rows=[];drows=[]
for name,p in oof.items():
    bs=[roc_auc_score(y_all[ix],p[ix]) for ix in (rng.integers(0,len(y_all),len(y_all)) for _ in range(2000))]
    rows.append(dict(model=name,auc=roc_auc_score(y_all,p),ci_low=np.percentile(bs,2.5),ci_high=np.percentile(bs,97.5)))
for a,b in itertools.combinations(oof.keys(),2):
    d,pv=delong(y_all,oof[a],oof[b]);drows.append(dict(a=a,b=b,delta_auc=d,p=pv))
pd.DataFrame(rows).to_csv(W+"revA_summary.csv",index=False);pd.DataFrame(drows).to_csv(W+"revA_delong.csv",index=False)
print(pd.DataFrame(rows).round(3).to_string())
