"""Audit UCSF-PDGM external cohort metadata without training or changing labels.
Usage: python 01_audit_ucsf_metadata.py /path/to/UCSF-PDGM-metadata_v5.csv
"""
import argparse, json, pathlib
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('csv'); p.add_argument('--out',default='ucsf_metadata_audit.json'); args=p.parse_args()
df=pd.read_csv(args.csv,low_memory=False)
print('Rows:',len(df)); print('Columns:',list(df.columns))
for keyword in ('id','age','sex','idh','grade','braTS','mri'):
    matches=[c for c in df if keyword.lower() in c.lower()]
    print('\n',keyword,'candidate columns:',matches)
    for c in matches[:5]:
        if df[c].nunique(dropna=True)<30: print(' ',c,df[c].fillna('<MISSING>').astype(str).value_counts().head(30).to_dict())
report={'rows':len(df),'columns':list(df.columns),'missing':{str(c):int(v) for c,v in df.isna().sum().items()},'candidate_columns':{k:[str(c) for c in df if k in c.lower()] for k in ('id','age','sex','idh','grade','brats')}}
pathlib.Path(args.out).write_text(json.dumps(report,indent=2)); print('\nSaved',args.out)
