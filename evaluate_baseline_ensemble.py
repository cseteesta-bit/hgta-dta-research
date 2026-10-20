#!/usr/bin/env python3
"""Evaluate a mean-prediction ensemble from seed_*/test_predictions.csv files."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TRUTH=("y_true","true","label","affinity","y","Y")
PRED=("y_pred","pred","prediction","pred_mean","mean","mu")

def ci(y, p):
    # Exact concordance index in O(n log n), counting half-credit prediction ties.
    order=np.argsort(y, kind="mergesort"); y,p=y[order],p[order]
    ranks=np.searchsorted(np.unique(p),p)+1; size=int(ranks.max())+1; bit=np.zeros(size+1); seen=0; score=0.; pairs=0
    def add(i):
        while i<len(bit): bit[i]+=1; i+=i&-i
    def query(i):
        s=0.
        while i: s+=bit[i]; i-=i&-i
        return s
    start=0
    while start<len(y):
        end=start+1
        while end<len(y) and y[end]==y[start]: end+=1
        for i in range(start,end):
            lower=query(ranks[i]-1); equal=query(ranks[i])-lower; score+=lower+.5*equal; pairs+=seen
        for i in range(start,end): add(ranks[i])
        seen+=end-start; start=end
    return float(score/pairs) if pairs else float("nan")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--results-dir",required=True,type=Path); ap.add_argument("--out-dir",required=True,type=Path); ap.add_argument("--seeds",nargs="+",type=int,default=[4221,17,71,314,2025]); a=ap.parse_args()
    tables=[]; keys=None
    for seed in a.seeds:
        path=a.results_dir/f"seed_{seed}"/"test_predictions.csv"; df=pd.read_csv(path)
        truth=next((x for x in TRUTH if x in df),None); pred=next((x for x in PRED if x in df),None)
        trial_keys=next((list(k) for k in (("drug_id","target_id"),("canonical_smiles","target_sequence")) if set(k)<=set(df)),None)
        if not all((truth,pred,trial_keys)): raise ValueError(f"{path}: cannot identify keys/truth/prediction. Columns: {list(df.columns)}")
        if keys is None: keys=trial_keys
        if trial_keys!=keys: raise ValueError("Different pair keys across seeds")
        tables.append(df[keys+[truth,pred]].rename(columns={truth:"y_true",pred:f"pred_{seed}"}))
    merged=tables[0]
    for table in tables[1:]: merged=merged.merge(table,on=keys+["y_true"],how="outer",validate="one_to_one")
    if merged.isna().any().any(): raise ValueError("Seed files contain different test pairs or labels")
    prediction=merged[[f"pred_{s}" for s in a.seeds]].mean(axis=1).to_numpy(float); y=merged.y_true.to_numpy(float); error=prediction-y
    mse=float(np.mean(error**2)); report={"seeds":a.seeds,"n_pairs":len(y),"ensemble":{"MSE":mse,"RMSE":float(np.sqrt(mse)),"MAE":float(np.mean(np.abs(error))),"R2":float(1-np.sum(error**2)/np.sum((y-y.mean())**2)),"Pearson":float(np.corrcoef(y,prediction)[0,1]),"CI":ci(y,prediction)}}
    a.out_dir.mkdir(parents=True,exist_ok=True); (a.out_dir/"baseline_ensemble_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2)); print(f"Saved: {(a.out_dir/'baseline_ensemble_report.json').resolve()}")
if __name__=="__main__": main()
