#!/usr/bin/env python3
"""Audit raw-ID KIBA CSVs and create fixed train/validation/test files."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

def col(df, names):
    return next((x for x in names if x in df.columns), None)

def read(path, split):
    x = pd.read_csv(path)
    smi, seq, y = col(x,("compound_iso_smiles","smiles")), col(x,("target_sequence","sequence")), col(x,("affinity","label","y"))
    if not all((smi,seq,y)): raise ValueError(f"{path}: expected SMILES, target sequence, affinity columns; found {list(x.columns)}")
    out = x[[smi,seq,y]].copy(); out.columns=["smiles","target_sequence","affinity"]
    out["drug_id"] = x["drug_id"].astype(str) if "drug_id" in x else out.smiles.astype(str)
    out["target_id"] = x["target_id"].astype(str) if "target_id" in x else out.target_sequence.astype(str)
    out["canonical_smiles"] = out.smiles.map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
    if out.canonical_smiles.isna().any(): raise ValueError(f"{path}: invalid SMILES")
    out["target_sequence"] = out.target_sequence.astype(str).str.strip().str.upper(); out["affinity"] = pd.to_numeric(out.affinity,errors="raise")
    out["source_split"] = split
    return out[["drug_id","target_id","canonical_smiles","target_sequence","affinity","source_split"]]

def clean(df, name, audit):
    keys=["drug_id","target_id"]; groups=df[df.duplicated(keys,False)].groupby(keys,sort=False)
    conflict=sum(g.affinity.nunique()>1 for _,g in groups); audit[f"{name}_duplicate_pair_groups"]=len(groups); audit[f"{name}_conflicting_pair_groups"]=conflict
    if conflict: raise ValueError(f"{name}: {conflict} duplicate raw-ID pairs have conflicting labels. Do not aggregate; inspect the source.")
    return df.drop_duplicates(keys).reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train-csv",required=True,type=Path); ap.add_argument("--test-csv",required=True,type=Path); ap.add_argument("--out-dir",required=True,type=Path); ap.add_argument("--val-fraction",type=float,default=.15); ap.add_argument("--seed",type=int,default=20260915); a=ap.parse_args()
    audit={"seed":a.seed,"val_fraction":a.val_fraction,"input_train":str(a.train_csv),"input_test":str(a.test_csv)}
    train,test=clean(read(a.train_csv,"train"),"train",audit),clean(read(a.test_csv,"test"),"test",audit); keys=["drug_id","target_id"]
    overlap=train.merge(test[keys],on=keys,how="inner"); audit["overlap_removed_from_train"]=len(overlap)
    if len(overlap): train=train.merge(test[keys],on=keys,how="left",indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    rng=np.random.default_rng(a.seed); valid=np.zeros(len(train),bool); valid[rng.choice(len(train),round(len(train)*a.val_fraction),replace=False)]=True
    parts={"train":train.loc[~valid].drop(columns="source_split"),"val":train.loc[valid].drop(columns="source_split"),"test":test.drop(columns="source_split")}; a.out_dir.mkdir(parents=True,exist_ok=True)
    for name,frame in parts.items(): frame.to_csv(a.out_dir/f"{name}.csv",index=False); audit[f"{name}_rows"]=len(frame); audit[f"{name}_sha256"]=hashlib.sha256((a.out_dir/f"{name}.csv").read_bytes()).hexdigest()
    audit["test_drugs"],audit["test_targets"]=parts["test"].drug_id.nunique(),parts["test"].target_id.nunique(); (a.out_dir/"audit_manifest.json").write_text(json.dumps(audit,indent=2),encoding="utf8"); print(json.dumps(audit,indent=2)); print(f"Saved: {a.out_dir.resolve()}")
if __name__=="__main__": main()
