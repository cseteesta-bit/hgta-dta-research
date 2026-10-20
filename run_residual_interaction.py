#!/usr/bin/env python3
"""Gated residual drug--protein interaction model for locked DTA splits.

Prediction = drug main effect + protein main effect + gated interaction residual.
The main-effect heads make shortcut structure explicit; the residual head must
earn any additional accuracy from the joint drug--protein representation.
Checkpoint selection uses validation MSE only.  Test labels are evaluated only
after the checkpoint is fixed.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer


def hash_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for part in iter(lambda:f.read(1<<20),b""): h.update(part)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def concordance(y,p):
    order=np.argsort(y,kind="mergesort"); y,p=y[order],p[order]; ranks=np.searchsorted(np.unique(p),p)+1; bit=np.zeros(ranks.max()+1,np.int64)
    def query(i):
        total=0
        while i: total+=bit[i]; i-=i&-i
        return total
    def add(i):
        while i<len(bit): bit[i]+=1; i+=i&-i
    score=pairs=previous=0.; i=0
    while i<len(y):
        j=i
        while j<len(y) and y[j]==y[i]: j+=1
        for k in range(i,j):
            less,upto=query(ranks[k]-1),query(ranks[k]); score+=less+.5*(upto-less); pairs+=previous
        for k in range(i,j): add(ranks[k]); previous+=1
        i=j
    return float(score/pairs) if pairs else float("nan")


def metrics(y,p):
    e=p-y; mse=float(np.mean(e*e)); denom=float(np.sum((y-y.mean())**2))
    return {"MSE":mse,"RMSE":math.sqrt(mse),"MAE":float(np.mean(np.abs(e))),
            "R2":float(1-np.sum(e*e)/denom) if denom else float("nan"),
            "Pearson":float(np.corrcoef(y,p)[0,1]) if np.std(y) and np.std(p) else float("nan"),"CI":concordance(y,p),
            "Within0.5":float(100*np.mean(np.abs(e)<=.5)),"Within1.0":float(100*np.mean(np.abs(e)<=1))}


class ESMMeanCache:
    def __init__(self, name, cache_dir, device):
        self.name=name; self.cache_dir=Path(cache_dir); self.device=device; self.cache_dir.mkdir(parents=True,exist_ok=True); self.tokenizer=self.model=None
    def _path(self, sequence):
        token=hashlib.sha256(("esm-mean-v1\0"+self.name+"\0"+sequence).encode()).hexdigest(); return self.cache_dir/(token+".pt")
    @torch.inference_mode()
    def get(self, sequence):
        sequence=str(sequence).strip().upper(); path=self._path(sequence)
        if path.exists(): return torch.load(path,map_location="cpu",weights_only=True).float()
        if self.model is None:
            self.tokenizer=AutoTokenizer.from_pretrained(self.name); self.model=AutoModel.from_pretrained(self.name,add_pooling_layer=False).to(self.device).eval()
        total=None; count=0
        for start in range(0,len(sequence),1022):
            tok=self.tokenizer(sequence[start:start+1022],return_tensors="pt",add_special_tokens=True).to(self.device)
            h=self.model(**tok).last_hidden_state[0]
            special=self.tokenizer.get_special_tokens_mask(tok["input_ids"][0].tolist(),already_has_special_tokens=True)
            keep=torch.tensor([not x for x in special],device=h.device,dtype=torch.bool); values=h[keep].float()
            total=values.sum(0) if total is None else total+values.sum(0); count+=len(values)
        if not count: raise ValueError("Empty protein sequence")
        result=(total/count).cpu(); torch.save(result,path); return result


class ResidualInteraction(nn.Module):
    def __init__(self, drug_dim, protein_dim, hidden=512, dropout=.2):
        super().__init__(); half=hidden//2
        self.drug=nn.Sequential(nn.Linear(drug_dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,half),nn.GELU())
        self.protein=nn.Sequential(nn.Linear(protein_dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,half),nn.GELU())
        self.drug_main=nn.Linear(half,1); self.protein_main=nn.Linear(half,1)
        self.interaction=nn.Sequential(nn.Linear(4*half,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,1))
        self.gate=nn.Sequential(nn.Linear(4*half,half),nn.GELU(),nn.Linear(half,1))
        # Start near a main-effects model; interaction contribution is learned only if useful.
        nn.init.constant_(self.gate[-1].bias,-2.0)
    def forward(self, drug, protein, return_parts=False):
        d=self.drug(drug); p=self.protein(protein); z=torch.cat((d,p,d*p,(d-p).abs()),dim=-1)
        dm=self.drug_main(d).squeeze(-1); pm=self.protein_main(p).squeeze(-1); residual=self.interaction(z).squeeze(-1); gate=torch.sigmoid(self.gate(z).squeeze(-1))
        prediction=dm+pm+gate*residual
        return (prediction,dm,pm,residual,gate) if return_parts else prediction


@torch.inference_mode()
def predict(model, loader, device, y_mean, y_std):
    model.eval(); yy=[]; pp=[]; gates=[]
    for drug,protein,y in loader:
        pred,_,_,_,gate=model(drug.to(device),protein.to(device),return_parts=True)
        pp.append(pred.cpu().numpy()*y_std+y_mean); yy.append(y.numpy()); gates.append(gate.cpu().numpy())
    return np.concatenate(yy),np.concatenate(pp),np.concatenate(gates)


def train_seed(seed, data, labels, test_frame, args, root, device):
    seed_all(seed); run=root/f"seed_{seed}"; run.mkdir()
    dx_train,px_train=data["train"]; dx_val,px_val=data["validation"]; dx_test,px_test=data["test"]
    y_train,y_val,y_test=labels["train"],labels["validation"],labels["test"]
    dm,ds=dx_train.mean(0),dx_train.std(0); ds[ds<1e-6]=1
    pm,ps=px_train.mean(0),px_train.std(0); ps[ps<1e-6]=1
    dx_train,dx_val,dx_test=(dx_train-dm)/ds,(dx_val-dm)/ds,(dx_test-dm)/ds
    px_train,px_val,px_test=(px_train-pm)/ps,(px_val-pm)/ps,(px_test-pm)/ps
    ym,ys=float(y_train.mean()),float(y_train.std())
    train=DataLoader(TensorDataset(torch.from_numpy(dx_train),torch.from_numpy(px_train),torch.from_numpy((y_train-ym)/ys)),batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    val=DataLoader(TensorDataset(torch.from_numpy(dx_val),torch.from_numpy(px_val),torch.from_numpy(y_val)),batch_size=1024)
    test=DataLoader(TensorDataset(torch.from_numpy(dx_test),torch.from_numpy(px_test),torch.from_numpy(y_test)),batch_size=1024)
    model=ResidualInteraction(dx_train.shape[1],px_train.shape[1],args.hidden_dim,args.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=10,min_lr=1e-6)
    best,stale,history=float("inf"),0,[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=n=0
        for drug,protein,y in train:
            drug,protein,y=drug.to(device),protein.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            pred,_,_,residual,_=model(drug,protein,return_parts=True); loss=nn.functional.mse_loss(pred,y)+args.residual_l2*residual.square().mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()*len(y); n+=len(y)
        vy,vp,vg=predict(model,val,device,ym,ys); vm=metrics(vy,vp); schedule.step(vm["MSE"])
        history.append({"epoch":epoch,"train_objective":total/n,"lr":opt.param_groups[0]["lr"],"val_mean_gate":float(vg.mean()),**{f"val_{k}":v for k,v in vm.items()}})
        if vm["MSE"]<best-1e-8:
            best,stale=vm["MSE"],0; torch.save({"model":model.state_dict(),"epoch":epoch,"validation":vm,"validation_mean_gate":float(vg.mean())},run/"best.pt")
        else: stale+=1
        if epoch==1 or epoch%10==0: print(f"seed={seed} epoch={epoch} train={total/n:.4f} val={vm['MSE']:.4f} gate={vg.mean():.3f}")
        if stale>=args.patience: break
    pd.DataFrame(history).to_csv(run/"history.csv",index=False); ckpt=torch.load(run/"best.pt",map_location=device,weights_only=True); model.load_state_dict(ckpt["model"])
    ty,tp,tg=predict(model,test,device,ym,ys); tm=metrics(ty,tp); tm["MeanInteractionGate"]=float(tg.mean())
    out=test_frame.copy(); out["prediction"]=tp; out["absolute_error"]=np.abs(tp-ty); out["interaction_gate"]=tg; out.to_csv(run/"test_predictions.csv",index=False)
    (run/"metrics.json").write_text(json.dumps({"seed":seed,"best_epoch":ckpt["epoch"],"validation":ckpt["validation"],"validation_mean_gate":ckpt["validation_mean_gate"],"test":tm},indent=2))
    return tm


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir",required=True); ap.add_argument("--out-dir",required=True); ap.add_argument("--seeds",nargs="+",type=int,default=[4221,17,71,314,2025])
    ap.add_argument("--epochs",type=int,default=300); ap.add_argument("--patience",type=int,default=35); ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--learning-rate",type=float,default=1e-3); ap.add_argument("--weight-decay",type=float,default=1e-5); ap.add_argument("--dropout",type=float,default=.2); ap.add_argument("--hidden-dim",type=int,default=512); ap.add_argument("--residual-l2",type=float,default=.001); ap.add_argument("--protein-model",default="facebook/esm2_t12_35M_UR50D"); args=ap.parse_args()
    root=Path(args.out_dir)
    if root.exists(): raise ValueError(f"Output directory exists: {root}")
    source=Path(args.data_dir); paths={n:source/f"{n}.csv" for n in ("train","validation","test")}; frames={n:pd.read_csv(p) for n,p in paths.items()}
    needed={"canonical_smiles","target_sequence","affinity"}
    for n,f in frames.items():
        if needed-set(f): raise ValueError(f"{n} missing {sorted(needed-set(f))}")
        f.target_sequence=f.target_sequence.astype(str).str.strip().str.upper(); f.affinity=pd.to_numeric(f.affinity,errors="raise")
    root.mkdir(parents=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048); drug_cache={}
    def fp(s):
        if s not in drug_cache:
            mol=Chem.MolFromSmiles(s)
            if mol is None: raise ValueError(f"Invalid SMILES: {s}")
            array=np.zeros(2048,dtype=np.float32); DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol),array); drug_cache[s]=array
        return drug_cache[s]
    esm=ESMMeanCache(args.protein_model,root/"protein_cache",device); sequences=sorted(set().union(*(set(f.target_sequence) for f in frames.values()))); protein_cache={s:esm.get(s).numpy().astype(np.float32) for s in sequences}
    data={n:(np.stack([fp(s) for s in f.canonical_smiles]),np.stack([protein_cache[s] for s in f.target_sequence])) for n,f in frames.items()}; labels={n:frames[n].affinity.to_numpy(np.float32) for n in frames}
    manifest={"model":"gated_residual_interaction_morgan_esm2","formula":"drug_main + protein_main + sigmoid(gate) * interaction_residual","args":vars(args),"source_sha256":{n:hash_file(p) for n,p in paths.items()},"n_unique_drugs":len(drug_cache),"n_unique_proteins":len(protein_cache)}; (root/"manifest.json").write_text(json.dumps(manifest,indent=2))
    results=[train_seed(seed,data,labels,frames["test"],args,root,device) for seed in args.seeds]; table=pd.DataFrame(results); table.insert(0,"seed",args.seeds); table.to_csv(root/"all_seed_metrics.csv",index=False)
    summary={c:{"mean":float(table[c].mean()),"std":float(table[c].std(ddof=1))} for c in table.columns if c!="seed"}; (root/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
