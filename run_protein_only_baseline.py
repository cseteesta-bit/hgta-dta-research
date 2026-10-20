#!/usr/bin/env python3
"""Frozen-split protein-only DTA control using frozen ESM2 residue embeddings.

The model receives a mean-pooled ESM2 protein representation and no ligand or
SMILES information.  Validation selects the checkpoint; test labels are used
only for the final, post-selection evaluation.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer


SEQ_CANDIDATES = ("target_sequence", "sequence", "protein_sequence")
Y_CANDIDATES = ("affinity", "label", "y")


def column(frame, candidates, label):
    for item in candidates:
        if item in frame.columns: return item
    raise ValueError(f"Missing {label}; expected one of {candidates}, got {list(frame.columns)}")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""): h.update(part)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


class ProteinEmbedder:
    def __init__(self, model_name, cache_dir, device, chunk_residues=1022):
        self.model_name, self.cache_dir, self.device = model_name, Path(cache_dir), device
        self.chunk_residues = chunk_residues; self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = self.model = None

    def path(self, sequence):
        identity = f"mean-residue-pooling-v1\0{self.model_name}\0{sequence}".encode()
        return self.cache_dir / (hashlib.sha256(identity).hexdigest() + ".pt")

    @torch.inference_mode()
    def get(self, sequence):
        sequence = str(sequence).strip().upper(); path = self.path(sequence)
        if path.exists(): return torch.load(path, map_location="cpu", weights_only=True).float()
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name, add_pooling_layer=False).to(self.device).eval()
        total, residues = None, 0
        for start in range(0, len(sequence), self.chunk_residues):
            token = self.tokenizer(sequence[start:start+self.chunk_residues], return_tensors="pt", add_special_tokens=True).to(self.device)
            hidden = self.model(**token).last_hidden_state[0]
            special = self.tokenizer.get_special_tokens_mask(token["input_ids"][0].tolist(), already_has_special_tokens=True)
            keep = torch.tensor([not v for v in special], device=hidden.device, dtype=torch.bool)
            values = hidden[keep].float(); total = values.sum(0) if total is None else total + values.sum(0); residues += len(values)
        if not residues: raise ValueError("Empty protein sequence")
        embedding = (total / residues).cpu(); torch.save(embedding, path); return embedding


class ProteinMLP(nn.Module):
    def __init__(self, dimension, hidden, dropout):
        super().__init__(); self.net = nn.Sequential(
            nn.Linear(dimension, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def scores(y, p):
    e = p-y; mse = float(np.mean(e*e)); denom = float(np.sum((y-y.mean())**2))
    pearson = float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else float("nan")
    return {"MSE": mse, "RMSE": math.sqrt(mse), "MAE": float(np.mean(np.abs(e))),
            "R2": float(1-np.sum(e*e)/denom) if denom else float("nan"), "Pearson": pearson,
            "Within0.5": float(100*np.mean(np.abs(e)<=.5)), "Within1.0": float(100*np.mean(np.abs(e)<=1))}


@torch.inference_mode()
def predict(model, loader, device, mean, std):
    model.eval(); yy, pp = [], []
    for x, y in loader:
        pp.append((model(x.to(device)).cpu().numpy()*std)+mean); yy.append(y.numpy())
    return np.concatenate(yy), np.concatenate(pp)


def train_seed(seed, x_train, y_train, x_val, y_val, x_test, y_test, test_frame, args, root, device):
    seed_all(seed); run = root / f"seed_{seed}"; run.mkdir()
    mean_x, std_x = x_train.mean(0), x_train.std(0); std_x[std_x < 1e-6] = 1.0
    x_train, x_val, x_test = (x_train-mean_x)/std_x, (x_val-mean_x)/std_x, (x_test-mean_x)/std_x
    mean_y, std_y = float(y_train.mean()), float(y_train.std())
    if std_y < 1e-8: raise ValueError("Training affinities have zero variance")
    train = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy((y_train-mean_y)/std_y)),
                       batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=1024)
    test = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size=1024)
    model = ProteinMLP(x_train.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=10, min_lr=1e-6)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs+1):
        model.train(); total=n=0
        for x, y in train:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(model(x),y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            total += loss.item()*len(y); n += len(y)
        vy,vp=predict(model,val,device,mean_y,std_y); vm=scores(vy,vp); schedule.step(vm["MSE"])
        history.append({"epoch":epoch,"train_mse_standardized":total/n,"lr":optimizer.param_groups[0]["lr"],**{f"val_{k}":v for k,v in vm.items()}})
        if vm["MSE"] < best-1e-8:
            best,stale=vm["MSE"],0; torch.save({"model":model.state_dict(),"epoch":epoch,"validation":vm},run/"best.pt")
        else: stale += 1
        if epoch==1 or epoch%10==0: print(f"seed={seed} epoch={epoch} train={total/n:.4f} val={vm['MSE']:.4f}")
        if stale >= args.patience: break
    pd.DataFrame(history).to_csv(run/"history.csv",index=False)
    checkpoint=torch.load(run/"best.pt",map_location=device,weights_only=True); model.load_state_dict(checkpoint["model"])
    ty,tp=predict(model,test,device,mean_y,std_y); tm=scores(ty,tp)
    out=test_frame.copy(); out["prediction"]=tp; out["absolute_error"]=np.abs(tp-ty); out.to_csv(run/"test_predictions.csv",index=False)
    (run/"metrics.json").write_text(json.dumps({"seed":seed,"best_epoch":checkpoint["epoch"],"validation":checkpoint["validation"],"test":tm},indent=2))
    return tm


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir",required=True); ap.add_argument("--out-dir",required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=[4221,17,71,314,2025]); ap.add_argument("--epochs",type=int,default=300); ap.add_argument("--patience",type=int,default=35)
    ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--learning-rate",type=float,default=1e-3); ap.add_argument("--weight-decay",type=float,default=1e-5); ap.add_argument("--dropout",type=float,default=.2); ap.add_argument("--hidden-dim",type=int,default=512)
    ap.add_argument("--protein-model",default="facebook/esm2_t12_35M_UR50D"); args=ap.parse_args()
    root=Path(args.out_dir)
    if root.exists(): raise ValueError(f"Output directory exists: {root}")
    data=Path(args.data_dir); paths={n:data/f"{n}.csv" for n in ("train","validation","test")}
    frames={n:pd.read_csv(p) for n,p in paths.items()}; seq=column(frames["train"],SEQ_CANDIDATES,"protein sequence column"); label=column(frames["train"],Y_CANDIDATES,"affinity label column")
    for n,f in frames.items():
        f[seq]=f[seq].astype(str).str.strip().str.upper(); f[label]=pd.to_numeric(f[label],errors="raise")
        if not np.isfinite(f[label]).all() or f[seq].eq("").any(): raise ValueError(f"Bad sequence or affinity in {n}")
    root.mkdir(parents=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); embed=ProteinEmbedder(args.protein_model,root/"protein_cache",device)
    unique=sorted(set().union(*(set(f[seq]) for f in frames.values()))); table={s:embed.get(s).numpy().astype(np.float32) for s in unique}
    xs={n:np.stack([table[s] for s in f[seq]]) for n,f in frames.items()}; ys={n:frames[n][label].to_numpy(np.float32) for n in frames}
    (root/"manifest.json").write_text(json.dumps({"model":"protein_only_frozen_esm2_mlp","input":"mean-pooled frozen ESM2 protein embedding only; no ligand information","args":vars(args),"data_sha256":{n:file_hash(p) for n,p in paths.items()},"unique_proteins":len(unique)},indent=2))
    results=[train_seed(s,xs["train"],ys["train"],xs["validation"],ys["validation"],xs["test"],ys["test"],frames["test"],args,root,device) for s in args.seeds]
    report=pd.DataFrame(results); report.insert(0,"seed",args.seeds); report.to_csv(root/"all_seed_metrics.csv",index=False)
    summary={c:{"mean":float(report[c].mean()),"std":float(report[c].std(ddof=1))} for c in report.columns if c!="seed"}; (root/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
