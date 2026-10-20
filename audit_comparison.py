"""Read supplied prediction archive; paired bootstrap of fixed ensembles."""
import csv
import io
import json
import zipfile
from pathlib import Path
import numpy as np

ARCHIVE = Path('C:/Users/HP/Desktop/hgta_comparison.zip')
SEEDS = [4221, 17, 71, 314, 2025]

def main():
    with zipfile.ZipFile(ARCHIVE) as z:
        def js(p): return json.loads(z.read(p))
        def rows(p): return list(csv.DictReader(io.StringIO(z.read(p).decode('utf-8-sig'))))
        roots = ['results_davis_hgta', 'results_uq_hgta_full']
        manifests = [js(r+'/manifest.json') for r in roots]
        predictions=[]; master=None; labels=None; checkpoints={}
        for root in roots:
            ensemble=[]; checkpoints[root]=[]
            for seed in SEEDS:
                data=rows(root+'/seed_%d/test_predictions.csv'%seed)
                lookup={(r['canonical_smiles'],r['target_sequence']):r for r in data}
                assert len(lookup)==len(data), 'Duplicate prediction keys'
                keys=sorted(lookup)
                y=np.array([float(lookup[k]['affinity']) for k in keys])
                if master is None: master,labels=keys,y
                assert master==keys and np.allclose(labels,y,atol=1e-7,rtol=0), 'Different test pairs/labels'
                ensemble.append([float(lookup[k]['prediction']) for k in keys])
                m=js(root+'/seed_%d/metrics.json'%seed)
                checkpoints[root].append({'seed':seed,'best_epoch':m['best_epoch'],'validation_mse':m['validation']['MSE'],'test_mse':m['test']['MSE']})
            predictions.append(np.mean(ensemble,axis=0))
        differences={}
        for section in ['config','environment','sha256']:
            a,b=[m.get(section,{}) for m in manifests]
            differences[section]={k:[a.get(k),b.get(k)] for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
        excluded=[rows(r+'/excluded_train_test_overlaps.csv') for r in roots]
        exkeys=[{(r['canonical_smiles'],r['target_sequence']) for r in records} for records in excluded]
    err0=predictions[0]-labels; err1=predictions[1]-labels
    delta=np.column_stack((err1**2-err0**2,abs(err1)-abs(err0)))
    bootstrap={}
    for mode,col in [('pairs',None),('drug_clusters',0),('target_clusters',1)]:
        groups=list(range(len(master))) if col is None else [k[col] for k in master]
        _,index=np.unique(groups,return_inverse=True)
        n=int(index.max()+1)
        counts=np.bincount(index)
        totals=np.column_stack([np.bincount(index,weights=delta[:,j]) for j in range(2)])
        rng=np.random.default_rng(20260909)
        values=[]
        for _ in range(100):
            selected=rng.integers(0,n,size=(100,n))
            values.append(totals[selected].sum(axis=1)/counts[selected].sum(axis=1)[:,None])
        vals=np.concatenate(values)
        bootstrap[mode]={'n_groups':n,'replicates':len(vals),'difference_UQ_minus_baseline':dict(zip(['MSE','MAE'],delta.mean(axis=0).tolist())),
            'percentile_95_CI':{m:np.quantile(vals[:,j],[.025,.975]).tolist() for j,m in enumerate(['MSE','MAE'])}}
    report={'n_pairs':len(master),'manifest_differences':differences,'manifests':manifests,
            'excluded_pair_sets_equal':exkeys[0]==exkeys[1], 'checkpoints':checkpoints,'bootstrap':bootstrap,
            'metrics':{name:{'MSE':float(np.mean(e**2)),'MAE':float(np.mean(abs(e)))} for name,e in zip(['baseline','uq'],[err0,err1])},
            'limits':'Fixed trained ensembles; drug and target cluster bootstraps run separately, not joint two-way resampling. No retraining or model-selection uncertainty included. Exact training/validation membership not archived.'}
    Path('comparison_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
