#!/usr/bin/env python3
"""Analyze saved HGTA predictions without retraining (Python 3.8+, standard library)."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

KEY = ("canonical_smiles", "target_sequence")
SEEDS = [4221, 17, 71, 314, 2025]


def read_predictions(path, uncertainty):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = set(KEY) | {"affinity", "prediction"}
        if uncertainty:
            required.add("predicted_std")
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing columns in " + str(path))
        rows = {}
        for r in reader:
            key = tuple(r[k] for k in KEY)
            if key in rows:
                raise ValueError("Duplicate pair in " + str(path))
            y, p = float(r["affinity"]), float(r["prediction"])
            sd = float(r["predicted_std"]) if uncertainty else 0.0
            if not all(math.isfinite(v) for v in (y, p, sd)) or sd < 0:
                raise ValueError("Invalid numeric value in " + str(path))
            rows[key] = (y, p, sd)
    if len(rows) < 2:
        raise ValueError("Too few predictions in " + str(path))
    return rows


def mean(x):
    return sum(x) / len(x)


def concordance(y, p):
    ranks = {v: i + 1 for i, v in enumerate(sorted(set(p)))}
    bit = [0] * (len(ranks) + 1)
    def query(i):
        result = 0
        while i:
            result += bit[i]
            i -= i & -i
        return result
    ordered = sorted(zip(y, p))
    i = previous = pairs = 0
    score = 0.0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        for _, pred in ordered[i:j]:
            r = ranks[pred]
            less = query(r - 1)
            score += less + 0.5 * (query(r) - less)
            pairs += previous
        for _, pred in ordered[i:j]:
            r = ranks[pred]
            while r < len(bit):
                bit[r] += 1
                r += r & -r
            previous += 1
        i = j
    return score / pairs if pairs else None


def metrics(y, p, sd=None):
    errors = [a-b for a, b in zip(p, y)]
    mse = mean([e*e for e in errors])
    ym, pm = mean(y), mean(p)
    yy = sum((v-ym)**2 for v in y)
    pp = sum((v-pm)**2 for v in p)
    result = {"MSE": mse, "RMSE": math.sqrt(mse),
              "MAE": mean([abs(e) for e in errors]),
              "R2": 1-sum(e*e for e in errors)/yy if yy else None,
              "Pearson": sum((a-ym)*(b-pm) for a,b in zip(y,p))/math.sqrt(yy*pp) if yy*pp else None,
              "CI": concordance(y,p)}
    if sd is not None:
        for level in (0.90, 0.95):
            z = NormalDist().inv_cdf((1+level)/2)
            result["Coverage%d" % (100*level)] = 100*mean([float(abs(e)<=z*s) for e,s in zip(errors,sd)])
            result["MeanWidth%d" % (100*level)] = mean([2*z*s for s in sd])
        result["GaussianApproxNLL"] = mean([0.5*(math.log(2*math.pi*max(s*s,1e-12))+e*e/max(s*s,1e-12)) for e,s in zip(errors,sd)])
    return result


def combine(root, seeds, filename, uncertainty):
    runs = [read_predictions(root/('seed_%d' % seed)/filename, uncertainty) for seed in seeds]
    keys = sorted(runs[0])
    for run in runs[1:]:
        if set(run) != set(keys):
            raise ValueError("Seed files contain different pairs; refusing to align by row number")
        if any(not math.isclose(run[k][0], runs[0][k][0], rel_tol=1e-7, abs_tol=1e-7) for k in keys):
            raise ValueError("Affinity labels differ across seed files")
    y, prediction, aleatoric, epistemic = [], [], [], []
    for k in keys:
        preds = [r[k][1] for r in runs]
        mu = mean(preds)
        y.append(runs[0][k][0]); prediction.append(mu)
        aleatoric.append(mean([r[k][2]**2 for r in runs]))
        # Population variance is the exact between-component variance of this finite mixture.
        epistemic.append(mean([(v-mu)**2 for v in preds]))
    return keys, y, prediction, aleatoric, epistemic, runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', default='results_uq_hgta_full')
    ap.add_argument('--baseline-dir', default=None)
    ap.add_argument('--out-dir', default='ensemble_analysis')
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--calibration-dir', default=None,
                    help='Optional separate held-out calibration predictions, named calibration_predictions.csv per seed')
    args = ap.parse_args()
    if len(args.seeds)<2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Supply at least two distinct seeds')
    root, out = Path(args.results_dir), Path(args.out_dir)
    keys, y, p, av, ev, runs = combine(root,args.seeds,'test_predictions.csv',True)
    sd = [math.sqrt(a+e) for a,e in zip(av,ev)]
    report = {'seeds':args.seeds, 'n_pairs':len(keys),
              'interpretation':'Post-hoc exploratory ensemble; normal intervals approximate a Gaussian mixture. No SOTA or calibration guarantee.',
              'ensemble':metrics(y,p,sd),
              'per_seed':{str(seed):metrics(y,[r[k][1] for k in keys],[r[k][2] for k in keys]) for seed,r in zip(args.seeds,runs)},
              'mean_aleatoric_variance':mean(av), 'mean_epistemic_variance':mean(ev)}
    scale = None
    if args.calibration_dir:
        ck, cy, cp, ca, ce, _ = combine(Path(args.calibration_dir),args.seeds,'calibration_predictions.csv',True)
        if set(ck) & set(keys):
            raise ValueError('Calibration and test pairs overlap')
        scale = math.sqrt(mean([(a-b)**2/max(v+w,1e-12) for a,b,v,w in zip(cy,cp,ca,ce)]))
        report['calibration_scale'] = scale
        report['calibrated_ensemble'] = metrics(y,p,[s*scale for s in sd])
        report['calibration_note'] = 'Scale fitted only on supplied calibration labels. User must verify these pairs were excluded from model training.'
    else:
        report['calibration_note'] = 'Not fitted: original trainer does not save calibration predictions. No test labels used to fit a scale.'
    if args.baseline_dir:
        bk, by, bp, _, _, br = combine(Path(args.baseline_dir),args.seeds,'test_predictions.csv',False)
        if bk != keys or any(not math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-7) for a,b in zip(by,y)):
            raise ValueError('Baseline pairs/labels do not match UQ experiment')
        report['baseline_ensemble'] = metrics(y,bp)
        report['paired_seed_differences_uq_minus_baseline'] = {
            str(seed):{m:metrics(y,[ur[k][1] for k in keys])[m]-metrics(y,[b[k][1] for k in keys])[m] for m in ('MSE','MAE')}
            for seed,ur,b in zip(args.seeds,runs,br)}
    out.mkdir(parents=True,exist_ok=False)
    with (out/'report.json').open('w') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    with (out/'ensemble_predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(list(KEY)+['affinity','prediction','aleatoric_variance','epistemic_variance','total_std','lower95_normal','upper95_normal'])
        for k,t,mu,a,e,s in zip(keys,y,p,av,ev,sd):
            writer.writerow(list(k)+[t,mu,a,e,s,mu-1.959964*s,mu+1.959964*s])
    print(json.dumps(report,indent=2,allow_nan=False))
    print('Saved:',out.resolve())


if __name__ == '__main__':
    main()
