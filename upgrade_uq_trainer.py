#!/usr/bin/env python3
"""Upgrade the supplied uq_hgta.py without changing its HGTA architecture.

Creates uq_hgta_calibrated.py.  It changes checkpoint selection to validation
Gaussian NLL, records validation predictions, and fits one multiplicative
uncertainty scale on validation data only.  Test labels are never used to fit
the scale or select a checkpoint.
"""
from pathlib import Path
import argparse


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Could not safely locate {label} (found {count} matches). "
                           "Use the original uq_hgta.py supplied for this upgrade.")
    return text.replace(old, new, 1)


PREDICT_OLD = '''@torch.inference_mode()
def predict(model, loader, device, amp):
    model.eval(); ps, stds, ys, ids = [], [], [], []
    for idx, g, p, pm, y, _ in loader:
        g, p, pm = g.to(device), p.to(device), pm.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=amp): mu, log_var = model(g, p, pm)
        ids.extend(idx); ps.extend(mu.float().cpu().tolist())
        stds.extend((0.5 * log_var.float()).exp().cpu().tolist()); ys.extend(y.tolist())
    return np.asarray(ids), np.asarray(ys), np.asarray(ps), np.asarray(stds)
'''

PREDICT_NEW = '''def predictive_variance(std, assay_var, cfg):
    """One variance definition shared by NLL, calibration, and intervals."""
    observed = assay_var if cfg.use_replicate_uncertainty else np.zeros_like(assay_var)
    return np.clip(np.square(std) + observed + cfg.min_variance, cfg.min_variance, 100.0)


def gaussian_nll_numpy(y, mu, std, assay_var, cfg):
    var = predictive_variance(std, assay_var, cfg)
    return float(0.5 * np.mean(np.log(var) + np.square(y - mu) / var))


def fit_validation_scale(y, mu, std, assay_var, cfg):
    """Closed-form scale fit using validation data only; bounded for stability."""
    base = predictive_variance(std, assay_var, cfg)
    scale = float(np.sqrt(np.mean(np.square(y - mu) / base)))
    return float(np.clip(scale, 0.10, 10.0))


@torch.inference_mode()
def predict(model, loader, device, amp):
    model.eval(); ps, stds, ys, assay_vars, ids = [], [], [], [], []
    for idx, g, p, pm, y, assay_var in loader:
        g, p, pm = g.to(device), p.to(device), pm.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=amp): mu, log_var = model(g, p, pm)
        ids.extend(idx); ps.extend(mu.float().cpu().tolist())
        # Keep this as the raw model standard deviation; predictive_variance()
        # applies the one shared variance floor for NLL and calibration.
        stds.extend((0.5 * log_var.float()).exp().cpu().tolist())
        ys.extend(y.tolist()); assay_vars.extend(assay_var.tolist())
    return (np.asarray(ids), np.asarray(ys), np.asarray(ps), np.asarray(stds),
            np.asarray(assay_vars))
'''


TRAIN_OLD = '''        _, vy, vp, vs = predict(model, val_ld, device, amp); vm = metrics(vy, vp)
        vm["Coverage90"] = float(100*np.mean(np.abs(vy-vp) <= 1.644854*vs))
        vm["Coverage95"] = float(100*np.mean(np.abs(vy-vp) <= 1.959964*vs))
        vm["MeanPredStd"] = float(vs.mean()); sched.step(vm["MSE"])
        row = {"epoch": epoch, "train_mse": losses/count, "lr": opt.param_groups[0]["lr"], **{f"val_{k}":v for k,v in vm.items()}}
        history.append(row)
        if vm["MSE"] < best - 1e-8:
            best, stale = vm["MSE"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": vm}, run/"best.pt")
        else: stale += 1
        if epoch == 1 or epoch % 10 == 0: print(f"seed={seed} epoch={epoch} train={losses/count:.4f} val={vm['MSE']:.4f}")
'''

TRAIN_NEW = '''        _, vy, vp, vs, va = predict(model, val_ld, device, amp); vm = metrics(vy, vp)
        vm["NLL"] = gaussian_nll_numpy(vy, vp, vs, va, cfg)
        vm["Coverage90"] = float(100*np.mean(np.abs(vy-vp) <= 1.644854*vs))
        vm["Coverage95"] = float(100*np.mean(np.abs(vy-vp) <= 1.959964*vs))
        vm["MeanPredStd"] = float(vs.mean()); sched.step(vm["NLL"])
        row = {"epoch": epoch, "train_nll": losses/count, "lr": opt.param_groups[0]["lr"], **{f"val_{k}":v for k,v in vm.items()}}
        history.append(row)
        if vm["NLL"] < best - 1e-8:
            best, stale = vm["NLL"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": vm,
                        "selection_metric": "validation_nll"}, run/"best.pt")
        else: stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"seed={seed} epoch={epoch} train_nll={losses/count:.4f} val_nll={vm['NLL']:.4f} val_mse={vm['MSE']:.4f}")
'''

TEST_OLD = '''    # First and only test evaluation for this seed.
    ids, ty, tp, ts = predict(model, test_ld, device, amp); tm = metrics(ty, tp)
    tm["Coverage90"] = float(100*np.mean(np.abs(ty-tp) <= 1.644854*ts))
    tm["Coverage95"] = float(100*np.mean(np.abs(ty-tp) <= 1.959964*ts))
    tm["MeanPredStd"] = float(ts.mean())
    out = test_df.iloc[ids].copy(); out["prediction"] = tp; out["predicted_std"] = ts
    out["lower95"] = tp - 1.959964*ts; out["upper95"] = tp + 1.959964*ts
    out["absolute_error"] = abs(tp-ty)
    out.to_csv(run/"test_predictions.csv", index=False)
    with open(run/"metrics.json", "w") as f: json.dump({"seed": seed, "best_epoch": ckpt["epoch"], "validation": ckpt["val_metrics"], "test": tm}, f, indent=2)
    return tm
'''

TEST_NEW = '''    # Fit calibration on the locked validation split only, after checkpoint selection.
    vi, vy, vp, vs, va = predict(model, val_ld, device, amp)
    scale = fit_validation_scale(vy, vp, vs, va, cfg)
    cvs = vs * scale
    val_out = val_df.iloc[vi].copy(); val_out["prediction"] = vp
    val_out["raw_predicted_std"] = vs; val_out["calibrated_predicted_std"] = cvs
    val_out["absolute_error"] = np.abs(vp-vy)
    val_out.to_csv(run/"validation_predictions.csv", index=False)
    calibration = {"method": "validation_only_multiplicative_std_scale", "scale": scale,
                   "raw_validation_nll": gaussian_nll_numpy(vy, vp, vs, va, cfg),
                   "calibrated_validation_nll": gaussian_nll_numpy(vy, vp, cvs, va, cfg)}

    # First and only test evaluation for this seed.  No test value affects fitting.
    ids, ty, tp, raw_ts, ta = predict(model, test_ld, device, amp)
    model_ts = raw_ts * scale
    raw_interval_std = np.sqrt(predictive_variance(raw_ts, ta, cfg))
    ts = np.sqrt(predictive_variance(model_ts, ta, cfg)); tm = metrics(ty, tp)
    tm["NLL"] = gaussian_nll_numpy(ty, tp, model_ts, ta, cfg)
    tm["RawNLL"] = gaussian_nll_numpy(ty, tp, raw_ts, ta, cfg)
    tm["Coverage90"] = float(100*np.mean(np.abs(ty-tp) <= 1.644854*ts))
    tm["Coverage95"] = float(100*np.mean(np.abs(ty-tp) <= 1.959964*ts))
    tm["MeanPredStd"] = float(ts.mean()); tm["RawMeanPredStd"] = float(raw_interval_std.mean())
    out = test_df.iloc[ids].copy(); out["prediction"] = tp
    out["raw_predicted_std"] = raw_interval_std; out["predicted_std"] = ts
    out["lower95"] = tp - 1.959964*ts; out["upper95"] = tp + 1.959964*ts
    out["absolute_error"] = np.abs(tp-ty)
    out.to_csv(run/"test_predictions.csv", index=False)
    with open(run/"metrics.json", "w") as f:
        json.dump({"seed": seed, "best_epoch": ckpt["epoch"], "selection_metric": "validation_nll",
                   "validation": ckpt["val_metrics"], "calibration": calibration, "test": tm}, f, indent=2)
    return tm
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="uq_hgta.py")
    ap.add_argument("--output", default="uq_hgta_calibrated.py")
    args = ap.parse_args()
    source = Path(args.input)
    text = source.read_text(encoding="utf-8")
    text = replace_once(text, PREDICT_OLD, PREDICT_NEW, "predict()")
    text = replace_once(text, TRAIN_OLD, TRAIN_NEW, "validation/training loop")
    text = replace_once(text, TEST_OLD, TEST_NEW, "test evaluation block")
    text = text.replace('"""Leakage-free HGTA training', '"""HGTA training with validation-only uncertainty calibration', 1)
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"Created {args.output}. Original source was not modified.")


if __name__ == "__main__":
    main()
