# Implementing uncertainty-aware HGTA

## 1. Install into the existing environment

```bash
conda activate Sho2
pip install torch-geometric rdkit pandas numpy transformers
```

Use the PyTorch/CUDA versions already working for the baseline. Do not reinstall
PyTorch merely to follow this command.

## 2. Copy and smoke-test

Copy `uq_hgta.py` to the Linux project directory. It already contains the Davis
paths used in the existing experiment.

```bash
cd /home/cse1000/PycharmProjects/PythonProject2
python uq_hgta.py --seeds 4221 --epochs 2 --patience 2 --out-dir smoke_uq_hgta
```

The smoke test must finish with exit code 0 and produce `summary.json`,
`seed_4221/metrics.json`, and `seed_4221/test_predictions.csv`.

## 3. Development experiment

Use a new validation split and do not inspect the locked test output while
developing hyperparameters:

```bash
python uq_hgta.py \
  --seeds 4221 17 71 314 2025 \
  --val-split drug \
  --epochs 300 \
  --patience 35 \
  --out-dir results_uq_hgta_cold_drug
```

The model predicts both affinity and aleatoric standard deviation. It trains with
Gaussian negative log-likelihood but still selects checkpoints by validation MSE.
Prediction files contain `prediction`, `predicted_std`, `lower95`, and `upper95`.

## 4. Replicate-aware experiment

Run this only after confirming that repeated labels are comparable experimental
replicates rather than different assay conditions or data-construction errors:

```bash
python uq_hgta.py \
  --seeds 4221 17 71 314 2025 \
  --val-split drug \
  --use-replicate-uncertainty \
  --out-dir results_uq_hgta_replicate_aware
```

This adds the estimated variance of the aggregated measurement mean to the model's
predicted variance during Gaussian-NLL training.

## 5. Required comparison

Run the original `journal_hgta.py` and `uq_hgta.py` on the same frozen splits.
Compare MSE, MAE, CI, Pearson, 90%/95% coverage, and mean predicted uncertainty.
An uncertainty model is useful only if intervals are calibrated and errors grow
with predicted uncertainty; lower MSE alone is insufficient.

## 6. Next architecture stage

Do not add the 3D confidence gate until the uncertainty model passes the smoke test
and matched baseline experiment. The next dataset must contain stable compound and
target identifiers, ligand conformer coordinates, pocket-residue coordinates,
AlphaFold pLDDT/PAE confidence, and atom/residue mappings. That stage cannot be
implemented correctly from the three-column CSV alone.
