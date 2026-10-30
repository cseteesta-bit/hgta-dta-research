# HGTA-v2 gated atom-protein residual: benchmark report

## Claim supported by these experiments

HGTA-v2 is a **gated atom-protein interaction residual** added to a frozen,
validation-selected GraphDTA GIN predictor.  Under the fixed benchmark files
and protocol recorded below, it consistently improves the matched frozen GIN
baseline on KIBA and Davis.

This is **not a state-of-the-art claim**.  A SOTA claim would require matched
reimplementations or published results under exactly the same data version,
split, preprocessing, training budget, and metric implementation, together
with a prospectively untouched confirmation evaluation.

## Locked architecture and selection rule

* Base: frozen GraphDTA `GINConvNet` checkpoint, selected only by validation
  MSE from an 80/20 seeded split of the supplied training CSV.
* Candidate: frozen base prediction plus a gated residual.  The residual uses
  GIN atom-node representations as queries over protein CNN motif tokens.
* The base has no trainable gradients.  The residual output is zero-initialized,
  so epoch 0 exactly reproduces the base model.
* Candidate checkpoint selection uses validation MSE only.  The benchmark test
  set is evaluated once after selecting the checkpoint.
* Fixed candidate settings: 64-dimensional interaction space, dropout 0.1,
  Adam learning rate 0.001, batch size 512, at most 400 epochs, patience 80.
* Five preselected seeds: 4221, 17, 71, 314, 2025.
* Ensemble: arithmetic mean of the five seed predictions.

## Standard KIBA fixed-fold benchmark

Test rows: 19,709.

| Model ensemble | MSE | RMSE | MAE | R2 | Pearson | CI |
|---|---:|---:|---:|---:|---:|---:|
| Frozen GraphDTA-GIN | 0.135216 | 0.367717 | 0.213815 | 0.801055 | 0.896640 | 0.890609 |
| HGTA-v2 gated residual | **0.131651** | **0.362837** | **0.205685** | **0.806300** | **0.898315** | **0.891752** |
| HGTA-v2 minus base | -0.003565 | — | -0.008130 | — | +0.001675 | +0.001142 |

Paired row bootstrap (10,000 repetitions; seed 20261028):

| Difference, HGTA-v2 minus base | Estimate | 95% CI | Two-sided p |
|---|---:|---:|---:|
| MSE | -0.003565 | [-0.004171, -0.002954] | 0.000200 |
| MAE | -0.008130 | [-0.008728, -0.007537] | 0.000200 |

## Standard Davis fixed-fold benchmark

Test rows: 5,010.

| Model ensemble | MSE | RMSE | MAE | R2 | Pearson | CI |
|---|---:|---:|---:|---:|---:|---:|
| Frozen GraphDTA-GIN | 0.237459 | 0.487297 | 0.267903 | 0.703737 | 0.839945 | 0.899255 |
| HGTA-v2 gated residual | **0.233612** | **0.483335** | **0.258596** | **0.708536** | **0.842511** | **0.904198** |
| HGTA-v2 minus base | -0.003846 | — | -0.009307 | — | +0.002565 | +0.004943 |

Paired row bootstrap (10,000 repetitions; seed 20261030):

| Difference, HGTA-v2 minus base | Estimate | 95% CI | Two-sided p |
|---|---:|---:|---:|
| MSE | -0.003846 | [-0.005815, -0.001952] | 0.000400 |
| MAE | -0.009307 | [-0.010543, -0.008073] | 0.000200 |

## Interpretation

The candidate improves the matched base in every one of the five individual
seeds on both datasets, and both ensemble-level paired bootstrap confidence
intervals exclude zero.  The learned gates are small (approximately 0.01–0.03),
which indicates a conservative correction of the base predictor rather than a
replacement of it.

The bootstrap is row-level because the stored GraphDTA prediction CSVs do not
retain drug and target IDs.  It quantifies uncertainty over these benchmark
rows, not generalization to clusters of new drugs or targets.

## Reproduction artifacts

* `training_validation_clean.py`: validation-only GraphDTA baseline runner.
* `run_hgta_gated_residual_graphdta.py`: frozen-base HGTA-v2 candidate runner.
* `evaluate_hgta_v2_gated_ensemble.py`: verified five-seed ensemble evaluator.
* `paired_bootstrap_prediction_files.py`: paired row-bootstrap evaluator.

The output directories contain `metrics.json`, `test_predictions.csv`,
ensemble prediction files, and bootstrap reports.  Data, caches, checkpoints,
and generated results are intentionally ignored by Git.

## Next research step

Do not use either benchmark test set for further architecture or hyperparameter
choices.  Freeze this implementation and evaluate it on a newly defined,
prospectively locked split or external dataset.  For a stronger generalization
claim, retain drug/target identifiers in prediction files and use drug- and
target-clustered paired bootstraps.
