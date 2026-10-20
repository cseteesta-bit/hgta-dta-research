# Leakage-Aware, Uncertainty-Aware Drug–Target Affinity Prediction

## Research progress report

**Project aim.** Develop and evaluate an HGTA-based drug–target affinity (DTA) predictor that is scientifically defensible: no test-set model selection, explicit data-integrity checks, fixed partitions, multi-seed reporting, and uncertainty quantification (UQ).

## Why the workflow was redesigned

Initial Davis CSV experiments exposed two data risks that can inflate published DTA results:

1. Duplicate drug–target pairs had conflicting affinity labels.
2. Train and test sets contained overlapping canonical molecular/protein inputs.

Investigation of the raw Davis source showed that several named kinase variants map to identical protein sequences. A model receiving only SMILES and sequence cannot distinguish such source identities. Median-aggregating those labels would hide the ambiguity; retaining cross-split duplicate inputs would create leakage.

**Decision:** do not merge conflicting labels. Instead, build locked, input-unambiguous benchmark subsets and retain explicit source IDs wherever available.

## Model comparison

| Component | Baseline HGTA | UQ-HGTA |
|---|---|---|
| Drug encoder | Bond-aware graph encoder | Same |
| Protein encoder | Frozen protein representation with atom–residue interaction | Same |
| Output | Affinity mean | Affinity mean and predictive variance |
| Training objective | Regression loss | Gaussian negative log-likelihood |
| Uncertainty | None | Aleatoric uncertainty; ensembles add between-seed epistemic uncertainty |
| Checkpoint selection | Validation loss only | Validation loss only |

Both systems use the same architecture, data partitions, seeds, epochs, early-stopping patience, and evaluation metrics. The output head and probabilistic objective are the planned experimental difference.

## Evaluation protocol

- Five fixed random seeds: **4221, 17, 71, 314, 2025**.
- Fixed train/validation/test partitions are hashed in manifests before training.
- Test labels are not used for checkpoint selection, calibration fitting, or hyperparameter selection.
- Final prediction is a five-seed mean ensemble.
- Performance: MSE, RMSE, MAE, R², Pearson correlation, and concordance index (CI).
- UQ evaluation additionally reports interval coverage and predictive uncertainty.
- Paired bootstrap comparisons resample pairs, drugs, and targets. Drug/target clustering addresses correlated DTA observations.

## Davis: completed corrected benchmark

### Dataset integrity

The final Davis restricted subset retained only sequence-unambiguous target identities:

- Train: **16,475** interactions
- Validation: **4,042** interactions
- Test: **4,031** interactions
- Retained target identities: **361**

### Five-seed ensemble comparison

| Metric | Baseline ensemble | UQ ensemble | Change (UQ − baseline) |
|---|---:|---:|---:|
| MSE | 0.3638 | 0.3240 | −0.0399 (−11.0%) |
| RMSE | 0.6032 | 0.5692 | −0.0340 |
| MAE | 0.3582 | 0.3067 | −0.0515 (−14.4%) |
| R² | 0.4346 | 0.4966 | +0.0619 |
| Pearson | 0.6798 | 0.7062 | +0.0264 |
| CI | 0.8229 | 0.8508 | +0.0280 |

### Davis statistical evidence

Paired ensemble bootstrap results support the UQ gain:

| Resampling unit | MSE difference, UQ − baseline (95% CI) | MAE difference, UQ − baseline (95% CI) |
|---|---|---|
| Pair | −0.0399 [−0.0548, −0.0253] | −0.0515 [−0.0585, −0.0443] |
| Drug (68 clusters) | −0.0399 [−0.0637, −0.0177], p = 0.0002 | −0.0515 [−0.0654, −0.0375], p < 0.0002 |
| Target (361 clusters) | −0.0399 [−0.0561, −0.0241] | −0.0515 [−0.0591, −0.0437] |

Negative differences favor UQ. The clustered analyses are the appropriate conservative evidence for a manuscript.

## KIBA: independent replication benchmark

KIBA was exported directly from the official GraphDTA fold-setting-1 raw files while preserving raw drug and target identifiers. A global input audit then removed any interaction whose `(canonical SMILES, target sequence)` appeared more than once across the three partitions.

| KIBA audit outcome | Value |
|---|---:|
| Source interactions | 118,254 |
| Repeated model-input interactions removed | 1,070 (0.91%) |
| Retained interactions | 117,184 |
| Train | 82,979 |
| Validation | 14,665 |
| Test | 19,540 |

### Completed KIBA baseline result

The five-seed baseline ensemble has completed on the locked KIBA subset:

| MSE | RMSE | MAE | R² | Pearson | CI |
|---:|---:|---:|---:|---:|---:|
| 0.4075 | 0.6384 | 0.4131 | 0.3954 | 0.6650 | 0.7688 |

### Current status

The matched five-seed UQ-HGTA KIBA run is in progress. Its results must be evaluated using the same ensemble and paired cluster-bootstrap protocol before any cross-dataset conclusion is made.

## What has been achieved

- Replaced exploratory, potentially leaky CSV handling with reproducible and auditable locked splits.
- Established a fair baseline-vs-UQ comparison under identical experimental conditions.
- Demonstrated a statistically supported UQ advantage on corrected Davis.
- Created a large, independently prepared, input-unambiguous KIBA replication benchmark.
- Completed the KIBA baseline ensemble; the UQ replication is underway.

## Remaining work before publication claims

1. Finish KIBA UQ training for all five seeds.
2. Compute the KIBA UQ ensemble and compare it to the baseline ensemble.
3. Run paired pair/drug/target bootstrap confidence intervals on KIBA.
4. Report calibration/coverage results without fitting on test labels.
5. Include dataset provenance, exclusion rules, manifests, code version hashes, seeds, and hardware/software details in the reproducibility appendix.

## Current conclusion

The method has **promising, statistically supported evidence on corrected Davis**. It is not yet appropriate to claim state-of-the-art performance. A publishable claim depends on completing the preregistered-style KIBA replication and presenting both successes and limitations transparently.
