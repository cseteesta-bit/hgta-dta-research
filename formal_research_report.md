# A Leakage-Aware and Uncertainty-Aware Framework for Drug–Target Affinity Prediction

## Progress report and presentation brief

### Executive summary

Drug–target affinity (DTA) prediction estimates the strength of interaction between a drug molecule and a protein target. It can help prioritize experimental assays, but benchmark scores are meaningful only when the data split, identity handling, and model-selection protocol are credible. This project develops an HGTA-based DTA model with predictive uncertainty and evaluates it under a deliberately conservative protocol.

The central contribution is not simply a new neural-network head. It is an **auditable experimental pipeline**: source-aware identity reconstruction, removal of duplicate model inputs across partitions, validation-only checkpoint selection, five-seed ensembles, and cluster-aware statistical comparison. These choices address common risks in DTA benchmarks, where different biological records can become indistinguishable after they are represented solely as a molecular string and a protein sequence.

The work has produced statistically supported improvement of the uncertainty-aware model on a corrected Davis benchmark. An independent KIBA replication is underway; the baseline ensemble is complete and the matched UQ ensemble is still training. Therefore, the appropriate current claim is *promising evidence under leakage-aware evaluation*, not state of the art.

## 1. Scientific motivation

DTA is naturally a regression problem: a drug–protein pair is mapped to a continuous affinity value. DeepDTA established a widely used benchmark formulation that learns directly from drug and protein sequences [1]. GraphDTA later represented drugs as molecular graphs so that chemical bonds are available to the model rather than being inferred from a character string [2]. Both Davis and KIBA are standard kinase-inhibitor benchmarks: Davis reports a kinase selectivity panel, whereas KIBA integrates multiple kinase bioactivity measurements into a common score [3,4].

However, high accuracy without uncertainty can be unsafe for experimental prioritization. The UQ model therefore predicts both an affinity mean and an input-dependent variance. This follows the standard distinction between **aleatoric uncertainty** (measurement/data noise) and **epistemic uncertainty** (model uncertainty), while a multi-seed ensemble supplies a practical estimate of between-model variation [5].

## 2. Problem discovered during initial analysis

The initial Davis CSV files were not safe to use directly. Two issues were observed:

1. Multiple rows represented the same canonical drug–protein input but had different labels.
2. Some model inputs appeared on both sides of the intended train/test split.

Source reconstruction showed why this matters. Certain named kinase variants share an identical stored protein sequence. A model whose input is only `(SMILES, sequence)` cannot recover the original target label after that identity has been collapsed. Treating those records as independent test examples or averaging their labels can yield misleading performance estimates.

### Research decision

We do **not** median-aggregate conflicting labels. Instead, we preserve raw identifiers during auditing and create a restricted benchmark that excludes ambiguous or repeated model inputs. This makes the benchmark smaller but makes the reported result more interpretable.

## 3. Proposed model and fair comparison

| Element | HGTA baseline | UQ-HGTA |
|---|---|---|
| Drug representation | Bond-aware molecular graph | Same |
| Protein representation | Frozen protein representation | Same |
| Drug–target fusion | Atom–residue interaction/attention | Same |
| Prediction | Affinity mean | Affinity mean + log-variance |
| Training criterion | Regression objective | Gaussian negative log-likelihood |
| Model uncertainty | Not reported | Variance decomposition across five seeds |
| Checkpoint choice | Validation loss only | Validation loss only |

The comparison is intentionally controlled. Architecture, dataset partitions, random seeds, batch size, maximum epochs, patience, and evaluation code are matched. The difference under test is the uncertainty-aware prediction objective and head.

## 4. How the study is conducted

```text
Official raw benchmark source
            ↓
ID-preserving export and audit
            ↓
Remove repeated/ambiguous model inputs (no label averaging)
            ↓
Lock train / validation / test CSVs and record SHA-256 hashes
            ↓
Train baseline and UQ-HGTA with five fixed seeds
            ↓
Choose checkpoint using validation only
            ↓
Average five test predictions (ensemble)
            ↓
Evaluate errors, ranking, uncertainty coverage, and paired cluster bootstrap
```

### Reproducibility controls

- Fixed seeds: 4221, 17, 71, 314, and 2025.
- Partitions are stored as CSV files with manifests and SHA-256 hashes.
- Test labels are not used for early stopping or to fit calibration scales.
- Every model writes per-seed test predictions, enabling independent re-analysis.
- Ensemble comparisons use paired predictions on exactly the same test pairs.
- Bootstrap resampling is performed at pair, drug, and target levels. Drug/target clustering is emphasized because multiple observations share a compound or protein.

## 5. Completed Davis experiment

### Corrected Davis subset

| Partition | Interactions |
|---|---:|
| Training | 16,475 |
| Validation | 4,042 |
| Test | 4,031 |
| Retained target identities | 361 |

### Five-seed ensemble results

| Metric | Baseline HGTA | UQ-HGTA | UQ change |
|---|---:|---:|---:|
| MSE | 0.3638 | 0.3240 | −0.0399 (−11.0%) |
| RMSE | 0.6032 | 0.5692 | −0.0340 |
| MAE | 0.3582 | 0.3067 | −0.0515 (−14.4%) |
| R² | 0.4346 | 0.4966 | +0.0619 |
| Pearson correlation | 0.6798 | 0.7062 | +0.0264 |
| Concordance index | 0.8229 | 0.8508 | +0.0280 |

Lower MSE/RMSE/MAE is better; higher R², Pearson correlation, and CI is better.

### Statistical comparison

The UQ gain remains after conservative clustered bootstrap resampling:

| Cluster unit | MSE difference: UQ − baseline | MAE difference: UQ − baseline |
|---|---|---|
| Pair | −0.0399; 95% CI [−0.0548, −0.0253] | −0.0515; 95% CI [−0.0585, −0.0443] |
| Drug (68 clusters) | −0.0399; 95% CI [−0.0637, −0.0177], p = 0.0002 | −0.0515; 95% CI [−0.0654, −0.0375], p < 0.0002 |
| Target (361 clusters) | −0.0399; 95% CI [−0.0561, −0.0241] | −0.0515; 95% CI [−0.0591, −0.0437] |

All confidence intervals lie below zero. Thus, on this corrected Davis subset, the five-seed UQ ensemble has significantly lower error than the matched baseline. A reported numerical p-value of `0.0` at 10,000 bootstrap draws is expressed properly as `p < 0.0002`, not as p = 0.

## 6. Independent KIBA replication

KIBA was exported from the official GraphDTA fold-setting-1 raw files while retaining raw drug and target identifiers. A global audit then removed every repeated `(canonical SMILES, target sequence)` input across train, validation, and test. No labels were averaged.

| KIBA filtering outcome | Interactions |
|---|---:|
| Source rows | 118,254 |
| Removed repeated-input rows | 1,070 (0.91%) |
| Retained rows | 117,184 |
| Training | 82,979 |
| Validation | 14,665 |
| Test | 19,540 |

### Completed KIBA baseline ensemble

| MSE | RMSE | MAE | R² | Pearson | CI |
|---:|---:|---:|---:|---:|---:|
| 0.4075 | 0.6384 | 0.4131 | 0.3954 | 0.6650 | 0.7688 |

### Current status and next analysis

The matched UQ-HGTA KIBA training is in progress. Once all five seeds finish, the next fixed sequence is:

1. form the UQ ensemble using the same 19,540 test pairs;
2. compare UQ and baseline ensemble metrics;
3. run paired pair-, drug-, and target-cluster bootstrap confidence intervals;
4. report interval coverage and width without test-set calibration fitting.

The KIBA result will act as an independent replication, not a dataset selected after observing the Davis result.

## 7. Current contribution and limitations

### Contribution to date

- A source-aware, leakage-resistant DTA evaluation pipeline.
- A controlled baseline-versus-UQ comparison with multi-seed ensembles.
- Statistically supported UQ improvement on corrected Davis.
- A large, locked, input-unambiguous KIBA benchmark and completed baseline.

### Limitations to state explicitly

- The corrected subsets are not identical to the conventional, potentially ambiguous CSV benchmark; direct comparison to papers using different preprocessing is not a head-to-head SOTA claim.
- A five-seed ensemble improves stability but is more computationally expensive than a single model.
- Predicted uncertainty must be judged by calibration and coverage, not only by mean-error reduction.
- KIBA UQ results and statistical testing remain incomplete at the time of this report.

## 8. Present conclusion

The evidence currently supports the statement: **UQ-HGTA significantly improves affinity prediction over a matched HGTA baseline on a leakage-aware, sequence-unambiguous Davis benchmark.** The study is progressing through a pre-specified independent KIBA replication. A broader claim—including state-of-the-art performance—will be considered only after KIBA completes and all outcomes are reported transparently.

## References

1. Öztürk H, Özgür A, Ozkirimli E. DeepDTA: deep drug–target binding affinity prediction. *Bioinformatics*. 2018;34(17):i821–i829. doi: [10.1093/bioinformatics/bty593](https://doi.org/10.1093/bioinformatics/bty593).
2. Nguyen T, Le H, Quinn TP, Nguyen T, Le TD, Venkatesh S. GraphDTA: predicting drug–target binding affinity with graph neural networks. *Bioinformatics*. 2021;37(8):1140–1147. doi: [10.1093/bioinformatics/btaa921](https://doi.org/10.1093/bioinformatics/btaa921).
3. Davis MI, Hunt JP, Herrgard S, et al. Comprehensive analysis of kinase inhibitor selectivity. *Nature Biotechnology*. 2011;29:1046–1051. doi: [10.1038/nbt.1990](https://doi.org/10.1038/nbt.1990).
4. Tang J, Szwajda A, Shakyawar S, et al. Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis. *Journal of Chemical Information and Modeling*. 2014;54(3):735–743. doi: [10.1021/ci400709d](https://doi.org/10.1021/ci400709d).
5. Kendall A, Gal Y. What uncertainties do we need in Bayesian deep learning for computer vision? In: *Advances in Neural Information Processing Systems 30*. 2017:5574–5584. [Open proceedings record](https://papers.nips.cc/paper_files/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html).
