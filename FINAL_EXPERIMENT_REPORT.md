# Drug--Target Affinity Experiments: Final Technical Report

**Prepared:** 14 October 2026  
**Status:** completed experimental record; not a claim of state-of-the-art performance

## Executive conclusion

The project established a leakage-aware evaluation pipeline for drug--target affinity (DTA) prediction and tested four point-prediction model families: original HGTA, a gated residual interaction model, ligand-only, and protein-only controls. The central result is split- and dataset-dependent.

1. On two independently seeded **KIBA strict double-cold** splits, the gated residual interaction model consistently outperformed original HGTA and both single-modality controls. The result held under pair-, drug-, and target-cluster bootstrap analyses.
2. On **Davis strict double-cold**, this advantage did not transfer. Original HGTA outperformed the residual model on the untouched external split. On a second Davis development split, MSE was indistinguishable and the residual MAE advantage was not robust to drug-cluster resampling.
3. Therefore, the work supports a **KIBA-specific, reproducible improvement under this protocol**, not a general DTA or SOTA claim.

## 1. Reproducibility and data safeguards

All experiments used five fixed seeds: `4221, 17, 71, 314, 2025`. For every locked partition, the workflow recorded split hashes, trained only on the train partition, selected checkpoints on validation MSE, and evaluated the test partition only after checkpoint selection. Model-level comparisons used prediction-aligned paired bootstrap resampling (10,000 repetitions) at three levels: pairs, drugs, and targets.

### Input handling

- Molecules were canonicalized with RDKit.
- Duplicate model inputs were removed before locked KIBA experiments; labels were not silently converted into training examples across partitions.
- Exact canonical drug and protein-sequence overlap were audited for all cold splits.
- Frozen ESM2 protein representations and Morgan fingerprints were used in the residual and single-modality baselines.

### Important interpretation rule

`p_two_sided = 0.0` from 10,000 bootstrap repetitions means no resample crossed zero. It must be reported as **p < 0.0001**, never as an exact zero p-value.

## 2. Models

| Name | Inputs | Prediction design | Purpose |
|---|---|---|---|
| Original HGTA | Molecular graph + ESM2 residue representations | GINE graph encoder plus bidirectional cross-attention | Main historical baseline |
| UQ-HGTA | Same as HGTA | Gaussian mean and variance head | Exploratory uncertainty baseline |
| Ligand-only | Morgan fingerprint | MLP | Tests drug-only shortcut signal |
| Protein-only | Mean-pooled frozen ESM2 | MLP | Tests protein-only shortcut signal |
| Gated residual interaction | Morgan fingerprint + mean-pooled frozen ESM2 | `drug_main + protein_main + sigmoid(gate) × interaction_residual` | Tests whether an explicit joint residual improves over main effects |
| Floor-aware residual | Same as gated residual | One-sided loss at the hypothesized Davis floor of 5.0 | Exploratory Davis-only loss ablation; rejected |

The gated residual model is not called HGTA. It is a separate, lighter model whose gate starts near a main-effects solution and learns an interaction residual only when validation performance supports it.

## 3. Early pair-style KIBA results

The pair-style KIBA split retained shared entities across training and test. It is useful but does not test generalization to unseen drugs or targets.

| KIBA pair-style ensemble | MSE | MAE | Pearson | CI |
|---|---:|---:|---:|---:|
| Original HGTA baseline | 0.4075 | 0.4131 | 0.6650 | 0.7688 |
| Original UQ-HGTA | 0.4495 | 0.4253 | 0.6180 | 0.7585 |
| Validation-calibrated UQ-HGTA | 0.4212 | 0.4216 | 0.6474 | 0.7621 |
| Ligand-only | 0.4204 | 0.4520 | 0.6137 | 0.6990 |
| Protein-only | 0.5798 | 0.5016 | 0.3742 | 0.6842 |

Calibration improved the uncertainty procedure relative to the original UQ model, but it did not beat the point-prediction HGTA baseline. The calibrated UQ ensemble was over-conservative (90% coverage 95.91%; 95% coverage 97.73%). It must not be presented as a superior point predictor or as calibrated uncertainty without further work.

## 4. KIBA cold-drug and cold-target controls

### 4.1 Cold-drug split

Drugs were unseen in test while targets could be shared. The five-model results showed that target effects dominated this protocol.

| KIBA cold-drug ensemble | MSE | MAE | Pearson | CI |
|---|---:|---:|---:|---:|
| Original HGTA | 0.4282 | 0.4589 | 0.4321 | 0.7193 |
| Ligand-only | 0.5029 | 0.5146 | 0.3214 | 0.5890 |
| Protein-only | **0.4181** | **0.4276** | **0.4397** | 0.7191 |
| Train-only target mean | 0.4174 | 0.4122 | 0.4405 | 0.7209 |

Protein-only was significantly better than HGTA: ΔMSE = -0.0101 and ΔMAE = -0.0314, with pair-, drug-, and target-cluster bootstrap intervals below zero. This is evidence that target/main-effect signal can dominate cold-drug evaluation.

### 4.2 Cold-target split

Targets were unseen in test while drugs could be shared.

| KIBA cold-target ensemble | MSE | MAE | Pearson | CI |
|---|---:|---:|---:|---:|
| Original HGTA | 0.6209 | 0.5269 | 0.4502 | 0.6739 |
| Ligand-only | **0.4392** | **0.4537** | **0.6438** | 0.7181 |
| Protein-only | 0.7226 | 0.6004 | 0.2027 | 0.6045 |
| Train-only drug mean | 0.4391 | 0.4483 | 0.6445 | 0.7182 |

Ligand-only was significantly better than HGTA: ΔMSE = -0.1817 and ΔMAE = -0.0732, with all three cluster bootstrap intervals below zero. This exposes a drug/main-effect shortcut in cold-target evaluation.

## 5. KIBA strict double-cold study

Double-cold splits required both test drugs and test target sequences to be absent from training. Cross-partition pairs were excluded. This is the primary interaction-generalization protocol.

### 5.1 Development split v2

The v2 split contained 31,349 train, 4,331 validation, and 9,107 test pairs. It used 1,286/284/479 mutually disjoint drugs and 141/33/55 mutually disjoint proteins.

| KIBA double-cold v2 ensemble | MSE | MAE | Pearson | CI |
|---|---:|---:|---:|---:|
| Train global mean | 0.6250 | 0.5553 | 0.0000 | 0.5000 |
| Protein-only | 0.6181 | 0.5656 | 0.1628 | 0.5909 |
| Original HGTA | 0.6029 | 0.5796 | 0.2424 | 0.6310 |
| Ligand-only | 0.5903 | 0.5511 | 0.3052 | 0.5951 |
| **Gated residual interaction** | **0.5224** | **0.5002** | **0.4333** | **0.6492** |

Residual interaction improved over HGTA by ΔMSE = -0.0805 and ΔMAE = -0.0794. Drug- and target-cluster confidence intervals excluded zero. It also significantly beat ligand-only and protein-only controls.

### 5.2 Independent confirmation split v3

The v3 split was created after locking the residual architecture and hyperparameters. It contained 31,379 train, 4,355 validation, and 9,093 test pairs, with 1,286/284/477 disjoint drugs and 141/33/55 disjoint proteins.

| KIBA double-cold v3 ensemble | MSE | MAE | Pearson | CI |
|---|---:|---:|---:|---:|
| Protein-only | 0.6218 | 0.5525 | 0.2165 | 0.6091 |
| Original HGTA | 0.6224 | 0.5807 | 0.2695 | 0.6300 |
| Ligand-only | 0.6026 | 0.5466 | 0.3288 | 0.5924 |
| **Gated residual interaction** | **0.5223** | **0.5003** | **0.4572** | **0.6412** |

The residual ensemble was nearly identical on v2 and v3 (MSE 0.5224 vs. 0.5223), supporting reproducibility within KIBA.

### 5.3 v3 paired bootstrap evidence

Negative differences favor residual interaction.

| Comparator | ΔMSE residual - comparator | Drug-cluster 95% CI | Target-cluster 95% CI | ΔMAE | Interpretation |
|---|---:|---:|---:|---:|---|
| Original HGTA | -0.1001 | [-0.1343, -0.0686] | [-0.1636, -0.0416] | -0.0804 | Robust residual improvement |
| Ligand-only | -0.0802 | [-0.1061, -0.0566] | [-0.1356, -0.0252] | -0.0463 | Robust residual improvement |
| Protein-only | -0.0995 | [-0.1346, -0.0671] | [-0.1578, -0.0476] | -0.0522 | Robust residual improvement |

### 5.4 v3 integrity audit

There was zero exact drug and target overlap between every pair of partitions. The test train-global-mean MSE was 0.6487; residual interaction improved it to 0.5223. Label means were close (train 11.71, test 11.68). However, test drugs were not fully scaffold-disjoint: median maximum Morgan similarity to training drugs was 0.657. Future work should include scaffold- or similarity-cluster partitions.

## 6. Davis strict double-cold study

The Davis external-to-KIBA split contained 6,804 train, 923 validation, and 1,919 test pairs, with 36/13/19 disjoint drugs and 189/71/101 disjoint proteins. This is dramatically lower-data than KIBA.

### 6.1 Davis v1 external evaluation

| Davis double-cold v1 ensemble | MSE | MAE | Pearson | CI |
|---|---:|---:|---:|---:|
| Train global mean | 0.7014 | 0.5880 | 0.0000 | 0.5000 |
| Original HGTA | **0.6889** | **0.5051** | **0.2433** | **0.6247** |
| Gated residual interaction | 0.7500 | 0.5524 | 0.1270 | 0.5761 |

Residual interaction was significantly worse than HGTA:

| Bootstrap unit | ΔMSE residual - HGTA | 95% CI | ΔMAE | 95% CI |
|---|---:|---:|---:|---:|
| Pairs | +0.0611 | [0.0355, 0.0870] | +0.0473 | [0.0337, 0.0612] |
| Drugs | +0.0611 | [0.0202, 0.1026] | +0.0473 | [0.0026, 0.0974] |
| Targets | +0.0611 | [0.0317, 0.0906] | +0.0473 | [0.0282, 0.0672] |

The v1 audit confirmed zero exact entity overlap. It also showed a much harsher chemical novelty regime than KIBA: median maximum drug similarity to train was 0.238. Davis labels had a strong pile-up at 5.0 (test median 5.0), which motivated an exploratory floor-aware loss experiment.

### 6.2 Davis v2 development experiments

| Davis double-cold v2 ensemble | MSE | MAE | Pearson | CI |
|---|---:|---:|---:|---:|
| Original HGTA | **0.6509** | 0.6104 | 0.1049 | 0.6164 |
| Original residual | 0.6541 | **0.5661** | **0.1851** | **0.6205** |
| Floor-aware residual | 0.6952 | 0.6460 | 0.1728 | 0.6186 |

Residual versus HGTA on v2 was inconclusive for MSE (ΔMSE = +0.0032; drug-cluster CI [-0.1788, 0.2321]). Residual MAE was lower at pair and target level, but the drug-cluster MAE interval crossed zero: [-0.1470, 0.0657]. Thus it is not robust across unseen drugs.

The floor-aware loss worsened both ensemble MSE (+0.0411) and MAE (+0.0799) relative to the original residual model. It is rejected for this project.

## 7. Final interpretation

### What the experiments establish

- The leakage-aware split and audit workflow is functioning and records reproducible partitions.
- Original pair-style metrics alone are insufficient because single-modality controls can win on cold-drug or cold-target protocols.
- The gated residual interaction model has a reproducible advantage on the current KIBA strict double-cold protocol.
- That advantage was independently reproduced on a second locked KIBA split and survived entity-cluster bootstrap tests.

### What the experiments do not establish

- A general DTA improvement across datasets.
- State-of-the-art performance.
- A clinically actionable affinity predictor.
- A causal biological explanation from the learned gate.
- Fully novel chemical generalization; KIBA strict double-cold still permits high fingerprint similarity for some test drugs.

## 8. Recommended manuscript claim

> In leakage-aware KIBA double-cold experiments, a gated residual interaction model improved affinity prediction relative to original HGTA and ligand-only/protein-only controls across two independently seeded partitions. The advantage did not transfer to a strict double-cold Davis evaluation, demonstrating that apparent interaction gains are dataset- and split-dependent.

This is a credible and useful research contribution if presented as a rigorous evaluation/ablation study, not as SOTA.

## 9. Next research steps

1. Freeze all current results and scripts; do not tune on Davis v1 external data.
2. If pursuing a new model, use only Davis v2 as development data and create a fresh Davis v3 confirmation split before reporting results.
3. Investigate transfer learning from KIBA representations to Davis, with a newly initialized Davis prediction head and strict source/target provenance.
4. Add scaffold-cluster drug partitions and homology-cluster protein partitions; exact-identity cold splits are necessary but insufficient.
5. Reproduce official published baselines under identical data processing and explicitly distinguish official-fold results from custom strict cold-split results.
6. Release split manifests, SHA-256 hashes, all per-seed predictions, histories, code versions, and bootstrap reports.

## 10. Key references for the manuscript

1. Öztürk H, Özgür A, Ozkirimli E. *DeepDTA: deep drug-target binding affinity prediction.* Bioinformatics, 2018.
2. Nguyen T, Le H, Quinn TP, Nguyen T, Le TD, Venkatesh S. *GraphDTA: predicting drug-target binding affinity with graph neural networks.* Bioinformatics, 2021.
3. Davis MI, et al. *Comprehensive analysis of kinase inhibitor selectivity.* Nature Biotechnology, 2011.
4. Tang J, et al. *Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative study.* Journal of Chemical Information and Modeling, 2014. (KIBA)
5. Efron B, Tibshirani RJ. *An Introduction to the Bootstrap.* Chapman & Hall/CRC, 1993.

