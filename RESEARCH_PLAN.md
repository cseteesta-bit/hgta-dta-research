# HGTA: journal-oriented research plan

## Claim boundary

This implementation predicts sequence-based drug--target affinity. It does not
predict a physical binding pose and should not be described as a docking or
structure-based free-energy method. Attention weights are not causal explanations.

## Primary contribution to test

**Residue--atom interaction modeling improves cold-start affinity prediction when
combined with frozen protein-language-model representations and bond-aware molecular
graphs.** The publishable contribution is the controlled test of this hypothesis,
not merely a lower MSE on the familiar random Davis split.

## Required experiments

1. Pre-register pair-random, unseen-drug, unseen-target, and both-unseen splits.
2. Use validation data only for checkpoint and hyperparameter selection; touch each
   test set once per locked seed.
3. Run five seeds and report mean, standard deviation, paired bootstrap 95% CIs, and
   paired significance/effect sizes against every reimplemented baseline.
4. Include DeepDTA, GraphDTA, a Morgan-fingerprint MLP, ESM mean-pooling + molecular
   graph late fusion, and a modern interaction model as controlled baselines.
5. Ablate bond features, ESM, bidirectional attention, atom-level tokens, protein
   length pooling, and the multiplicative/difference fusion terms.
6. Report parameter count, train/inference time, peak memory, invalid-input policy,
   and results by target/drug frequency and sequence/chemical similarity.
7. Repeat beyond Davis: KIBA plus a non-kinase or structure-grounded benchmark.
   Davis alone is too small and kinase-specific for a broad generalization claim.
8. Publish split indices, raw predictions, configuration, environment, dataset
   hashes, and code. Do not compare headline numbers copied from papers using
   different splits or checkpoint-selection rules.

## Suggested title

**HGTA: Bond-aware atom--residue co-attention for robust drug--target affinity
prediction under cold-start evaluation**

## Commands

```bash
python journal_hgta.py --train-csv data/davis_train.csv --test-csv data/davis_test.csv --val-split pair
python journal_hgta.py --train-csv data/davis_train.csv --test-csv data/davis_test.csv --val-split drug --out-dir results_davis_cold_drug
python journal_hgta.py --train-csv data/davis_train.csv --test-csv data/davis_test.csv --val-split target --out-dir results_davis_cold_target
```

The supplied official test set determines the actual test regime. `--val-split`
only controls leakage-free checkpoint selection inside the provided training data.
For a true cold test, generate and freeze entity-disjoint train/validation/test CSVs.
