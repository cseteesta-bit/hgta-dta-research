# HGTA architecture comparison

## Previous architecture

```mermaid
flowchart LR
  S[SMILES] --> R[RDKit molecular graph]
  R --> A[16 scalar atom features]
  A --> G[3 GIN layers]
  G --> M[Global mean pooling]
  M --> D[One drug vector]

  P[Protein sequence] --> E[ESM-2 frozen encoder]
  E --> X[Pad/truncate to 1024]
  X --> AP[Adaptive pooling to 512]
  AP --> T[Protein token matrix]

  D --> Q[Single drug query token]
  T --> CA[2 protein cross-attention blocks]
  Q --> CA
  CA --> QV[Conditioned drug vector]
  T --> PM[Masked protein mean]
  D --> F[Concatenate d, q, p, d times p]
  QV --> F
  PM --> F
  F --> H[BatchNorm MLP]
  H --> Y[Affinity]
```

## Current journal-oriented architecture

```mermaid
flowchart LR
  S[Canonical SMILES] --> R[Validated RDKit graph]
  R --> A[12 atom features]
  R --> B[7 bond features]
  A --> GI[Atom projection]
  B --> EI[Edge projection]
  GI --> GE[Residual GINE layers]
  EI --> GE
  GE --> AT[Atom-token matrix and mask]

  P[Validated protein sequence] --> C[Chunk long sequences]
  C --> E[ESM-2 frozen encoder]
  E --> V[Remove special and padding tokens]
  V --> AP[Valid-residue adaptive pooling when needed]
  AP --> PT[Residue-token projection and mask]

  AT --> DP[Atoms query residues]
  PT --> DP
  AT --> PD[Residues query atoms]
  PT --> PD
  DP --> DB[Residual attention and FFN blocks]
  PD --> DB
  DB --> DM[Masked atom mean]
  DB --> PM[Masked residue mean]
  DM --> F[Concatenate d, p, d times p, absolute difference]
  PM --> F
  F --> H[LayerNorm MLP]
  H --> Y[Affinity]
```

## Evaluation architecture

```mermaid
flowchart LR
  O[Official training CSV] --> A[Validation and overlap audit]
  A --> TR[Training subset]
  A --> VA[Validation subset]
  TR --> FIT[Fit model]
  VA --> SEL[Early stopping and checkpoint selection]
  FIT --> SEL
  SEL --> LOCK[Lock best checkpoint]
  TE[Untouched official test CSV] --> ONCE[One final evaluation per seed]
  LOCK --> ONCE
  ONCE --> OUT[Predictions, metrics, histories, hashes and environment]
```

## Direct comparison

| Dimension | Previous HGTA | Current HGTA | Consequence |
|---|---|---|---|
| Drug tokens at fusion | One pooled graph vector | Every atom | Current model can represent local atom--residue relationships |
| Molecular messages | Connectivity only | Bond-aware GINE | Bond order and chemical context affect message passing |
| Protein preparation | Pad to 1024, then pool | Remove padding/special tokens, chunk long sequences | Avoids padding contamination and silent truncation |
| Interaction | Drug-to-protein only | Bidirectional atom--residue co-attention | Both modalities condition each other |
| Normalization | BatchNorm | LayerNorm | Robust to small and final batches |
| Invalid SMILES | Silent zero graph | Hard data-quality error | Prevents hidden corrupted observations |
| Checkpoint criterion | Test MSE | Validation MSE | Removes checkpoint-selection leakage |
| Test usage | Repeated every 20 epochs | Once after model selection | Test remains an independent estimate |
| Repetitions | One seed | Five predefined seeds | Allows variability reporting |
| Audit trail | Aggregate JSON | Hashes, environment, histories, predictions and checkpoints | Supports reproducibility and paired statistics |

## Tensor flow

For hidden width `D=256`, batch size `B`, maximum atoms `A`, and maximum retained
residues `R`:

| Stage | Previous | Current |
|---|---:|---:|
| Drug representation before fusion | `[B, 1, 480]` after projection from one graph vector | `[B, A, 256]` |
| Protein representation | `[B, 512, 480]` | `[B, R, 256]`, `R <= 512` |
| Main attention map | `[B, 1, 512]` | Two maps: `[B, A, R]` and `[B, R, A]` |
| Fused predictor input | `[B, 1920]` | `[B, 1024]` |

The current architecture is more expressive but has attention complexity
`O(B*A*R*D)`. For unusually large ligands or proteins, pocket selection or sparse
top-k attention is the appropriate next optimization.
