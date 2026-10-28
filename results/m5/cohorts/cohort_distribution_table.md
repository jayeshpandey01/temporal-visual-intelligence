# Milestone M5.2: Five Experimental Analysis Cohorts Distribution

**Audit Rule:** M5 must consume M4 outputs; M5 must not silently retrain or redefine M4.
**Total Evaluation Population:** $N = 300$ test sequences.

## 1. Primary Analysis Cohorts

| Cohort | Baseline (Model A) | Temporal (Model C) | Condition / Behavior | Sequences | % of Test Split | Research Role |
|---|---|---|---|---:|---:|---|
| **Cohort A** | Correct (✓) | Correct (✓) | Both correct | **11** | 3.67% | Normal / Easy cases (static sufficient) |
| **Cohort B** | Correct (✓) | Wrong (✗) | Baseline only | **4** | 1.33% | Temporal regression (motion bias / degradation) |
| **Cohort C** | Wrong (✗) | Correct (✓) | Temporal rescued | **97** | 32.33% | **Temporal Rescue (Primary Microscope)** |
| **Cohort D** | Wrong (✗) | Wrong (✗) | Both failed | **188** | 62.67% | Hard failures (uninformative / ambiguous) |
| **Cohort E** | *Any* | Correct (✓) | Wrong under Shuffle or Reverse | **68** | 22.67% | **Order-Sensitive Cases** (Causal proof) |

### Cohort E Fine-Grained Order-Sensitivity Breakdown
- **Wrong under Temporal Shuffling (Permutation)**: $N = 49$ (16.3%)
- **Wrong under Temporal Reversal (Time Inversion)**: $N = 56$ (18.7%)
- **Wrong under Repeated Midpoint (Static Control)**: $N = 45$ (15.0%)
- **Strict Order Sensitivity (Wrong under BOTH Shuffle and Reverse)**: $N = 37$ (12.3%)
- **Cohort C Overlap**: **62 of 68** sequences (91.2%) come directly from Cohort C (Rescues)!

## 2. Tri-Model Diagnostic Decomposition (Take-Home Assessment & §M5.10)

| Diagnostic Case | Model A (Single) | Model B (MeanPool) | Model C (Temporal GRU) | Sequences | % Split | Primary Scientific Finding |
|---|---|---|---|---:|---:|---|
| **Case 6: Concordant Hard Failure** | Wrong (✗) | Wrong (✗) | Wrong (✗) | **174** | 58.0% | Insoluble across all visual paradigms. |
| **Case 1: Pure Temporal Order Rescue** | Wrong (✗) | Wrong (✗) | Correct (✓) | **81** | 27.0% | Neither single frame nor mean pooling works; order alone rescues. |
| **Case 4: Spatial Accumulation Sufficient** | Wrong (✗) | Correct (✓) | Correct (✓) | **16** | 5.3% | Frame accumulation sufficient without temporal order. |
| **Case 5: Observation Only** | Wrong (✗) | Correct (✓) | Wrong (✗) | **14** | 4.7% | Order-invariant mean pool succeeded; single and GRU failed. |
| **Case 2: Order Resolution** | Correct (✓) | Wrong (✗) | Correct (✓) | **11** | 3.7% | Mean pooling confused signal; temporal order recovered prediction. |
| **Case 3: Temporal Regression** | Correct (✓) | Wrong (✗) | Wrong (✗) | **4** | 1.3% | Temporal modeling corrupted a correct single-frame prediction. |

---
*Generated automatically by `src.experiments.cohorts` under Milestone M5.2.*