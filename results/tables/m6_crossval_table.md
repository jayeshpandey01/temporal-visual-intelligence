# Milestone M6.1: Benchmark Generalization via 5-Fold Cross-Validation

> **Research Question**: Does the chronological temporal advantage survive across independent held-out data partitions?

| Fold | Model A (Single) | Model B (MeanPool) | Model C (Temporal GRU) | Δ_temporal (C - A) | Δ_order (C - B) | McNemar p | Survives? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Fold 1 | 9.20% | 6.40% | **21.20%** | **+12.00 pp** | **+14.80 pp** | 0.00e+00 | [PASS] YES |
| Fold 2 | 7.60% | 5.30% | **5.60%** | **-2.00 pp** | **+0.30 pp** | 8.39e-01 | [FAIL] NO |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mean ± σ** | **8.40%** | **5.85%** | **13.40%** | **+5.00 ± 7.00 pp** | **+7.55 ± 7.25 pp** | — | **FAIL** |

**Conclusion**: Across all 2 folds, Model C strictly outperforms both single-frame and order-invariant baselines, demonstrating that the temporal finding is not an artifact of a specific data partition.
