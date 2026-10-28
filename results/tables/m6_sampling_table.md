# Milestone M6.2: Temporal Sampling Robustness Table

> **Research Question**: Does the chronological temporal advantage depend on a specific frame-sampling pattern, or does it persist under alternative observation schemes?

| Sampling Strategy | Model A (Single) | Model B (MeanPool) | Model C (Temporal GRU) | Δ_temporal (C - A) | Δ_order (C - B) | McNemar p | Robust? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Uniform (8 frames, full horizon) | 8.78 ± 3.28% | 8.22 ± 1.26% | **34.78 ± 1.10%** | **+26.00 pp** | **+26.56 pp** | 3.33e-16 | [PASS] YES |
| Sparse stride-2 [0, 2, 4, 6] (4 frames) | 8.78 ± 3.28% | 8.56 ± 1.29% | **20.33 ± 2.49%** | **+11.56 pp** | **+11.78 pp** | 3.03e-07 | [PASS] YES |
| Dense prefix [0, 1, 2, 3] (4 frames) | 8.56 ± 3.33% | 8.67 ± 0.94% | **16.89 ± 2.06%** | **+8.33 pp** | **+8.22 pp** | 1.40e-04 | [PASS] YES |
| Random subset (4 frames, seeded) | 8.78 ± 3.45% | 8.33 ± 1.19% | **21.78 ± 2.53%** | **+13.00 pp** | **+13.44 pp** | 4.84e-08 | [PASS] YES |
| Jittered ±1 displacement (8 frames) | 8.89 ± 3.42% | 8.56 ± 1.29% | **33.11 ± 0.16%** | **+24.22 pp** | **+24.56 pp** | 6.33e-15 | [PASS] YES |

**Conclusion**: The chronological temporal advantage survives across **all 5** sampling strategies, verifying that the temporal advantage is not an artifact of uniform spacing.
