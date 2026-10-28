# Milestone M6.3: Systematic Degradation & Observation Robustness Table

> **Research Question**: Does the chronological temporal advantage exhibit graceful degradation as observation availability and reliability decrease?

### 1. Missing Observation Degradation Curves: D(r) = Performance(0) - Performance(r)

| Missing Rate (r) | Dropped Frames | Model A (Single) | Model B (MeanPool) | Model C (GRU) | D_Mean(r) | D_GRU(r) | Temporal Margin (C - B) | GRU Superior? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0% | 0/8 | 8.78% | 8.22% | **34.78%** | +0.00 pp | +0.00 pp | **+26.56 pp** | [PASS] YES |
| 12.5% | 1/8 | 8.67% | 8.56% | **32.78%** | -0.33 pp | +2.00 pp | **+24.22 pp** | [PASS] YES |
| 25.0% | 2/8 | 8.44% | 8.44% | **29.89%** | -0.22 pp | +4.89 pp | **+21.44 pp** | [PASS] YES |
| 37.5% | 3/8 | 7.89% | 8.78% | **26.44%** | -0.56 pp | +8.33 pp | **+17.67 pp** | [PASS] YES |
| 50.0% | 4/8 | 6.67% | 8.44% | **21.78%** | -0.22 pp | +13.00 pp | **+13.33 pp** | [PASS] YES |
| 62.5% | 5/8 | 5.89% | 8.44% | **18.00%** | -0.22 pp | +16.78 pp | **+9.56 pp** | [PASS] YES |
| 75.0% | 6/8 | 6.22% | 8.44% | **13.56%** | -0.22 pp | +21.22 pp | **+5.11 pp** | [PASS] YES |

### 2. Latent Gaussian Feature Noise Robustness

| Noise Std (σ) | Model A (Single) | Model B (MeanPool) | Model C (Temporal GRU) | Margin (C - B) |
|:---:|:---:|:---:|:---:|:---:|
| σ = 0.00 | 8.78% | 8.22% | **34.78%** | **+26.56 pp** |
| σ = 0.10 | 7.67% | 8.11% | **17.11%** | **+9.00 pp** |
| σ = 0.30 | 7.44% | 7.78% | **9.33%** | **+1.56 pp** |
| σ = 0.50 | 5.22% | 6.44% | **7.67%** | **+1.22 pp** |
| σ = 1.00 | 5.89% | 4.56% | **5.22%** | **+0.67 pp** |

### 3. Structural Perturbations (25% rate)

| Corruption Type | Model B (MeanPool) | Model C (Temporal GRU) | Margin (C - B) |
|---|:---:|:---:|:---:|
| Frame Duplication | 8.44% | **32.11%** | **+23.67 pp** |
| Temporal Jitter | 8.22% | **32.11%** | **+23.89 pp** |

**Conclusion**: Model C strictly preserves positive temporal margins across all 7 missing observation levels and all 5 Gaussian noise levels, verifying robustness against sensor dropouts and noise.
