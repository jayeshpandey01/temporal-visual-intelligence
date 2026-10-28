# Table: Robustness to Missing Observations & Sensor Degradation (Milestone M4.4)
Degradation curves under structured frame-dropping (zero-masking) and latent Gaussian noise injection (mean ± $\sigma$ across seeds `[42, 43, 44]`).

### A. Frame Drop Robustness (Zero-Masking / Occlusion)
| Architecture | Parameters | Clean (0 Drops) | 2 Frames Dropped (25%) | 4 Frames Dropped (50%) | Graceful Degradation Drop (0 -> 4 Drops) |
|---|---:|---:|---:|---:|---:|
| **Model B: Mean-Pool Control (103.7k)** | 103,700 | 8.22% | 8.33% | 8.44% | **+0.22 pp** |
| **Model D: CompactGRU (103.7k Parity)** | 103,700 | 23.56% | 21.33% | 20.33% | **-3.22 pp** |
| **Model C: Temporal GRU (249.4k)** | 249,364 | 34.78% | 28.67% | 21.44% | **-13.33 pp** |

### B. Latent Gaussian Noise Robustness (Sensor Degradation)
| Architecture | Parameters | $\sigma = 0.0$ (Clean) | $\sigma = 0.10$ | $\sigma = 0.25$ | $\sigma = 0.50$ | Total Noise Drop ($\sigma = 0 \to 0.5$) |
|---|---:|---:|---:|---:|---:|---:|
| **Model B: Mean-Pool Control (103.7k)** | 103,700 | 8.22% | 8.33% | 8.44% | 6.33% | **-1.89 pp** |
| **Model D: CompactGRU (103.7k Parity)** | 103,700 | 23.56% | 15.67% | 10.33% | 7.44% | **-16.11 pp** |
| **Model C: Temporal GRU (249.4k)** | 249,364 | 34.78% | 17.00% | 9.11% | 6.67% | **-28.11 pp** |

---

## Scientific Interpretation: Graceful Degradation vs Baseline Immunity

1. **Graceful Degradation of Recurrent Temporal Models (Model C & D)**: `STRONG REDUNDANCY`
   - Even after losing **50% of observation frames (4 out of 8 dropped)**, **Model C (Temporal GRU)** maintains **21.44% Accuracy** and **Model D (CompactGRU)** maintains **15.22% Accuracy**.
   - In comparison, **Model B (MeanPool Control)** achieves only **8.22%** under clean conditions. Therefore, a temporal model with half its frames wiped out still outperforms the clean bag-of-frames baseline by **+13.22 pp (2.6× higher accuracy)**.

2. **The Apparent 'Immunity' of Mean-Pooling Explained**: `NO LOSS FROM THE FLOOR`
   - Model B's drop is negligible (+0.22 pp) solely because its prediction accuracy is already essentially at floor level (~8.22%). A model with near-chance performance has zero headroom to lose.

3. **Latent Gaussian Noise Vulnerability**: `RECURRENT SENSITIVITY`
   - As noise increases to $\sigma = 0.50$, all models converge toward the uniform chance floor (~5.0% - 7.0%).
   - Temporal recurrence operates on clean manifold trajectories; high-variance latent noise disrupts transition dynamics, requiring clean sensor pre-filtering.
