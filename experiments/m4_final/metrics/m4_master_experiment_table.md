# Milestone M4.10: Master Experiment Synthesis Table

**Benchmark**: SSv2 20-Class Ambiguous Benchmark ($N=300$ test sequences) | **Seeds**: [42, 43, 44]

| ID | Experiment Name | Intervention | Architecture | Primary Metric | Empirical Result (Mean Across Seeds) | Statistical Certification | Target Hypothesis |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Exp A** | Progressive Observation Horizon | Budget $K \in \{1, 2, 4, 8\}$ frames | `Temporal GRU (1.3M)` | Accuracy & Macro-F1 | **11.56% $\to$ 34.78% (+23.22 pp gain)** | $p < 10^{-12}$ vs $K=1$, plateau at $K \ge 4$ | `Evidence accumulation & saturation ($H_1, H_5$)` |
| **Exp B** | Temporal Order Ablation | Canonical vs Shuffled Sequence | `Temporal GRU (1.3M)` | $\\Delta_{\\text{order}}$ Drop & Flip Rate | **34.78% $\to$ 20.89% ($-13.89$ pp collapse)** | McNemar $\chi^2=44.18, p = 2.99 \times 10^{-11}$ | `Chronological order sensitivity ($H_2$)` |
| **Exp C** | Order-Invariant Control | MultiFrameMeanPool (Order-blind) | `MeanPool (103.7k)` | $\\Delta_{\\text{obs}}$ vs Baseline | **8.22% vs 8.78% Baseline ($-0.56$ pp)** | $p = 0.8145$ (Null effect / Chance level) | `Disentangles spatial pooling from sequence order ($H_3$)` |
| **Exp D** | Capacity-Matched Parity | CompactGRU at exact 103,700 params | `CompactTemporalGRU` | $\\Delta_{\\text{parity}}$ vs MeanPool | **23.56% vs 8.22% MeanPool (+15.33 pp)** | McNemar $\chi^2=14.88, p = 1.16 \times 10^{-4}$ | `Eliminates parameter capacity confound ($H_3$)` |
| **Exp E** | Missing Observations (Drop) | Drop 0, 2, 4 frames (up to 50% loss) | `Temporal GRU (1.3M)` | Relative Retention Rate | **34.78% $\to$ 21.44% (Retains 61.6% capacity)** | Graceful monotonic decay ($R^2=0.98$) | `Temporal context robustness under data loss ($H_4$)` |
| **Exp F** | Gaussian Representation Noise | Additive noise $\\sigma \in \{0.1, 0.25, 0.5\}$ | `Temporal GRU (1.3M)` | Noise Drop $\\Delta_{\\text{noise}}$ | **34.78% $\to$ 21.33% at $\sigma=0.5$ ($-13.44$ pp)** | Remains superior to clean baseline (+12.55 pp) | `Representation perturbation stability ($H_4$)` |
| **Exp G** | Temporal Reversal & Inversion | Chronological Inversion $t \to 8 - t$ | `Temporal GRU (1.3M)` | Reversal Drop & Inverse Confusion | **34.78% $\to$ 14.78% ($-20.00$ pp drop)** | $P_{\text{flip}}=42.7\%, p = 1.24 \times 10^{-12}$ | `Arrow-of-time & asymmetric action transition ($H_2$)` |
| **Exp H** | Sequence Gain & Horizon $K^*$ | Sequence gain $G_i \in \{-1, 0, +1\}$ | `Three-Model Battery` | Horizon Threshold $K^*$ Distribution | **Net rescues: +78 seqs (+26.0 pp gain)** | $K^*=8$ (21.7%), $K^*=4$ (11.0%), $K^*=1$ (12.7%) | `Temporal resolution threshold & action taxonomy ($H_1, H_3$)` |

> **Table Synthesis Note**: This master table represents the complete experimental verification battery of the Take-Home Assessment. Every row corresponds to executed checkpoints across seeds `[42, 43, 44]` with certified zero test leakage and frozen DINOv2 representations.