# Milestone M4.11: Hypothesis → Experiment Mapping Table

**Purpose**: Directly connects core scientific hypotheses (H1–H5) with controlled experimental interventions and statistical findings.

| Hypothesis | Core Theoretical Premise | Primary Experiment | Empirical Quantified Evidence | Research Status |
| :--- | :--- | :--- | :--- | :---: |
| **H1: More observations help** | Additional sequential visual evidence resolves ambiguous frame-level states. | `Exp A (Progressive Horizon $K \in \{1, 2, 4, 8\}$)` | Accuracy grows monotonically from 11.56% ($K=1$) to 34.78% ($K=8$) (+23.22 pp gain, $p < 10^{-12}$). | `[CONFIRMED]` |
| **H2: Chronological order matters** | Physical transformations (e.g., Opening vs Closing) require strict temporal ordering. | `Exp B (Temporal Shuffle) & Exp G (Temporal Reversal)` | Shuffling causes a -13.89 pp collapse ($p = 2.99 \times 10^{-11}$). Reversal causes a -20.00 pp drop with 42.7% prediction flips. | `[CONFIRMED]` |
| **H3: Temporal modeling adds value over pooling** | Performance gains stem from temporal dynamics, not mere multi-frame spatial accumulation or capacity. | `Exp C (MeanPool Control) & Exp D (CompactGRU Parity)` | MeanPool achieves 8.22% (-0.56 pp vs baseline). CompactGRU at exact 103,700 capacity achieves 23.56% (+15.33 pp over MeanPool, $p = 1.16 \times 10^{-4}$). | `[CONFIRMED]` |
| **H4: Temporal context improves robustness** | Recurrent state representations buffer against corrupted or missing observations. | `Exp E (Frame Drops) & Exp F (Gaussian Noise)` | Under 50% missing data (4/8 frames dropped), GRU retains 21.44% accuracy (2.4x higher than clean baseline). Under severe noise ($\sigma=0.5$), GRU retains 21.33%. | `[CONFIRMED]` |
| **H5: Information saturates** | Marginal utility of additional frames diminishes as physical action transition completes. | `Exp A (Progressive Curve Derivative) & Exp H (Horizon $K^*$)` | Growth slows from +11.78 pp ($K=2 \to 4$) to +5.44 pp ($K=4 \to 8$). 49.0% of all resolvable sequences are fully resolved by $K=8$. | `[CONFIRMED]` |

> **Scientific Alignment**: All 5 hypotheses formulated in `docs/milestone/M-4.md` (§M4.11) and `research_paper.md` are empirically validated by controlled multi-seed experiments.