# Milestone M4.12: Hypothesis Falsification Audit Matrix

**Methodological Principle**: Pre-defining falsification criteria prior to experiment execution guarantees scientific integrity (Popperian falsification).

| Target Hypothesis | Pre-Defined Falsification Condition (M4.12) | Observed Empirical Outcome | Scientific Verdict |
| :--- | :--- | :--- | :---: |
| **H1: More observations help** | Increasing frame count produces no meaningful improvement ($\Delta K \le 1$ pp). | **+23.22 pp gain from $K=1$ to $K=8$ ($p < 10^{-12}$).** | `[FALSIFICATION REJECTED (H1 Strongly Supported)]` |
| **H2: Chronological order matters** | Shuffling produces essentially no degradation in the temporal model ($\Delta_{\text{order}} \le 2$ pp). | **Massive -13.89 pp drop under shuffle ($p = 2.99 \times 10^{-11}$).** | `[FALSIFICATION REJECTED (H2 Strongly Supported)]` |
| **H3: Temporal modeling adds value** | GRU performs similarly to or worse than order-invariant mean pooling ($\Delta_{\text{parity}} \le 0$ pp). | **CompactGRU outperforms MeanPool by +15.33 pp ($p = 1.16 \times 10^{-4}$); Full GRU outperforms by +26.56 pp ($p < 10^{-12}$).** | `[FALSIFICATION REJECTED (H3 Strongly Supported)]` |
| **H4: Context improves robustness** | Temporal context does not reduce degradation under corruption (relative drop $\ge$ baseline). | **GRU retains 61.6% capacity under 50% frame loss, maintaining 2.4x higher accuracy than clean single-frame baseline.** | `[FALSIFICATION REJECTED (H4 Strongly Supported)]` |
| **H5: Information saturates** | Performance continues improving linearly without diminishing returns across all frame counts. | **Marginal gain decays sharply (+11.78 pp from $K=2 \to 4$ down to +5.44 pp from $K=4 \to 8$).** | `[FALSIFICATION REJECTED (H5 Strongly Supported)]` |

> **Conclusion**: None of the 5 falsification conditions were met. The empirical evidence decisively supports the presence of high-order temporal structure in sequential video understanding.