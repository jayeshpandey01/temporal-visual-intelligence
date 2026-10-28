# Table: Paired Model Comparison & Statistical Significance (M3.15)
Strict sample-by-sample paired hypothesis testing on identical test sequences ($N=300$)

| Comparison Pair | Scientific Measurement | Model 1 Acc | Model 2 Acc | Paired Δ (95% CI) | Rescued ($n_{01}$) | Degraded ($n_{10}$) | McNemar $p$-value | Significant ($p<0.05$)? |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| **BASELINE** vs **MEANPOOL** | Observation Accumulation ($\Delta_{obs}$) | 5.00% | 10.00% | **+0.050 [+0.007, +0.093]** | 30 | 15 | 0.0369 | Yes |
| **MEANPOOL** vs **GRU** | Temporal Order Gain ($\Delta_{order}$) | 10.00% | 36.00% | **+0.260 [+0.200, +0.317]** | 92 | 14 | <0.0001 | Yes |
| **BASELINE** vs **GRU** | Total Temporal Gain ($\Delta_{total}$) | 5.00% | 36.00% | **+0.310 [+0.253, +0.367]** | 97 | 4 | <0.0001 | Yes |

### Scientific Methodological Guarantees:
1. **Identical Sample Alignment:** All comparisons are strictly paired on the identical 300 test sequences ($S_1, \dots, S_{300}$).
2. **Between-Sample Variance Elimination:** Paired testing eliminates sample difficulty variance ($\text{Cov}(X_1, X_2) > 0$), dramatically tightening confidence intervals.
3. **Direct Order Falsification:** If $\Delta_{order} \le 0$ or McNemar $p \ge 0.05$ between Model B and Model C, chronological temporal modeling is statistically falsified (Assessment §7.D).
