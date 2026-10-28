# Milestone M7.3: Headline Findings Reproduction & Reconciliation Table

> **Objective**: Record diagnostic recomputation from currently available checkpoints. This does not prove the original protocol or resolve conflicting historical artifacts.

| ID | Headline Empirical Finding | Historical Reference | Current Checkpoint Recompute | Status |
|:---:|---|---|:---:|:---:|
| **F1** | Model C Canonical Accuracy (K=8) | 34.78 ± 1.10% | **34.78 ± 1.10%** | RECOMPUTED |
| **F2** | Model B MeanPool Accuracy (K=8) | 8.22 ± 1.26% | **8.22 ± 1.26%** | RECOMPUTED |
| **F3** | Model A SingleFrame Baseline Accuracy | 8.78 ± 3.28% | **8.78 ± 3.28%** | RECOMPUTED |
| **F4** | Capacity-Matched Parity Gain (Model D vs Model B) | Checkpoint means differ from M4 summary (parity protocol to reconcile) | **+15.33 pp** | RECOMPUTED |
| **F5** | Temporal Order Destruction Drop (OrderDrop under shuffle) | Historical M4 artifacts report 13.89 pp | **14.11 pp** | DIFFERS FROM CLAIM |
| **F6** | Observation Horizon Scaling: K=1 to K=8 | Historical M4 artifacts report 11.56% to 34.78% (+23.22 pp) | **11.56% to 34.78% (+23.22 pp)** | RECOMPUTED |

**Summary**: This script recomputed values from current checkpoints. Conflicting historical values are reported as differences; matching a stored canonical score does not establish full protocol provenance or fresh independent replication.
