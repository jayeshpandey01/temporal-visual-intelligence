# Table: High-Accuracy Model Escalation Comparison
Empirical benchmarks across model capacity, temporal mechanisms, and multi-sensor upgrades.

| Model Hierarchy | Observation | Order-Aware? | Trainable Params | Test Accuracy (Mean ± σ) | Macro-F1 (Mean ± σ) | Architectural Mechanism |
|---|:---:|:---:|---:|---:|---:|---|
| **Chance / Majority Baseline** | — | No | 0 | 5.00% | 0.48% | Random guessing on 20 balanced classes |
| Model A: Single-Frame MLP | 1 Frames | No | 103,700 | **8.78% ± 4.02%** | 3.27% ± 3.07% | Baseline: static visual features without motion |
| Model B: Mean-Pooling Control | 8 Frames | No (Symmetric) | 103,700 | **8.22% ± 1.54%** | 2.16% ± 1.09% | Evidence accumulation control without chronological order |
| Model D: Compact GRU | 8 Frames | Yes (Causal) | 103,700 | **23.56% ± 2.69%** | 18.00% ± 4.15% | Exact parameter capacity parity with baseline |
| Model C: Chronological GRU | 8 Frames | Yes (Causal) | 249,364 | **34.78% ± 1.35%** | 32.04% ± 0.69% | Diagnostic sequential head (suffers recency forgetting) |
| **Level 1: Attentive Bi-GRU + Contrastive** | 8 Frames | Yes (Bidirectional) | 482,580 | **46.11% ± 1.07%** | 43.24% ± 1.08% | Temporal attention pooling + InfoNCE directional loss |
| **Level 2: TimeSformer (Space-Time Patches)** | 8 Frames | Yes (Space-Time) | 772,116 | **99.89% ± 0.19%** | 99.89% ± 0.19% | Divided Space-Time Attention across spatial patches + temporal recurrence |
| **Level 3: Multimodal XR Fusion** | 8 Frames | Yes (Cross-Attention) | 356,884 | **95.22% ± 0.39%** | 93.88% ± 0.96% | RGB + 6-DoF IMU (ego-motion fix) + Metric Depth LiDAR + 3D Hand Pose |

### Key Quantitative Findings:
- **Level 1 Empirical Gain over Baseline GRU:** `+11.33 pp` (from 34.78% to 46.11%)
- **Recency Decay Elimination:** Attention pooling enables dynamic weighting on pivotal contact frames (e.g. frame 4 touch points) rather than suffering from terminal frame memory decay.
- **Directional Disentanglement:** Directional contrastive loss directly minimizes mirror-confusion between opposite pairs (`Pushing Left` vs `Pushing Right`).
- **Real-Time Edge Budget:** Multimodal fusion inference is empirically measured at **0.22 ms** on local CPU/MPS hardware, well within the 5.0 ms spatial computing budget.
