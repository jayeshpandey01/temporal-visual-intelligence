# Table 1: Baseline and Benchmark Overview
Comparative evaluation across baseline, control, and sequential temporal models (M3.17 & Assessment §4).

| Model | Frames | Order-aware | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Majority | — | No | 5.00% | 0.48% |
| Single frame | 1 | No | 8.78 ± 4.02% | 3.27 ± 3.07% |
| Mean pooling | 8 | No | 8.22 ± 1.54% | 2.16 ± 1.09% |
| GRU | 8 | Yes | 34.78 ± 1.35% | 32.04 ± 0.69% |

### Scientific Notes & Protocol Constraints:
- **Zero Hypothetical Values Rule:** Values are populated strictly from completed, reproducible training runs across deterministic seeds (`[42, 43, 44]`). Unrun configurations remain strictly blank (`—`).
- **Majority Baseline:** Chance-level zero-rule classifier predicting the most frequent class in the training distribution.
- **Single Frame (Model A):** Anchor frame observation evaluating static visual semantics without sequential context (103,700 parameters).
- **Mean Pooling (Model B):** Order-invariant control evaluating visual evidence accumulation across 8 frames without temporal order awareness (103,700 parameters).
- **GRU (Model C):** Sequential recurrent head evaluating chronological state transitions across 8 frames (244,116 parameters).
