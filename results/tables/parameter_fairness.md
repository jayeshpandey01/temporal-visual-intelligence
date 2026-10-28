# Table: Parameter Fairness and Capacity Allocation Audit (M3.10)
Controlled Model Capacity Allocation Across Baseline, Control, and Temporal Heads (Take-Home Assessment §4)

| Architecture | Scientific Role | Trainable Params | Frozen Params | Total Params | Ratio vs. Baseline | Trainable % | Order-Aware? |
|---|---|---:|---:|---:|---:|---:|:---:|
| **Model A (Single-Frame MLP)** | Baseline | 103,700 | 21,800,000 | 21,903,700 | **1.00×** | 0.47% | No |
| **Model B (Mean-Pooling Control)** | Control | 103,700 | 21,800,000 | 21,903,700 | **1.00×** | 0.47% | No |
| **Model C (Temporal GRU, H=128)** | Primary Probe (M3.9) | 249,364 | 21,800,000 | 22,049,364 | **2.40×** | 1.13% | Yes (Chronological) |
| **Model C (Temporal GRU, H=256)** | Expanded Probe | 498,452 | 21,800,000 | 22,298,452 | **4.81×** | 2.24% | Yes (Chronological) |
| **Model D (Temporal Transformer)** | Ablation Head | 1,691,668 | 21,800,000 | 23,491,668 | **16.31×** | 7.20% | Yes (Chronological) |

### Scientific Fairness Guarantees:
1. **Baseline-to-Control Parity (Model A vs Model B):** Exactly `103,700` trainable parameters each (`1.00×`). Zero parameter capacity confounding in measuring observation gain (Δ_obs).
2. **Lightweight Temporal Model (Model C, H=128):** `249,364` parameters (`2.4×` baseline), representing only `1.13%` of the total visual perception system.
3. **Visual Representation Constancy:** The DINOv2 ViT-S/14 visual encoder (21.8M parameters) remains frozen across all runs, guaranteeing that the visual representation space is constant.
4. **Compliance with Take-Home Assessment §4:** The primary GRU temporal head is kept strictly lightweight (<2.5× baseline), ensuring empirical gains reflect temporal sequence modeling rather than model bloat.
