# Milestone M4.8: Temporal Gain Analysis & Horizon Dynamics Report Table

**Date**: 2026-10-28 | **Dataset**: SSv2 Subset ($N=300$ test sequences) | **Seeds**: [42, 43, 44]

## 1. Sequence-Level Temporal Gain Disentanglement

| Gain Metric | Contrast Formulation | Accuracy Impact (Mean ± Std) | Net Sequences Rescued (Mean) | Direction / Certification |
| :--- | :--- | :--- | :--- | :--- |
| **$G_total$** | Full Temporal: GRU vs Single-Frame Baseline | **+26.00 ± 3.74 pp** | +78.0 / 300 sequences | `[STRONG RESCUE]` |
| **$G_obs$** | Spatial Accumulation: MeanPool vs Baseline | **-0.56 ± 4.37 pp** | -1.7 / 300 sequences | `[NEUTRAL / CHANCE]` |
| **$G_order$** | Chronological Order: GRU vs MeanPool | **+26.56 ± 0.79 pp** | +79.7 / 300 sequences | `[STRONG RESCUE]` |
| **$G_parity$** | Capacity Parity: CompactGRU vs MeanPool | **+15.33 ± 2.37 pp** | +46.0 / 300 sequences | `[POSITIVE GAIN]` |
| **$G_scale$** | Architectural Scale: Full GRU vs CompactGRU | **+11.22 ± 2.91 pp** | +33.7 / 300 sequences | `[POSITIVE GAIN]` |

## 2. Observation Horizon Threshold ($K^*$) Distribution (Seed 42)

| Horizon Threshold $K^*$ | Physical Interpretation | Sequences ($N$) | Proportion (%) | Cumulative Resolved (%) |
| :--- | :--- | :---: | :---: | :---: |
| **K*=1 (Instant)** | Instantaneous spatial appearance sufficient | **38** | 12.7% | 12.7% |
| **K*=2 (Early)** | Early velocity / motion onset reveals action | **11** | 3.7% | 16.3% |
| **K*=4 (Mid)** | Mid-trajectory object interaction / contact point | **33** | 11.0% | 27.3% |
| **K*=8 (Late)** | Terminal state outcome / final object placement required | **65** | 21.7% | 49.0% |
| **K*=Inf (Unresolved)** | Unresolved with 8 frames (severe visual ambiguity / fine manipulation) | **153** | 51.0% | — |

## 3. Physical Action Mechanism Breakdown

| Action Mechanism Category | Total Sequences | Baseline Right | GRU Right | Net Gain Sequences | Net Gain (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Container Physics** | 30 | 0 | 26 | **+26** | **+86.7%** |
| **Directional Vector** | 60 | 0 | 6 | **+6** | **+10.0%** |
| **Fine-Grained Manipulation** | 105 | 15 | 36 | **+21** | **+20.0%** |
| **Physical State Transition** | 75 | 0 | 29 | **+29** | **+38.7%** |
| **Rotational Kinematics** | 30 | 0 | 11 | **+11** | **+36.7%** |

## 4. Kinematic Trajectory Statistics Stratified by Resolution Horizon

| Horizon $K^*$ | Mean Stepwise Velocity ($v_{\text{mean}}$) | Max Stepwise Velocity ($v_{\text{max}}$) | Mean Path Length ($L$) | Peak Frame $t^*$ (Median) |
| :--- | :---: | :---: | :---: | :---: |
| **K*=1 (Instant)** | 0.0850 | 0.0986 | 0.5950 | Frame 3 |
| **K*=2 (Early)** | 0.1058 | 0.1399 | 0.7404 | Frame 1 |
| **K*=4 (Mid)** | 0.1110 | 0.1476 | 0.7769 | Frame 6 |
| **K*=8 (Late)** | 0.0953 | 0.1228 | 0.6670 | Frame 5 |
| **K*=Inf (Unresolved)** | 0.0961 | 0.1284 | 0.6727 | Frame 3 |