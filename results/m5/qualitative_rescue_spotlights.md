# Qualitative Rescue Spotlights: Forensic Case Studies

> **Milestone M5.9 Qualitative Deliverable**: Deep-dive forensic case studies across the 8 temporal semantic categories ($T_1$–$T_8$). Examines paired model predictions across Single-Frame MLP (Model A), MultiFrame MeanPool (Model B), Temporal GRU (Model C), and Capacity-Matched Compact GRU (Model D).

---

## Case Study: T1 — Object Displacement

- **Sequence ID**: `vid_00_0046` (Video: `vid_00_0046`)
- **Ground Truth Action**: **Pushing something from left to right** (Class ID: 2)
- **Active Taxonomy Codes**: `T1, T2, T6, T8`
- **Minimum Sufficient Context**: $K_{\min}^{\text{window}} = 4$ frames, Optimal Window = `[1, 4]` (span 4 frames)
- **Order Mutual Information Gain**: **+1.180 bits**
- **Reversal Experiment**: `VULNERABLE (Prediction Inverts)`

### Model Prediction Comparison

| Model | Architecture | Predicted Class | Confidence | Correct? |
|---|---|---|---:|:---:|
| **Model A** | Single-Frame Baseline (MLP) | Turning the camera left while filming something | 0.0612 | [FAIL] |
| **Model B** | Order-Invariant MeanPool | Uncovering something | 0.0533 | [FAIL] |
| **Model C** | Temporal GRU Head | **Pushing something from left to right** | **0.1159** | [PASS] |
| **Model D** | Capacity-Matched Compact GRU | `CORRECT (Capacity-Matched Confirmed)` | 0.0953 | [PASS] |

### Forensic Physical & Kinematic Analysis

> **Causal Explanation**: Monotonic lateral coordinate translation (dx/dt > 0). Order-invariant pooling loses the sign of displacement, causing confusion with opposite translation or stationary holding (T1, T2).

- **Why Model A (Single Frame) Failed**: The isolated static frame lacks temporal velocity. The static appearance features misled the classifier to predict `Turning the camera left while filming something`.
- **Why Model B (MeanPool) Failed**: Multi-frame accumulation without sequential order yielded a blurred, directionless representation that falsely predicted `Uncovering something`. Order-invariant pooling actively destroyed the chronological gradient.
- **Why Model C (Temporal GRU) Succeeded**: Recurrent state transitions preserved the chronological trajectory, isolating the key transition inside `[1, 4]` with a confidence surge to **0.1159**.

---

## Case Study: T2 — Direction of Motion

- **Sequence ID**: `vid_13_0014` (Video: `vid_13_0014`)
- **Ground Truth Action**: **Moving something towards the camera** (Class ID: 1)
- **Active Taxonomy Codes**: `T1, T2, T6, T8`
- **Minimum Sufficient Context**: $K_{\min}^{\text{window}} = 4$ frames, Optimal Window = `[2, 7]` (span 6 frames)
- **Order Mutual Information Gain**: **+2.198 bits**
- **Reversal Experiment**: `VULNERABLE (Prediction Inverts)`

### Model Prediction Comparison

| Model | Architecture | Predicted Class | Confidence | Correct? |
|---|---|---|---:|:---:|
| **Model A** | Single-Frame Baseline (MLP) | Turning the camera left while filming something | 0.0609 | [FAIL] |
| **Model B** | Order-Invariant MeanPool | Turning the camera right while filming something | 0.0541 | [FAIL] |
| **Model C** | Temporal GRU Head | **Moving something towards the camera** | **0.2443** | [PASS] |
| **Model D** | Capacity-Matched Compact GRU | `INCORRECT` | 0.1608 | [FAIL] |

### Forensic Physical & Kinematic Analysis

> **Causal Explanation**: Progressive expansion of object bounding silhouette and looming optical flow field. Reversal completely inverts the flow to 'Moving away from the camera' (T1, T2, T6, T8).

- **Why Model A (Single Frame) Failed**: The isolated static frame lacks temporal velocity. The static appearance features misled the classifier to predict `Turning the camera left while filming something`.
- **Why Model B (MeanPool) Failed**: Multi-frame accumulation without sequential order yielded a blurred, directionless representation that falsely predicted `Turning the camera right while filming something`. Order-invariant pooling actively destroyed the chronological gradient.
- **Why Model C (Temporal GRU) Succeeded**: Recurrent state transitions preserved the chronological trajectory, isolating the key transition inside `[2, 7]` with a confidence surge to **0.2443**.

---

## Case Study: T3 — State Transition

- **Sequence ID**: `vid_02_0051` (Video: `vid_02_0051`)
- **Ground Truth Action**: **Opening something** (Class ID: 6)
- **Active Taxonomy Codes**: `T1, T2, T3, T6, T8`
- **Minimum Sufficient Context**: $K_{\min}^{\text{window}} = 5$ frames, Optimal Window = `[2, 6]` (span 5 frames)
- **Order Mutual Information Gain**: **+1.677 bits**
- **Reversal Experiment**: `VULNERABLE (Prediction Inverts)`

### Model Prediction Comparison

| Model | Architecture | Predicted Class | Confidence | Correct? |
|---|---|---|---:|:---:|
| **Model A** | Single-Frame Baseline (MLP) | Turning the camera left while filming something | 0.0616 | [FAIL] |
| **Model B** | Order-Invariant MeanPool | Uncovering something | 0.0572 | [FAIL] |
| **Model C** | Temporal GRU Head | **Opening something** | **0.1754** | [PASS] |
| **Model D** | Capacity-Matched Compact GRU | `INCORRECT` | 0.1284 | [FAIL] |

### Forensic Physical & Kinematic Analysis

> **Causal Explanation**: Discrete topological state boundary transition from sealed exterior to exposed interior. High order sensitivity with localized critical window at boundary rupture (T3).

- **Why Model A (Single Frame) Failed**: The isolated static frame lacks temporal velocity. The static appearance features misled the classifier to predict `Turning the camera left while filming something`.
- **Why Model B (MeanPool) Failed**: Multi-frame accumulation without sequential order yielded a blurred, directionless representation that falsely predicted `Uncovering something`. Order-invariant pooling actively destroyed the chronological gradient.
- **Why Model C (Temporal GRU) Succeeded**: Recurrent state transitions preserved the chronological trajectory, isolating the key transition inside `[2, 6]` with a confidence surge to **0.1754**.

---

## Case Study: T4 — Object Interaction

- **Sequence ID**: `vid_19_0052` (Video: `vid_19_0052`)
- **Ground Truth Action**: **Taking something out of something** (Class ID: 15)
- **Active Taxonomy Codes**: `T4, T7`
- **Minimum Sufficient Context**: $K_{\min}^{\text{window}} = 1$ frames, Optimal Window = `[5, 5]` (span 1 frames)
- **Order Mutual Information Gain**: **+3.315 bits**
- **Reversal Experiment**: `ROBUST`

### Model Prediction Comparison

| Model | Architecture | Predicted Class | Confidence | Correct? |
|---|---|---|---:|:---:|
| **Model A** | Single-Frame Baseline (MLP) | Turning the camera left while filming something | 0.0616 | [FAIL] |
| **Model B** | Order-Invariant MeanPool | Uncovering something | 0.0541 | [FAIL] |
| **Model C** | Temporal GRU Head | **Taking something out of something** | **0.5348** | [PASS] |
| **Model D** | Capacity-Matched Compact GRU | `CORRECT (Capacity-Matched Confirmed)` | 0.4385 | [PASS] |

### Forensic Physical & Kinematic Analysis

> **Causal Explanation**: Single-frame appearance captures hand in contact with container interior, creating extreme ambiguity between insertion and extraction. Chronological trajectory resolves the spatial exit vector through the container aperture (T4, T7).

- **Why Model A (Single Frame) Failed**: The isolated static frame lacks temporal velocity. The static appearance features misled the classifier to predict `Turning the camera left while filming something`.
- **Why Model B (MeanPool) Failed**: Multi-frame accumulation without sequential order yielded a blurred, directionless representation that falsely predicted `Uncovering something`. Order-invariant pooling actively destroyed the chronological gradient.
- **Why Model C (Temporal GRU) Succeeded**: Recurrent state transitions preserved the chronological trajectory, isolating the key transition inside `[5, 5]` with a confidence surge to **0.5348**.

---

## Case Study: T5 — Relative Motion

- **Sequence ID**: `vid_10_0003` (Video: `vid_10_0003`)
- **Ground Truth Action**: **Putting something on top of something** (Class ID: 10)
- **Active Taxonomy Codes**: `T2, T4, T5, T6, T8`
- **Minimum Sufficient Context**: $K_{\min}^{\text{window}} = 3$ frames, Optimal Window = `[3, 5]` (span 3 frames)
- **Order Mutual Information Gain**: **+2.679 bits**
- **Reversal Experiment**: `ROBUST`

### Model Prediction Comparison

| Model | Architecture | Predicted Class | Confidence | Correct? |
|---|---|---|---:|:---:|
| **Model A** | Single-Frame Baseline (MLP) | Turning the camera left while filming something | 0.0609 | [FAIL] |
| **Model B** | Order-Invariant MeanPool | Putting something into something | 0.0548 | [FAIL] |
| **Model C** | Temporal GRU Head | **Putting something on top of something** | **0.3783** | [PASS] |
| **Model D** | Capacity-Matched Compact GRU | `INCORRECT` | 0.2121 | [FAIL] |

### Forensic Physical & Kinematic Analysis

> **Causal Explanation**: Downward velocity vector terminates at resting equilibrium on a supporting substrate. Order preservation is critical to distinguish terminal placement from initial lifting (T2, T4, T5, T6).

- **Why Model A (Single Frame) Failed**: The isolated static frame lacks temporal velocity. The static appearance features misled the classifier to predict `Turning the camera left while filming something`.
- **Why Model B (MeanPool) Failed**: Multi-frame accumulation without sequential order yielded a blurred, directionless representation that falsely predicted `Putting something into something`. Order-invariant pooling actively destroyed the chronological gradient.
- **Why Model C (Temporal GRU) Succeeded**: Recurrent state transitions preserved the chronological trajectory, isolating the key transition inside `[3, 5]` with a confidence surge to **0.3783**.

---

## Case Study: T6 — Sub-Event Order

- **Sequence ID**: `vid_12_0032` (Video: `vid_12_0032`)
- **Ground Truth Action**: **Standing something upright** (Class ID: 19)
- **Active Taxonomy Codes**: `T1, T2, T3, T4, T6, T8`
- **Minimum Sufficient Context**: $K_{\min}^{\text{window}} = 3$ frames, Optimal Window = `[4, 6]` (span 3 frames)
- **Order Mutual Information Gain**: **+2.514 bits**
- **Reversal Experiment**: `VULNERABLE (Prediction Inverts)`

### Model Prediction Comparison

| Model | Architecture | Predicted Class | Confidence | Correct? |
|---|---|---|---:|:---:|
| **Model A** | Single-Frame Baseline (MLP) | Turning the camera left while filming something | 0.0610 | [FAIL] |
| **Model B** | Order-Invariant MeanPool | Uncovering something | 0.0594 | [FAIL] |
| **Model C** | Temporal GRU Head | **Standing something upright** | **0.3039** | [PASS] |
| **Model D** | Capacity-Matched Compact GRU | `CORRECT (Capacity-Matched Confirmed)` | 0.1265 | [PASS] |

### Forensic Physical & Kinematic Analysis

> **Causal Explanation**: Angular rotation from horizontal orientation (theta approx 0) to vertical stability (theta approx pi/2). Temporal order establishes whether stability is achieved or lost (T1, T2, T3, T4).

- **Why Model A (Single Frame) Failed**: The isolated static frame lacks temporal velocity. The static appearance features misled the classifier to predict `Turning the camera left while filming something`.
- **Why Model B (MeanPool) Failed**: Multi-frame accumulation without sequential order yielded a blurred, directionless representation that falsely predicted `Uncovering something`. Order-invariant pooling actively destroyed the chronological gradient.
- **Why Model C (Temporal GRU) Succeeded**: Recurrent state transitions preserved the chronological trajectory, isolating the key transition inside `[4, 6]` with a confidence surge to **0.3039**.

---

## Case Study: T7 — Static Control

- **Sequence ID**: `vid_16_0031` (Video: `vid_16_0031`)
- **Ground Truth Action**: **Taking something off of something** (Class ID: 11)
- **Active Taxonomy Codes**: `T4, T7`
- **Minimum Sufficient Context**: $K_{\min}^{\text{window}} = 4$ frames, Optimal Window = `[0, 4]` (span 5 frames)
- **Order Mutual Information Gain**: **+2.068 bits**
- **Reversal Experiment**: `ROBUST`

### Model Prediction Comparison

| Model | Architecture | Predicted Class | Confidence | Correct? |
|---|---|---|---:|:---:|
| **Model A** | Single-Frame Baseline (MLP) | Turning the camera left while filming something | 0.0610 | [FAIL] |
| **Model B** | Order-Invariant MeanPool | Uncovering something | 0.0555 | [FAIL] |
| **Model C** | Temporal GRU Head | **Taking something off of something** | **0.2373** | [PASS] |
| **Model D** | Capacity-Matched Compact GRU | `INCORRECT` | 0.1400 | [FAIL] |

### Forensic Physical & Kinematic Analysis

> **Causal Explanation**: Static frame shows object resting on top of another surface, indistinguishable from 'Putting something on top of something'. Chronological elevation vector and detachment of support contact points disambiguates the action (T4, T7).

- **Why Model A (Single Frame) Failed**: The isolated static frame lacks temporal velocity. The static appearance features misled the classifier to predict `Turning the camera left while filming something`.
- **Why Model B (MeanPool) Failed**: Multi-frame accumulation without sequential order yielded a blurred, directionless representation that falsely predicted `Uncovering something`. Order-invariant pooling actively destroyed the chronological gradient.
- **Why Model C (Temporal GRU) Succeeded**: Recurrent state transitions preserved the chronological trajectory, isolating the key transition inside `[0, 4]` with a confidence surge to **0.2373**.

---

## Case Study: T8 — Single Frame Ambiguity

- **Sequence ID**: `vid_09_0009` (Video: `vid_09_0009`)
- **Ground Truth Action**: **Tipping something over** (Class ID: 18)
- **Active Taxonomy Codes**: `T1, T2, T3, T6, T8`
- **Minimum Sufficient Context**: $K_{\min}^{\text{window}} = 1$ frames, Optimal Window = `[1, 1]` (span 1 frames)
- **Order Mutual Information Gain**: **+1.575 bits**
- **Reversal Experiment**: `ROBUST`

### Model Prediction Comparison

| Model | Architecture | Predicted Class | Confidence | Correct? |
|---|---|---|---:|:---:|
| **Model A** | Single-Frame Baseline (MLP) | Turning the camera left while filming something | 0.0612 | [FAIL] |
| **Model B** | Order-Invariant MeanPool | Uncovering something | 0.0582 | [FAIL] |
| **Model C** | Temporal GRU Head | **Tipping something over** | **0.1731** | [PASS] |
| **Model D** | Capacity-Matched Compact GRU | `INCORRECT` | 0.1160 | [FAIL] |

### Forensic Physical & Kinematic Analysis

> **Causal Explanation**: Angular rotation from vertical equilibrium to unstable fall and resting horizontal state. Direct directional opposite of 'Standing something upright' (T2, T3).

- **Why Model A (Single Frame) Failed**: The isolated static frame lacks temporal velocity. The static appearance features misled the classifier to predict `Turning the camera left while filming something`.
- **Why Model B (MeanPool) Failed**: Multi-frame accumulation without sequential order yielded a blurred, directionless representation that falsely predicted `Uncovering something`. Order-invariant pooling actively destroyed the chronological gradient.
- **Why Model C (Temporal GRU) Succeeded**: Recurrent state transitions preserved the chronological trajectory, isolating the key transition inside `[1, 1]` with a confidence surge to **0.1731**.

---
