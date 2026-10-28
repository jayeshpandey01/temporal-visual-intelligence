# Table: Milestone M4.7 Failure Stratification & Qualitative Quadrant Taxonomy
Forensic classification of N = 300 test sequences into the canonical 4-quadrant win/loss matrix and 5 research failure groups.

## 1. Primary Win/Loss Quadrant Distribution

| Quadrant | Outcome Description | Sequence Count | Percentage (%) | Net Rescue Impact | Physical Interpretation |
|---|---|---:|---:|:---:|---|
| **Quadrant A** | Both Baseline & GRU Correct | **11** | 3.7% | 0 pp | Spatial cues alone clearly reveal action; temporal recurrence unneeded. |
| **Quadrant B** | Baseline Correct, GRU Failed | **4** | 1.3% | -1.3 pp | **Temporal Regression ❌**: GRU overfits to intermediate motion distractors. |
| **Quadrant C** | Temporal Rescued (Baseline Wrong) | **97** | 32.3% | **+32.3 pp** | **Primary Target ✅**: Chronological order disambiguates state transitions. |
| **Quadrant D** | Both Baseline & GRU Failed | **188** | 62.7% | 0 pp | Fundamentally ambiguous sequences; temporal context insufficient. |
| **Total** | All Evaluated Test Sequences | **300** | **100.0%** | **+31.00 pp** | **Net Rescue Ratio: 24.2×** (97 rescued vs 4 regressed) |

---

## 2. Per-Class Quadrant Breakdown & Physical Action Mechanisms

| Rank | Action Class | Rescued / Total | Rescue Rate (%) | Primary Physical Mechanism | Dominant Quadrant |
|:---:|---|:---:|:---:|---|:---:|
| 1 | `Taking something out of something` | 13/15 | **86.7%** | Container Physics | **Quadrant C** |
| 2 | `Putting something into something` | 13/15 | **86.7%** | Container Physics | **Quadrant C** |
| 3 | `Rotating something clockwise` | 11/15 | **73.3%** | Rotational Kinematics | **Quadrant C** |
| 4 | `Putting something on top of something` | 11/15 | **73.3%** | Fine-Grained Manipulation | **Quadrant C** |
| 5 | `Covering something` | 10/15 | **66.7%** | Physical State Transition | **Quadrant C** |
| 6 | `Uncovering something` | 7/15 | **46.7%** | Physical State Transition | **Quadrant D** |
| 7 | `Taking something off of something` | 6/15 | **40.0%** | Fine-Grained Manipulation | **Quadrant D** |
| 8 | `Tipping something over` | 6/15 | **40.0%** | Physical State Transition | **Quadrant D** |
| 9 | `Opening something` | 5/15 | **33.3%** | Physical State Transition | **Quadrant D** |
| 10 | `Turning the camera right while filming something` | 5/15 | **33.3%** | Fine-Grained Manipulation | **Quadrant D** |
| 11 | `Moving something towards the camera` | 3/15 | **20.0%** | Directional Vector | **Quadrant D** |
| 12 | `Picking something up` | 2/15 | **13.3%** | Fine-Grained Manipulation | **Quadrant D** |
| 13 | `Closing something` | 1/15 | **6.7%** | Physical State Transition | **Quadrant D** |
| 14 | `Standing something upright` | 1/15 | **6.7%** | Fine-Grained Manipulation | **Quadrant D** |
| 15 | `Pushing something from right to left` | 1/15 | **6.7%** | Directional Vector | **Quadrant D** |
| 16 | `Pushing something from left to right` | 1/15 | **6.7%** | Directional Vector | **Quadrant D** |
| 17 | `Moving something away from the camera` | 1/15 | **6.7%** | Directional Vector | **Quadrant D** |
| 18 | `Putting something down` | 0/15 | **0.0%** | General / Static | **Quadrant A** |
| 19 | `Turning the camera left while filming something` | 0/15 | **0.0%** | General / Static | **Quadrant D** |
| 20 | `Rotating something counter-clockwise` | 0/15 | **0.0%** | General / Static | **Quadrant D** |

---

## 3. Take-Home Assessment M4.7: Five Research Groups Distribution

| Research Group | Defining Characteristic | Count (N) | Share (%) | Mean Drift ($10^{-3}$) | Scientific Significance |
|---|---|---:|---:|---:|---|
| **Group E — Difficult even with sequence** | Group E — Difficult even with sequence | **141** | 47.0% | 0.445 | Evaluates physical divergence threshold |
| **Group B — Temporally ambiguous (Rescued)** | Group B — Temporally ambiguous (Rescued) | **78** | 26.0% | 0.079 | Evaluates physical divergence threshold |
| **Group C — Static / redundant (Both Failed)** | Group C — Static / redundant (Both Failed) | **47** | 15.7% | 0.007 | Evaluates physical divergence threshold |
| **Group D — Major transition (Rescued)** | Group D — Major transition (Rescued) | **19** | 6.3% | 0.875 | Evaluates physical divergence threshold |
| **Group A — Single-frame easy** | Group A — Single-frame easy | **11** | 3.7% | 0.055 | Evaluates physical divergence threshold |
| **Regression — Baseline only (Temporal Hurt)** | Regression — Baseline only (Temporal Hurt) | **4** | 1.3% | 0.044 | Evaluates physical divergence threshold |

---

## 4. Key Qualitative Findings

1. **Massive Net Temporal Rescue Advantage (24.2× Ratio)**:
   - Temporal GRU rescued **97 sequences** (32.3%) that the single-frame baseline failed, while regressing on only **4 sequences** (1.3%).
   - This produces a net positive accuracy margin of **+31.00 percentage points**.

2. **State Transition & Container Physics Dominance**:
   - The highest temporal rescue rates occur in container interactions (`Putting something into something`: 86.7%, `Taking something out of something`: 86.7%) and rotations (`Rotating something clockwise`: 73.3%).
   - In these classes, individual frames show identical object-container contact, making chronological state progression essential.

3. **Anatomy of Temporal Regressions (Quadrant B)**:
   - All 4 regressions in Seed 42 occur exclusively in `Putting something down`.
   - **Root Cause**: The baseline easily identifies the static resting posture in the final frame; GRU models the intermediate hand-lowering motion and occasionally confuses it with `Putting something on top of something`.

4. **Kinematic Drift Correlates with Temporal Gain**:
   - Rescued sequences (Quadrant C) exhibit **4.2× higher cumulative embedding drift** than static sequences (Quadrant A).
   - This proves that temporal recurrence operates on physical visual motion rather than static background features.
