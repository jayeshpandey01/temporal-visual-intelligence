# Milestone 2: Visual Representation & Embedding Pipeline Analysis

**Author:** Senior ML Research Engineer / Principal Scientist  
**Date:** October 28, 2026  
**Project:** Temporal Visual Intelligence  
**Milestone:** M2 Deliverable (`docs/milestone/M-2.md` §M2.1 – §M2.16)  
**Artifact Dependencies:** `embeddings/manifest.json`, `results/representation/cache_validation_report.json`, `results/representation/cosine_similarity.csv`, `results/representation/embedding_statistics.csv`

---

## Executive Summary

Milestone 2 bridges the transition from raw video sequences to fixed, high-throughput numerical manifolds. Its primary objective is to answer:
> **How do we convert each observation into a representation that allows us to study temporal information without confounding the experiment with a complicated vision model?**

By enforcing an immutable data interface contract (`src/embeddings/contract.py`), a frozen self-supervised vision encoder (DINOv2 `dinov2_vits14`), a deterministic sampling protocol, and persistent array caching (`src/embeddings/cache.py`), we ensure that downstream model performance deltas in Milestone 3 (Single-Frame MLP vs. Multi-Frame Mean-Pooling vs. Temporal GRU) reflect purely temporal aggregation mechanisms rather than visual covariate shift or fine-tuning artifacts.

---

## 1. The Experimental Interface Contract (M2.1)

Downstream sequence models must evaluate strictly identical visual representations. We formalize this via two dataclass contracts:

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              M2.1 FROZEN EXPERIMENTAL INTERFACE CONTRACT                               │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ INPUT CONTRACT (SequenceInput):                                                                        │
│   • sequence_id: str                      - Deterministic unique sequence identifier                   │
│   • frame_paths: List[Path]               - Chronologically ordered file paths to extracted frames     │
│   • label: str                            - Categorical action class label                             │
│   • frame_indices: List[int]              - Strictly increasing natural frame indices in source video  │
│   • metadata: Dict[str, Any]              - Frame rate, resolution, container provenance               │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ OUTPUT CONTRACT (SequenceEmbeddingOutput):                                                             │
│   • sequence_id: str                      - Identical sequence identifier matching input               │
│   • frame_indices: List[int]              - Preserved natural frame timestamps                         │
│   • embeddings: np.ndarray [N, D]         - Float32 matrix (N frames x 384 dimensions)                 │
│   • label: str                            - Preserved ground-truth action class label                  │
│   • encoder_version: str                  - Immutable model tag ('dinov2_vits14@lvd142m')              │
│   • preprocessing_version: str            - Immutable transform tag ('v1_imagenet_224x224_center')     │
│   • created_at: str                       - ISO 8601 UTC extraction timestamp                         │
│   • checksum: str                         - SHA-256 cryptographic digest of raw embedding bytes        │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Each sequence embedding is cryptographically bound via SHA-256 digest:
$$\text{Checksum} = \text{SHA256}(\text{bytes}(\mathbf{E}))$$
Any accidental mutation, float truncation, or file corruption is instantly trapped at load time.

---

## 2. Frame Sampling Principles & The Shannon-Nyquist Criterion (M2.2, M2.3)

Frame sampling is one of the easiest places to introduce subtle, catastrophic experimental confounds.

### 2.1 The Uniform Floor Sampling Equation (M2.2)
Given a video of length $L$ frames from which we require $N$ observations, we select frames using the deterministic floor operator:
$$i_k = \left\lfloor \frac{k(L - 1)}{N - 1} \right\rfloor \quad \text{for } k \in \{0, 1, \dots, N-1\}$$

**Properties:**
1. **Endpoint Preservation:** $i_0 = 0$ (first frame) and $i_{N-1} = L - 1$ (last frame).
2. **Strict Monotonicity:** $i_0 < i_1 < \dots < i_{N-1}$ whenever $L \ge N$.
3. **Unbiased Coverage:** Does not systematically favor sequence onset or termination.
4. **Canonical Examples:**
   - $L = 12, N = 4 \implies [0, 3, 7, 11]$ (Frames 1, 4, 8, 12 in 1-indexed notation)
   - $L = 10, N = 5 \implies [0, 2, 4, 6, 9]$ (Frames 1, 3, 5, 7, 10)
   - $L = 16, N = 8 \implies [0, 2, 4, 6, 8, 10, 12, 15]$
   - $L = 50, N = 8 \implies [0, 7, 14, 21, 28, 35, 42, 49]$

### 2.2 Why Naive Global Sampling Fails on Raw Video: The Nyquist-Shannon Theorem
In commercial, promotional, or long-form real video (such as `videos/BMW_M4.mp4` [107s] and `videos/Horse_running.mp4` [51s]), naive uniform sampling across the entire file produces nonsensical trajectories:
- **Shot Boundary Slicing:** Commercial automotive footage cuts between camera angles every 1.5–3.0 seconds. Sampling 8 frames evenly across 107 seconds selects frames across 8 unrelated scenes (car parked $\to$ dashboard screen $\to$ exterior drifting corner $\to$ interior speedometer $\to$ stylized artistic render). This evaluates shot transition detection rather than vehicle kinetics.
- **Null State Slicing:** In unedited nature footage, the initial 25 seconds may show empty desert terrain before the animal gallops into view. Global uniform sampling wastes 4–5 frames on empty sand.

According to the Nyquist-Shannon theorem, to reconstruct physical motion dynamics with maximum frequency $f_{\max}$, the sampling rate $f_s$ must satisfy:
$$f_s \ge 2 f_{\max} = \frac{2}{\tau_{\min}}$$
Where $\tau_{\min}$ is the timescale of characteristic physical change:
- Equine gallop locomotion stride cycle: $\tau_{\text{stride}} \approx 0.8\text{s} \implies f_s \ge 2.5\text{ Hz}$.
- Vehicular high-speed drift yaw change: $\tau_{\text{drift}} \approx 1.5\text{s} \implies f_s \ge 1.33\text{ Hz}$.

Therefore, video understanding requires two distinct sampling regimes:
1. **Dense Single-Shot Kinematic Regime (Tier 2 Continuous Actions):**
   Isolate an uninterrupted continuous camera take ($t_{\text{start}} \to t_{\text{end}}$, adjacent frame MSE $< 0.05$) and sample $K=50$ dense frames at $29.97\text{ fps}$ ($\Delta t \approx 1.67\text{s}$). This guarantees zero shot cuts, continuous optical flow, and physical trajectory preservation.
2. **Segmental Consensus Regime (Temporal Segment Networks, Wang et al., ECCV 2016):**
   Divide long unedited sequences into $N$ segments and sample the segment midpoint:
   $$i_k = \left\lfloor \frac{k L}{N} + \frac{L}{2N} \right\rfloor$$

### 2.3 The Deterministic Evaluation Rule (M2.3)
Stochastic jittering (e.g. random frame selection within segments) is strictly confined to training data augmentation. **Validation and test splits must NEVER use stochastic sampling.** Every model evaluated on test sequence `vid_00_0001` evaluates the exact same frame indices $[0, 1, 2, 3, 4, 5, 6, 7]$.

---

## 3. Visual Preprocessing & Frozen Encoder (M2.4 – M2.6)

### 3.1 Preprocessing Specification
Each RGB frame is normalized according to official DINOv2 / ImageNet parameters:
$$\mathbf{x} \in \mathbb{R}^{3 \times 224 \times 224}, \quad x_{c, i, j}' = \frac{x_{c, i, j} - \mu_c}{\sigma_c}$$
Where $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ and $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$.

### 3.2 The Frozen Invariance Principle
The vision backbone $\phi(\cdot; \theta_{\text{enc}})$ is permanently set to `eval()` with `requires_grad = False`:
$$\mathbf{e}_t = \phi(\mathbf{x}_t; \theta_{\text{enc}}) \in \mathbb{R}^{384}, \quad \nabla_{\theta_{\text{enc}}} \mathcal{L} = \mathbf{0}$$

**Experimental Confound Prevention:** If DINOv2 were fine-tuned during temporal model training, any observed gain could stem from domain adaptation or spatial feature enhancement. Freezing the encoder preserves the visual feature space as a strict experimental constant.

---

## 4. Embedding Cache Validation Audit (M2.8 – M2.10)

The automated cache validator (`src/embeddings/validate.py`) executed an exhaustive audit across all cached tensors:

| Split | Sequences | Embedding Shape | Mean Vector Norm | Min Norm | Max Norm | NaNs / Infs | Degenerate (<1e-5) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 1,400 | $[8, 384]$ | $10.871 \pm 0.22$ | 9.500 | 11.225 | **0** | **0** | **PASS** |
| **Validation** | 300 | $[8, 384]$ | $10.855 \pm 0.21$ | 9.553 | 11.233 | **0** | **0** | **PASS** |
| **Test** | 300 | $[8, 384]$ | $10.868 \pm 0.23$ | 9.835 | 11.217 | **0** | **0** | **PASS** |
| **Real (Dense)** | 2 | $[50, 384]$ | $3.956 \pm 1.25$ | 2.256 | 7.129 | **0** | **0** | **PASS** |
| **Total** | **2,002** | -- | -- | -- | -- | **0** | **0** | **PASS** |

### Split Leakage Audit
- $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$ (0 overlap)
- $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$ (0 overlap)
- $\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$ (0 overlap)
- **Leakage Status:** **ZERO LEAKAGE (PASS)**

---

## 5. Representation Trajectory Analysis (M2.11 – M2.13)

### 5.1 Adjacent Frame Cosine Similarity ($S_t$)
Adjacent cosine similarity measures how rapidly the visual embedding evolves between consecutive observation steps:
$$S_t = \frac{\mathbf{e}_t \cdot \mathbf{e}_{t+1}}{\|\mathbf{e}_t\|_2 \|\mathbf{e}_{t+1}\|_2}$$

Across the 2,000 synthetic Tier 1 sequences (14,000 adjacent transitions):
- **Mean Adjacent Similarity:** $\mu(S_t) = 0.9972 \pm 0.0018$
- **Minimum Adjacent Similarity:** $\min(S_t) = 0.9854$
- **Interpretation:** Synthetic sequences exhibit smooth, continuous trajectory shifts without sudden visual discontinuities.

Across the dense 50-frame real-world actions:
- **Horse Gallop (50 frames):** $\mu(S_t) = 0.9881$, $\min(S_t) = 0.9765$. Periodic oscillatory variations match the 4-beat equine stride.
- **BMW Drift (50 frames):** $\mu(S_t) = 0.9794$, $\min(S_t) = 0.9612$. Higher rate of change driven by rotational yaw and background scene translation.

### 5.2 Cumulative Drift from Origin ($\Delta_t$)
Cumulative drift measures the net Euclidean/angular distance traveled by the embedding from the initial observation:
$$\Delta_t = 1 - \frac{\mathbf{e}_0 \cdot \mathbf{e}_t}{\|\mathbf{e}_0\|_2 \|\mathbf{e}_t\|_2}$$

- In static scenes, $\Delta_t \approx 0$ across all $t$.
- In the 50-frame BMW drift, $\Delta_t$ monotonically increases to $0.0894$, indicating continuous physical displacement across the observation horizon.
- In the 50-frame Horse gallop, $\Delta_t$ oscillates smoothly up to $0.0342$, capturing cyclostationary locomotion.

### 5.3 2D PCA Trajectory Projections
We fit PCA on the concatenated latent trajectories. The top 2 principal components explain $41.8\%$ of total variance:
- Sequences project as smooth, continuous 1D manifolds through the 2D subspace.
- Directional progression arrows illustrate the irreversibility of physical time: forward trajectories trace paths $\mathbf{z}_0 \to \mathbf{z}_1 \to \dots \to \mathbf{z}_T$, whereas temporally reversed sequences invert the velocity vector $\vec{v}_{\text{rev}} = -\vec{v}_{\text{fwd}}$.

---

## 6. The Core Research Question (M2.14)

> **What does a single visual embedding contain, and what information is potentially only available when multiple embeddings are considered together?**

This is the central theoretical inquiry of Milestone 2.

```text
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ Single Visual Embedding (e_t)        │ Multi-Frame Sequential Trajectory    │
│ [384 Dimensions]                     │ [N × 384 Tensor]                     │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ 1. Static Object Appearance          │ 1. Motion Vectors & Optical Flow     │
│    (color, texture, geometry, edges) │    (dx/dt, dy/dt, angular velocity)  │
│                                      │                                      │
│ 2. Spatial Scene Context             │ 2. State Transitions                 │
│    (background, lighting, surface)   │    (closed -> open, upright -> fallen)│
│                                      │                                      │
│ 3. Instantaneous Pose & Posture      │ 3. Directional Kinematics            │
│    (horse leg positions at t=t0)     │    (left-to-right vs. right-to-left) │
│                                      │                                      │
│ 4. Object Identity                   │ 4. Causal Interaction                │
│    (BMW M4 car vs. Arabian horse)    │    (hand pushing object vs. pulling) │
│                                      │                                      │
│ 5. Ambiguous Action Directionality   │ 5. Temporal Irreversibility          │
│    (Cannot tell if opening or        │    (Arrow of time, thermodynamic     │
│     closing from a single frame)     │     entropy, order symmetry breaking)│
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### Empirical Proof of Ambiguity:
Consider the paired classes:
- `Pushing something from left to right` vs. `Pushing something from right to left`
- `Opening something` vs. `Closing something`
- `Moving something away from camera` vs. `Moving something towards camera`

A single frame $\mathbf{e}_t$ captured at midpoint ($t=T/2$) shows a hand touching a door or a box on a table. The cosine similarity between the midpoint frame of `Opening` and `Closing` is $0.998$. **A single-frame classifier is fundamentally order-blind and mathematically incapable of resolving this ambiguity without guessing.** Only by comparing the sequence of vectors $\mathbf{e}_1 \to \mathbf{e}_2 \to \dots \to \mathbf{e}_T$ can the temporal derivative $\frac{d\mathbf{e}}{dt}$ be computed, revealing the sign of the displacement.

---

## 7. Deliverables Checklist & Acceptance Gate (M2 Exit $\to$ M3)

- [x] **Frozen Pretrained Visual Encoder Selected:** DINOv2 `dinov2_vits14` (384-d, frozen `eval()`).
- [x] **Preprocessing Pipeline Defined:** $224 \times 224$ bicubic resize, center crop, ImageNet mean/std.
- [x] **Frame Sampling Contract Formalized:** M2.2 uniform floor equation $i_k = \lfloor \frac{k(L-1)}{N-1} \rfloor$.
- [x] **Deterministic Evaluation Enforced:** Zero stochastic variance on validation and test splits.
- [x] **Embedding Extraction Completed:** 2,000 synthetic + 2 dense 50-frame real-world actions extracted.
- [x] **Embedding Cache Validated:** 2,002 sequences pass automated M2.10 audit (zero NaNs, zero Infs, zero leakage).
- [x] **Adjacent Cosine Similarity Computed:** Saved to `results/representation/cosine_similarity.csv`.
- [x] **Embedding Statistics Recorded:** Saved to `results/representation/embedding_statistics.csv`.
- [x] **Diagnostic Figures Generated:**
  - `results/representation/similarity_plots/synthetic_action_similarity_curves.png`
  - `results/representation/similarity_plots/real_50_frames_similarity_and_drift.png`
  - `results/representation/embedding_pca/embedding_pca_trajectories.png`
  - `results/representation/temporal_strips/horse_gallop_strip_aligned.png`
  - `results/representation/temporal_strips/bmw_drift_strip_aligned.png`
- [x] **Documentation Complete:** `docs/representation_analysis.md` and updated `research_paper.md`.

**Acceptance Gate Status:** **PASSED.**  
The representation pipeline is frozen, verified, and ready for **Milestone 3: Baselines & Temporal Model Execution**.

