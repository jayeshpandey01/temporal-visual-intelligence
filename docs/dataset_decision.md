# M1 Deliverable — Dataset Decision Document

**Project:** Temporal & Multimodal Visual Intelligence  
**Milestone:** M1 (Dataset Selection, Candidate Discovery & Integrity Audit)  
**Status:** Approved & Frozen  
**Selected Dataset:** Something-Something V2 (SSv2) — 20-Class Directionally Paired Subset (~2,000 sequences)  

---

## 1. Why Did We Choose This Dataset?

Standard video benchmarks such as UCF-101 and HMDB-51 suffer from severe **static scene bias**: over 80% of their action classes can be accurately identified from a single static snapshot because scene context (e.g., an ice rink for figure skating, a swimming pool for swimming) gives away the label (Sevilla-Lara et al., WACV 2021 / Meta AI; Huang et al., CVPR 2018). Using such datasets would allow a static 2D classifier to solve the task without temporal reasoning, rendering the core research question unfalsifiable.

We selected **Something-Something V2 (SSv2)** (Goyal et al., ICCV 2017) because it was specifically engineered to isolate physical dynamics and visual common sense. In SSv2, everyday objects (cups, boxes, bottles, books) and domestic backgrounds are held constant across categories, while labels correspond to opposing physical action verbs:
- *"Moving something away from the camera"* vs. *"Moving something towards the camera"*
- *"Pushing something from left to right"* vs. *"Pushing something from right to left"*
- *"Opening something"* vs. *"Closing something"*
- *"Picking something up"* vs. *"Putting something down"*
- *"Tipping something over"* vs. *"Standing something upright"*

An isolated static frame from any of these pairs is fundamentally ambiguous. Only observing the continuous chronological trajectory enables the model to disambiguate the class, satisfying the critical **Temporal Ambiguity Gate (Requirement 4)**.

---

## 2. What Exactly Is One Sample?

* **Raw Input:** One complete video sequence $\mathcal{V} = (f_1, f_2, \dots, f_T)$, where $T \ge 8$ chronologically ordered frames ($224 \times 224 \times 3$ RGB) sampled at uniform temporal intervals.
* **Latent Input:** A fixed matrix of frozen DINOv2 self-supervised embeddings $E = [E_1, E_2, \dots, E_T] \in \mathbb{R}^{T \times 384}$, extracted with zero gradients.
* **Ground Truth Target:** A single categorical sequence label $y \in \{0, 1, \dots, 19\}$ corresponding to the 20 locked action classes.

---

## 3. What Constitutes Temporal Information?

In our experiment, temporal information is explicitly defined as:
1. **Directional Trajectory & Velocity:** The sign and magnitude of spatial displacement over time ($\vec{v} = \frac{\Delta \vec{x}}{\Delta t}$), distinguishing opposing translation vectors (e.g. left-to-right vs. right-to-left).
2. **Chronological State Transition:** Irreversible transformations in object geometry, aperture, or containment (e.g. open $\to$ closed, uncovered $\to$ covered, upright $\to$ tipped over).
3. **Contextual Continuity:** The temporal auto-correlation that allows surrounding observations $\{E_t\}_{t \neq \tau}$ to reconstruct or compensate for corrupted observations at step $\tau$.

Temporal information does **not** mean static multi-view redundancy; this distinction is strictly enforced by benchmarking against an order-invariant multi-frame mean-pooling control ($\frac{1}{T}\sum E_t$).

---

## 4. What Is Our Evaluation Unit?

* **Primary Evaluation Unit:** Sequence-level prediction $\hat{y} = \Psi(E)$. Every model outputs exactly one categorical prediction per sequence.
* **Primary Metric:** **Macro-F1** across all held-out test sequences, ensuring that each of the 20 classes contributes equally without class-frequency distortion.
* **Secondary Metrics:** Top-1 Accuracy, per-class F1, paired difference 95% bootstrap confidence intervals, and the 4-quadrant contingency matrix.

---

## 5. How Did We Prevent Data Leakage?

Following the rigorous standards of Kapoor & Narayanan (Patterns 2023):
1. **Video-Level Group Partitioning:** Splits are generated using `GroupKFold` on the source video capture ID (`video_id`).
2. **Zero Cross-Split Overlap:** All frames, sub-clips, backgrounds, and object appearances from a single video recording session are allocated exclusively to **one** split:
   $$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset, \quad \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset, \quad \mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$$
3. **Partition Ratios:** 70% Train (~1,400 sequences), 15% Validation (~300 sequences), and 15% Test (~300 sequences).
4. **Reproducible Split Freezing:** The partition is locked into `data/metadata/split_manifest.json` using fixed seed `42`. The test set is strictly quarantined until final evaluation.

---

## 6. What Are the Dataset Limitations?

1. **Subsampling Redundancy:** At high raw frame rates (30 fps), adjacent frames exhibit high cosine similarity ($S_t > 0.95$), requiring uniform temporal striding ($T=8$) to capture meaningful state changes.
2. **Visual Simplicity in Synthetic Fallback:** While synthetic sequences guarantee mathematically clean temporal ambiguity and zero static shortcuts, real-world SSv2 clips introduce natural hand occlusions and complex object geometries. Both ingestion modes are supported.
3. **Scope of Physical Reasoning:** High performance on SSv2 demonstrates sensitivity to 2D trajectory direction and state change; it does not prove general 3D intuitive physics or open-world robotic reasoning.

