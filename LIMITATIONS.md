# Scientific Limitations, Failure Boundaries, and Honest Disclosure

In adherence to rigorous scientific standards, this document explicitly articulates the boundaries of empirical claims, structural limitations of the methodology, and documented failure modes identified during evaluation.

> **Provisional counts:** Failure and cohort frequencies below are copied from stored analysis artifacts that use different cohort definitions. Reconcile each count against source prediction rows and its stated denominator before treating it as a final prevalence estimate. They do not establish generalization beyond this benchmark.

---

## 1. Scope & Generalization Boundaries

### A. Pre-Trained Representation Sensitivity
- All primary evaluations rely on frozen representations extracted from **DINOv2 ViT-S/14** (384-dimensional latents).
- Cross-encoder evaluations on **CLIP ViT-B/32** (512-dimensional latents) are present, but exact comparisons must be read from matching frozen result artifacts and should not be generalized beyond the tested subset and protocol.
- However, feature extractors trained with explicit temporal objectives (e.g., VideoMAE, TimeSformer) were not evaluated under this frozen benchmark and could alter the baseline spatial encoding strength.

### B. Horizon and Sampling Constraints
- The primary synthetic benchmark operates over an observation window of $K=8$ uniformly sampled frames.
- The repository does not establish a measured physical frequency bound for the continuous clips. Treat Nyquist-based statements as unvalidated design rationale; rapid motions below the sampling interval may be missed.

---

## 2. Failure Mode Taxonomy & Analysis

Stored failure-mining artifacts describe the following candidate archetypes across the held-out test split ($N=300$). Their counts and denominators have not been reconciled against the current canonical prediction rows; treat these as qualitative leads, not verified prevalence estimates:

### Candidate Archetype 1: Camera Ego-Motion Parallax (Quadrant B Failures)
- **Observed Frequency**: 11 sequences (3.7% of test split).
- **Failure Mechanism**: When an egocentric camera pans or rolls simultaneously with an object movement, the background optical flow dominates the recurrent hidden state dynamics.
- **Example**: In sequence `vid_05_0002` (*"Turning the camera left while filming something"*), Model A and Model B rely on static scene context and correctly identify the class, whereas Model C overfits to foreground motion and mispredicts.
- **Remediation**: Decouple camera ego-motion via IMU sensor integration or background stabilization before recurrent modeling.

### Candidate Archetype 2: Subtle Fine-Grained Object Interactions (Quadrant D Failures)
- **Observed Frequency**: 179 sequences (59.7% of test split).
- **Failure Mechanism**: Subtle state transitions involving small objects (e.g., *"Plugging something into something"* vs *"Unplugging something"*) where the 224x224 spatial resolution and frozen patch tokens do not resolve precise finger-object contact.
- **Remediation**: High-resolution foveated patches or explicit 3D hand-object pose inputs.

---

## 3. Methodological Bounds & Negative Controls

### A. Absence of 3D Spatial Geometry
- Visual inputs are projected 2D monocular frames without explicit depth point clouds or tactile feedback.
- Directional ambiguities along the camera optical axis (moving away vs moving closer) exhibit high cosine similarity ($S_t \approx 1.0$) when object scale variation is minimal.

### B. Capacity Parity Invariance
- While The current checkpoint diagnostic reports Model D (CompactGRU) at 23.56% with 103,700 parameters (+15.33 pp over MeanPool), Model C (Scaled GRU, 249.4k parameters) achieves 34.78%.
- The stored final checkpoint summary reports a CompactGRU advantage, while the M4 summary lists a different CompactGRU mean. Reconcile those sources before treating capacity attribution as established. The paired H14 comparison is recomputed from saved per-sequence predictions, conditional on the saved checkpoints; it is not an independent training replication.

---

## 4. Responsible AI & Application Constraints

- This benchmark is designed as a foundational diagnostic system for temporal visual intelligence.
- Deployments in safety-critical autonomous navigation or medical robotics must incorporate redundant sensor streams (Depth, IMU, Audio) and rigorous runtime fallback systems to handle ego-motion parallax and visual occlusion.

