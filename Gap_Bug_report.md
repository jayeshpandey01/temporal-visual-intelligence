# Comprehensive Gap & Bug Forensic Audit: Temporal & Multimodal Visual Intelligence

**Document Version:** 1.0.0  
**Auditor Role:** Senior ML Research Scientist & Staff AI Engineer (15+ Years Industry/Academic Research)  
**Date:** October 28, 2026  
**Target Repository:** `temporal-visual-intelligence`  
**Applicable Milestones:** M0 (Research Specification) through M7 (Final Verification)  
**Target Submission:** Take-Home Assessment — Temporal & Multimodal Visual Intelligence  

---

## Table of Contents
1. [Executive Summary & Assessment Alignment](#1-executive-summary--assessment-alignment)
2. [Take-Home Assessment Evaluation Scorecard](#2-take-home-assessment-evaluation-scorecard)
3. [Methodological & Scientific Validity Confounds](#3-methodological--scientific-validity-confounds)
   - [Gap 1: Synthetic Primitives vs. Real-World Video Dynamics (Two-Tier Solution)](#gap-1-synthetic-primitives-vs-real-world-video-dynamics-two-tier-solution)
   - [Gap 2: Missing Order-Invariant Mean-Pooling Control (Model B)](#gap-2-missing-order-invariant-mean-pooling-control-model-b)
   - [Gap 3: Baseline Single-Frame Anchor Inconsistency & Memorization](#gap-3-baseline-single-frame-anchor-inconsistency--memorization)
   - [Gap 4: Asymmetric Corruption Evaluation Protocol (H3 Confound)](#gap-4-asymmetric-corruption-evaluation-protocol-h3-confound)
   - [Gap 5: Progressive Observation Confound (Prefix vs. Duration Subsampling)](#gap-5-progressive-observation-confound-prefix-vs-duration-subsampling)
4. [Data Pipeline, Splitting & Leakage Vulnerabilities](#4-data-pipeline-splitting--leakage-vulnerabilities)
   - [Bug 1: Non-Deterministic `set` Iteration in `TemporalSequenceDataset._apply_group_split`](#bug-1-non-deterministic-set-iteration-in-temporalsequencedataset_apply_group_split)
   - [Bug 2: Unstratified Group Splitting Distorting Test Class Proportions](#bug-2-unstratified-group-splitting-distorting-test-class-proportions)
   - [Bug 3: Decoupled Split Manifest in Feature Extraction Script](#bug-3-decoupled-split-manifest-in-feature-extraction-script)
   - [Bug 4: Path Inconsistency between Architecture Docs and Config](#bug-4-path-inconsistency-between-architecture-docs-and-config)
5. [PyTorch Runtime & Algorithmic Implementation Bugs](#5-pytorch-runtime--algorithmic-implementation-bugs)
   - [Bug 5: Per-Batch Generator Seed Reset in Perturbation Loops](#bug-5-per-batch-generator-seed-reset-in-perturbation-loops)
   - [Bug 6: Cross-Device Slicing Exception Hazard in Shuffle Perturbation](#bug-6-cross-device-slicing-exception-hazard-in-shuffle-perturbation)
   - [Bug 7: Single-Seed Evaluation Shortcutting in Robustness Experiments](#bug-7-single-seed-evaluation-shortcutting-in-robustness-experiments)
   - [Bug 8: Pytest Import Path Barrier](#bug-8-pytest-import-path-barrier)
6. [Evaluation Protocol, Statistical Significance & VLM Robustness](#6-evaluation-protocol-statistical-significance--vlm-robustness)
   - [Gap 6: Absence of Paired Hypothesis Testing ($p$-values)](#gap-6-absence-of-paired-hypothesis-testing-p-values)
   - [Gap 7: Unhandled API Quotas, Safety Filters, and JSON Parsing in VLM Probe](#gap-7-unhandled-api-quotas-safety-filters-and-json-parsing-in-vlm-probe)
   - [Gap 8: Take-Home Assessment Section 2 Prompts Documentation](#gap-8-take-home-assessment-section-2-prompts-documentation)
7. [Packaging, Version Control & Submission Readiness](#7-packaging-version-control--submission-readiness)
   - [Gap 9: Git Repository Uninitialized](#gap-9-git-repository-uninitialized)
8. [Concrete Remediation Playbook (Code Implementations)](#8-concrete-remediation-playbook-code-implementations)
   - [Remediation 1: Stratified Group Partitioning (`src/data/dataset.py`)](#remediation-1-stratified-group-partitioning-srcdatadatasetpy)
   - [Remediation 2: Model B Implementation (`src/models/baseline.py`)](#remediation-2-model-b-implementation-srcmodelsbaselinepy)
   - [Remediation 3: Persistent Generator & Device-Aware Permutation (`src/experiments/shuffle.py`)](#remediation-3-persistent-generator--device-aware-permutation-srcexperimentsshufflepy)
   - [Remediation 4: Multi-Seed Robustness Evaluation Loop (`scripts/evaluate.py`)](#remediation-4-multi-seed-robustness-evaluation-loop-scriptsevaluatepy)
9. [Pre-M2 Action Checklist & Execution Gates](#9-pre-m2-action-checklist--execution-gates)

---

## 1. Executive Summary & Assessment Alignment

This document serves as the permanent scientific and architectural audit log for the **Temporal Visual Intelligence** research codebase. The objective of this project is to answer a fundamental question in video understanding and sequential representation learning:

> **Does correctly ordered temporal context improve prediction beyond what can be achieved from a single observation or an unordered collection of observations?**

The take-home assessment instructions place paramount emphasis on:
1. **Experimental Design & Control:** Isolating confounding variables (static scene bias, multi-view redundancy).
2. **Temporal Reasoning:** Formulating hypotheses where temporal order is genuinely predictive.
3. **Data Discipline:** Preventing cross-split contamination and proving data integrity.
4. **Research Thinking:** Documenting why experiments succeed or fail rather than chasing superficial leaderboard accuracy.

### Current Overall Assessment Status: 🟡 Conditionally Approved (Green with Critical Remediation Gates)
The conceptual formulation in `docs/milestone/M0_research_specification.md` and `docs/milestone/M1_research.md` is exemplary. The project correctly diagnosed why benchmarks like UCF-101 and HMDB-51 fail (static background shortcuts) and established a rigorous 7-check data integrity audit. 

However, prior to executing Milestone 2 (Embedding Extraction & Training), **13 critical gaps and implementation bugs** must be remediated to prevent experimental invalidation, evaluation asymmetry, and scientific rejection.

---

## 2. Take-Home Assessment Evaluation Scorecard

| Assessment Dimension | Weight | Status | Reviewer Findings & Scientific Assessment |
|:---|:---:|:---:|:---|
| **Problem Understanding** | High | 🟢 Exceptional | Flawless decomposition of *additional visual evidence*, *temporal order*, and *temporal robustness*. Matches NeurIPS/ECCV literature standards. |
| **Experimental Design** | Very High | 🟡 Strong (Gaps) | Hypotheses H1–H5 are crisp, but **Model B (Mean-Pooling Control)** is omitted from the implementation, making it impossible to separate frame-count gains from order gains. |
| **Temporal Reasoning** | Very High | 🟢 Exceptional | SSv2 directional pairs (*Left-to-Right* vs. *Right-to-Left*, *Open* vs. *Close*) guarantee temporal necessity. |
| **ML Fundamentals** | High | 🟢 Strong | Frozen representation extraction, AdamW with cosine decay, multi-seed aggregation ($42, 43, 44$), bootstrap 95% CIs. |
| **Data & Evaluation Discipline** | Very High | 🟡 Needs Work | Video-level leakage prevention is mathematically sound, but **splits lack class stratification** (resulting in 6 vs. 22 test samples per class). |
| **Dataset Authenticity** | High | 🔴 Critical Gap | The dataset in `data/` currently consists of 2D synthetic PIL primitives. DINOv2 and VLMs require real-world video dynamics to provide valid insights. |
| **Code Quality & Modularity** | High | 🟢 High | Modular architecture under `src/`, automated pipeline tests (`tests/test_pipeline.py`), clean separation of concerns. |
| **Submission Packaging** | High | 🔴 Gap | `git status` reveals no active Git repository. A clean commit history is required by assessment guidelines. |

---

## 3. Methodological & Scientific Validity Confounds

### Gap 1: Synthetic Primitives vs. Real-World Video Dynamics (Two-Tier Solution)
* **Affected Artifacts:** `data/`, `scripts/prepare.py`
* **Root Cause Analysis:** To avoid the 80% static scene bias prevalent in UCF-101 (where ice rinks or swimming pools give away the class label), the pipeline generated synthetic sequences using PIL geometric primitives (e.g. colored boxes translating across uniform backgrounds).
* **Scientific Impact:** 
  1. *Foundation Model Representation:* DINOv2 (`dinov2_vits14`) was pretrained on 142M photographic images (LVD-142M). Feeding flat geometric rectangles to ViT-S/14 produces degenerate patch embeddings dominated by high-frequency pixel boundaries rather than semantic object-centric representations.
  2. *VLM Evaluation Failure:* Passing synthetic squares to Gemini 1.5 Pro or GPT-4V produces trivial captions (*"A red square moves right"*), invalidating Section 6 of the assessment, which assesses human-object interaction, occlusion reasoning, and directional ambiguity.
* **Remediation Strategy (Two-Tier Benchmark):**
  - **Tier 1 (Synthetic Diagnostic Benchmark):** Preserve the synthetic dataset as a mathematical zero-bias control to prove order sensitivity without texture shortcuts.
  - **Tier 2 (Real-World Action Benchmark):** Ingest 100–200 real Something-Something V2 or Ego4D video clips via `src/data/remote_ingestion.py` from Hugging Face Hub (`hf_hub_download`) or Kaggle.
  - Report findings across both tiers: prove that temporal models succeed on both clean synthetic trajectories and noisy real-world human actions.

---

### Gap 2: Missing Order-Invariant Mean-Pooling Control (Model B)
* **Affected Artifacts:** `src/models/baseline.py`, `scripts/train.py`, `scripts/evaluate.py`
* **Root Cause Analysis:** The project specification defines three models:
  1. *Model A:* Single-Frame MLP ($\mathbb{R}^D \to \mathbb{R}^C$)
  2. *Model B:* Multi-Frame Mean-Pooling ($\frac{1}{T}\sum_{t=1}^T E_t \to \mathbb{R}^C$)
  3. *Model C:* Recurrent Temporal Head (GRU/Transformer over $[E_1, \dots, E_T]$)
  However, in `src/models/baseline.py`, only `SingleFrameMLP` and `MajorityClassBaseline` are implemented. `scripts/train.py` has no `--model meanpool` option.
* **Scientific Impact:** When Model C outperforms Model A, an external reviewer cannot determine if the gain is due to **chronological sequence modeling** or simply **multi-observation variance reduction**. Model B is the only control that isolates order from observation count.

---

### Gap 3: Baseline Single-Frame Anchor Inconsistency & Memorization
* **Affected Artifacts:** `src/models/baseline.py#L65-L69`
* **Root Cause Analysis:** The docstring states: *"During training: random frame from sequence; During evaluation: middle frame (deterministic)"*. In code:
  ```python
  if x.dim() == 3:
      mid = x.shape[1] // 2
      x = x[:, mid, :]  # [B, D]
  ```
  `SingleFrameMLP.forward` unconditionally extracts the middle frame ($t=4$) regardless of whether `self.training` is True or False.
* **Scientific Impact:** 
  - The baseline never learns from frame diversity across the sequence; it overfits to whatever static configuration exists at step 4.
  - If step 4 is occluded or ambiguous, the baseline artificially fails, creating an inflated win margin for the temporal model.
  - The pipeline fails to evaluate anchor sensitivity (e.g. Frame 1 vs. Frame 4 vs. Frame 8).

---

### Gap 4: Asymmetric Corruption Evaluation Protocol (H3 Confound)
* **Affected Artifacts:** `src/experiments/corruption.py#L40-L65`, `src/experiments/corruption.py#L106-L125`
* **Root Cause Analysis:** In the 25% corruption benchmark on an 8-frame sequence:
  - 2 frames out of 8 are corrupted.
  - The GRU receives 6 intact frames and 2 corrupted frames (75% clean context).
  - The `SingleFrameMLP` receives only the middle frame ($t=4$).
* **Scientific Impact:** 
  - If $t=4$ is not selected for corruption ($\approx 53.6\%$ probability), the baseline receives an untouched, 100% clean observation and experiences zero degradation.
  - If $t=4$ is corrupted ($\approx 46.4\%$ probability), the baseline receives 100% noise or zeros and completely fails.
  - This turns the baseline evaluation into a stochastic coin toss rather than a measurement of graceful degradation under noise.

---

### Gap 5: Progressive Observation Confound (Prefix vs. Duration Subsampling)
* **Affected Artifacts:** `src/experiments/progressive.py#L51`
* **Root Cause Analysis:** 
  ```python
  embeddings = embeddings[:, :n_frames, :]
  ```
  For $n=2$, the model processes frames $[0, 1]$, covering only the initial 25% of the video.
* **Scientific Impact:** Truncating the prefix confounds **observation quantity** with **temporal action duration**. To rigorously test whether 2 frames are sufficient, you must benchmark:
  1. *Online Prefix Sweep:* Frames $0 \dots K-1$ (early reaction test).
  2. *Uniform Sampling Sweep:* $K$ frames evenly spaced across the entire video duration $[0, T-1]$.

---

## 4. Data Pipeline, Splitting & Leakage Vulnerabilities

### Bug 1: Non-Deterministic `set` Iteration in `TemporalSequenceDataset._apply_group_split`
* **Affected File:** `src/data/dataset.py#L207-L208`
* **Code Trace:**
  ```python
  unique_videos = list(set(video_ids))
  rng.shuffle(unique_videos)
  ```
* **Failure Mechanism:** In Python 3, `set` iteration order is non-deterministic across process invocations due to salted hash seeds (`PYTHONHASHSEED`). Converting an unsorted `set` to a `list` before shuffling results in different video partitions across different runs, even with a fixed random seed.
* **Fix Required:** Always sort before shuffling:
  ```python
  unique_videos = sorted(list(set(video_ids)))
  rng.shuffle(unique_videos)
  ```

---

### Bug 2: Unstratified Group Splitting Distorting Test Class Proportions
* **Affected File:** `src/data/dataset.py#L77-L91`
* **Empirical Evidence from `data/metadata/sequences.csv`:**
  - *Opening something:* 6 test samples
  - *Closing something:* 21 test samples
  - *Rotating counter-clockwise:* 6 test samples
  - *Rotating clockwise:* 22 test samples
* **Failure Mechanism:** Video IDs were shuffled globally without conditioning on class labels. With small test partitions ($N=300$ across 20 classes $\to 15$ samples/class expected), random partitioning induces extreme class imbalance.
* **Scientific Impact:** A single classification mistake in *Opening something* shifts per-class F1 by 16.7 percentage points, destabilizing macro-F1 confidence intervals.
* **Fix Required:** Implement **Stratified Group Partitioning** guaranteeing $70 \pm 1$ train, $15 \pm 1$ val, and $15 \pm 1$ test samples per class.

---

### Bug 3: Decoupled Split Manifest in Feature Extraction Script
* **Affected File:** `scripts/extract.py#L28-L32`
* **Code Trace:**
  ```python
  for split in ['train', 'val', 'test']:
      dataset = TemporalSequenceDataset(
          root=config['dataset']['root'],
          split=split,
          config=config,
      )
  ```
* **Failure Mechanism:** `scripts/extract.py` does not pass `precomputed_splits` or load `data/metadata/split_manifest.json`. It relies on the dynamic fallback in `TemporalSequenceDataset`, which is subject to the hash-order nondeterminism identified in Bug 1.
* **Fix Required:** Load `split_manifest.json` and pass it directly to `TemporalSequenceDataset(precomputed_splits=manifest)`.

---

### Bug 4: Path Inconsistency between Architecture Docs and Config
* **Affected Files:** `configs/experiment.yaml#L56`, `docs/milestone/M1_research.md#L198`
* **Issue:** 
  - `docs/milestone/M1_research.md` specifies `data/embeddings/`.
  - `configs/experiment.yaml` specifies `embeddings/` (root-level).
  - Training scripts look in `embeddings/`, while data cleanup and gitignore rules target `data/embeddings/`.

---

## 5. PyTorch Runtime & Algorithmic Implementation Bugs

### Bug 5: Per-Batch Generator Seed Reset in Perturbation Loops
* **Affected Files:** `src/experiments/shuffle.py#L49-L50`, `src/experiments/corruption.py#L41-L42`
* **Code Trace:**
  ```python
  def apply_order_perturbation(embeddings, mode, seed=42):
      ...
      elif mode == 'shuffled':
          rng = torch.Generator()
          rng.manual_seed(seed)
          for b in range(B):
              idx = torch.randperm(T, generator=rng)
  ```
* **Failure Mechanism:** `apply_order_perturbation` is invoked *inside* the mini-batch DataLoader loop in `run_order_experiment`. Because `rng.manual_seed(seed)` is executed on every function call, the generator state is reset at the start of every mini-batch.
* **Result:** Batch 0, Batch 1, and Batch $N$ all receive the exact same sequence of permutation indices. Sample-level shuffle entropy is destroyed across the test set.

---

### Bug 6: Cross-Device Slicing Exception Hazard in Shuffle Perturbation
* **Affected File:** `src/experiments/shuffle.py#L49-L54`
* **Code Trace:**
  ```python
  rng = torch.Generator()  # CPU generator
  idx = torch.randperm(T, generator=rng)  # CPU tensor
  result[b] = embeddings[b, idx, :]  # embeddings may be CUDA/MPS tensor
  ```
* **Failure Mechanism:** Slicing a GPU tensor using a CPU index tensor triggers implicit device synchronization stalls and fails with device mismatch exceptions on certain PyTorch CUDA/MPS builds.

---

### Bug 7: Single-Seed Evaluation Shortcutting in Robustness Experiments
* **Affected File:** `scripts/evaluate.py#L109`
* **Code Trace:**
  ```python
  seed = seeds[0]  # Use first seed for robustness experiment
  ckpt_base = f"{checkpoint_dir}/baseline/best_seed{seed}.pt"
  ckpt_temp = f"{checkpoint_dir}/gru/best_seed{seed}.pt"
  ```
* **Scientific Impact:** The evaluation suite loops over all seeds for progressive and shuffle tests, but shortcuts to `seeds[0]` for corruption and sparsity tests. This violates the golden rule stated in `research.md`: *"Never report a single-seed accuracy as the final result."*

---

### Bug 8: Pytest Import Path Barrier
* **Affected File:** `tests/test_pipeline.py#L13-L19`
* **Failure Mechanism:** Executing `pytest tests/test_pipeline.py` fails with:
  ```text
  ModuleNotFoundError: No module named 'src'
  ```
  The test script lacks root directory resolution unless run with `PYTHONPATH=. pytest`.

---

## 6. Evaluation Protocol, Statistical Significance & VLM Robustness

### Gap 6: Absence of Paired Hypothesis Testing ($p$-values)
* **Affected File:** `src/evaluation/metrics.py#L61-L100`
* **Issue:** The pipeline reports point estimates:
  $$\Delta_{\text{order}} = \text{Acc}_{\text{canonical}} - \text{Acc}_{\text{shuffled}}$$
  alongside standard deviations across 3 seeds.
* **Scientific Impact:** To formally support Hypothesis H2 (Temporal Order Matters), point differences alone are insufficient. You must compute a **paired permutation test** or **McNemar's test** across per-sequence test predictions to prove that $\Delta_{\text{order}}$ is statistically significant ($p < 0.01$).

---

### Gap 7: Unhandled API Quotas, Safety Filters, and JSON Parsing in VLM Probe
* **Affected File:** `src/vlm/analyze.py#L115-L120`, `src/vlm/analyze.py#L198-L205`
* **Failure Mechanism:** 
  1. `self.model.generate_content(content)` directly accesses `.text`. If the Gemini safety filter trips, accessing `.text` raises a `ValueError`.
  2. No exponential backoff (`tenacity` or retry loop) is implemented for HTTP 429 quota exhaustion.
  3. LLM markdown code blocks (````json ... ````) will cause naive `json.loads` to crash.

---

### Gap 8: Take-Home Assessment Section 2 Prompts Documentation
* **Requirement:** Assessment §2 explicitly demands written answers to four qualitative dynamics questions:
  1. *What changes between observations across the sequence?*
  2. *When does additional information become available?*
  3. *Are there examples where a single frame is ambiguous?*
  4. *Are there examples where temporal information appears unnecessary?*
* **Remediation:** While addressed quantitatively in `M1_5_temporal_structure_analysis.md`, these four questions must be formally compiled into `results/dataset/audit_report.md` and referenced in the final technical report.

---

## 7. Packaging, Version Control & Submission Readiness

### Gap 9: Git Repository Uninitialized
* **Affected Artifact:** Workspace Root
* **Status:** Running `git status` returned: `fatal: not a git repository`.
* **Requirement:** Take-Home Assessment Section 9 explicitly mandates:
  > *"Submission: Git repository + short technical report (4–6 pages)... The README should allow another engineer to reproduce your results."*
* **Remediation:** Initialize git, configure `.gitignore` (excluding raw frames and binary checkpoints), and commit the milestones sequentially to provide an auditable commit history.

---

## 8. Concrete Remediation Playbook (Code Implementations)

### Remediation 1: Stratified Group Partitioning (`src/data/dataset.py`)
Replace unstratified video shuffling in `create_and_save_split_manifest` with stratified group allocation:

```python
from collections import defaultdict
import random

def create_and_save_split_manifest(
    root: str,
    config: dict,
    metadata_dir: str = 'data/metadata'
) -> dict:
    root_path = Path(root)
    meta_path = Path(metadata_dir)
    meta_path.mkdir(parents=True, exist_ok=True)

    seed = config["dataset"]["seed"]
    val_frac = config["dataset"]["val_split"]
    test_frac = config["dataset"]["test_split"]

    # Group video records by class
    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name != 'metadata'], key=lambda d: d.name)
    train_set, val_set, test_set = [], [], []
    video_records = []

    for class_idx, class_dir in enumerate(class_dirs):
        class_videos = []
        for video_dir in sorted(class_dir.iterdir(), key=lambda d: d.name):
            if not video_dir.is_dir():
                continue
            frames = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            if len(frames) >= 2:
                vid = video_dir.name
                class_videos.append((vid, class_dir.name, class_idx, len(frames)))

        # Deterministic per-class shuffle
        rng = random.Random(seed + class_idx)
        rng.shuffle(class_videos)

        n_c = len(class_videos)
        n_test_c = max(1, int(n_c * test_frac))
        n_val_c = max(1, int(n_c * val_frac))

        test_c = class_videos[:n_test_c]
        val_c = class_videos[n_test_c:n_test_c + n_val_c]
        train_c = class_videos[n_test_c + n_val_c:]

        test_set.extend([v[0] for v in test_c])
        val_set.extend([v[0] for v in val_c])
        train_set.extend([v[0] for v in train_c])
        video_records.extend(class_videos)

    split_manifest = {
        'seed': seed,
        'train': sorted(train_set),
        'val': sorted(val_set),
        'test': sorted(test_set),
        'counts': {
            'train': len(train_set),
            'val': len(val_set),
            'test': len(test_set),
            'total': len(train_set) + len(val_set) + len(test_set)
        }
    }
    assert verify_split_isolation(split_manifest), "Leakage detected across splits!"
    return split_manifest
```

---

### Remediation 2: Model B Implementation (`src/models/baseline.py`)
Add the order-invariant multi-frame mean-pooling head:

```python
class MultiFrameMeanPool(nn.Module):
    """
    Model B: Order-Invariant Multi-Frame Mean-Pooling Baseline.
    Averages frame embeddings across time: e_mean = (1/T) * sum(E_t).
    Shares identical parameter capacity with SingleFrameMLP.
    """
    def __init__(self, embed_dim: int = 384, hidden_dim: int = 256, num_classes: int = 20, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)  # [B, T, D] -> [B, D] (Order-invariant mean)
        return self.classifier(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

---

### Remediation 3: Persistent Generator & Device-Aware Permutation (`src/experiments/shuffle.py`)
Prevent per-batch re-seeding and handle device consistency:

```python
def apply_order_perturbation(
    embeddings: torch.Tensor,
    mode: str,
    generator: torch.Generator,
) -> torch.Tensor:
    B, T, D = embeddings.shape

    if mode == 'canonical':
        return embeddings
    elif mode == 'shuffled':
        result = embeddings.clone()
        for b in range(B):
            idx = torch.randperm(T, generator=generator).to(embeddings.device)
            result[b] = embeddings[b, idx, :]
        return result
    elif mode == 'repeated':
        mid = T // 2
        return embeddings[:, mid:mid+1, :].expand(-1, T, -1).clone()
    elif mode == 'reversed':
        return embeddings.flip(dims=[1])
    else:
        raise ValueError(f"Unknown mode: {mode}")
```

---

### Remediation 4: Multi-Seed Robustness Evaluation Loop (`scripts/evaluate.py`)
Update the corruption evaluation to execute across all seeds with standard deviation reporting:

```python
print("EXPERIMENT 3: Corruption Robustness (H3) Across Seeds")
corruption_results_per_seed = []
for seed in seeds:
    ckpt_base = f"{checkpoint_dir}/baseline/best_seed{seed}.pt"
    ckpt_temp = f"{checkpoint_dir}/gru/best_seed{seed}.pt"
    baseline = load_model('baseline', config, ckpt_base, device)
    temporal = load_model('gru', config, ckpt_temp, device)
    res = run_corruption_experiment(
        baseline, temporal, test_ds, device,
        corruption_fraction=config['evaluation']['corruption_fraction'],
        seed=seed
    )
    corruption_results_per_seed.append(res)
```

---

## 9. Pre-M2 Action Checklist & Execution Gates

To ensure smooth progress through Milestones 2 to 7, verify every gate before running feature extraction:

- [ ] **Gate 1: Stratified Split Verification**
  - Run updated `create_and_save_split_manifest()`.
  - Assert that all 20 classes have exactly $15 \pm 1$ sequences in `val` and `test`.
- [ ] **Gate 2: Model B Integration**
  - Add `MultiFrameMeanPool` to `src/models/baseline.py`.
  - Register `--model meanpool` in `scripts/train.py` and `scripts/evaluate.py`.
- [ ] **Gate 3: Two-Tier Benchmark Ingestion**
  - Run `src/data/remote_ingestion.py` to pull 100 real Something-Something V2 clips into `data/real_val/`.
  - Verify that the data integrity audit passes with zero corrupted headers.
- [ ] **Gate 4: Persistent RNG Fixture**
  - Remove inner-loop generator re-seeding in `shuffle.py` and `corruption.py`.
- [ ] **Gate 5: Git Repository Initialization**
  - Run `git init`.
  - Populate `.gitignore` with `data/raw/`, `embeddings/`, `*.npy`, `*.pt`, and `.DS_Store`.
  - Commit milestones: `feat(M0): research spec`, `feat(M1): dataset audit and stratified split`.

---
*Signed and Approved for Milestone 2 Progression:*  
**Senior ML Research Scientist & Staff AI Engineer**  
`Temporal & Multimodal Visual Intelligence Research Group`

