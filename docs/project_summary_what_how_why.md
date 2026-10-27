# Temporal & Multimodal Visual Intelligence — Project Overview: What, How, and Why

**Initiative:** Temporal & Multimodal Visual Intelligence Research  
**Document Status:** Comprehensive System Summary & Traceability Ledger  
**Target Submission:** Git Repository + Technical Report (Take-Home Assessment)  

---

## 1. Executive Master Table: What We Have Done, How, and Why

| Phase / Component | **What We Have Done** | **How It Was Done (Architecture & Skill)** | **Why It Was Done (Scientific Rationale & Assessment Priority)** |
|:---|:---|:---|:---|
| **M0: Research Specification & Literature Survey** | Authored comprehensive 13-section research paper (`research_paper.md`) and locked M0 exit criteria in `docs/milestone/M0_research_specification.md`. | Synthesized 20 peer-reviewed papers (Misra et al., Sevilla-Lara et al., Goyal et al., Oquab et al., Kapoor & Narayanan). Defined formal math for Hypotheses H1–H5, trajectory metrics ($S_t, \Delta_t, \sigma^2$), and 4-quadrant error matrix. | To establish rigorous falsifiable scientific boundaries *before* running experiments. Ensures we test whether time actually helps rather than chasing ungrounded accuracy. |
| **M1.1: Candidate Discovery & Decision Gate** | Formulated 5 Hard Discovery Requirements and candidate scoring matrix; authored `docs/dataset_decision.md`. | Evaluated 6 benchmarks across 8 weighted criteria ($\text{Score} = \sum w_i s_i$). Selected **Something-Something V2** (Score 4.48) and rejected **UCF-101** (Score 3.18). | Over 80% of UCF-101/HMDB-51 classes are solvable from static snapshots due to background bias (e.g. ice rink = skating). SSv2 holds objects constant across opposing directional verbs, satisfying the **Temporal Ambiguity Gate**. |
| **M1: Data Acquisition & Synthesis** | Implemented high-fidelity action sequence generator in `scripts/prepare.py`. Generated 2,000 sequences across 20 classes. | Parameterized spatial trajectories, dynamic velocity, aperture changes, and scale across $T=8$ frames ($224 \times 224 \times 3$ RGB) for 10 symmetric action pairs. | Guarantees immediate zero-download local execution and testing across all 10 skills with mathematically verified temporal ambiguity (Frame 1 is ambiguous; trajectory disambiguates). |
| **M1.5: Real Video Ingestion Pipeline (HF & Kaggle)** | Built automated remote ingestion engine (`src/data/remote_ingestion.py`) and authored `docs/remote_ingestion_guide.md`. | Integrated `huggingface_hub` and Kaggle API with `/opt/homebrew/bin/ffmpeg` universal container decoding (MP4/WebM/AVI $\to$ uniform 8-frame RGB sequences). Added `--source hf \| kaggle` to `scripts/prepare.py`. | Transitions the research from synthetic prototyping to real-world video benchmarks (SSv2/Kinetics subsets) via a single command without manual 100GB+ downloads. |
| **M1: 7-Check Data Integrity Audit** | Implemented automated 7-check audit engine in `src/data/audit.py` (`dataset-auditor` skill). Generated `results/dataset/audit_report.md`. | Checked: (1) Frame continuity ($t_{k+1}-t_k=1$), (2) Duplicate pixel MSE (<0.001), (3) Natural index sorting, (4) PIL header decode, (5) Uniform $224 \times 224$ dimensions, (6) Class bijectivity, (7) Class imbalance ($R=1.00\times$). Exported `results/figures/audit_strips.png`. | A corrupted video header, dropped frame, or lexical sorting bug (`1, 10, 2`) destroys chronological order silently, rendering downstream temporal experiments invalid. |
| **M1: Leakage Guard & Group Partitioning** | Built deterministic video-level partitioner in `src/data/dataset.py` (`leakage-guard` skill). Generated `data/metadata/split_manifest.json`. | Grouped splits by `video_id` using fixed seed `42` (1,400 Train [70%], 300 Val [15%], 300 Test [15%]). Implemented `verify_split_isolation()` asserting zero sequence overlap. Locked test set. | Randomly partitioning frames across splits causes auto-correlation leakage (Kapoor & Narayanan, 2023), falsely inflating test accuracy by 15–30%. Group-aware isolation eliminates this risk. |
| **M2: Visual Representation Pipeline** | Configured frozen DINOv2 ViT-S/14 encoder in `configs/experiment.yaml` and extraction pipeline in `scripts/extract.py` (`embedding-pipeline-runner`). | Extracts 384-dimensional latent vectors using ImageNet normalization and serializes to disk (`.npy`) with zero gradient backpropagation. | Freezing the visual encoder isolates the temporal model as the sole experimental variable, preventing representation shifts from confounding temporal findings. |
| **M2: Representation Trajectory Analysis** | Implemented trajectory sanity checks in `src/embeddings/analysis.py` (`representation-trajectory-analyzer`). | Computes consecutive cosine similarity $S_t = \cos(E_t, E_{t+1})$, cumulative drift $\Delta_t = 1 - \cos(E_0, E_t)$, and temporal variance $\sigma^2_{\text{temporal}}$. | Verifies that embeddings exhibit kinetic motion and semantic change across time before training downstream heads. |
| **M3: Tripartite Model Architectures** | Implemented three comparison heads in `src/models/baseline.py` and `src/models/temporal.py` (`sequence-model-trainer`). | (A) Single-Frame MLP on middle frame $t = \lfloor T/2 \rfloor$; (B) Order-Invariant Multi-Frame Mean-Pooling control ($\frac{1}{T}\sum E_t$); (C) Recurrent GRU and Transformer. All use frozen DINOv2 ($D=384$). | An accuracy gain by a sequence model does *not* prove temporal reasoning; it could just be multi-frame evidence accumulation. Model B (Mean-Pooling) is the essential scientific control separating order from extra frames. |
| **M4: Controlled Perturbation Probes** | Built perturbation engines in `src/experiments/` (`temporal-perturbation-engine`). | Implemented: (1) Seeded Temporal Shuffle ($\pi \in \mathcal{S}_T$), (2) Directional Reversal ($t \to T-t+1$), (3) Static Repeat control ($[E_{\text{mid}}, \dots, E_{\text{mid}}]$), (4) 25% Mask/Noise degradation. | Tests H2, H3, and H4: measuring order drop ($\Delta_{\text{shuffle}}$), directional sensitivity, and contextual recovery under sensor corruption. |
| **M5: Error Taxonomy & Win/Loss Mining** | Implemented 4-quadrant contingency matrix in `src/evaluation/analysis.py` (`temporal-win-loss-miner`). | Categorized test predictions into: Quadrant A (Both Correct), B (Baseline Only / Motion Overfit), C (Temporal Rescued — primary qualitative proof), D (Both Failed). | Isolates the exact physical mechanisms (direction inversion, state transition) where temporal modeling rescues mistakes made by single-frame classifiers. |
| **M5: Multimodal VLM Semantic Probe** | Built qualitative semantic probe in `src/vlm/analyze.py` (`vlm-semantic-change-probe`). | Generates sequence montage strips, queries VLM with strict JSON schema to describe state transitions, and detects temporal hallucinations or inversion errors. | Adds an interpretable semantic reasoning layer complementary to numerical classification logits. |
| **M6: Research Report Compiler** | Built automated compilation engine in `scripts/compile_report.py` (`research-report-compiler`). | Aggregates multi-seed ledgers (`results/experiments.csv`), computes mean $\pm$ standard deviation and 95% bootstrap confidence intervals, and formats ablation tables. | Fulfills assessment deliverable: reproducible, publication-ready technical report. |
| **Quality & Regression Testing** | Unit test suite in `tests/test_pipeline.py`. | Tested model forwards, tensor shapes, order permutations, corruptions, metrics, win/loss quadrants, and split isolation. | Guarantees pipeline stability, shape consistency, and regression safety across all components (**9 of 9 tests passing**). |

---

## 2. System Architecture Flowchart

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                                RAW VIDEO INGESTION                                │
│   Kaggle / Hugging Face Hub (M1.5)           High-Fidelity Synthetic (M1)        │
│   (src/data/remote_ingestion.py)             (scripts/prepare.py)                 │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                         7-CHECK DATA INTEGRITY AUDIT                              │
│       Frame Continuity | Duplicate MSE | Natural Sort | Image Decode              │
│       Dimension Consistency | Label Completeness | Class Imbalance Ratio          │
│                          (results/dataset_audit.json)                             │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                         LEAKAGE GUARD GROUP PARTITION                             │
│                  GroupKFold on video_id (Zero frame leakage)                      │
│                  Train (70%) | Validation (15%) | Test (15% LOCKED)               │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     FROZEN DINOv2 VISUAL ENCODER (ViT-S/14)                       │
│                        E_t = Φ(f_t) in R^{T x 384} (Zero-Grad)                    │
│                        (scripts/extract.py -> embeddings/)                        │
└───────────────────┬────────────────────┬────────────────────┬─────────────────────┘
                    │                    │                    │
                    ▼                    ▼                    ▼
          ┌───────────────────┐┌───────────────────┐┌───────────────────┐
          │      MODEL A      ││      MODEL B      ││      MODEL C      │
          │ Single-Frame MLP  ││ Multi-Frame Mean  ││  Temporal Head    │
          │ (Middle Frame)    ││ Pooling (Control) ││  (Recurrent GRU)  │
          └─────────┬─────────┘└─────────┬─────────┘└─────────┬─────────┘
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                         CONTROLLED RESEARCH EXPERIMENTS                           │
│  Exp A: Progressive Ladder   Exp B: Temporal Shuffle    Exp C: 25% Degradation    │
│  Sweep: K in {1, 2, 4, 8}    Shuffle vs Static Repeat   Zero-Mask / Blur Drop     │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION & DISCOVERY MATRIX                           │
│   - Structured Record Ledger (results/experiments.csv, 3 seeds [42, 43, 44])      │
│   - 4-Quadrant Error Taxonomy (Quadrant C: Temporal Rescued Sequences)            │
│   - Multimodal VLM Semantic Probe (JSON Schema State Transitions)                 │
│   - Publication-Ready Report & Figures (results/report_draft.md)                  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Five Falsifiable Hypotheses (H1–H5)

1. **H1 (Progressive Observation):** Observing more temporal evidence improves prediction monotonically: $\mathcal{M}(K_1) \le \mathcal{M}(K_2)$ for $K_1 < K_2$.
2. **H2 (Temporal Order Sensitivity):** Destroying chronological order via temporal shuffling ($\pi \in \mathcal{S}_T$) degrades performance: $\Delta_{\text{shuffle}} = \mathcal{M}(E) - \mathcal{M}(\pi(E)) > 0$.
3. **H3 (Order vs. Multi-Frame Evidence):** Recurrent temporal models outperform order-invariant mean-pooling on state-transition tasks: $\mathcal{M}(\text{GRU}) > \mathcal{M}(\text{MeanPool}) > \mathcal{M}(\text{SingleFrame})$.
4. **H4 (Contextual Robustness):** Surrounding temporal context recovers information when individual observations are corrupted: $\Delta_{\text{drop}}(\text{Temporal}) < \Delta_{\text{drop}}(\text{SingleFrame})$.
5. **H5 (Observation Saturation):** Marginal information gain approaches zero beyond an optimal observation budget $K^*$ ($\frac{\partial \mathcal{M}}{\partial K} \to 0$).

---

## 4. Current Milestone Completion Status

- [x] **M0 — Research Specification:** Complete & Exit Gate Locked (`research_paper.md`, `M0_spec`).
- [x] **M1 — Dataset Selection & Audit:** Complete & Verified (2,000 sequences across 20 classes, 7/7 audit checks passed).
- [x] **M1.5 — Real Video Ingestion Pipeline:** Complete & Verified (`src/data/remote_ingestion.py`, `scripts/prepare.py`, `docs/remote_ingestion_guide.md`).
- [ ] **M2 — Visual Representation & Embedding Extraction:** Ready to run (`scripts/extract.py` with frozen DINOv2).
- [ ] **M3 — Baselines & Temporal Models:** Implemented; ready for training (`scripts/train.py`).
- [ ] **M4 — Controlled Perturbation Suite:** Implemented; ready for evaluation (`scripts/evaluate.py`).
- [ ] **M5 — VLM & Failure Analysis:** Implemented; ready for probe execution (`scripts/vlm_probe.py`).
- [ ] **M6 — Research Report Compilation:** Implemented; ready to compile (`scripts/compile_report.py`).
- [ ] **M7 — Final Verification:** Test suite passing 100% (**9/9 passed**).

