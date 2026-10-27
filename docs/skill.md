To execute this research workflow rigorously, your AI engineering workflows, agent pipelines, and evaluation loops (via Antigravity / agentic skill toolkits) should map directly to each experimental phase.

Here is the structured list of agent skills and tool definitions configured for this project, categorized by research phase:

---

### 1. Data Audit & Integrity Skills

* **`dataset-auditor`**
* **Purpose:** Inspects raw video/sequence directories prior to feature extraction.
* **Capabilities:** Computes frame-count distributions, detects dropped/duplicate frames (via structural similarity or pixel MSE), identifies corrupted video headers, checks class distributions, and validates cross-split isolation (ensuring zero frame leakage across video IDs).
* **Deliverable Generated:** `dataset_audit.json` and strip montage grids (`audit_strips.png`).


* **`leakage-guard`**
* **Purpose:** Deterministic partitioner and split validator.
* **Capabilities:** Enforces group-aware splits (`GroupKFold` or hash-partitioning on sequence identifiers) to guarantee that temporally contiguous sub-sequences or multi-view captures of the same physical event never bridge `train`, `val`, and `test`.



---

### 2. Feature Extraction & Embedding Operations

* **`embedding-pipeline-runner`**
* **Purpose:** High-throughput frozen representation extraction.
* **Capabilities:** Loads DINOv2 (`dinov2_vits14`, `dinov2_vitb14`, or `dinov2_vitl14`), sets `eval()` mode, applies exact normalization parameters, processes batch sequences with zero-grad, and serializes fixed embeddings to disk (`.npy` or memory-mapped HDF5/Zarr) with associated frame metadata.


* **`representation-trajectory-analyzer`**
* **Purpose:** Pre-modeling representational sanity checks (Step 9 in your workflow).
* **Capabilities:** Calculates consecutive frame cosine similarity ($S_t = \cos(E_t, E_{t+1})$), cumulative cosine drift from $E_0$, and temporal variance. Automatically highlights sequences exhibiting near-static similarity profiles vs. abrupt delta spikes.



---

### 3. Model Training & Hypothesis Probing

* **`sequence-model-trainer`**
* **Purpose:** Controlled model execution engine.
* **Capabilities:** Trains the baseline MLP (single-frame) and the GRU temporal head with identical optimizer configurations (AdamW, cosine learning rate scheduler, weight decay, gradient clipping). Implements deterministic multi-seed training loops (`seeds: [42, 43, 44]`).


* **`temporal-perturbation-engine`**
* **Purpose:** Generates controlled intervention datasets to test H2, H3, and H4.
* **Capabilities:**
* *Temporal Shuffle:* Applies seeded permutation matrices across the time dimension ($T$).
* *Temporal Reversal:* Flips sequence index ($t \to T - t - 1$).
* *Static Position Control:* Replicates a single frame across $T$ positions ($E_t \to [E_k, E_k, \dots, E_k]$).
* *Corruption Injection:* Injects frame-level Gaussian blur, random zero-masking, or temporal frame-drop dropouts.





---

### 4. Evaluation, Statistics & Qualitative Discovery

* **`experiment-ledger-manager`**
* **Purpose:** Automates structured record-keeping (Step 33).
* **Capabilities:** Logs run configurations, metrics (Accuracy, Macro-F1, per-class F1, Expected Calibration Error), and parameter counts into `results/experiments.csv`. Computes statistical dispersion ($\mu \pm \sigma$, 95% bootstrap confidence intervals) across seeds.


* **`temporal-win-loss-miner`**
* **Purpose:** Qualitative error taxonomy and quadrant analysis (Step 21).
* **Capabilities:** Evaluates prediction vectors from the baseline vs. GRU model to filter samples into the four quadrants:
* *Quadrant A (Both Correct)*
* *Quadrant B (Baseline Correct, Temporal Wrong — motion overfit/noise)*
* *Quadrant C (Temporal Rescued — primary qualitative targets)*
* *Quadrant D (Both Failed)*


* Exports visual strip assets for all Quadrant C and B samples.



---

### 5. Multimodal / VLM Qualitative Synthesis

* **`vlm-semantic-change-probe`**
* **Purpose:** Semantic interpretation of multi-frame dynamics (Step 23–25).
* **Capabilities:** Formats sequence frame grids into a structured VLM call with strict JSON schema outputs. Asks the model to describe what state transition occurs between early and late observations, and compares VLM descriptions against ground-truth labels to isolate visual hallucinations or directional confusion.


* **`research-report-compiler`**
* **Purpose:** Compiles raw experiment ledgers, LaTeX ablation tables, and diagnostic plots into publication-ready figures and structured Markdown/PDF sections.



---

### Recommended Skill Integration Mapping

| Workflow Phase | Antigravity Skill | Primary Output |
| --- | --- | --- |
| **Data Audit (Step 3–5)** | `dataset-auditor`, `leakage-guard` | `audit.json`, leakage report, frame strips |
| **Embeddings (Step 7–9)** | `embedding-pipeline-runner`, `representation-trajectory-analyzer` | Frozen `.npy` caches, cosine similarity plots |
| **Experiments (Step 10–20)** | `sequence-model-trainer`, `temporal-perturbation-engine` | Multi-frame ladder curves, shuffle/corruption deltas |
| **Analysis (Step 21–27)** | `temporal-win-loss-miner`, `experiment-ledger-manager` | Quadrant matrix, confidence intervals, ablation table |
| **VLM & Synthesis (Step 23–35)** | `vlm-semantic-change-probe`, `research-report-compiler` | Semantic change logs, failure taxonomy, report draft |

---

### 6. Skill Deployment Status

All 10 skills are now deployed as Antigravity-discoverable `SKILL.md` files under `.agents/skills/` (workspace-level discovery path). Each skill is a fully actionable runbook with YAML frontmatter, activation criteria, exact CLI commands, and expected outputs.

| Skill | Deployed Path | Pipeline Day | Hypothesis |
| --- | --- | --- | --- |
| `dataset-auditor` | `.agents/skills/dataset-auditor/SKILL.md` | Day 1 | Pre-experiment |
| `leakage-guard` | `.agents/skills/leakage-guard/SKILL.md` | Day 1 | Pre-experiment |
| `embedding-pipeline-runner` | `.agents/skills/embedding-pipeline-runner/SKILL.md` | Day 2 | Pre-experiment |
| `representation-trajectory-analyzer` | `.agents/skills/representation-trajectory-analyzer/SKILL.md` | Day 2 | Pre-experiment |
| `sequence-model-trainer` | `.agents/skills/sequence-model-trainer/SKILL.md` | Day 3 | H1–H5 (setup) |
| `temporal-perturbation-engine` | `.agents/skills/temporal-perturbation-engine/SKILL.md` | Day 4–5 | H2, H3, H4 |
| `experiment-ledger-manager` | `.agents/skills/experiment-ledger-manager/SKILL.md` | Day 4–5 | All |
| `temporal-win-loss-miner` | `.agents/skills/temporal-win-loss-miner/SKILL.md` | Day 5 | H1, H5 |
| `vlm-semantic-change-probe` | `.agents/skills/vlm-semantic-change-probe/SKILL.md` | Day 6 | Qualitative |
| `research-report-compiler` | `.agents/skills/research-report-compiler/SKILL.md` | Day 7 | All |

#### Discovery Note
Antigravity automatically discovers these skills because `.agents/` is the workspace-level discovery root. No additional registration is needed — skills appear in the agent's available toolkit immediately.