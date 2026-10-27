I would manage this assessment like a senior ML engineering project: clear milestones, explicit deliverables, measurable exit criteria, risk management, and research decisions documented at every stage.

The goal is not to build the most complicated model. It is to deliver a reproducible experiment that demonstrates whether temporal information provides measurable predictive value. The assessment places particular emphasis on experimental design, temporal reasoning, data discipline, and research thinking.&#x20;

Take-Home Assessment.pdf



# Project roadmap

M0 — Research specification

Define the question, hypotheses, evaluation protocol and scope.

Day 1

M1 — Dataset selection and audit

Choose the dataset and validate its sequence structure and labels.

Days 1–2

M1.5 — Real Video Pipeline (Kaggle & Hugging Face Hub)

Build automated ingestion, container decoding, and frame extraction for real action videos from Hugging Face Hub and Kaggle.

Day 2

M2 — Data pipeline and visual embeddings

Build reliable preprocessing, embedding extraction and caching.

Days 2–3

M3 — Baselines and temporal model

Establish fair comparisons between single-frame and sequence-based approaches.

Days 3–4

M4 — Controlled experiments

Measure multi-frame gains, order sensitivity and robustness.

Days 4–6

M5 — VLM and failure analysis

Study semantic interpretation and inspect representative errors.

Days 5–6

M6 — Research report and repository

Make the evidence reproducible, explain the findings and finalize deliverables.

Days 6–7

M7 — Final verification and demo

Test setup instructions, inspect results, remove secrets and verify submission.

Days 8–9

This is an eight-milestone plan with overlap between suitable activities. We should target a complete submission by Day 7 and reserve Days 8–9 as contingency, consistent with the assessment's stated submission window.&#x20;

Take-Home Assessment.pdf



# M0 — Research specification

Objective: Establish exactly what the project must prove before implementation starts.

### Tasks

1. Write the primary research question.
2. Define the experimental hypotheses.
3. Define the prediction target and evaluation unit.
4. Identify confounding factors.
5. Freeze the initial experiment plan.

Our primary question:

> Does correctly ordered temporal context improve prediction beyond what can be achieved from a single observation or an unordered collection of observations?

Initial hypotheses:

| ID | Hypothesis                                 | Evidence required                              |
| -- | ------------------------------------------ | ---------------------------------------------- |
| H1 | Additional observations improve prediction | Performance as the number of frames increases  |
| H2 | Temporal order can matter                  | Correct-order versus shuffled-order comparison |
| H3 | Temporal context can help under corruption | Clean versus corrupted comparisons             |
| H4 | Temporal benefit varies by example         | Per-sequence error analysis                    |

These are hypotheses to test, not outcomes to assume.

### Deliverables

- `docs/research_plan.md`
- `configs/experiment.yaml`
- Initial evaluation protocol
- A risk and assumptions log

### Definition of done

The research plan identifies the dataset decision criteria, the baselines, the experiments, the metrics, and what evidence would count against the hypothesis.

Senior engineering rule: Do not change the primary hypothesis after seeing the test results merely to make the model look successful.

# M1 — Dataset selection and audit

Objective: Find a dataset in which temporal information is genuinely testable.

The assessment requires ordered sequences, integrity checks, basic statistics, temporal frame strips, and discussion of temporal ambiguity.&#x20;

Take-Home Assessment.pdf



### Tasks

First, compare candidate datasets against these criteria:

- Labels correspond to a well-defined prediction task.
- Sequence boundaries and temporal ordering are available.
- Individual frames can be ambiguous for at least some examples.
- The dataset can be downloaded and processed within the available compute budget.
- There are enough independent sequences to split training, validation and testing sensibly.

Next, implement the dataset loader and audit.

For each sequence, record:

```
sequence_id
label
ordered frame paths
number of frames
frame dimensions
timestamps or frame indices
source video / subject identity, when available
```

Generate statistics for sequence count, frames per sequence, class distribution, missing observations, duplicate frames and any detected integrity problems.

### Deliverables

```
src/data/
    dataset.py
    audit.py
    visualization.py

results/
    dataset_statistics.json
    sequence_strips/
    dataset_summary.png
```

### Definition of done

- Sequences load in the correct order.
- Labels map consistently to sequences.
- Representative sequences have been visually inspected.
- Train/validation/test splits are fixed and reproducible.
- Correlated frames from the same source sequence cannot leak across splits.

Go/no-go gate: If the data cannot support a meaningful temporal comparison, change the dataset before building the models. A sophisticated architecture cannot repair a poorly formulated experiment.

# M1.5 — Real Video Pipeline (Kaggle & Hugging Face Hub)

Objective: Build an automated, robust data pipeline to fetch, stream, decode, and extract real-world action video and image sequences directly from Hugging Face Hub and Kaggle.

While synthetic generation establishes mathematical ground truth for temporal ambiguity, empirical scientific rigor requires evaluating real-world human-object interactions (e.g. Something-Something V2, UCF-101, Kinetics).

### Tasks

1. **Remote Repository Ingestion Engine:**
   - Integrate `huggingface_hub` (`hf_hub_download`, `snapshot_download`) with optional `HF_TOKEN` support.
   - Integrate Kaggle API (`kaggle datasets download`) with `~/.kaggle/kaggle.json` authentication.
   - Support streaming or targeted archive extraction for specific action classes without downloading full 100GB+ datasets.

2. **Video Container & Stream Decoding:**
   - Implement universal container decoding (`ffmpeg` / `torchvision.io` / PIL) supporting MP4, AVI, WebM, and compressed image archives (tar/zip).
   - Uniformly sample $T=8$ or $T=16$ frames across sequence duration.
   - Resize and center-crop all extracted frames to standard ImageNet dimensions ($224 \times 224 \times 3$).

3. **Data Integrity & Immutability Enforcement:**
   - Preserve original downloaded containers in `data/raw/` as immutable artifacts.
   - Structure extracted frames into standardized format: `data/{class_name}/{video_id}/frame_{index:04d}.jpg`.
   - Run the 7-check data integrity audit (`src/data/audit.py`) and group-aware split generator (`src/data/dataset.py`).

### Deliverables

```
src/data/
    remote_ingestion.py

scripts/
    prepare.py (extended with --source hf | kaggle)

docs/
    remote_ingestion_guide.md
```

### Definition of done

- At least one real-world video dataset/subset from Hugging Face or Kaggle can be ingested via a single command.
- Video container decoding extracts ordered frames without dropping or corrupting headers.
- The 7-check data integrity audit runs and confirms zero missing frame gaps and group-aware split isolation.
- Offline fallback to synthetic generator remains available if API credentials or network connections are absent.

# M2 — Data pipeline and visual embeddings

Objective: Convert the dataset into validated, reusable numerical representations.

### Tasks

Choose a suitable pretrained vision encoder, such as DINOv2 or CLIP, and initially keep it frozen. The assignment explicitly permits pretrained visual representations.&#x20;

Take-Home Assessment.pdf



Implement the following pipeline:

```
Ordered sequence
      ↓
Frame sampling
      ↓
Image preprocessing
      ↓
Pretrained vision encoder
      ↓
Frame embeddings
      ↓
Validated embedding cache
```

For a sequence of \\(N\\) frames, the embedding tensor has the form:

\\[ E \in \mathbb{R}^{N \times D} \\]

where \\(N\\) is the number of sampled frames and \\(D\\) is the embedding dimension.

Cache the results instead of recomputing embeddings during every training run.

Then analyze representation changes:

\\[ s_t = \frac{E_t^\top E\_{t+1}} {\lVert E_t\rVert\\,\lVert E\_{t+1}\rVert} \\]

Plot adjacent-frame cosine similarity and investigate the corresponding frames when similarity changes sharply. Remember that a change in embedding similarity does not, by itself, prove that temporal reasoning has occurred.

### Deliverables

```
src/embeddings/
    extractor.py
    cache.py
    analysis.py

results/
    embeddings/
    embedding_similarity.png
    embedding_trajectory.png
```

### Definition of done

- Embeddings correspond to the correct frame and sequence IDs.
- Missing or invalid frames are handled explicitly.
- Cached features can be reused without rerunning the encoder.
- At least one visualization has an evidence-based interpretation.
- Model checkpoint, preprocessing and embedding dimensions are recorded.

Engineering gate: No model training until the data and embeddings pass integrity checks.

# M3 — Baselines and temporal model

Objective: Build the smallest set of models needed to isolate the value of temporal information.

I would implement three approaches, not just two. The third is an important experimental control.

### Model A — Single-frame baseline

Question: How well can the task be solved from one observation?

One frame → Frozen encoder → Small classifier

### Model B — Multi-frame, order-invariant control

Question: Does receiving more visual evidence help without modeling sequence order explicitly?

Multiple frames → Embeddings → Mean pooling → Classifier

### Model C — Temporal model

Question: Does modeling relationships among ordered observations provide additional value?

Ordered embeddings → GRU → Classifier

The single-frame baseline and temporal model fulfil the assessment's core model requirement. The mean-pooling control strengthens the scientific comparison by helping distinguish extra visual evidence from explicit temporal modeling.

### Tasks

1. Implement a majority-class baseline as a sanity check.
2. Train the single-frame classifier.
3. Train the mean-pooling control.
4. Train a small GRU-based temporal model.
5. Use the same data splits and visual embeddings.
6. Save training configuration, checkpoints, metrics and predictions.

Use a fixed training protocol where possible. Tune hyperparameters on the validation set only, and reserve the test set for final evaluation.

### Deliverables

```
src/models/
    baseline.py
    temporal.py

src/training/
    train.py

results/
    model_checkpoints/
    baseline_metrics.json
    predictions/
```

### Definition of done

All three approaches can be trained and evaluated using a common interface. The evaluation code produces comparable metrics from the same test sequences.

# M4 — Controlled research experiments

Objective: Establish what kind of temporal information helps and whether the temporal model actually uses it.

This is the project's most important milestone.

## Experiment 1 — Progressive observation

Evaluate the temporal model with increasing numbers of frames:

\\[ K \in \\{1,2,3,4,\ldots,N\\} \\]

Record accuracy, macro-F1 and the number of frames observed.

Expected output:

Illustrative progressive-observation experiment

Example structure for the plot only; values are hypothetical and must be replaced by actual measurements.

00.250.50.751123456

Illustrative values, not assessment results.

The goal is to discover where useful information appears and whether performance eventually saturates. The chart above demonstrates the format only; it contains no actual results.

## Experiment 2 — Correct order versus shuffled order

Use the same frames and the same frame count in both conditions.

```
Correct:   F1 → F2 → F3 → F4

Shuffled:  F3 → F1 → F4 → F2
```

Change the order and nothing else. Evaluate the trained temporal model without retraining it specifically for the shuffled test condition.

A meaningful performance drop is evidence of order sensitivity, although it does not by itself prove general temporal reasoning.

## Experiment 3 — Corruption and missing observations

Compare clean sequences with sequences in which selected frames are blurred, masked, or dropped.

Measure whether temporal context mitigates the effect of degraded observations.

## Experiment 4 — Temporal reversal (optional)

Compare the original sequence against the reversed sequence. This can expose sensitivity to motion direction when the task's labels depend on direction or state transition.

### Deliverables

```
src/experiments/
    progressive.py
    shuffle.py
    corruption.py
    reversal.py

results/
    progressive_results.csv
    order_results.csv
    corruption_results.csv
    figures/
```

### Definition of done

- At least two controlled experiments have been completed, as required by the assessment.
- Results are generated from a repeatable evaluation procedure.
- Comparisons use consistent sequences and metrics.
- Each experiment has an interpretation, including plausible alternative explanations.
- Negative or inconclusive findings are documented rather than hidden.

Priority: Complete progressive observation and temporal-order evaluation first. Add corruption evaluation next. Reversal is optional.

# M5 — VLM and failure analysis

Objective: Understand what semantic information a Vision-Language Model adds beyond the embedding-based models.

The assessment requires a small VLM experiment, but it does not require VLM fine-tuning.&#x20;

Take-Home Assessment.pdf



### Tasks

Select a small, representative set of sequences, ideally including:

- cases where temporal modeling improves prediction
- cases where it does not help
- ambiguous or corrupted cases
- cases where both models fail

Give the sequence to a suitable VLM and ask it to describe the observed changes, compare early and later observations, or identify what remains ambiguous.

Record:

```
sequence_id
ground_truth
single_frame_prediction
temporal_prediction
VLM_response
observed_failure
researcher_interpretation
```

Assess whether the VLM accurately describes the observations. Do not treat its response as ground truth.

### Deliverables

```
src/vlm/
    analyze.py

results/
    vlm_observations.json
    qualitative_analysis/
    failure_cases/
```

### Definition of done

The report explains what the VLM contributed, where its interpretations were useful, and where it made mistakes or failed to resolve ambiguity.

# M6 — Research report and repository

Objective: Turn the implementation and findings into a reproducible technical submission.

The assessment expects a Git repository, runnable code, a technical report of approximately 4–6 pages, metrics, plots, sequence visualizations, comparisons and failure cases.&#x20;

Take-Home Assessment.pdf



### Repository structure

```
temporal-visual-intelligence/
├── README.md
├── requirements.txt
├── report.pdf
├── configs/
│   └── experiment.yaml
├── src/
│   ├── data/
│   ├── embeddings/
│   ├── models/
│   ├── experiments/
│   ├── evaluation/
│   └── vlm/
├── scripts/
│   ├── prepare_data.py
│   ├── extract_embeddings.py
│   ├── train.py
│   └── evaluate.py
└── results/
    ├── tables/
    ├── figures/
    └── failure_cases/
```

### Report structure

| Section                        | Main question answered                                          |
| ------------------------------ | --------------------------------------------------------------- |
| 1. Problem and dataset         | What are we investigating, and why is this dataset suitable?    |
| 2. Method and architecture     | How do the baseline, pooling control and temporal model differ? |
| 3. Experiments                 | What controlled interventions were tested?                      |
| 4. Results                     | What did the measured results show?                             |
| 5. Research analysis           | When did temporal information help or fail?                     |
| 6. Limitations and future work | What conclusions are justified, and what remains unresolved?    |

### Definition of done

A second engineer should be able to understand the design decisions, reproduce the principal experiments, and trace each reported metric back to its saved evaluation output.

# M7 — Final verification

Objective: Ensure that the submission is complete, honest and executable.

Use this checklist on the final day.

## Submission readiness

### 0/14

Research

Primary question and hypotheses are documented

Dataset choice is justified

Train/validation/test isolation is verified

Models and experiments

Single-frame baseline and temporal model are implemented

At least two controlled experiments are complete

Metrics and figures match the saved evaluation outputs

Failure cases and negative findings are included

VLM and communication

VLM experiment is documented

System architecture diagram is present

Technical report is within the requested length

Engineering

README contains setup and reproduction instructions

Dependencies and configuration files are included

No API keys, passwords or unrelated confidential information are present

Final code and report have been checked together

The demo is optional but preferred. The assessment also mentions I/O screenshots in its submission instructions, so include those with your other supporting results.

# Project management rules I would enforce

These are the rules that keep the milestones from turning into an uncontrolled research project.

1\. Every milestone must produce a tangible artifact. Not just code written, but a validated dataset, a result table, a figure, or a documented decision.

2\. Keep one source of truth for experimental results. Save run configurations, random seeds, checkpoints, predictions, and metrics. Never manually invent or copy illustrative numbers into the report as measured results.

3\. Use stage gates. Do not start temporal-model experimentation while the dataset split is unreliable. Do not interpret results before the baselines run correctly. Do not write conclusions before the controlled experiments finish.

4\. Protect the core deliverables. If time runs short, reduce optional experiments and model complexity before sacrificing reproducibility, the required VLM component, the two controlled experiments, or the research report.

5\. Report what the evidence supports. A performance improvement alone does not prove temporal reasoning. A performance drop after shuffling is useful evidence, but it must be interpreted alongside the order-invariant control, dataset characteristics, and failure cases.

## What we should do first

We should start with M0: Research Specification, then move to M1 once we have a clear selection criterion. The first actual decision is which prediction task and dataset allow us to isolate the value of temporal information without unnecessary compute or implementation risk.

Would a daily milestone check-in help keep the assessment on track during the 7–9-day window?

Set up daily assessment milestone check-ins