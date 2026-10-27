# M0 — Research Specification

**Project:** Temporal & Multimodal Visual Intelligence  
**Document status:** Complete (M0 Exit Criteria Approved & Locked); dataset-specific choices locked during M1  
**Purpose:** Define the research question, hypotheses, scope, controls, and evaluation protocol before implementation.

## 1. Objective

Determine whether ordered temporal context provides predictive value beyond a single visual observation, and identify the conditions under which it helps, does not help, or improves robustness to degraded observations.

The aim is not to maximize a headline metric or claim general temporal reasoning from one experiment. The aim is a small, reproducible study with conclusions proportionate to the evidence.

## 2. Primary research question

> **Does correctly ordered temporal context improve prediction beyond what can be achieved from a single observation or from an unordered collection of the same observations?**

This question separates three mechanisms that must not be conflated:

1. **Additional visual evidence:** more frames expose more content.
2. **Temporal structure:** the order and relationships among frames contain information.
3. **Contextual robustness:** surrounding observations can compensate for an ambiguous, missing, or corrupted frame.

## 3. Provisional task definition

The working plan is a **sequence-level classification task** using a labelled video or sequential-vision dataset. Each source sequence has one target label, and all models produce one prediction per sequence. This is provisional: M1 must confirm that the selected dataset's label semantics support this task. If not, revise this definition before training and record the reason.

**Unit of prediction and primary unit of evaluation:** one sequence, not one frame.  
**Input:** one or more chronologically ordered frames sampled from a sequence.  
**Output:** the sequence label.  
**Primary comparison:** a single-frame baseline versus a temporal sequence model.  
**Additional control:** an order-invariant multi-frame baseline using the same sampled frame embeddings, such as mean pooling followed by a classifier.

## 4. Hypotheses and falsifiable expectations

### H1 — Additional observations

**Hypothesis:** Performance improves when more observations are available, at least for examples whose labels depend on changes across time.

**Test:** Measure performance at increasing observation counts and compare against the single-frame reference.

**Evidence against H1:** Additional observations consistently produce negligible or negative improvement on a dataset with enough temporal ambiguity and adequate statistical power.

### H2 — Temporal order

**Hypothesis:** If a trained temporal model uses order-sensitive relationships, evaluation on shuffled versions of the same test frames will be worse than evaluation on the correct order.

**Test:** Hold the frames, frame count, preprocessing, trained weights, and test examples fixed; change only the order. Use deterministic, recorded permutations.

**Caution:** A shuffle-related performance drop is evidence of order sensitivity, not definitive proof of general temporal reasoning. Shuffling can also create unusual inputs outside the training distribution.

### H3 — Order versus additional visual evidence

**Hypothesis:** An order-sensitive temporal model may outperform both a single-frame classifier and an order-invariant multi-frame control on tasks with meaningful temporal dependencies.

**Test:** Compare a single-frame classifier, a mean-pooled multi-frame model, and a GRU-based temporal model using the same underlying frozen frame embeddings and data splits.

**Evidence against H3:** The temporal model does not outperform the order-invariant control, or the difference is too uncertain or too small to matter for the selected task.

### H4 — Robustness to degraded observations (core if time allows)

**Hypothesis:** Context from neighboring observations can reduce the performance loss caused by selected corrupted or missing frames.

**Test:** Compare clean and corrupted test conditions for the single-frame and temporal approaches. Predefine the corruption type and severity, and apply matched corruption conditions.

**Evidence against H4:** The temporal model has an equal or larger relative performance drop than the single-frame baseline under the tested corruption condition.

### H5 — Observation saturation (descriptive)

**Hypothesis:** Performance eventually plateaus as additional observations become redundant.

**Test:** Plot performance against observation count and report where gains become negligible. Do not assume saturation must occur within the tested range.

## 5. Model plan

Keep the visual encoder frozen initially so the experiment isolates the value of temporal modeling rather than simultaneously changing the visual representation.

- **Sanity baseline:** majority-class prediction.
- **Model A — Single-frame:** pretrained visual encoder embedding plus a small classifier. Use a deterministic, predeclared frame-selection rule; check anchor-frame sensitivity if the label may depend strongly on when the frame was captured.
- **Model B — Multi-frame, order-invariant control:** encode the same sampled frames, mean-pool their embeddings, then classify. This tests whether extra frames help without explicit order modeling.
- **Model C — Temporal:** pass the ordered frame embeddings through a small GRU and classify from its sequence representation.
- **VLM component:** use a suitable pretrained/API/open-source VLM for a small qualitative sequence-level description and temporal-change analysis. It is complementary analysis, not ground truth and not a substitute for the quantitative comparison.

Do not fine-tune the vision encoder or VLM in the initial experiment unless a clear, recorded blocker justifies changing scope.

## 6. Dataset and split protocol

Dataset selection is deferred to M1, but the dataset must satisfy these gates:

1. Ordered observations or reliable frame indices/timestamps are available.
2. Labels and sequence boundaries are understandable.
3. Some examples plausibly require information across time; this must be checked by inspecting actual examples, not assumed.
4. The subset is manageable with available compute and the assessment timeline.
5. There are enough independent sequences for a meaningful held-out evaluation.

**Split rules:**

- Split by source video, subject, session, or another available grouping variable to prevent correlated frames from leaking across splits.
- Never randomly distribute frames from the same source sequence across train and test.
- Fix and record train/validation/test identifiers before model comparison.
- Use validation for model/parameter choices; reserve test data for the locked final evaluation.
- If the dataset is too small for one reliable split, consider group-aware cross-validation and document the trade-off.
- Check duplicates and near-duplicates across splits where feasible.

## 7. Evaluation protocol

### Primary metric

Use **macro-F1** as the primary metric for a multi-class classification task, so each class contributes equally. Confirm this choice after the dataset audit. Report accuracy as a secondary metric, plus per-class F1 and a confusion matrix. If M1 reveals a different task type or label structure, revise the metrics before training and record the decision.

### Fair comparisons

- Evaluate all main models on the same held-out sequences and target labels.
- Reuse the same frozen visual embeddings, preprocessing, and sequence samples where applicable.
- Keep training/validation/test splits fixed.
- Record model configuration, checkpoint, random seed, frame sampling, and evaluation code for every run.
- Compare sequence-level predictions, not a mixture of frame-level and sequence-level metrics.
- Report paired performance differences on the same test examples.
- Where compute and sample size permit, repeat training across multiple seeds and report mean and standard deviation. For uncertainty estimates, consider a sequence-level bootstrap confidence interval; avoid implying precision unsupported by a small dataset.

### Experiment A — Progressive observation (required priority)

Evaluate performance at increasing numbers of frames, starting at one and increasing to a manageable maximum. Define sampling before running the test.

Two distinct questions may be tested, and must be labelled correctly:

- **Online prefix:** use the first K ordered observations to study how evidence becomes available over time.
- **Frame-count control:** uniformly sample K frames across the full sequence to study the effect of the number of observations while preserving temporal coverage.

Do not mix these interpretations in one curve without clearly distinguishing them. Compare the GRU to the single-frame reference and, when feasible, the mean-pooling control. Report the gain relative to the one-frame condition and whether/where performance saturates.

### Experiment B — Correct order versus shuffled order (required priority)

Use the same trained temporal model and the same exact test frames. Compare correct order with one or more recorded deterministic shuffles. Keep frame count, image content, preprocessing, and weights unchanged. Report the absolute performance difference and inspect examples affected by shuffling. The order-invariant mean-pooling control should be invariant to permutation apart from numerical implementation effects; this is a useful sanity check.

### Experiment C — Corruption / missing observations (recommended core extension)

Choose one primary intervention, such as blur, partial masking, or dropping a selected observation. Specify severity and affected-frame selection before final evaluation. Compare clean-to-corrupted performance drops for the baseline and temporal model. Do not claim general robustness from one corruption type.

### Qualitative and VLM analysis

Select a small set of representative sequences including temporal wins, temporal losses, and cases both models get wrong. For the VLM, record the input frames, prompt, response, any ground-truth comparison, and observed errors. Separate model outputs from researcher interpretation. Do not infer causality merely from a fluent VLM description.

## 8. Interpretation rules

- Better temporal-model accuracy alone does not prove that the model uses temporal order.
- Better performance with more frames may reflect additional visual content rather than temporal relationships.
- A drop under shuffling supports order sensitivity but can be affected by distribution shift caused by unnatural permutations.
- A failure to improve may mean the task is largely solvable from individual frames, the dataset lacks temporal ambiguity, the model is undertrained, or the experiment lacks statistical power. Inspect these alternatives before concluding temporal information is inherently useless.
- Only claim robustness for the corruption types and severity actually tested.
- Report negative and inconclusive results as well as positive results.

## 9. Scope boundaries

### In scope

- One suitable labelled sequential-vision dataset and a documented manageable subset.
- One frozen pretrained visual encoder.
- A small single-frame classifier, order-invariant pooling control, and one temporal architecture (GRU by default).
- Progressive-observation and shuffled-order experiments.
- One controlled corruption experiment if time permits.
- A small VLM sequence-analysis experiment.
- Reproducible code, figures, failure cases, an architecture diagram, README, and a 4–6 page report.

### Out of scope for the initial submission

- Training a vision encoder or VLM from scratch.
- Large-scale hyperparameter sweeps or chasing state-of-the-art results.
- Building a production video-understanding service.
- Adding depth, audio, IMU, pose, or other modalities to the implemented experiment; these may appear in the future-system architecture if justified.
- Claiming general temporal reasoning or general robustness from a single dataset.

Optional work is added only after all required deliverables and controlled experiments are reproducible.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Dataset labels are recognizable from one frame | Inspect representative sequences before locking the dataset; report this limitation honestly if found. |
| Source-sequence leakage | Group splits by source video/subject/session and audit split identifiers. |
| Extra frames confound temporal value | Include an order-invariant multi-frame pooling control and same-frame shuffling. |
| Single-frame anchor is unusually informative | Predeclare the anchor rule and run a limited first/centre/last sensitivity check if needed. |
| Small test set creates noisy metrics | Report per-class results, paired differences, and uncertainty where feasible. |
| VLM hallucinations are mistaken for evidence | Use VLM outputs for semantic analysis only and verify against visible observations. |
| Compute or time overruns | Freeze the encoder, cache embeddings, use a small GRU, prioritize the two required experiments, and defer optional work. |

## 11. M0 deliverables and exit criteria

M0 is complete when the following are agreed and recorded:

- [x] **Primary research question and provisional task definition.** (Agreed in §2–§3; formal mathematical mapping in `research_paper.md` §1–§2. Evaluates sequence-level classification from frozen visual representations.)
- [x] **Testable hypotheses with outcomes that could contradict them.** (Agreed in §4; formal falsification criteria established for H1–H5 including null-effect and negative-transfer boundaries in `research_paper.md` §2.)
- [x] **Candidate dataset selection criteria and M1 decision gate.** (Agreed in §6 & §12; gated on temporal ambiguity, Something-Something V2 / TDS 20-class subset selected to prevent static shortcuts per Sevilla-Lara et al., 2021.)
- [x] **Initial model comparison: single frame, order-invariant multi-frame, temporal model.** (Agreed in §5; Majority Baseline, Model A Single-Frame MLP, Model B Multi-Frame Mean-Pooling Control, and Model C GRU/Transformer documented with identical frozen DINOv2 backbone and capacity match in `research_paper.md` §4.)
- [x] **Primary and secondary metrics with sequence-level evaluation.** (Agreed in §7; Primary: Macro-F1 across sequences; Secondary: Accuracy, per-class F1, paired difference bootstrap 95% CIs, confusion matrices.)
- [x] **Data-splitting and leakage-prevention rules.** (Agreed in §6; Group-aware partitioning enforced on source video/session IDs via `leakage-guard` with zero sequence overlap between train/val/test.)
- [x] **At least two controlled experiments specified in enough detail to reproduce.** (Agreed in §7; Experiment A Progressive Sweep [1, 2, 3, 4, 6, 8 frames], Experiment B Correct vs Shuffled Order with Static-Repeat control, and Experiment C Controlled Corruption [25% blur/mask/drop] detailed in `research_paper.md` §2 & `configs/experiment.yaml`.)
- [x] **In-scope and out-of-scope decisions.** (Agreed in §9; Strictly limits scope to frozen encoder, controlled head comparisons, and qualitative VLM probe; excludes unconstrained pretraining or unverified benchmark chasing.)
- [x] **Risks and mitigations recorded.** (Agreed in §10; Risk matrix with 7 documented failure modes and operational mitigations logged.)

## 12. Decisions intentionally deferred to M1

The following cannot be responsibly fixed before the dataset is selected and inspected:

- final dataset and subset size;
- final label mapping and whether sequence-level classification is appropriate;
- frame rate, maximum frames, and exact sampling parameters;
- final model input dimensions and compute budget;
- whether macro-F1 is suitable or a different metric is required;
- corruption type/severity and the size of the qualitative VLM sample;
- realistic effect-size and uncertainty expectations given the dataset size.

These decisions must be made before inspecting final test performance, recorded in the experiment configuration, and changed only with an explicit note explaining why.
