Yes. If you approach this as a **senior ML Research Engineer / Scientist**, I would not start by writing the GRU. I would treat the assessment as a **small research study** whose implementation is the instrument used to answer a hypothesis.

The assessment itself explicitly prioritizes **experimental design, temporal reasoning, data/evaluation discipline, and research thinking** over maximizing accuracy. Take-Home Assessment

Below is the exact workflow I would follow.

---

# 0. Define the research question before touching the code

Your primary research question should be:

> **Does ordered temporal context provide predictive information that is unavailable from an individual visual observation?**

Then break it into hypotheses.

### H1 — Multi-frame information

> Increasing the number of observations improves performance on temporally dependent examples.

### H2 — Temporal order

> If the model uses temporal relationships, destroying the original order should reduce performance.

### H3 — Robustness

> Temporal context should help recover information when individual observations are corrupted or ambiguous.

### H4 — Temporal saturation

> Performance should eventually plateau once sufficient temporal evidence has been observed.

### H5 — Task dependence

> Temporal modeling should provide little benefit for examples whose label is already identifiable from a single frame.

This gives you a scientific framework.

---

# 1. Decide what exactly you are measuring

Do **not** begin with:

> "I will use DINOv2 + GRU."

Begin with:

```text
                RESEARCH QUESTION
                       │
                       ↓
             What information is
             contained in time?
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
   Static information         Temporal information
          │                         │
          ↓                         ↓
   Single frame              Sequence/order/change
          │                         │
          └────────────┬────────────┘
                       ↓
                 Prediction
                       ↓
                 Evaluation
```

Your independent variable is primarily:

> **amount and structure of temporal information**

Your dependent variable:

> **prediction performance**

Controlled variables:

- dataset
- label
- visual encoder
- classifier capacity
- training split
- test samples
- preprocessing
- evaluation metric

This is how you avoid an apples-to-oranges experiment.

---

# 2. Select the dataset based on the research question

The assessment only requires a suitable sequential/video/egocentric dataset and explicitly says a small subset is acceptable. Take-Home Assessment

Don't choose a dataset because it is famous.

Choose it because it has **temporal ambiguity**.

A useful way to think about datasets is:

### Static-heavy dataset

```text
Frame alone → label
```

Example concept:

```text
person + guitar → playing guitar
```

Temporal information may barely matter.

### Temporal-heavy dataset

```text
Frame alone → ambiguous
Sequence → label
```

Example:

```text
A → B → C

opening something
vs
closing something
```

The distinction can depend on motion/order.

This distinction is not merely theoretical; temporal-action research has specifically found that many video datasets contain actions recognizable from a single frame, while some classes genuinely require temporal information. [Meta AI](https://ai.meta.com/blog/introducing-the-temporal-data-set-a-benchmark-for-recognizing-actions-in-videos/?utm_source=chatgpt.com)

### Senior-level decision

Before committing to the dataset, perform a **dataset audit**.

---

# 3. Dataset audit

Create:

```text
dataset_audit.ipynb
```

or preferably:

```text
src/data/audit.py
```

Measure:

```text
Number of sequences
Number of frames
Frames / sequence
Sequence duration
Classes
Samples / class
Missing frames
Duplicate frames
Frame dimensions
Frame rate if available
Class imbalance
```

Then inspect samples.

For each class:

```text
Sequence 1:
F1 F2 F3 F4 F5

Sequence 2:
F1 F2 F3 F4 F5
```

Create temporal strips.

The assessment explicitly asks for these visualizations. Take-Home Assessment

---

# 4. Establish whether temporal information is plausibly necessary

This is where I would go beyond a basic assignment implementation.

Take representative sequences and perform:

### Test A — single-frame inspection

Show one frame.

Ask:

> Could a human reasonably determine the label?

### Test B — temporal inspection

Show:

```text
F1 → F2 → F3 → F4 → F5
```

Ask:

> Does the sequence resolve ambiguity?

Create categories:

```text
Temporal relevance

0 = unnecessary
1 = mildly useful
2 = strongly useful
```

This doesn't need to become a formal ground-truth label unless you have time.

It is exploratory analysis.

---

# 5. Define your data split BEFORE experimentation

This is extremely important.

You want:

```text
TRAIN
  ↓
VALIDATION
  ↓
TEST
```

But don't randomly split individual frames from the same sequence across train/test.

Bad:

```text
Video A:
F1 → train
F2 → test
F3 → train
F4 → test
```

This causes leakage because nearly identical frames from the same video appear in both sets.

Instead:

```text
Video A → TRAIN
Video B → TRAIN
Video C → VALIDATION
Video D → TEST
```

Ideally, keep correlated observations from the same underlying sequence together.

This should be explicitly documented in your report.

---

# 6. Establish the simplest possible baseline

Before DINOv2 + GRU, establish:

```text
majority-class baseline
```

and ideally:

```text
random baseline
```

Then:

```text
single-frame visual baseline
```

This tells you whether your system is actually learning.

---

# 7. Freeze the visual encoder initially

Use:

```text
Image
 ↓
DINOv2
 ↓
embedding
```

Do **not** fine-tune it initially.

Why?

Because your research question is:

> Does temporal modeling help?

If you simultaneously fine-tune:

- vision encoder
- temporal encoder
- classifier

you introduce too many variables.

Instead:

```text
             F1
             ↓
          DINOv2
             ↓
             E1

             F2
             ↓
          DINOv2
             ↓
             E2

             ...

             FN
             ↓
          DINOv2
             ↓
             EN
```

Now your controlled experiment is much cleaner.

---

# 8. Store embeddings separately

Do not repeatedly run DINOv2 during every experiment.

Create:

```text
embeddings/
    train/
    val/
    test/
```

Conceptually:

```text
sequence_001.npy
sequence_002.npy
sequence_003.npy
```

Each:

```text
[N, D]
```

where:

- N = number of frames
- D = embedding dimension

Example:

```text
sequence_001
    ↓
[32 frames, 768 dimensions]
```

Now your temporal experiments are cheap.

---

# 9. Perform representation analysis before training the temporal model

This is explicitly required by the assessment. Take-Home Assessment

Calculate:

\[
S_t = cosine(E_t,E_{t+1})
\]

Then plot:

```text
Similarity
1.0 ┤●──●──●
    │
0.8 ┤         ●
    │
0.6 ┤             ●
    │
0.4 ┤
    └──────────────────
       t1 t2 t3 t4 t5
```

Now investigate:

> What happened visually when similarity changed?

That connects:

```text
embedding behavior
       ↓
visual behavior
       ↓
temporal interpretation
```

That's much stronger than simply producing a graph.

---

# 10. Build the single-frame baseline

Architecture:

```text
Frame
  ↓
Frozen DINOv2
  ↓
Embedding
  ↓
Linear / small MLP
  ↓
Class
```

For each sequence, you need a clearly defined frame selection strategy.

For example:

```text
first frame
```

or:

```text
random frame during training
fixed representative frame during evaluation
```

But whatever you choose, document it.

---

# 11. Build the temporal model

Now:

```text
E1
E2
E3
...
EN
 │
 ↓
GRU
 │
 ↓
Temporal hidden state
 │
 ↓
MLP
 │
 ↓
Class
```

Start with a small GRU.

Something conceptually like:

```text
Embedding D
     ↓
Linear projection
     ↓
Temporal dimension
     ↓
GRU
     ↓
Last hidden state
     ↓
Classifier
```

Do not make it huge.

The assessment explicitly says the temporal architecture should remain reasonably small and that state-of-the-art architecture is not the objective. Take-Home Assessment

---

# 12. Make the comparison fair

This is where senior-level thinking matters.

You need:

```text
Baseline
    vs
Temporal
```

with as much as possible held constant.

For example:

| Property | Single | Temporal |
|---|---|---|
| Visual encoder | DINOv2 | DINOv2 |
| Encoder weights | Frozen | Frozen |
| Embedding | Same | Same |
| Classifier family | MLP | MLP |
| Dataset | Same | Same |
| Train split | Same | Same |
| Test split | Same | Same |
| Label | Same | Same |

The main difference:

```text
single:
E_t

temporal:
E_1 ... E_N
```

That makes your conclusion defensible.

---

# 13. Experiment 1 — Information ladder

This should be your first major experiment.

Evaluate:

```text
1 frame
2 frames
3 frames
4 frames
...
N frames
```

For example:

```text
N = 8
```

Generate:

```text
Accuracy vs number of frames
F1
F2
F3
...
F8
```

Also calculate:

\[
\Delta_N = Performance_N - Performance_{1}
\]

This gives you **multi-frame gain**.

Recent temporal evaluation work similarly emphasizes distinguishing whether a task itself requires multiple frames from whether the model genuinely exploits temporal information. [Academus](https://academ.us/article/2607.12304/?utm_source=chatgpt.com)

---

# 14. Don't only report accuracy

Depending on the task, report:

```text
Accuracy
Macro F1
Per-class F1
Confusion matrix
```

If classes are imbalanced, macro-F1 becomes especially important.

Also report:

```text
mean ± standard deviation
```

if you repeat experiments across seeds/splits.

For example:

```text
Single frame:
82.1 ± 0.8

4-frame:
87.4 ± 0.6
```

This is much stronger than:

```text
82.1 vs 87.4
```

because you have some indication of variability.

---

# 15. Experiment 2 — Temporal order test

Now construct:

### Correct

```text
F1 → F2 → F3 → F4 → F5
```

### Shuffled

```text
F3 → F1 → F5 → F2 → F4
```

Critically:

```text
same frames
same number of frames
same model
same preprocessing
```

Only:

```text
ORDER
```

changes.

Then:

\[
OrderDrop =
Performance_{correct}
-
Performance_{shuffled}
\]

For example:

```text
Correct     88%
Shuffled    73%

OrderDrop   15 pp
```

That's strong evidence that order matters.

Frame shuffling has been used specifically as a way of destroying temporal structure while preserving the set of visual observations, which makes it a useful controlled test for temporal dependence. [Meta AI](https://ai.meta.com/blog/introducing-the-temporal-data-set-a-benchmark-for-recognizing-actions-in-videos/?utm_source=chatgpt.com)

---

# 16. But add one more control

This is where I would make your assessment stronger.

Compare:

```text
Correct:
F1 F2 F3 F4

Shuffled:
F3 F1 F4 F2
```

with:

```text
Repeated:
F2 F2 F2 F2
```

Why?

The repeated-frame condition asks:

> Does having multiple input positions help, even when no new visual information is introduced?

You can think of:

```text
Single frame
      ↓
Repeated same frame
      ↓
True sequence
```

This starts separating:

```text
more input positions
vs
new visual information
vs
temporal relationships
```

That's a very strong research control.

---

# 17. Experiment 3 — Corruption

Take:

```text
F1 F2 F3 F4 F5
```

Corrupt one or more frames:

```text
F1 F2 BLUR(F3) F4 F5
```

or:

```text
F1 F2 [DROP] F4 F5
```

or:

```text
F1 F2 MASK(F3) F4 F5
```

Then compare:

```text
single-frame degradation
vs
temporal-model degradation
```

Your research question becomes:

> Can temporal context compensate for unreliable observations?

The assessment explicitly asks for corruption/occlusion experiments. Take-Home Assessment

---

# 18. Measure corruption sensitivity

Don't simply say:

> "Temporal model was more robust."

Calculate:

\[
RobustnessDrop =
Performance_{clean}
-
Performance_{corrupted}
\]

Example:

```text
                 Clean    Corrupted    Drop

Single frame      84%       62%        22 pp

Temporal          88%       79%         9 pp
```

Now you have a quantitative result.

---

# 19. Experiment 4 — Temporal reversal

This is an optional experiment I would seriously consider.

Correct:

```text
F1 → F2 → F3 → F4
```

Reverse:

```text
F4 → F3 → F2 → F1
```

This is different from random shuffling.

Why?

Because reversal preserves **all frames and local adjacency relationships**, but reverses direction.

If the task has directional semantics, this can be revealing.

Recent work also discusses reversal-based evaluation as a way to probe whether a model is actually sensitive to temporal ordering rather than merely seeing a collection of frames. [Academus](https://academ.us/article/2607.12304/?utm_source=chatgpt.com)

Don't make this your primary required experiment; make it a controlled supplementary experiment.

---

# 20. Experiment 5 — Temporal sparsity

Suppose you have:

```text
F1 F2 F3 F4 F5 F6 F7 F8
```

Compare:

```text
Dense:
F1 F2 F3 F4 F5 F6 F7 F8

Sparse:
F1 F3 F5 F7

Early:
F1 F2 F3 F4

Late:
F5 F6 F7 F8
```

This tells you:

> **When is the information actually available?**

That directly addresses the assessment's question about when additional information becomes available. Take-Home Assessment

---

# 21. Perform per-example analysis

This is something I would absolutely include.

Don't only analyze:

```text
overall accuracy
```

Find:

### Temporal wins

```text
Baseline: wrong
Temporal: correct
```

### Temporal losses

```text
Baseline: correct
Temporal: wrong
```

### Both correct

```text
Baseline: correct
Temporal: correct
```

### Both wrong

```text
Baseline: wrong
Temporal: wrong
```

Then create:

```text
                 Temporal
               Correct Wrong
Baseline
Correct          A       B
Wrong            C       D
```

Especially inspect **C**:

> cases where temporal information rescued the prediction.

These are probably your strongest qualitative examples.

---

# 22. Build a temporal information taxonomy

From those examples, categorize why temporal information helped.

For example:

```text
Temporal benefit
│
├── Motion
├── State transition
├── Object interaction
├── Occlusion resolution
├── Direction
├── Context accumulation
└── Disambiguation
```

Then show representative sequences.

That turns your result into a research story.

---

# 23. VLM experiment

Now introduce the VLM **after** the quantitative experiments.

Don't make it the central model.

The assessment specifically asks for a small VLM experiment and says fine-tuning is unnecessary. Take-Home Assessment

Use it for:

```text
Sequence description
Temporal change detection
Ambiguous example analysis
Early vs late comparison
```

For example:

```text
INPUT

Frame 1
Frame 2
Frame 3
Frame 4

QUESTION

"What changes across these observations?
What information becomes available only after
observing multiple frames?"
```

---

# 24. Treat VLM as a semantic analysis layer

Your architecture should be:

```text
                 Visual sequence
                       │
              ┌────────┴────────┐
              ↓                 ↓
         DINOv2              VLM
              ↓                 ↓
        Embeddings        Language semantics
              ↓                 ↓
         GRU / Temporal      Change explanation
              ↓                 ↓
              └────────┬────────┘
                       ↓
                 Understanding
```

Don't claim:

> VLM is better.

Instead:

> The VLM provides a complementary semantic interpretation that numerical embeddings and temporal classifiers do not directly provide.

---

# 25. VLM failure analysis

Record:

```text
Sequence ID
Ground truth
VLM description
Correct?
What did VLM miss?
```

Look for:

- hallucination
- incorrect ordering
- missed subtle motion
- object confusion
- background bias
- incorrect causal interpretation

This demonstrates that you understand VLM limitations.

---

# 26. Statistical discipline

This is where you can make your submission look substantially more senior.

If possible:

```text
3–5 random seeds
```

or repeated evaluation.

Report:

```text
mean ± std
```

And where practical:

```text
95% confidence interval
```

For example:

```text
             Accuracy

Baseline     81.9 ± 0.7
Temporal     86.4 ± 0.5
```

Don't overdo statistical testing on a tiny dataset, but quantify uncertainty honestly.

---

# 27. Ablation matrix

I would create one final table like:

| Experiment | Baseline | Temporal | Δ |
|---|---:|---:|---:|
| 1 frame | 82.1 | 82.1 | 0 |
| 2 frames | — | 84.7 | +2.6 |
| 4 frames | — | 87.3 | +5.2 |
| 8 frames | — | 87.5 | +5.4 |
| Shuffled | — | 76.2 | -11.3 |
| Corrupted | 61.4 | 78.2 | +16.8 |
| Reversed | — | 80.1 | -7.2 |

Obviously, these numbers are examples, not expected results.

The important thing is the **structure of the evidence**.

---

# 28. Determine the actual conclusion

At the end, don't ask:

> "Did GRU beat baseline?"

Ask:

### Question 1

Does more temporal context help?

### Question 2

Does correct order matter?

### Question 3

Does temporal context help under corruption?

### Question 4

How many frames are actually necessary?

### Question 5

Which types of examples benefit?

### Question 6

When is temporal modeling unnecessary?

Now your conclusion becomes something like:

> Temporal information provided a meaningful improvement primarily on sequences involving state transitions and directional interactions. Performance saturated after four observations, while frame shuffling caused a substantial degradation, suggesting that the temporal model exploited sequence order rather than merely receiving additional visual observations. However, examples with visually discriminative single frames showed negligible temporal benefit.

**That is a research conclusion.**

Not:

> "GRU accuracy was 89%."

---

# 29. Define failure cases before writing the report

You need examples where:

```text
Baseline correct
Temporal wrong
```

Ask why.

Possibilities:

```text
Temporal model overfits motion
Noisy frames
Incorrect temporal sampling
Long sequence loses important signal
Class imbalance
Background changes
GRU state contamination
```

Also:

```text
Baseline wrong
Temporal wrong
```

These tell you where temporal information isn't sufficient.

---

# 30. Build the final system architecture

I would show two diagrams.

### Experimental architecture

```text
Video
 │
 ├── Frame Sampling
 │
 └── Frames
       │
       ↓
   Frozen DINOv2
       │
       ↓
Frame Embeddings
       │
       ├───────────────┐
       ↓               ↓
Single Frame         GRU
Baseline              │
       │               ↓
       │           Temporal State
       │               │
       └───────┬───────┘
               ↓
          Classification
               ↓
          Evaluation
```

### Future multimodal architecture

```text
RGB ──→ Visual Encoder ──┐
                         │
Depth ─→ Depth Encoder ──┤
                         │
IMU ───→ IMU Encoder ────┤
                         ↓
                   Multimodal Fusion
                         ↓
                   Temporal Model
                         ↓
                  Semantic / VLM
                         ↓
               Prediction / Planning
```

The latter satisfies the system-design requirement in the assessment. Take-Home Assessment

---

# 31. Engineering implementation

Only after the research protocol is fixed should you finalize the repository.

I'd use:

```text
temporal-visual-intelligence/
│
├── README.md
├── requirements.txt
├── report.pdf
│
├── configs/
│   └── experiment.yaml
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── sampling.py
│   │   └── audit.py
│   │
│   ├── embeddings/
│   │   └── extractor.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   └── temporal.py
│   │
│   ├── experiments/
│   │   ├── progressive.py
│   │   ├── shuffle.py
│   │   ├── corruption.py
│   │   └── reversal.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── analysis.py
│   │
│   └── vlm/
│       └── analyze.py
│
├── scripts/
│   ├── prepare.py
│   ├── extract.py
│   ├── train.py
│   └── evaluate.py
│
└── results/
    ├── figures/
    ├── tables/
    └── qualitative/
```

---

# 32. Make every experiment configuration-driven

Don't hardcode:

```python
num_frames = 8
```

Instead:

```yaml
sequence:
  max_frames: 8
  sampling: uniform

encoder:
  name: dinov2
  freeze: true

temporal:
  model: gru
  hidden_dim: 256
  layers: 1

evaluation:
  seeds: [42, 43, 44]
```

Then:

```bash
python scripts/train.py --config configs/experiment.yaml
```

This makes the work reproducible.

---

# 33. Maintain an experiment ledger

This is something I would genuinely do as a research engineer.

Create:

```text
results/experiments.csv
```

Columns:

```text
experiment_id
dataset_version
seed
model
num_frames
sampling
order
corruption
accuracy
macro_f1
notes
```

Example:

```text
E001 | baseline | 1 | normal | 82.1 | ...
E002 | GRU      | 2 | normal | 84.7 | ...
E003 | GRU      | 4 | normal | 87.3 | ...
E004 | GRU      | 4 | shuffled | 76.2 | ...
```

Now your final report is generated from recorded experiments rather than manually copied numbers.

---

# 34. Your 7-day research execution

I would divide the assessment like this.

## Day 1 — Problem + dataset

```text
Dataset selection
Dataset audit
Sequence inspection
Train/val/test split
Temporal examples
```

Deliverable:

```text
dataset_audit.png
sequence_visualizations.png
```

---

## Day 2 — Representation

```text
DINOv2 setup
Embedding extraction
Embedding caching
Cosine similarity
Temporal trajectories
```

Deliverable:

```text
embedding_analysis.png
```

---

## Day 3 — Baselines

```text
Majority baseline
Single-frame baseline
GRU baseline
```

Deliverable:

```text
baseline_results.csv
```

---

## Day 4 — Main experiments

```text
1 → N frames
correct vs shuffled
```

Deliverable:

```text
progressive_curve.png
shuffle_comparison.png
```

---

## Day 5 — Robustness

```text
blur
masking
frame dropping
possibly reversal
```

Deliverable:

```text
corruption_results.png
failure_cases/
```

---

## Day 6 — VLM + analysis

```text
VLM experiment
qualitative analysis
failure analysis
architecture diagram
```

---

## Day 7 — Research report

```text
README
report
tables
figures
reproduction
demo
final cleanup
```

The assessment expects a Git repository, technical report, results and optionally a short demo. Take-Home Assessment

---

# 35. What I would put in the 4–6 page report

### Page 1

**Problem + Dataset**

```text
Research question
Hypotheses
Dataset
Sequence characteristics
```

### Page 2

**Method**

```text
DINOv2
Single-frame baseline
GRU
Architecture diagram
```

### Page 3

**Main experiments**

```text
Progressive observation
Temporal order
```

### Page 4

**Robustness + VLM**

```text
Corruption
VLM
Qualitative examples
```

### Page 5

**Research analysis**

```text
When temporal helps
When it doesn't
Evidence for temporal reasoning
Failure cases
```

### Page 6

**Limitations + future work**

```text
Dataset limitations
Model limitations
Experimental limitations
Multimodal future architecture
```

---

# 36. Your final research argument should follow this structure

This is probably the single most important framework I'd use:

```text
QUESTION
   ↓
Does temporal information provide additional predictive value?
   ↓
HYPOTHESIS
   ↓
Sequence information should improve ambiguous cases.
   ↓
BASELINE
   ↓
Single-frame model
   ↓
INTERVENTION
   ↓
Add ordered observations
   ↓
CONTROL
   ↓
Shuffle / repeat / corrupt / reverse
   ↓
MEASUREMENT
   ↓
Accuracy + Macro-F1 + confidence/variance
   ↓
QUALITATIVE ANALYSIS
   ↓
Temporal wins + failures
   ↓
VLM
   ↓
Semantic interpretation
   ↓
CONCLUSION
   ↓
When does temporal information actually matter?
```

That is the mindset I would expect from a **senior ML Research Scientist**.

---

# 37. The three strongest pieces of evidence

If we have limited time, I would prioritize these three:

### Evidence #1 — Information curve

```text
Performance
   ↑
   │                 ●──●
   │             ●
   │         ●
   │     ●
   │ ●
   └────────────────────→
      1  2  3  4  5  6
          frames
```

Shows **how much information becomes available over time**.

### Evidence #2 — Shuffle experiment

```text
Correct sequence  ███████████ 88%

Shuffled sequence █████████    75%
```

Shows whether **order matters**.

### Evidence #3 — Temporal rescue cases

```text
               Baseline   Temporal
Example A         ✗          ✓
Example B         ✗          ✓
Example C         ✗          ✓
```

Then show the actual sequences.

This answers:

> **What kind of information did temporal modeling actually recover?**

Together, these three are much more convincing than a single accuracy number.

---

# 38. The senior-level standard

At the end, I want your project to let a reviewer trace this chain:

```text
Dataset
   ↓
Why this dataset?
   ↓
Why should temporal information matter?
   ↓
How did you verify that?
   ↓
What representation did you use?
   ↓
Why?
   ↓
What is the baseline?
   ↓
What exactly changed in the temporal model?
   ↓
What controlled experiments did you run?
   ↓
Did order matter?
   ↓
Did additional observations help?
   ↓
Did temporal context improve robustness?
   ↓
Where did it fail?
   ↓
What did the VLM add?
   ↓
What can you actually conclude?
   ↓
What can you NOT conclude?
```

That last question — **what can you not conclude** — is particularly important.

A senior researcher doesn't turn:

> “GRU improved accuracy by 4%”

into:

> “GRU understands temporal reasoning.”

Instead:

> “Under these controlled conditions, the temporal model demonstrated sensitivity to sequence information, as evidenced by the performance gap between correctly ordered and shuffled observations. However, this does not establish general temporal reasoning beyond the selected dataset and task.”

**That is the level of reasoning I would aim for.**

And it aligns very closely with the assessment's own instruction that technically honest conclusions are valuable even when temporal modeling does **not** improve the selected task. Take-Home Assessment

### One important next step

I would **not start coding yet**. First, let's do **Step 1 as a real research exercise: select and justify the dataset**. We should compare 2–4 suitable sequential datasets specifically against the assessment's criteria — temporal ambiguity, manageable size, labels, sequence structure, ease of downloading, compute requirements, and ability to run the progressive/shuffle/corruption experiments — and then lock the dataset before designing the model.