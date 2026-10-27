# M1 — Dataset Selection & Audit

Now we move from **research specification → evidence about the data**.

As a senior ML researcher/engineer, I would treat M1 as a **gate**, not simply "download a dataset."

The output of M1 should let us answer:

> **Is this dataset actually capable of answering our M0 research question?**

We should **not choose the dataset because it is convenient**.

---

# M1 objective

We need to select **one sequential visual dataset** that gives us a credible opportunity to test:

1. Does additional visual context help?
2. Does temporal ordering matter?
3. Does temporal context help under corruption/missing information?
4. Can we perform these experiments within the available compute/time?

The assessment specifically expects a dataset with sequential observations and asks for dataset statistics, representative sequences, frame strips, temporal ambiguity analysis and discussion of when temporal information becomes useful. Take-Home Assessment

---

# M1 structure

I would divide M1 into **6 engineering tasks**.

```text
M1
│
├── M1.1 Candidate discovery
│
├── M1.2 Dataset selection
│
├── M1.3 Dataset acquisition
│
├── M1.4 Data integrity audit
│
├── M1.5 Temporal structure analysis
│
└── M1.6 Real data collection & split freeze
```

The important part is the final one.

Once M1.6 is complete:

> **The real dataset and group-aware split are frozen.**

We don't keep changing datasets because an experiment isn't producing the result we hoped for.

---

# M1.1 — Candidate discovery

Before downloading anything, define our requirements.

## Hard requirements

The dataset should have:

### 1. Ordered observations

Something like:

```text
Sequence A

F1 → F2 → F3 → F4 → F5 → ...
```

Not merely a collection of unrelated images.

---

### 2. Sequence-level labels

Ideally:

```text
Sequence → Label
```

rather than:

```text
Frame → Label
```

because our core question concerns temporal context.

---

### 3. Multiple observations per example

We need enough temporal length to perform:

```text
1 frame
2 frames
4 frames
8 frames
...
```

The exact values will depend on the dataset.

---

### 4. Some temporal ambiguity

This is extremely important.

We want examples where:

```text
F1
```

may be ambiguous, but:

```text
F1 → F2 → F3
```

provides additional information.

If every frame independently reveals the answer, then our temporal experiment becomes weak.

---

### 5. Manageable compute

We don't need a million-frame dataset.

For this assessment, a carefully selected subset is acceptable. Take-Home Assessment

A smaller dataset with excellent experimental control is preferable to a huge dataset that prevents us from completing the research properly.

---

# M1.2 — Candidate scoring

Before selecting anything, I would create a decision matrix.

| Criterion | Weight | Dataset A | Dataset B | Dataset C |
|---|---:|---:|---:|---:|
| Temporal structure | 25% | | | |
| Temporal ambiguity | 20% | | | |
| Label quality | 15% | | | |
| Sequence length | 10% | | | |
| Split integrity | 10% | | | |
| Compute feasibility | 10% | | | |
| Ease of preprocessing | 5% | | | |
| Reproducibility | 5% | | | |

Score each from:

```text
1 = poor
2 = weak
3 = acceptable
4 = good
5 = excellent
```

Then:

\[
Score = \sum_i w_i s_i
\]

This prevents us from choosing a dataset based on intuition alone.

---

# M1.3 — Dataset acquisition

Once we select the candidate:

```text
raw/
```

should be treated as **immutable**.

I would structure the project:

```text
data/
├── raw/
├── metadata/
├── processed/
└── embeddings/
```

Never modify the raw data in place.

---

## Acquisition manifest

Create:

```text
data/metadata/dataset_manifest.json
```

Record:

```text
dataset_name
dataset_version
source
download_date
license
number_of_sequences
number_of_frames
class_count
original_resolution
frame_rate
```

Also record checksums if practical.

Example:

```json
{
  "dataset": "...",
  "version": "...",
  "download_date": "2026-10-28",
  "source": "...",
  "license": "...",
  "raw_checksum": "..."
}
```

This is boring engineering.

It's also exactly the sort of thing that saves you when something breaks later.

---

# M1.4 — Data integrity audit

Now we stop thinking about ML and act like data engineers.

For every sequence:

```text
sequence_id
    ↓
frames
    ↓
ordering
    ↓
label
```

We need to validate each relationship.

---

## Check 1 — Missing frames

Suppose:

```text
F001
F002
F003
F005
F006
```

What happened to F004?

Determine whether:

- it genuinely doesn't exist,
- it was intentionally removed,
- the downloader failed,
- or the dataset itself contains a gap.

Do not silently ignore it.

---

# Check 2 — Duplicate frames

Detect cases such as:

```text
F17 == F18
```

Some duplication can be legitimate.

The important thing is to know it exists.

---

# Check 3 — Frame ordering

Verify:

```text
001
002
003
004
...
```

rather than accidentally loading:

```text
001
010
002
020
003
```

A temporal experiment is meaningless if the data loader destroys temporal order.

---

# Check 4 — Corrupt images

Test every image.

```text
open(image)
decode(image)
validate(shape)
```

Record failures:

```text
bad_frames.csv
```

---

# Check 5 — Resolution

Calculate:

```text
width
height
aspect ratio
```

We need to know whether resizing/cropping will remove meaningful information.

---

# Check 6 — Label integrity

For each sequence:

```text
sequence_id → label
```

Check:

- missing labels
- unknown labels
- duplicate labels
- inconsistent label names
- classes with extremely few examples

---

# Check 7 — Class distribution

Generate:

```text
Class A █████████████
Class B ███████
Class C ███
Class D █████████
```

Then calculate:

\[
N_c
\]

for every class \(c\).

This matters because an apparently strong accuracy can be caused by class imbalance.

---

# M1.5 — Temporal structure analysis

This is the most important part of M1.

We need to determine whether the dataset actually contains the type of temporal information we're trying to study.

---

## Analysis A — Sequence length

Calculate:

```text
min
max
mean
median
p25
p75
p95
```

Example:

```text
Sequence length

minimum     8
median     32
mean       37
p95        61
maximum   120
```

This will later determine our frame-sampling strategy.

---

# Analysis B — Temporal strips

For representative examples, create:

```text
F1       F2       F3       F4       F5
┌────┐   ┌────┐   ┌────┐   ┌────┐   ┌────┐
│    │ → │    │ → │    │ → │    │ → │    │
└────┘   └────┘   └────┘   └────┘   └────┘
```

We want examples from different categories:

### Category 1 — Clearly temporal

The action/state changes over time.

### Category 2 — Frame-identifiable

A single frame already reveals the label.

### Category 3 — Ambiguous

Early observations are unclear, but later observations resolve the ambiguity.

### Category 4 — Difficult

Even the complete sequence is difficult to interpret.

This categorization will become extremely useful during M4.

---

# Analysis C — Temporal ambiguity

This is where we connect M1 directly to M0.

For representative sequences ask:

> If I only see the first frame, can I confidently identify the label?

Then:

> What about two frames?

Then:

> Four frames?

Then:

> The complete sequence?

Conceptually:

```text
Information available

1 frame
   ↓
low confidence
   ↓
2 frames
   ↓
more context
   ↓
4 frames
   ↓
label becomes apparent
```

We don't necessarily need to turn this into a human annotation benchmark.

We're establishing whether the dataset provides a plausible environment for testing H1/H2.

---

# Analysis D — When does information appear?

For each representative sequence, note the earliest useful observation.

Example:

```text
Frame 1–3
    ↓
ambiguous

Frame 4
    ↓
object interaction appears

Frame 5
    ↓
state transition becomes clear

Frame 6+
    ↓
mostly redundant
```

This gives us a preliminary concept of:

\[
t_{\text{informative}}
\]

Later, M4 can test whether the model exhibits a similar saturation point.

---

# Analysis E — Adjacent visual change

We can perform a preliminary version of the representation analysis here, although the full embedding analysis belongs to M2.

For now, inspect:

```text
F1 → F2
F2 → F3
F3 → F4
...
```

and identify where significant visual changes occur.

Later, with DINOv2 embeddings, we'll quantify this using cosine similarity.

---

# M1.6 — Real data collection & split freeze

Now we transition from local synthetic validation to **real-world action video collection** from public machine learning hubs (Hugging Face Hub & Kaggle), followed by deterministic group-aware split freezing.

While synthetic generation establishes mathematically guaranteed ambiguity, scientific completeness requires demonstrating that the pipeline functions seamlessly on real-world human-object interactions (Something-Something V2, UCF-101, Kinetics).

### Real Data Ingestion Architecture
- **Hugging Face Hub Ingestion:** Integrated via `src/data/remote_ingestion.py` using `huggingface_hub` (`hf_hub_download`, `snapshot_download`) with optional `HF_TOKEN`.
- **Kaggle Ingestion:** Automated downloading via `kaggle datasets download -d <dataset_slug>` authenticated by `~/.kaggle/kaggle.json`.
- **Universal Container Decoding:** `/opt/homebrew/bin/ffmpeg` decodes MP4, AVI, WebM, and MOV containers into uniform 8-frame RGB sequences at $224 \times 224$.
- **CLI Commands:**
  ```bash
  # Hugging Face
  python3 scripts/prepare.py --dataset real-video --source hf --repo-id "sayakpaul/something-something-v2-mini" --audit

  # Kaggle
  python3 scripts/prepare.py --dataset real-video --source kaggle --repo-id "pevogam/something-something-v2" --audit
  ```

---

## Split design & leakage prevention

This is a **critical gate**.

Never do this:

```text
Video A
├── F1 → train
├── F2 → train
├── F3 → test
├── F4 → train
└── F5 → test
```

Those frames are highly correlated.

The model could effectively see the same scene during training and testing.

Instead:

```text
Sequence A ─────→ TRAIN

Sequence B ─────→ TRAIN

Sequence C ─────→ VALIDATION

Sequence D ─────→ TEST
```

If the dataset has subject identities, users, videos or sessions, prefer splitting at the highest meaningful independent unit.

For example:

```text
Subject 1 → train
Subject 2 → train
Subject 3 → validation
Subject 4 → test
```

rather than splitting frames randomly.

---

# M1 split policy

Before training, write this down:

```text
Split unit:
    <video / sequence / subject / session>

Train:
    X%

Validation:
    Y%

Test:
    Z%

Random seed:
    XXXXX

Stratification:
    yes/no

Leakage prevention:
    <specific rule>
```

Once this is frozen:

> **DO NOT TOUCH THE TEST SET FOR MODEL DEVELOPMENT.**

That's one of the most important rules of this project.

---

# M1 deliverables

At the end of M1, our repository should contain something like:

```text
src/
└── data/
    ├── dataset.py
    ├── audit.py
    ├── split.py
    └── visualize.py

data/
├── raw/
├── metadata/
│   ├── dataset_manifest.json
│   ├── sequences.csv
│   ├── class_distribution.csv
│   └── split_manifest.csv
└── processed/

results/
└── dataset/
    ├── dataset_statistics.csv
    ├── class_distribution.png
    ├── sequence_length_distribution.png
    ├── temporal_strips/
    └── audit_report.md
```

And importantly:

```text
docs/
└── dataset_decision.md
```

---

# `dataset_decision.md`

This should answer six questions.

### 1. Why did we choose this dataset?

Not:

> "It was easy to download."

Instead:

> "It contains ordered sequences where the target cannot always be inferred from an individual observation, making it appropriate for evaluating the contribution of temporal context."

---

### 2. What exactly is one sample?

For example:

```text
One sample = one complete sequence + one target label
```

or whatever the selected dataset actually defines.

---

### 3. What constitutes temporal information?

For example:

```text
relative ordering
state transition
motion
object interaction
context accumulation
```

We should only claim what the dataset actually supports.

---

### 4. What is our evaluation unit?

Potentially:

```text
sequence-level prediction
```

Again, this depends on the selected dataset.

---

### 5. How did we prevent leakage?

Explicitly document:

```text
split level
grouping variable
random seed
duplicate handling
```

---

### 6. What are the dataset limitations?

This is important.

For example:

```text
class imbalance
limited sequence length
dataset-specific visual bias
weak temporal ambiguity
small number of subjects
```

Don't hide weaknesses.

They become part of the scientific interpretation.

---

# M1 acceptance criteria

I would not mark M1 complete until **all** of these are true.

### Data

- [ ] Dataset version/source recorded.
- [ ] Raw data preserved.
- [ ] Dataset loads successfully.
- [ ] Every usable sequence has a valid label.
- [ ] Frame ordering verified.
- [ ] Corrupt/missing frames quantified.
- [ ] Duplicate data investigated.
- [ ] Class distribution measured.
- [ ] Sequence-length distribution measured.

### Temporal validity

- [ ] Representative sequences visually inspected.
- [ ] Temporal strips generated.
- [ ] At least some temporally ambiguous examples identified, **if they exist**.
- [ ] Examples where a single frame appears sufficient identified.
- [ ] Preliminary temporal-information analysis documented.

### Evaluation

- [ ] Train/validation/test split defined.
- [ ] Leakage risks identified.
- [ ] Split generated reproducibly.
- [ ] Test set frozen.

### Documentation

- [ ] Dataset decision written.
- [ ] Limitations written.
- [ ] No model trained on the test set.

---

# The M1 gate

At the end we make one of three decisions:

### 🟢 GO

The dataset contains sufficient sequential structure and is suitable for testing our hypotheses.

→ Proceed to **M2 — Visual Representation & Embedding Pipeline**.

### 🟡 CONDITIONAL GO

The dataset is usable, but temporal ambiguity is weaker than expected.

→ Proceed, but explicitly adjust the research interpretation.

For example:

> "This dataset primarily tests whether additional visual context improves classification, with weaker evidence for genuinely order-dependent reasoning."

### 🔴 NO-GO

The dataset doesn't support the research question.

Examples:

```text
Almost every frame independently reveals the label.
No reliable sequence boundaries.
Severe leakage.
Insufficient sequence length.
Labels incompatible with temporal prediction.
Compute requirements unrealistic.
```

Then we select another dataset **before M2**.

---

# What I would NOT do in M1

This is equally important.

Don't:

```text
❌ train GRU
❌ fine-tune DINOv2
❌ perform hyperparameter sweeps
❌ compare 10 architectures
❌ tune on test data
❌ start VLM experiments
❌ chase accuracy
```

M1 is about answering:

> **"Do we have the right experimental material?"**

Not:

> **"Can we make a model achieve 90%?"**

---

# M1 final output

By the end of M1, we should have this chain:

```text
Dataset
   ↓
Sequence definition
   ↓
Label definition
   ↓
Integrity audit
   ↓
Temporal structure
   ↓
Temporal ambiguity
   ↓
Leakage-safe split
   ↓
Frozen dataset
   ↓
M2
```

**The next concrete action is candidate selection.** We should now take 3–4 realistic sequential datasets and score them against this M1 matrix before downloading/building anything. That way we don't accidentally spend half the assessment implementing a dataset that cannot answer our research question.