# Reproduction Runbook & Verification Protocol

This document describes available diagnostic and reproduction commands. The checkpoint script re-evaluates existing weights; it does not retrain the models and is not an independent replication. Historical significance reports remain unverified.

---

## 1. Environment Setup

### System Prerequisites
- **Python**: Version 3.10 or 3.11
- **PyTorch**: Version 2.0+ (CPU or CUDA/Apple Silicon MPS)
- **Memory**: Minimum 8 GB RAM

### Installation
Clone the repository and install core dependencies:

```bash
git clone https://github.com/jayeshpandey01/temporal-visual-intelligence.git
cd temporal-visual-intelligence
pip install -r requirements.txt
```

A normal Git clone does not contain the 12 model checkpoints or any `.npy` feature arrays required by the recorded headline diagnostic: `.pt` and `.npy` files are ignored, and this workspace’s weights/arrays are local-only. Consequently, `scripts/reproduce_results.py` cannot run from a clean clone as currently documented. The repository also tracks 189 frame images, but their dataset identity and redistribution rights are unresolved. Do not treat the following commands as sufficient acquisition or as a clean-clone reproduction until permission-cleared data and checkpoint acquisition are documented:

```bash
python3 scripts/prepare.py --dataset ssv2-synthetic --subset-classes 20 --n-per-class 100
python3 scripts/extract.py --config configs/experiment.yaml
```

The example preparation mode generates synthetic sequences. The checked-in config currently names `ssv2`, and the synthetic generator orders classes differently from the recorded prediction labels; this command does not reconstruct the checkpoint training inputs. The current dataset files’ identities and label mapping are unresolved, so do not compare this regenerated data with saved checkpoint metrics. The current configuration requests `dinov2_vits14`. The extractor may fall back to a different offline encoder if pretrained weights cannot be loaded; such a fallback does **not** reproduce the recorded DINOv2 predictions. Check the emitted encoder/provenance metadata and SHA-256 cache manifest before comparing metrics. Real-clip notebook cells additionally require permission-cleared videos/frames and their precomputed real-sequence embeddings. See [asset provenance](ASSET_PROVENANCE.md). The tracked dataset manifest's license field is not a substitute for source-level permission records.

---

## 2. One-Command Headline Findings Reproduction

Run the current checkpoint diagnostic only after providing the exact training checkpoints and feature arrays. A fresh clone is currently missing these ignored assets, so the command does not run from Git contents alone. It records hashes and writes its measurements and comparison with historical claims.

```bash
python3 scripts/reproduce_results.py
```

### Current diagnostic output
Local-workspace diagnostic (requires the ignored local checkpoints and feature arrays): the latest CPU run reports Model A/B/D/C accuracy means of 8.78%, 8.22%, 23.56%, and 34.78% across seeds [42, 43, 44]; Model C shuffle accuracy 20.67% (14.11 pp drop), reversed accuracy 15.89% (18.89 pp drop), and K=1/K=8 accuracy 11.56%/34.78% (+23.22 pp). Macro-F1 and population standard deviations are in `results/final/headline_metrics.json`, along with checkpoint and config hashes and protocol notes. These are measurements from the current checkpoint files. Historical M4 tables still disagree on shuffle. Five paired statistical contrasts (H1–H4, H14) have been recomputed from source prediction rows using sequence-level percentile bootstrap intervals, Monte Carlo sign-flip tests (100,000 permutations), and BH correction within this five-test family:

```bash
python3 scripts/recompute_core_statistics.py
```

Results are written to `results/final/paired_statistics.json` and `results/tables/paired_statistics.md`. The inference unit is the sequence, with correctness differences averaged across seeds; all intervals and tests condition on saved checkpoints and omit training-seed variability. Other M6 contrasts remain unverified because complete paired inputs were unavailable. See [the reconciliation table](results/tables/m7_reconciliation_table.md) before citing historical results.

---

## 3. Seven-Check Scientific Integrity Audit

Run the integrity audit as a diagnostic and inspect each check. It intentionally exits nonzero while statistical validity and claim reconciliation remain unverified. The current implementation checks parameter counts, not checkpoint hashes; checks a validation-monitor setting, not all test-set access; and uses a synthetic tensor-sum check for intervention behavior. It is not a scientific certification.

```bash
python3 scripts/verify_scientific_integrity.py
```

### Audited Dimensions:
- **1. Dataset Integrity**: Exactly 20 action classes, 100 sequences per class (N=2,000 total sequences, 16,000 frames).
- **2. Split Integrity**: Checks that sequence IDs are disjoint across Train (1,400), Validation (300), and Test (300); identity leakage beyond those IDs requires a separate audit.
- **3. Checkpoint Diagnostic**: Checks parameter counts for expected model sizes; hashes are recorded separately by the headline diagnostic.
- **4. Evaluation Configuration**: Checks that early stopping monitors a validation metric; does not establish all test-set access boundaries.
- **5. Intervention Integrity**: A synthetic tensor-sum check exercises shuffle/reversal conservation; the complete experiment pipeline requires separate review.
- **6. Statistical Integrity**: Review paired tests, their exact prediction inputs, and multiple-testing procedure.
- **7. Claim Traceability**: Deliberately remains failed until all headline claims are reconciled from source predictions.

The current tracked tree contains raw image frames. A clean-clone reproduction cannot be considered supported until those files are permission-cleared or excluded and the data acquisition instructions are complete.

---

## 4. Render the report PDF

Install the report-rendering dependencies from `requirements.txt`, then render the reviewed Markdown source:

```bash
python3 scripts/render_report_pdf.py
```

The script writes `report.pdf` at the repository root. It uses Markdown 2 and ReportLab, embeds local raster images referenced by the report, and keeps PDF text selectable. Mathematical expressions are emitted as readable source-style text rather than typeset LaTeX; inspect the PDF before distribution. To choose another destination, use `--output PATH`.

---

## 5. Full Unit Test Battery

Run the comprehensive unit test suite:

```bash
pytest --tb=short -q
```
Latest local checkout result (2026-10-29): 360 passed. This is not clean-clone verification; required ignored checkpoints/features are present in this workspace.

---

## 6. Live Interactive Sequence Demonstration

Run the interactive sequence inspector to verify predictions and counterfactual perturbations live:

```bash
# Showcase sequence (vid_07_0001: Pure Temporal Order Rescue)
python3 scripts/run_scientific_demo.py

# Multi-archetype automated suite
python3 scripts/run_scientific_demo.py --demo-suite

# Inspect arbitrary sequence by ID
python3 scripts/run_scientific_demo.py --sequence-id vid_07_0001 --intervention canonical
```

---

## 6. End-to-End Training Pipeline (Optional)

To retrain all models from scratch across seeds [42, 43, 44]:

```bash
# Model A: Single-Frame MLP
python3 scripts/train_single.py

# Model B: MultiFrame Mean-Pooling Control
python3 scripts/train_mean.py

# Model D: Compact TemporalGRU (103.7k Parity)
python3 scripts/train_compact.py

# Model C: Scaled TemporalGRU (249.4k)
python3 scripts/train_gru.py
```

