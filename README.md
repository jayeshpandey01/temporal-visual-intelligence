# Temporal & Multimodal Visual Intelligence

**ML Research Engineer / Scientist Take-Home Technical Assessment**

> **Research Question**: *Can a model extract and utilize useful information from a sequence of visual observations that may not be available from an individual frame?*

---

## Overview

This repository implements a rigorous experimental system to investigate temporal reasoning in visual sequences. The approach follows a scientific protocol: hypotheses → controlled experiments → qualitative analysis → evidence-based conclusions.

**Hypotheses tested:**
| ID | Statement |
|----|-----------|
| H1 | More frames → better performance on temporally dependent examples |
| H2 | Destroying temporal order reduces performance (model uses relationships) |
| H3 | Temporal context recovers information under frame corruption |
| H4 | Performance saturates after sufficient temporal evidence |
| H5 | Temporal modeling adds little value when a single frame is discriminative |

---

## Environment Setup

```bash
# Python 3.9+
pip install -r requirements.txt

# For VLM probe (choose one):
export GEMINI_API_KEY=your_gemini_key
# OR
export OPENAI_API_KEY=your_openai_key
```

---

## Dataset Preparation

### Something-Something v2 (Recommended)
1. Register and download from: https://developer.qualcomm.com/software/ai-datasets/something-something
2. Extract frames: `ffmpeg -i video.webm frames/%05d.jpg`
3. Organize as: `data/{class_name}/{video_id}/frame_*.jpg`
4. Run: `python scripts/prepare.py --dataset ssv2 --n-per-class 100`

### UCF-101 (Alternative)
1. Download from: https://www.crcv.ucf.edu/data/UCF101.php
2. Extract frames and organize same structure
3. Run: `python scripts/prepare.py --dataset ucf101 --n-per-class 150`

---

## Pipeline (7 Steps)

```bash
# Day 1: Dataset audit + temporal strip visualization
python src/data/audit.py --data-root data/ --output results/dataset_audit.json

# Day 2: Extract frozen DINOv2 embeddings (run ONCE, cached)
python scripts/extract.py --config configs/experiment.yaml

# Day 3: Train baseline (single-frame MLP)
python scripts/train.py --config configs/experiment.yaml --model baseline --seeds 42 43 44

# Day 3: Train temporal model (GRU)
python scripts/train.py --config configs/experiment.yaml --model gru --seeds 42 43 44

# Day 4-5: Run full experiment suite (H1-H5)
python scripts/evaluate.py --config configs/experiment.yaml --all-experiments --seeds 42 43 44

# Day 6: VLM semantic change probe
python scripts/vlm_probe.py --config configs/experiment.yaml --n-samples 20

# Day 7: Compile results (tables, figures, report draft)
python scripts/compile_report.py --config configs/experiment.yaml
```

---

## Reproduce Results

```bash
# Smoke test (3 epochs, 1 seed, validates entire pipeline)
python scripts/train.py --config configs/experiment.yaml --model gru --smoke-test
python scripts/evaluate.py --config configs/experiment.yaml --smoke-test

# Full reproduction (3 seeds, all experiments)
python scripts/train.py --config configs/experiment.yaml --model baseline --seeds 42 43 44
python scripts/train.py --config configs/experiment.yaml --model gru --seeds 42 43 44
python scripts/evaluate.py --config configs/experiment.yaml --all-experiments --seeds 42 43 44
```

---

## Repository Structure

```
temporal-visual-intelligence/
├── README.md
├── requirements.txt
├── configs/
│   └── experiment.yaml        # All experiment configs (config-driven)
├── src/
│   ├── data/
│   │   ├── dataset.py         # Video-level GroupKFold splits (no leakage)
│   │   ├── sampling.py        # Frame sampling strategies
│   │   └── audit.py           # Dataset audit + temporal strips
│   ├── embeddings/
│   │   ├── extractor.py       # Frozen DINOv2 extraction + caching
│   │   └── analysis.py        # Cosine similarity trajectory analysis
│   ├── models/
│   │   ├── baseline.py        # Single-frame MLP
│   │   └── temporal.py        # Temporal GRU + Transformer
│   ├── experiments/
│   │   ├── progressive.py     # H1: 1→N frame sweep
│   │   ├── shuffle.py         # H2: order test + repeated-frame control
│   │   ├── corruption.py      # H3: robustness experiments
│   │   └── sparsity.py        # H4: dense/sparse/early/late
│   ├── evaluation/
│   │   ├── metrics.py         # Accuracy, Macro-F1, mean±std, CI
│   │   └── analysis.py        # Win/loss quadrant miner (A/B/C/D)
│   └── vlm/
│       └── analyze.py         # VLM semantic change probe
├── scripts/
│   ├── prepare.py             # Dataset preparation
│   ├── extract.py             # Embedding extraction
│   ├── train.py               # Model training (multi-seed)
│   ├── evaluate.py            # Experimental evaluation
│   ├── vlm_probe.py           # VLM qualitative analysis
│   └── compile_report.py      # Report & ablation table compiler
├── results/
│   ├── experiments.csv        # Canonical experiment ledger
│   ├── figures/               # Auto-generated plots
│   ├── tables/                # Ablation tables
│   └── qualitative/           # Quadrant C/B frame strips
└── .agents/
    └── skills/                # 10 Antigravity agent skills
        ├── dataset-auditor/
        ├── leakage-guard/
        ├── embedding-pipeline-runner/
        ├── representation-trajectory-analyzer/
        ├── sequence-model-trainer/
        ├── temporal-perturbation-engine/
        ├── experiment-ledger-manager/
        ├── temporal-win-loss-miner/
        ├── vlm-semantic-change-probe/
        └── research-report-compiler/
```

---

## Key Design Decisions

**Why freeze DINOv2?** The research question is about temporal modeling, not representation quality. Fine-tuning the encoder introduces a confound — we'd be comparing encoder fine-tuning dynamics rather than temporal reasoning.

**Why video-level splits?** Frame-level splits cause catastrophic leakage: near-identical frames from the same video appear in both train and test, inflating metrics artificially.

**Why the repeated-frame control?** The core question is whether the model uses temporal *order/relationships*, not just whether it benefits from more input positions. The repeated-frame condition (same frame × N) tests this precisely.

---

## Third-Party Resources Used

- [DINOv2](https://github.com/facebookresearch/dinov2) — Oquab et al., 2023 (visual encoder, frozen)
- [PyTorch](https://pytorch.org/) — model implementation
- [scikit-learn](https://scikit-learn.org/) — GroupKFold splits, metrics
- [Google Gemini API](https://ai.google.dev/) — VLM semantic probe

