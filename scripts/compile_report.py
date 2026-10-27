"""
Research Report Compiler (Day 7 of pipeline).
(research.md §27, §35)

Compiles experiment ledgers, figures, and qualitative findings into publication-ready tables
and Markdown reports.

Usage:
    python scripts/compile_report.py --config configs/experiment.yaml
    python scripts/compile_report.py --table ablation
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_ablation_table(csv_path: str, output_path: str):
    """Aggregate experiments.csv into formatted ablation table."""
    if not os.path.exists(csv_path):
        print(f"No experiment ledger found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Experiment ledger is empty.")
        return

    # Group by key experimental variables
    group_cols = [c for c in ['model', 'num_frames', 'order', 'corruption'] if c in df.columns]
    agg_df = df.groupby(group_cols).agg(
        acc_mean=('accuracy', 'mean'),
        acc_std=('accuracy', 'std'),
        f1_mean=('macro_f1', 'mean'),
        f1_std=('macro_f1', 'std'),
        n_seeds=('seed', 'count')
    ).reset_index()

    print("\n--- ABLATION TABLE SUMMARY ---")
    print(agg_df.to_string(index=False))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("# Experimental Ablation Summary\n\n")
        f.write(agg_df.to_markdown(index=False))
    print(f"\nAblation table written to {output_path}")


def generate_report_draft(config: dict, output_path: str):
    """Generate Markdown draft synthesizing all results and analysis."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_content = f"""# Technical Report: Temporal & Multimodal Visual Intelligence

## 1. Executive Summary & Research Question
**Core Question:** Can a model extract and utilize useful information from a sequence of visual observations that may not be available from an individual frame?

### Hypotheses Tested
- **H1 (Multi-frame Gain):** Increasing observation count improves performance on temporally dependent actions.
- **H2 (Temporal Order):** Destroying sequential order (shuffling) reduces performance, verifying genuine temporal reasoning rather than set-based visual aggregation.
- **H3 (Corruption Robustness):** Temporal context compensates for visual noise, blur, or occluded frames.
- **H4 (Temporal Saturation):** Information gains plateau once discriminative state changes have been captured.
- **H5 (Task Dependence):** Single-frame visual bias dominates static actions; temporal modeling provides isolated lift on dynamic state-transition actions.

---

## 2. Methodology & Experimental Design
- **Visual Encoder:** Frozen `{config['encoder']['name']}` (dim={config['encoder']['embed_dim']}), ImageNet normalized.
- **Single-Frame Baseline:** MLP operating on representative frame observations.
- **Temporal Sequence Model:** GRU / Transformer head operating on temporal embedding trajectories.
- **Discipline:** Video-level group splits with GroupKFold (zero cross-split frame leakage). Identical optimizer (AdamW), LR, and random seeds across models.

---

## 3. Results & Ablation Summary
See `results/tables/ablation.md` for multi-seed aggregated metrics.

---

## 4. Qualitative Quadrant Analysis
- **Quadrant A (Both Correct):** Static-heavy classes where a single frame provides conclusive context.
- **Quadrant B (Baseline Correct, Temporal Wrong):** Motion noise or state overfitting in sequence modeling.
- **Quadrant C (Temporal Rescued):** Critical qualitative validation where sequence modeling resolved ambiguity.
- **Quadrant D (Both Failed):** Insufficient signal or fine-grained ambiguity.

---

## 5. VLM Semantic Synthesis & Multimodal Design
The VLM probe reveals state changes and directional cues that bridge numerical feature space and semantic explanations.
Future multimodal expansion fuses Depth, IMU, and Audio streams through cross-attention temporal aggregation.
"""
    with open(output_path, 'w') as f:
        f.write(report_content)
    print(f"Report draft generated at {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment.yaml')
    parser.add_argument('--table', choices=['ablation'], default='ablation')
    parser.add_argument('--output-table', default='results/tables/ablation.md')
    parser.add_argument('--output-report', default='results/report_draft.md')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ledger_path = config.get('paths', {}).get('experiments_csv', 'results/experiments.csv')
    generate_ablation_table(ledger_path, args.output_table)
    generate_report_draft(config, args.output_report)

