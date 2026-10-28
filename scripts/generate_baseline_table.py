#!/usr/bin/env python3
"""
CLI script to generate Table 1: Baseline and Benchmark Overview (Milestone M3.17).

Enforces the scientific rule:
"The numbers remain blank until we actually run the experiments.
Do not put hypothetical values into the report."

Usage:
    # Generate blank template mandated by M3.17
    python3 scripts/generate_baseline_table.py --blank --output results/tables/first_baseline_table_blank.md

    # Generate live populated table from completed runs
    python3 scripts/generate_baseline_table.py --output results/tables/first_baseline_table.md
"""

import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp')
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.tables import generate_first_baseline_table


def main():
    parser = argparse.ArgumentParser(description="Generate Milestone M3.17 Table 1: Baseline and Benchmark Overview")
    parser.add_argument(
        "--blank",
        action="store_true",
        help="Generate blank template with '—' placeholders matching M3.17 verbatim.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/tables/first_baseline_table.md",
        help="Path where markdown table will be saved.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/experiments.csv",
        help="Path to experiments.csv ledger.",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="data/metadata/splits/test.json",
        help="Path to test split JSON to evaluate Majority baseline.",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="data/metadata/splits/train.json",
        help="Path to train split JSON to identify Majority class mode.",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default="data/metadata/split_manifest.json",
        help="Path to split_manifest.json containing train/val/test video IDs.",
    )
    args = parser.parse_args()

    test_labels = None
    train_labels = None

    if not args.blank:
        # Check split manifest first
        if os.path.exists(args.split_manifest):
            try:
                with open(args.split_manifest) as f:
                    splits = json.load(f)
                if "test" in splits:
                    test_labels = np.array([int(v.split('_')[1]) for v in splits["test"] if '_' in v])
                if "train" in splits:
                    train_labels = np.array([int(v.split('_')[1]) for v in splits["train"] if '_' in v])
            except Exception as e:
                print(f"Warning: could not parse split manifest {args.split_manifest}: {e}")

        # Fallback to test_split file if specified and not yet loaded
        if test_labels is None and os.path.exists(args.test_split):
            try:
                with open(args.test_split) as f:
                    test_data = json.load(f)
                test_labels = np.array([item["label"] for item in test_data])
            except Exception:
                pass

    md_content, df_out = generate_first_baseline_table(
        experiments_source=args.csv if not args.blank else None,
        output_path=args.output,
        blank_template=args.blank,
        test_labels=test_labels,
        train_labels=train_labels,
    )

    mode_str = "Blank Template" if args.blank else "Populated Table"
    print(f"\nGenerated {mode_str} successfully at: {args.output}\n")
    print(md_content)


if __name__ == "__main__":
    main()
