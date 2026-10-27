"""
CLI runner for M1.5 Temporal Structure Analysis.

Usage:
    python3 scripts/analyze_temporal_structure.py --data-root data/
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.temporal_structure import run_temporal_structure_analysis


def main():
    parser = argparse.ArgumentParser(description="M1.5 Temporal Structure Analysis Runner")
    parser.add_argument("--data-root", type=str, default="data", help="Root directory of dataset")
    parser.add_argument("--output-dir", type=str, default="results/dataset", help="Output directory for reports")
    parser.add_argument("--figures-dir", type=str, default="results/figures", help="Output directory for figures")
    args = parser.parse_args()

    run_temporal_structure_analysis(
        data_root=args.data_root,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
    )


if __name__ == '__main__':
    main()

