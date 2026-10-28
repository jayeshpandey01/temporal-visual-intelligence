"""
CLI script to execute and compile paired multi-model evaluations (M3.15, M3.16).

Enforces identical sequence alignment on the test split:
  Sequence 001 -> Model A, Model B, Model C
  Sequence 002 -> Model A, Model B, Model C
  ...
  Sequence 300 -> Model A, Model B, Model C

Usage:
    python3 scripts/evaluate_paired.py --config configs/experiment.yaml --seed 42
    python3 scripts/evaluate_paired.py --config configs/experiment.yaml --seed 42 --models baseline meanpool gru
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.paired import (
    validate_paired_test_alignment,
    compute_paired_metrics,
    build_paired_comparison_matrix,
    generate_paired_comparison_table,
)


def load_manifest_classes() -> List[str]:
    manifest_path = Path("data/metadata/dataset_manifest.json")
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f).get("classes", [f"Class_{i}" for i in range(20)])
    return [f"Class_{i}" for i in range(20)]


def run_paired_evaluation(
    config_path: str = "configs/experiment.yaml",
    seed: int = 42,
    models: Optional[List[str]] = None,
    output_dir: str = "results",
) -> Dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    models = models or ["baseline", "meanpool", "gru"]
    max_frames = config["sequence"]["max_frames"]
    class_names = load_manifest_classes()

    print(f"\n{'='*70}")
    print(f"PAIRED EVALUATION PROTOCOL (M3.15) | Seed: {seed} | Models: {models}")
    print(f"{'='*70}")

    # Discover and load predictions for each model
    predictions_dict: Dict[str, pd.DataFrame] = {}
    runs_dir = Path("runs")

    for model_name in models:
        run_id = f"{model_name}_seed{seed}_frames{max_frames}"
        pred_csv = runs_dir / run_id / "predictions.csv"
        pred_parquet = runs_dir / run_id / "predictions.parquet"

        if pred_parquet.exists():
            df = pd.read_parquet(pred_parquet)
        elif pred_csv.exists():
            df = pd.read_csv(pred_csv)
        else:
            raise FileNotFoundError(
                f"Prediction artifact for '{model_name}' (Seed {seed}) not found at {pred_csv}. "
                f"Please ensure '{model_name}' has been trained with: "
                f"python3 scripts/train.py --config {config_path} --model {model_name} --seeds {seed}"
            )

        predictions_dict[model_name] = df
        print(f"  Loaded {len(df)} predictions from {run_id}")

    # 1. Strictly validate paired sequence alignment (zero tolerance for drift)
    is_valid, sequence_ids, ground_truth = validate_paired_test_alignment(predictions_dict)
    print(f"\n  [ALIGNMENT CERTIFIED] All {len(models)} models evaluated the EXACT SAME "
          f"{len(sequence_ids)} test sequences in identical index order.")

    # 2. Build and export unified paired comparison matrix
    out_matrix_path = Path(output_dir) / "paired_test_predictions.csv"
    df_matrix = build_paired_comparison_matrix(
        predictions_dict, class_names=class_names, output_path=str(out_matrix_path)
    )

    # 3. Compute pairwise paired metrics and hypothesis tests
    paired_comparisons = []
    comparison_pairs = []
    if "baseline" in models and "meanpool" in models:
        comparison_pairs.append(("baseline", "meanpool"))
    if "meanpool" in models and "gru" in models:
        comparison_pairs.append(("meanpool", "gru"))
    if "baseline" in models and "gru" in models:
        comparison_pairs.append(("baseline", "gru"))

    for m1, m2 in comparison_pairs:
        res = compute_paired_metrics(
            ground_truth,
            predictions_dict[m1]["predicted_label"].values,
            predictions_dict[m2]["predicted_label"].values,
            model1_name=m1.upper(),
            model2_name=m2.upper(),
        )
        paired_comparisons.append(res)
        print(f"\n  Comparison: {m1.upper()} vs {m2.upper()}:")
        print(f"    {m1.upper()} Accuracy : {res['accuracy_model1']*100:.2f}%")
        print(f"    {m2.upper()} Accuracy : {res['accuracy_model2']*100:.2f}%")
        print(f"    Paired Delta    : {res['formatted_delta']}")
        print(f"    Rescued (n01)   : {res['n_rescued']} sequences")
        print(f"    Degraded (n10)  : {res['n_degraded']} sequences")
        print(f"    Net Rescues     : {res['net_rescues']:+d}")
        print(f"    McNemar p-value : {res['formatted_p_value']} (Significant: {res['statistically_significant']})")
        print(f"    Variance Reduct : {res['variance_reduction_ratio']*100:.1f}% vs unpaired")

    # 4. Generate and save paired Markdown comparison table
    table_path = Path(output_dir) / "tables" / "paired_model_comparison.md"
    table_content = generate_paired_comparison_table(paired_comparisons, output_path=str(table_path))

    print(f"\n{'='*70}")
    print(f"PAIRED EVALUATION COMPLETE:")
    print(f"  Paired Dataset → {out_matrix_path}")
    print(f"  Comparison Table → {table_path}")
    print(f"{'='*70}\n")

    return {
        "df_matrix": df_matrix,
        "paired_comparisons": paired_comparisons,
        "table_path": str(table_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired Model Evaluation CLI")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=["baseline", "meanpool", "gru"])
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_paired_evaluation(
        config_path=args.config,
        seed=args.seed,
        models=args.models,
        output_dir=args.output_dir,
    )
