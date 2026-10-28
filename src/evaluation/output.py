"""
Multi-model evaluation output archival and failure analysis engine (M3.16).

Saves comprehensive sample-level prediction outputs:
  sequence_id
  ground_truth
  prediction
  confidence
  correct

Formats structured prediction cards:
  sequence_001
  true:     class_B
  single:   class_A
  mean:     class_B
  gru:      class_B

Serves as the raw material for M5 failure analysis and qualitative error taxonomy.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np

from src.evaluation.paired import validate_paired_test_alignment


def format_prediction_card(rec: Dict[str, Any]) -> str:
    """
    Format a sequence prediction record into the human-readable card layout (M3.16).

    Example output:
      sequence_id: vid_07_0042
      true:        Pushing something from left to right (ID: 2)
      single:      Pushing something from right to left (ID: 3, conf: 0.4520) [WRONG]
      mean:        Pushing something from right to left (ID: 3, conf: 0.4880) [WRONG]
      gru:         Pushing something from left to right (ID: 2, conf: 0.8420) [CORRECT]
      outcome:     ORDER_RESCUED (Quadrant C)
    """
    seq_id = rec.get("sequence_id", "unknown")
    gt_id = rec.get("ground_truth_id", rec.get("ground_truth", 0))
    gt_name = rec.get("ground_truth_name", f"Class_{gt_id}")

    lines = [
        f"sequence_id: {seq_id}",
        f"{'true:':12s} {gt_name} (ID: {gt_id})",
    ]

    # Model lines
    model_aliases = [
        ("single", "single", ["baseline", "single"]),
        ("mean", "mean", ["meanpool", "mean"]),
        ("gru", "gru", ["gru", "temporal"]),
    ]

    for display_prefix, key_prefix, candidate_prefixes in model_aliases:
        # Find prefix present in record
        found_pfx = None
        for pfx in candidate_prefixes:
            if f"{pfx}_pred_id" in rec or f"y_{pfx}" in rec:
                found_pfx = pfx
                break

        if found_pfx:
            pred_id = rec.get(f"{found_pfx}_pred_id", rec.get(f"y_{found_pfx}"))
            pred_name = rec.get(f"{found_pfx}_pred_name", f"Class_{pred_id}")
            conf = rec.get(f"{found_pfx}_conf", rec.get(f"conf_{found_pfx}", 0.0))
            is_corr = rec.get(f"{found_pfx}_correct", rec.get(f"correct_{found_pfx}", False))

            status = "CORRECT" if is_corr else "WRONG"
            pfx_str = f"{display_prefix}:"
            lines.append(f"{pfx_str:12s} {pred_name} (ID: {pred_id}, conf: {conf:.4f}) [{status}]")

    # Diagnostic outcome summary
    tag = rec.get("failure_tag", rec.get("quadrant_ac", ""))
    if tag:
        lines.append(f"{'outcome:':12s} {tag}")

    return "\n".join(lines)


def get_quadrant_candidates(
    multi_model_df: pd.DataFrame,
    quadrant: str = "C",
    baseline_col: str = "single_correct",
    temporal_col: str = "gru_correct",
) -> pd.DataFrame:
    """
    Extract subset of sequences matching a specific quadrant for M5 failure analysis.
      - A: Both Correct (temporal info not necessary)
      - B: Baseline Correct, Temporal Wrong (temporal regression)
      - C: Baseline Wrong, Temporal Correct (temporal rescued ← primary M5 targets)
      - D: Both Wrong (difficult / ambiguous sequence)
    """
    b_col = baseline_col if baseline_col in multi_model_df.columns else "correct_baseline"
    t_col = temporal_col if temporal_col in multi_model_df.columns else "correct_gru"

    if b_col not in multi_model_df.columns or t_col not in multi_model_df.columns:
        raise KeyError(f"Required columns '{b_col}' or '{t_col}' not found in DataFrame.")

    b_corr = multi_model_df[b_col].astype(bool)
    t_corr = multi_model_df[t_col].astype(bool)

    q = quadrant.upper()
    if q == "A":
        mask = b_corr & t_corr
    elif q == "B":
        mask = b_corr & ~t_corr
    elif q == "C":
        mask = ~b_corr & t_corr
    elif q == "D":
        mask = ~b_corr & ~t_corr
    else:
        raise ValueError(f"Unknown quadrant '{quadrant}'. Must be 'A', 'B', 'C', or 'D'.")

    return multi_model_df[mask].copy()


def compile_evaluation_outputs(
    runs_dir: Union[str, Path] = "runs",
    models: Optional[List[str]] = None,
    seed: int = 42,
    output_dir: Union[str, Path] = "results/evaluation_outputs",
    class_names: Optional[List[str]] = None,
    max_frames: int = 8,
) -> Dict[str, Any]:
    """
    Compile sample-level predictions from all models into the canonical M3.16 evaluation output repository.

    Outputs generated in output_dir:
      - multi_model_predictions.csv
      - multi_model_predictions.parquet
      - failure_analysis_manifest.json
      - sample_predictions_inspect.txt
    """
    runs_dir = Path(runs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = models or ["baseline", "meanpool", "gru"]

    # Load predictions per model
    predictions_dict: Dict[str, pd.DataFrame] = {}
    for m in models:
        run_id = f"{m}_seed{seed}_frames{max_frames}"
        pred_csv = runs_dir / run_id / "predictions.csv"
        pred_parquet = runs_dir / run_id / "predictions.parquet"

        if pred_parquet.exists():
            predictions_dict[m] = pd.read_parquet(pred_parquet)
        elif pred_csv.exists():
            predictions_dict[m] = pd.read_csv(pred_csv)
        else:
            raise FileNotFoundError(
                f"Prediction artifact for model '{m}' (seed {seed}) not found at {pred_csv}. "
                f"Run training first via scripts/train.py --model {m} --seeds {seed}"
            )

    # Validate alignment
    validate_paired_test_alignment(predictions_dict)

    ref_model = models[0]
    ref_df = predictions_dict[ref_model]
    n_samples = len(ref_df)

    # Standardize naming aliases
    alias_map = {"baseline": "single", "meanpool": "mean", "gru": "gru"}

    records = []
    for i in range(n_samples):
        seq_id = str(ref_df.iloc[i]["sequence_id"])
        gt_id = int(ref_df.iloc[i]["ground_truth"])
        gt_name = class_names[gt_id] if class_names and gt_id < len(class_names) else f"Class_{gt_id}"

        rec = {
            "sequence_id": seq_id,
            "video_id": str(ref_df.iloc[i].get("video_id", seq_id)),
            "ground_truth_id": gt_id,
            "ground_truth_name": gt_name,
        }

        # Populate each model
        for m in models:
            alias = alias_map.get(m, m)
            df = predictions_dict[m]
            pred_id = int(df.iloc[i]["predicted_label"])
            pred_name = class_names[pred_id] if class_names and pred_id < len(class_names) else f"Class_{pred_id}"
            conf = float(df.iloc[i]["confidence"])
            is_corr = bool(pred_id == gt_id)

            rec[f"{alias}_pred_id"] = pred_id
            rec[f"{alias}_pred_name"] = pred_name
            rec[f"{alias}_conf"] = round(conf, 4)
            rec[f"{alias}_correct"] = is_corr

        # Diagnostic failure classification
        if "single_correct" in rec and "gru_correct" in rec:
            c_single = rec["single_correct"]
            c_gru = rec["gru_correct"]

            if c_single and c_gru:
                rec["quadrant_single_vs_gru"] = "A_both_correct"
            elif c_single and not c_gru:
                rec["quadrant_single_vs_gru"] = "B_temporal_regression"
            elif not c_single and c_gru:
                rec["quadrant_single_vs_gru"] = "C_temporal_rescued"
            else:
                rec["quadrant_single_vs_gru"] = "D_both_wrong"

        if "mean_correct" in rec and "gru_correct" in rec:
            c_mean = rec["mean_correct"]
            c_gru = rec["gru_correct"]

            if c_mean and c_gru:
                rec["quadrant_mean_vs_gru"] = "A_both_correct"
            elif c_mean and not c_gru:
                rec["quadrant_mean_vs_gru"] = "B_order_regression"
            elif not c_mean and c_gru:
                rec["quadrant_mean_vs_gru"] = "C_order_rescued"
            else:
                rec["quadrant_mean_vs_gru"] = "D_both_wrong"

        # Comprehensive failure mode tag
        c_s = rec.get("single_correct")
        c_m = rec.get("mean_correct")
        c_g = rec.get("gru_correct")

        if c_s and c_m and c_g:
            tag = "CONCORDANT_CORRECT"
        elif not c_s and not c_m and not c_g:
            tag = "CONCORDANT_FAILURE"
        elif not c_m and c_g:
            tag = "ORDER_RESCUED"  # GRU rescued despite mean-pooling failing
        elif not c_s and c_m and not c_g:
            tag = "OBSERVATION_RESCUED_ONLY"  # Mean rescued, but GRU regressed
        elif c_m and not c_g:
            tag = "ORDER_REGRESSION"  # Mean pooling succeeded, GRU failed
        elif c_s and not c_g:
            tag = "TEMPORAL_REGRESSION"
        elif not c_s and c_g:
            tag = "TEMPORAL_RESCUED"
        else:
            tag = "MIXED_OUTCOME"

        rec["failure_tag"] = tag
        records.append(rec)

    df_out = pd.DataFrame(records)

    # 1. Save CSV and Parquet
    csv_path = output_dir / "multi_model_predictions.csv"
    parquet_path = output_dir / "multi_model_predictions.parquet"
    df_out.to_csv(csv_path, index=False)
    try:
        df_out.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    # 2. Save human-readable inspection cards
    inspect_path = output_dir / "sample_predictions_inspect.txt"
    with open(inspect_path, "w") as f:
        f.write("# Sample-Level Multi-Model Evaluation Cards (M3.16)\n")
        f.write("# Format: sequence_id -> true -> single -> mean -> gru -> outcome\n\n")
        separator = "=" * 80 + "\n"
        for rec in records:
            f.write(format_prediction_card(rec))
            f.write("\n" + separator)

    # 3. Build M5 failure analysis manifest
    manifest: Dict[str, Any] = {
        "benchmark_info": {
            "test_sequences": n_samples,
            "seed": seed,
            "models_evaluated": models,
        },
        "quadrant_counts": {
            "A_both_correct": int((df_out["quadrant_single_vs_gru"] == "A_both_correct").sum()) if "quadrant_single_vs_gru" in df_out else 0,
            "B_temporal_regression": int((df_out["quadrant_single_vs_gru"] == "B_temporal_regression").sum()) if "quadrant_single_vs_gru" in df_out else 0,
            "C_temporal_rescued": int((df_out["quadrant_single_vs_gru"] == "C_temporal_rescued").sum()) if "quadrant_single_vs_gru" in df_out else 0,
            "D_both_wrong": int((df_out["quadrant_single_vs_gru"] == "D_both_wrong").sum()) if "quadrant_single_vs_gru" in df_out else 0,
        },
        "failure_tag_counts": {
            tag: int((df_out["failure_tag"] == tag).sum()) for tag in df_out["failure_tag"].unique()
        },
        "m5_candidate_sequences": {
            "quadrant_C_temporal_rescued": df_out[df_out["quadrant_single_vs_gru"] == "C_temporal_rescued"]["sequence_id"].tolist() if "quadrant_single_vs_gru" in df_out.columns else [],
            "quadrant_B_temporal_regression": df_out[df_out["quadrant_single_vs_gru"] == "B_temporal_regression"]["sequence_id"].tolist() if "quadrant_single_vs_gru" in df_out.columns else [],
            "order_rescued": df_out[df_out["failure_tag"] == "ORDER_RESCUED"]["sequence_id"].tolist() if "failure_tag" in df_out.columns else [],
            "order_regression": df_out[df_out["failure_tag"] == "ORDER_REGRESSION"]["sequence_id"].tolist() if "failure_tag" in df_out.columns else [],
            "concordant_failures": df_out[df_out["failure_tag"] == "CONCORDANT_FAILURE"]["sequence_id"].tolist() if "failure_tag" in df_out.columns else [],
        },
    }

    manifest_path = output_dir / "failure_analysis_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported multi-model predictions → {csv_path}")
    print(f"Exported prediction cards → {inspect_path}")
    print(f"Exported M5 failure analysis manifest → {manifest_path}")

    return {
        "df": df_out,
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path),
        "inspect_path": str(inspect_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }
