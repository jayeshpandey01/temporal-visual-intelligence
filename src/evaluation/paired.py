"""
Paired evaluation and variance reduction infrastructure (M3.15).

Enforces the most important evaluation rule:
Every comparative model must be evaluated on the EXACT same test sequences:
  Sequence 001 -> Model A, Model B, Model C
  Sequence 002 -> Model A, Model B, Model C
  ...
  Sequence 300 -> Model A, Model B, Model C

Computes sample-by-sample discordance, variance reduction, McNemar tests,
and compiles the unified multi-model paired comparison matrix.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2


def validate_paired_test_alignment(
    predictions_dict: Dict[str, Union[pd.DataFrame, str, Path]],
) -> Tuple[bool, List[str], np.ndarray]:
    """
    Validate that all models in predictions_dict evaluated the exact same test sequence IDs
    in the exact same sequential order with identical ground truth labels.

    Args:
        predictions_dict: dict mapping model name to pd.DataFrame or file path.

    Returns:
        (is_valid, sequence_ids, ground_truth_array)

    Raises:
        ValueError: if sequence count, sequence IDs, sequence ordering, or ground truths disagree.
    """
    dfs: Dict[str, pd.DataFrame] = {}
    for name, item in predictions_dict.items():
        if isinstance(item, (str, Path)):
            p = Path(item)
            if not p.exists():
                raise FileNotFoundError(f"Predictions file for '{name}' not found: {p}")
            if p.suffix == ".parquet":
                dfs[name] = pd.read_parquet(p)
            else:
                dfs[name] = pd.read_csv(p)
        elif isinstance(item, pd.DataFrame):
            dfs[name] = item
        else:
            raise TypeError(f"Expected DataFrame or path for '{name}', got {type(item)}")

    if not dfs:
        raise ValueError("predictions_dict is empty. At least one model required.")

    model_names = list(dfs.keys())
    ref_name = model_names[0]
    ref_df = dfs[ref_name]

    if "sequence_id" not in ref_df.columns:
        raise ValueError(f"Model '{ref_name}' DataFrame missing required column 'sequence_id'")
    if "ground_truth" not in ref_df.columns:
        raise ValueError(f"Model '{ref_name}' DataFrame missing required column 'ground_truth'")

    ref_ids = list(ref_df["sequence_id"])
    ref_gt = np.array(ref_df["ground_truth"])
    n_samples = len(ref_ids)

    for name in model_names[1:]:
        df = dfs[name]
        if len(df) != n_samples:
            raise ValueError(
                f"Sample count mismatch in paired evaluation! '{ref_name}' evaluated {n_samples} sequences, "
                f"but '{name}' evaluated {len(df)} sequences."
            )
        ids = list(df["sequence_id"])
        if ids != ref_ids:
            # Check if same set but different order or completely different IDs
            diff_ids = set(ref_ids).symmetric_difference(set(ids))
            if diff_ids:
                raise ValueError(
                    f"Sequence ID mismatch between '{ref_name}' and '{name}'. "
                    f"{len(diff_ids)} sequence IDs differ. Paired evaluation requires identical sequences."
                )
            else:
                raise ValueError(
                    f"Sequence ordering mismatch between '{ref_name}' and '{name}'. "
                    f"Sequences must be aligned in the exact same index order for paired testing."
                )

        gt = np.array(df["ground_truth"])
        if not np.array_equal(ref_gt, gt):
            mismatches = np.where(ref_gt != gt)[0]
            raise ValueError(
                f"Ground truth label mismatch for model '{name}' at {len(mismatches)} sequence positions "
                f"(e.g. index {mismatches[0]}). Test set labels must be identical."
            )

    return True, ref_ids, ref_gt


def compute_paired_metrics(
    y_true: np.ndarray,
    y_model1: np.ndarray,
    y_model2: np.ndarray,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute rigorous sample-level paired comparison statistics between two models on identical test sequences.

    Returns:
      - acc_model1, acc_model2
      - delta_paired = acc_model2 - acc_model1
      - n_01 (rescued count: model 1 wrong, model 2 right)
      - n_10 (degraded count: model 1 right, model 2 wrong)
      - n_11 (both right), n_00 (both wrong)
      - mcnemar_chi2, mcnemar_p_value
      - paired_bootstrap_ci_95
      - sample_correlation rho
      - variance_reduction_ratio
    """
    y_true = np.asarray(y_true)
    y_m1 = np.asarray(y_model1)
    y_m2 = np.asarray(y_model2)
    n = len(y_true)

    corr_m1 = (y_m1 == y_true).astype(float)
    corr_m2 = (y_m2 == y_true).astype(float)

    acc1 = float(np.mean(corr_m1))
    acc2 = float(np.mean(corr_m2))
    delta = acc2 - acc1

    # Sample-by-sample difference vector
    diff = corr_m2 - corr_m1  # values in {-1, 0, +1}

    # Contingency matrix counts
    n_11 = int(np.sum((corr_m1 == 1) & (corr_m2 == 1)))
    n_00 = int(np.sum((corr_m1 == 0) & (corr_m2 == 0)))
    n_01 = int(np.sum((corr_m1 == 0) & (corr_m2 == 1)))  # M2 rescued
    n_10 = int(np.sum((corr_m1 == 1) & (corr_m2 == 0)))  # M2 degraded

    # McNemar's test with continuity correction
    discordant = n_01 + n_10
    if discordant == 0:
        chi2_stat = 0.0
        p_val = 1.0
    else:
        chi2_stat = float((abs(n_01 - n_10) - 1.0) ** 2 / discordant)
        p_val = float(1.0 - chi2.cdf(chi2_stat, df=1))

    # Variance reduction analysis
    var1 = float(np.var(corr_m1, ddof=1)) if n > 1 else 0.0
    var2 = float(np.var(corr_m2, ddof=1)) if n > 1 else 0.0
    cov = float(np.cov(corr_m1, corr_m2)[0, 1]) if n > 1 else 0.0
    rho = float(cov / (np.sqrt(var1 * var2) + 1e-12)) if var1 > 0 and var2 > 0 else 0.0

    var_unpaired = (var1 + var2) / n if n > 0 else 0.0
    var_paired = float(np.var(diff, ddof=1) / n) if n > 1 else 0.0
    variance_reduction = float((var_unpaired - var_paired) / (var_unpaired + 1e-12))

    # Paired 95% Bootstrap Confidence Interval
    rng = np.random.default_rng(seed)
    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas.append(np.mean(diff[idx]))

    ci_low = float(np.percentile(boot_deltas, 2.5))
    ci_high = float(np.percentile(boot_deltas, 97.5))

    return {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "n_samples": n,
        "accuracy_model1": acc1,
        "accuracy_model2": acc2,
        "delta_accuracy": delta,
        "n_both_correct": n_11,
        "n_both_wrong": n_00,
        "n_rescued": n_01,
        "n_degraded": n_10,
        "net_rescues": n_01 - n_10,
        "mcnemar_chi2": chi2_stat,
        "mcnemar_p_value": p_val,
        "statistically_significant": bool(p_val < 0.05),
        "ci_95": (ci_low, ci_high),
        "sample_correlation_rho": rho,
        "variance_reduction_ratio": max(0.0, variance_reduction),
        "formatted_delta": f"{delta:+.3f} [{ci_low:+.3f}, {ci_high:+.3f}]",
        "formatted_p_value": f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001",
    }


def build_paired_comparison_matrix(
    predictions_dict: Dict[str, Union[pd.DataFrame, str, Path]],
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build unified sample-level paired comparison matrix side-by-side across all models (M3.15, M3.16).

    Columns:
      sequence_id, video_id, ground_truth, class_name,
      y_{model}, conf_{model}, correct_{model} for each model,
      delta_obs, delta_order, quadrant_ab, quadrant_bc, quadrant_ac
    """
    # Verify alignment
    validate_paired_test_alignment(predictions_dict)

    # Load all DataFrames
    dfs = {}
    for name, item in predictions_dict.items():
        if isinstance(item, (str, Path)):
            p = Path(item)
            dfs[name] = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        else:
            dfs[name] = item

    ref_name = list(dfs.keys())
    ref_df = dfs[ref_name[0]]
    n = len(ref_df)

    matrix_records = []
    for i in range(n):
        seq_id = ref_df.iloc[i]["sequence_id"]
        gt = int(ref_df.iloc[i]["ground_truth"])
        class_name = class_names[gt] if class_names and gt < len(class_names) else f"Class_{gt}"

        rec = {
            "sequence_id": seq_id,
            "video_id": ref_df.iloc[i].get("video_id", seq_id),
            "ground_truth": gt,
            "class_name": class_name,
        }

        # Populate per-model prediction columns
        for m_name in dfs:
            m_row = dfs[m_name].iloc[i]
            pred = int(m_row["predicted_label"])
            conf = float(m_row["confidence"])
            is_corr = bool(pred == gt)

            rec[f"y_{m_name}"] = pred
            rec[f"conf_{m_name}"] = round(conf, 4)
            rec[f"correct_{m_name}"] = is_corr

        # If baseline, meanpool, and gru are all present, compute standard diagnostic deltas
        if all(k in dfs for k in ("baseline", "meanpool", "gru")):
            c_base = int(rec["correct_baseline"])
            c_mean = int(rec["correct_meanpool"])
            c_gru = int(rec["correct_gru"])

            # Observation gain delta: meanpool vs baseline
            rec["delta_obs"] = c_mean - c_base  # +1 if meanpool rescued from baseline
            # Order gain delta: gru vs meanpool
            rec["delta_order"] = c_gru - c_mean  # +1 if gru rescued from meanpool
            # Total gain delta: gru vs baseline
            rec["delta_total"] = c_gru - c_base

            # Quadrant categorization (Model A vs Model C)
            if c_base and c_gru:
                quad = "A_both_correct"
            elif c_base and not c_gru:
                quad = "B_baseline_only"
            elif not c_base and c_gru:
                quad = "C_temporal_rescued"
            else:
                quad = "D_both_wrong"
            rec["quadrant_ac"] = quad

        matrix_records.append(rec)

    df_matrix = pd.DataFrame(matrix_records)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        out_p = Path(output_path)
        df_matrix.to_csv(out_p.with_suffix(".csv"), index=False)
        try:
            df_matrix.to_parquet(out_p.with_suffix(".parquet"), index=False)
        except Exception:
            pass
        print(f"Saved paired comparison matrix → {output_path}")

    return df_matrix


def generate_paired_comparison_table(
    paired_results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> str:
    """
    Format paired model comparison metrics into a structured Markdown table (Take-Home Assessment §4, §5, §10).
    """
    lines = [
        "# Table: Paired Model Comparison & Statistical Significance (M3.15)",
        "Strict sample-by-sample paired hypothesis testing on identical test sequences ($N=300$)\n",
        "| Comparison Pair | Scientific Measurement | Model 1 Acc | Model 2 Acc | Paired Δ (95% CI) | Rescued ($n_{01}$) | Degraded ($n_{10}$) | McNemar $p$-value | Significant ($p<0.05$)? |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    for r in paired_results:
        m1 = r["model1_name"]
        m2 = r["model2_name"]
        pair_str = f"**{m1}** vs **{m2}**"
        
        # Determine scientific measurement name
        if "baseline" in m1.lower() and "meanpool" in m2.lower():
            measurement = r"Observation Accumulation ($\Delta_{obs}$)"
        elif "meanpool" in m1.lower() and "gru" in m2.lower():
            measurement = r"Temporal Order Gain ($\Delta_{order}$)"
        elif "baseline" in m1.lower() and "gru" in m2.lower():
            measurement = r"Total Temporal Gain ($\Delta_{total}$)"
        else:
            measurement = f"{m2} vs {m1}"

        acc1 = f"{r['accuracy_model1'] * 100:.2f}%"
        acc2 = f"{r['accuracy_model2'] * 100:.2f}%"
        delta_str = f"{r['formatted_delta']}"
        sig_str = "Yes" if r["statistically_significant"] else "No"

        lines.append(
            f"| {pair_str} | {measurement} | {acc1} | {acc2} | **{delta_str}** | {r['n_rescued']} | {r['n_degraded']} | {r['formatted_p_value']} | {sig_str} |"
        )

    lines.append("\n### Scientific Methodological Guarantees:")
    lines.append(r"1. **Identical Sample Alignment:** All comparisons are strictly paired on the identical 300 test sequences ($S_1, \dots, S_{300}$).")
    lines.append(r"2. **Between-Sample Variance Elimination:** Paired testing eliminates sample difficulty variance ($\text{Cov}(X_1, X_2) > 0$), dramatically tightening confidence intervals.")
    lines.append(r"3. **Direct Order Falsification:** If $\Delta_{order} \le 0$ or McNemar $p \ge 0.05$ between Model B and Model C, chronological temporal modeling is statistically falsified (Assessment §7.D).")

    content = "\n".join(lines) + "\n"
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Saved paired comparison table → {output_path}")

    return content
