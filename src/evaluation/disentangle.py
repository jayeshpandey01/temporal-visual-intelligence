"""
Three-Model Disentanglement & Diagnostic Outcome Classifier.
Fulfills Milestone M3.19 & M3.20 and Take-Home Assessment §4 & §7.C.

Disentangles:
1. Total Temporal Head Advantage: Delta_total = Model C - Model A
2. Observation Accumulation Gain: Delta_obs   = Model B - Model A
3. Chronological Order Gain:      Delta_order = Model C - Model B
4. Order Contribution Ratio:      R_order     = Delta_order / Delta_total

Classifies results against M3.20 Diagnostic Outcomes (A, B, C, D).
"""

import os
from typing import Any, Dict, Optional, Tuple
import numpy as np


def classify_m3_20_outcome(
    acc_a: float,
    acc_b: float,
    acc_c: float,
) -> Dict[str, str]:
    """
    Classify empirical results into Milestone M3.20 Diagnostic Outcomes.

    Outcomes:
        - Outcome A: GRU >> MeanPool >= SingleFrame
          (More observations help + order-aware modeling adds additional benefit)
        - Outcome B: MeanPool ≈ GRU >> SingleFrame
          (Visual observation accumulation helps, but order does not add value)
        - Outcome C: SingleFrame ≈ MeanPool ≈ GRU (High Acc)
          (Dataset does not require temporal context; task is static)
        - Outcome D: Small differences across all models
          (Ambiguous instances, subtle dynamics, or noisy regime)
    """
    delta_total = acc_c - acc_a
    delta_obs = acc_b - acc_a
    delta_order = acc_c - acc_b

    if delta_order >= 0.03 and delta_total >= 0.05:
        return {
            "outcome": "Outcome A",
            "verdict": "TEMPORAL ORDER & OBSERVATION ACCUMULATION CONFIRMED",
            "interpretation": (
                "Both observation count and chronological order provide substantial discriminative value. "
                "The temporal recurrent head successfully exploits sequential transitions (H2 confirmed)."
            ),
        }
    elif delta_obs >= 0.05 and delta_order < 0.03:
        return {
            "outcome": "Outcome B",
            "verdict": "OBSERVATION ACCUMULATION ONLY (ORDER-INVARIANT)",
            "interpretation": (
                "Additional visual glimpses provide benefit, but chronological order does not add value. "
                "The model functions primarily via spatial evidence accumulation (H2 not supported)."
            ),
        }
    elif abs(delta_total) <= 0.03 and acc_a >= 0.70:
        return {
            "outcome": "Outcome C",
            "verdict": "STATIC VISUAL SUFFICIENCY",
            "interpretation": (
                "Single-frame anchor is sufficient; dataset exhibits minimal temporal ambiguity."
            ),
        }
    else:
        return {
            "outcome": "Outcome D",
            "verdict": "DIAGNOSTIC AMBIGUITY REGIME",
            "interpretation": (
                "Margins are subtle or task exhibits high directional ambiguity; "
                "detailed failure quadrant analysis and M4 controlled perturbations required."
            ),
        }


def compute_three_model_disentanglement(
    acc_a: float,
    acc_b: float,
    acc_c: float,
    f1_a: Optional[float] = None,
    f1_b: Optional[float] = None,
    f1_c: Optional[float] = None,
    n_seeds: int = 3,
) -> Dict[str, Any]:
    """
    Compute quantitative disentanglement metrics across Model A, B, and C.

    Args:
        acc_a: Accuracy of SingleFrameMLP (Model A).
        acc_b: Accuracy of MultiFrameMeanPool (Model B).
        acc_c: Accuracy of TemporalGRU (Model C).
        f1_a: Macro-F1 of Model A.
        f1_b: Macro-F1 of Model B.
        f1_c: Macro-F1 of Model C.
        n_seeds: Number of evaluated seeds.

    Returns:
        dict with delta values, ratios, and outcome classification.
    """
    delta_total = acc_c - acc_a
    delta_obs = acc_b - acc_a
    delta_order = acc_c - acc_b

    # Order contribution ratio
    if abs(delta_total) > 1e-6:
        r_order = (delta_order / delta_total) * 100.0
        r_obs = (delta_obs / delta_total) * 100.0
    else:
        r_order = 0.0
        r_obs = 0.0

    outcome_meta = classify_m3_20_outcome(acc_a, acc_b, acc_c)

    res = {
        "acc_a": acc_a,
        "acc_b": acc_b,
        "acc_c": acc_c,
        "f1_a": f1_a,
        "f1_b": f1_b,
        "f1_c": f1_c,
        "delta_total": delta_total,
        "delta_obs": delta_obs,
        "delta_order": delta_order,
        "r_order": r_order,
        "r_obs": r_obs,
        "n_seeds": n_seeds,
        "outcome": outcome_meta["outcome"],
        "verdict": outcome_meta["verdict"],
        "interpretation": outcome_meta["interpretation"],
    }
    return res


def generate_three_model_comparison_table(
    disentangle_result: Dict[str, Any],
    output_path: Optional[str] = "results/tables/three_model_comparison.md",
    seeds_list: Optional[list] = None,
) -> str:
    """Format and save the scientific three-model comparison table."""
    r = disentangle_result
    seeds_str = str(len(seeds_list)) if seeds_list else str(r.get("n_seeds", 3))

    f1_a_str = f"{r['f1_a']*100:.2f}%" if r.get('f1_a') is not None else "—"
    f1_b_str = f"{r['f1_b']*100:.2f}%" if r.get('f1_b') is not None else "—"
    f1_c_str = f"{r['f1_c']*100:.2f}%" if r.get('f1_c') is not None else "—"

    lines = [
        "# Table: Three-Model Baseline & Control Comparison (M3.19)",
        "Disentangling Visual Evidence Accumulation vs. Chronological Order Dynamics (Take-Home Assessment §4 & §7.C)\n",
        "| Architecture | Frames (K) | Order-Aware? | Parameters | Accuracy | Macro-F1 | Seeds |",
        "|---|---:|:---:|---:|---:|---:|---:|",
        f"| Model A: Single-Frame MLP | 1 | No | 103,700 | {r['acc_a']*100:.2f}% | {f1_a_str} | {seeds_str} |",
        f"| Model B: Mean-Pooling Control | 8 | No (Symmetric sum) | 103,700 | {r['acc_b']*100:.2f}% | {f1_b_str} | {seeds_str} |",
        f"| Model C: Temporal GRU | 8 | Yes (Recurrent state) | 249,364 | {r['acc_c']*100:.2f}% | {f1_c_str} | {seeds_str} |\n",
        "### Quantitative Disentanglement Breakdown:",
        f"- **Total Temporal Head Advantage (Δ_total = Model C - Model A):** `{r['delta_total']*100:+.2f} pp`",
        f"- **Observation Accumulation Gain (Δ_obs = Model B - Model A):** `{r['delta_obs']*100:+.2f} pp`",
        f"- **Chronological Order Gain (Δ_order = Model C - Model B):** `{r['delta_order']*100:+.2f} pp`",
        f"- **Order Contribution Ratio (R_order = Δ_order / Δ_total):** `{r['r_order']:.1f}%`\n",
        "### Milestone M3.20 Scientific Verdict:",
        f"- **Outcome Classification:** **{r['outcome']} ({r['verdict']})**",
        f"- **Scientific Interpretation:** {r['interpretation']}",
    ]

    content = "\n".join(lines) + "\n"
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)

    return content

