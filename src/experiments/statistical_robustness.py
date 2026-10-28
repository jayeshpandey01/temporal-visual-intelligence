"""
Legacy M6.7 statistical summary template.

The current comparison values below are hard-coded historical entries. They are
not computed from prediction files in this module and must not be used for
inferential claims until the analyses are rebuilt from identified paired
predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.stats import chi2


def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """
    Benjamini-Hochberg procedure for False Discovery Rate (FDR) control (1995).
    Controls the expected proportion of false rejections among all rejections.

    Args:
        p_values: list of raw p-values from hypothesis tests
        alpha: FDR target threshold (default 0.05)

    Returns:
        (rejected_mask, adjusted_p_values)
    """
    m = len(p_values)
    if m == 0:
        return [], []

    p_arr = np.array(p_values, dtype=float)
    # Clip zero p-values to small positive value to avoid log issues
    p_arr = np.clip(p_arr, 1e-20, 1.0)

    # Sort indices
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    # Adjusted p-values: q_(i) = min_{j >= i} (p_(j) * m / j)
    q_values = np.zeros(m)
    current_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q_val = sorted_p[i] * m / rank
        current_min = min(current_min, q_val)
        q_values[i] = current_min

    q_values = np.clip(q_values, 0.0, 1.0)

    # Reorder back to original order
    adj_p = np.zeros(m)
    adj_p[sorted_indices] = q_values

    rejected = adj_p <= alpha
    return rejected.tolist(), adj_p.tolist()


def compute_cohens_d(
    acc_a: float,
    acc_b: float,
    n_samples: int = 300,
) -> float:
    """
    Compute Cohen's d effect size for two binary outcome proportions.
    Formula: d = (p1 - p2) / sqrt((p1(1-p1) + p2(1-p2)) / 2)
    """
    var_a = acc_a * (1.0 - acc_a)
    var_b = acc_b * (1.0 - acc_b)
    pooled_sd = np.sqrt((var_a + var_b) / 2.0)
    if pooled_sd < 1e-6:
        return 0.0
    return float((acc_a - acc_b) / pooled_sd)


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


class M6StatisticalReporter:
    """Formats legacy comparisons and marks them as unverified historical values."""
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def compile_family_report(self) -> Dict[str, Any]:
        # Pre-registered family of primary comparisons across M4 and M6
        raw_comparisons = [
            {
                "hypothesis_id": "H1",
                "comparison": "Canonical GRU vs SingleFrame MLP",
                "delta": 0.2600,
                "p_raw": 1.2e-14,
                "ci_95": [0.208, 0.312],
                "cohens_d": compute_cohens_d(0.3478, 0.0878),
            },
            {
                "hypothesis_id": "H2",
                "comparison": "Canonical GRU vs MultiFrame MeanPool",
                "delta": 0.2656,
                "p_raw": 3.3e-16,
                "ci_95": [0.214, 0.317],
                "cohens_d": compute_cohens_d(0.3478, 0.0822),
            },
            {
                "hypothesis_id": "H3",
                "comparison": "Temporal Order Destruction (Canonical vs Shuffled GRU)",
                "delta": 0.1389,
                "p_raw": 2.99e-11,
                "ci_95": [0.098, 0.180],
                "cohens_d": compute_cohens_d(0.3478, 0.2089),
            },
            {
                "hypothesis_id": "H4",
                "comparison": "Temporal Reversal Drop (Canonical vs Reversed GRU)",
                "delta": 0.0467,
                "p_raw": 4.1e-03,
                "ci_95": [0.015, 0.078],
                "cohens_d": compute_cohens_d(0.3478, 0.3011),
            },
            {
                "hypothesis_id": "H5",
                "comparison": "Sampling Robustness: Sparse (GRU vs MeanPool)",
                "delta": 0.1178,
                "p_raw": 3.03e-07,
                "ci_95": [0.073, 0.162],
                "cohens_d": compute_cohens_d(0.2033, 0.0856),
            },
            {
                "hypothesis_id": "H6",
                "comparison": "Sampling Robustness: Dense (GRU vs MeanPool)",
                "delta": 0.0822,
                "p_raw": 1.40e-04,
                "ci_95": [0.040, 0.124],
                "cohens_d": compute_cohens_d(0.1689, 0.0867),
            },
            {
                "hypothesis_id": "H7",
                "comparison": "Sampling Robustness: Random (GRU vs MeanPool)",
                "delta": 0.1344,
                "p_raw": 4.84e-08,
                "ci_95": [0.086, 0.182],
                "cohens_d": compute_cohens_d(0.2178, 0.0833),
            },
            {
                "hypothesis_id": "H8",
                "comparison": "Sampling Robustness: Jittered (GRU vs MeanPool)",
                "delta": 0.2456,
                "p_raw": 6.33e-15,
                "ci_95": [0.195, 0.296],
                "cohens_d": compute_cohens_d(0.3311, 0.0856),
            },
            {
                "hypothesis_id": "H9",
                "comparison": "Missing Observations: 25% Drop (GRU vs MeanPool)",
                "delta": 0.2144,
                "p_raw": 8.12e-13,
                "ci_95": [0.165, 0.264],
                "cohens_d": compute_cohens_d(0.2989, 0.0844),
            },
            {
                "hypothesis_id": "H10",
                "comparison": "Missing Observations: 50% Drop (GRU vs MeanPool)",
                "delta": 0.1333,
                "p_raw": 5.45e-08,
                "ci_95": [0.085, 0.181],
                "cohens_d": compute_cohens_d(0.2178, 0.0844),
            },
            {
                "hypothesis_id": "H11",
                "comparison": "Missing Observations: 75% Drop (GRU vs MeanPool)",
                "delta": 0.0511,
                "p_raw": 2.34e-02,
                "ci_95": [0.007, 0.095],
                "cohens_d": compute_cohens_d(0.1356, 0.0844),
            },
            {
                "hypothesis_id": "H12",
                "comparison": "Representation Robustness: CLIP ViT-B/32 (GRU vs Baseline)",
                "delta": 0.2367,
                "p_raw": 1.15e-11,
                "ci_95": [0.170, 0.303],
                "cohens_d": compute_cohens_d(0.3789, 0.1422),
            },
            {
                "hypothesis_id": "H13",
                "comparison": "Representation Robustness: CLIP ViT-B/32 (GRU vs MeanPool)",
                "delta": 0.2522,
                "p_raw": 4.56e-12,
                "ci_95": [0.185, 0.319],
                "cohens_d": compute_cohens_d(0.3789, 0.1267),
            },
            {
                "hypothesis_id": "H14",
                "comparison": "Capacity Parity Parity: CompactGRU vs MeanPool (103,700 params)",
                "delta": 0.1088,
                "p_raw": 2.15e-06,
                "ci_95": [0.063, 0.154],
                "cohens_d": compute_cohens_d(0.1890, 0.0803),
            },
        ]

        p_raw_list = [c["p_raw"] for c in raw_comparisons]
        rejected, adj_p = benjamini_hochberg(p_raw_list, alpha=self.alpha)

        for i, c in enumerate(raw_comparisons):
            c["historical_p_bh"] = adj_p[i]
            c["historical_rejected"] = rejected[i]
            c["p_bh"] = None
            c["rejected"] = None
            c["effect_magnitude"] = interpret_effect_size(c["cohens_d"])
            c["validation_status"] = "UNVERIFIED_HARD_CODED"

        historical_survivors = sum(1 for r in rejected if r)
        return {
            "status": "UNVERIFIED_HARD_CODED",
            "valid_for_inference": False,
            "warning": (
                "Comparison values are hard-coded historical entries; this module "
                "does not load paired prediction inputs or recompute tests."
            ),
            "alpha": self.alpha,
            "fdr_method": "Benjamini-Hochberg (1995)",
            "total_comparisons": len(raw_comparisons),
            "historical_survived_comparisons": historical_survivors,
            "survived_comparisons": None,
            "comparisons": raw_comparisons,
        }

