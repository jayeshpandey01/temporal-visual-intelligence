"""
Milestone M4.6: Statistical Analysis Across Seeds and Experimental Interventions.
Directly fulfills Take-Home Assessment Sections 5, 7, 10, & 11 and M4.6:
  - Cross-seed statistical dispersion (mean ± sigma across deterministic seeds).
  - Paired test sequence discordance analysis (n_00, n_11, n_01, n_10).
  - McNemar's paired test with Yates' continuity correction & exact binomial fallback.
  - 95% non-parametric paired bootstrap confidence intervals (B=2,000 resamples).
  - Effect size quantification (discordance odds ratio & Cohen's g).
  - Multiple hypothesis testing adjustment (Holm-Bonferroni).
  - Formal Take-Home Assessment Section 7.D falsification evaluation.
"""

from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd
from scipy.stats import chi2, binomtest


def compute_paired_contingency(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
) -> Dict[str, Any]:
    """
    Compute 2x2 contingency matrix and discordance metrics for paired sequence predictions.

    Args:
        y_true: Ground truth class indices [N]
        y_pred1: Model 1 predicted class indices [N]
        y_pred2: Model 2 predicted class indices [N]
        model1_name: Descriptive name for Model 1 (e.g. "Model B (MeanPool)")
        model2_name: Descriptive name for Model 2 (e.g. "Model C (TemporalGRU)")

    Returns:
        Dict with contingency counts, net rescues, discordance ratios, effect sizes.
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)
    n = len(y_true)

    corr1 = (y_pred1 == y_true).astype(int)
    corr2 = (y_pred2 == y_true).astype(int)

    # Contingency matrix
    n_11 = int(np.sum((corr1 == 1) & (corr2 == 1)))  # Concordant correct
    n_00 = int(np.sum((corr1 == 0) & (corr2 == 0)))  # Concordant error
    n_01 = int(np.sum((corr1 == 0) & (corr2 == 1)))  # Model 1 wrong, Model 2 right (M2 rescued)
    n_10 = int(np.sum((corr1 == 1) & (corr2 == 0)))  # Model 1 right, Model 2 wrong (M2 degraded)

    acc1 = float((n_11 + n_10) / n) if n > 0 else 0.0
    acc2 = float((n_11 + n_01) / n) if n > 0 else 0.0
    delta_acc = float(acc2 - acc1)

    discordant_count = n_01 + n_10
    discordant_rate = float(discordant_count / n) if n > 0 else 0.0
    net_rescues = n_01 - n_10

    # Odds ratio on discordance: OR = n_01 / n_10
    odds_ratio = float(n_01 / max(n_10, 1)) if discordant_count > 0 else 1.0

    # Cohen's g: paired effect size measuring deviation of discordant pairs from 50/50 null
    # g in [-0.5, +0.5], where |g| >= 0.05 is small, >= 0.15 is medium, >= 0.25 is large
    cohens_g = float((n_01 - n_10) / (2.0 * discordant_count)) if discordant_count > 0 else 0.0

    # Correlation between predictions
    var1 = float(np.var(corr1, ddof=1)) if n > 1 else 0.0
    var2 = float(np.var(corr2, ddof=1)) if n > 1 else 0.0
    cov = float(np.cov(corr1, corr2)[0, 1]) if n > 1 else 0.0
    rho = float(cov / (np.sqrt(var1 * var2) + 1e-12)) if var1 > 0 and var2 > 0 else 0.0

    return {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "n_samples": n,
        "n_11_both_correct": n_11,
        "n_00_both_wrong": n_00,
        "n_01_rescued_by_m2": n_01,
        "n_10_degraded_by_m2": n_10,
        "net_rescues": net_rescues,
        "discordant_count": discordant_count,
        "discordant_rate": discordant_rate,
        "accuracy_model1": acc1,
        "accuracy_model2": acc2,
        "delta_accuracy": delta_acc,
        "odds_ratio": odds_ratio,
        "cohens_g": cohens_g,
        "sample_correlation_rho": rho,
    }


def mcnemar_paired_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    exact_threshold: int = 25,
) -> Tuple[float, float, str]:
    """
    McNemar's test for paired classification discordance.
    Applies Yates' continuity correction for asymptotic chi-square,
    and falls back to exact two-sided binomial test when discordant count < exact_threshold.

    Returns:
        (statistic, p_value, test_type)
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    corr1 = (y_pred1 == y_true).astype(int)
    corr2 = (y_pred2 == y_true).astype(int)

    n_01 = int(np.sum((corr1 == 0) & (corr2 == 1)))
    n_10 = int(np.sum((corr1 == 1) & (corr2 == 0)))
    discordant = n_01 + n_10

    if discordant == 0:
        return 0.0, 1.0, "concordant_null"

    if discordant < exact_threshold:
        # Exact two-sided binomial test under H0: p = 0.5
        b_res = binomtest(k=n_01, n=discordant, p=0.5, alternative="two-sided")
        # Approximate chi2 equivalent for reporting
        chi2_approx = float((abs(n_01 - n_10) - 1.0) ** 2 / discordant) if abs(n_01 - n_10) > 1.0 else 0.0
        return chi2_approx, float(b_res.pvalue), "exact_binomial"
    else:
        # Asymptotic chi-square with Yates' continuity correction
        chi2_stat = float((abs(n_01 - n_10) - 1.0) ** 2 / discordant)
        p_val = float(1.0 - chi2.cdf(chi2_stat, df=1))
        return chi2_stat, p_val, "asymptotic_yates"


def bootstrap_paired_margin_ci(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute 95% non-parametric bootstrap confidence interval for paired accuracy delta:
        Delta = Acc(y_pred2) - Acc(y_pred1)

    Returns:
        (mean_delta, ci_low, ci_high)
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)
    n = len(y_true)

    corr1 = (y_pred1 == y_true).astype(float)
    corr2 = (y_pred2 == y_true).astype(float)
    diff = corr2 - corr1

    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas[b] = np.mean(diff[idx])

    ci_low = float(np.percentile(boot_deltas, (1.0 - alpha) / 2.0 * 100.0))
    ci_high = float(np.percentile(boot_deltas, (1.0 + alpha) / 2.0 * 100.0))
    mean_val = float(np.mean(boot_deltas))

    return mean_val, ci_low, ci_high


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply step-down Holm-Bonferroni correction to a list of p-values to control Family-Wise Error Rate (FWER).

    Args:
        p_values: List of raw uncorrected p-values

    Returns:
        List of adjusted p-values in the exact original order.
    """
    m = len(p_values)
    if m <= 1:
        return list(p_values)

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_indexed = [0.0] * m

    cum_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        multiplier = m - rank
        adj = min(1.0, p_val * multiplier)
        cum_max = max(cum_max, adj)
        adjusted_indexed[orig_idx] = min(1.0, cum_max)

    return adjusted_indexed


def compute_multi_seed_dispersion(
    seed_values: List[float],
) -> Dict[str, float]:
    """
    Compute sample mean, standard deviation, and standard error of the mean across seeds.
    """
    arr = np.asarray(seed_values, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr)) if n > 0 else 0.0
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem_val = float(std_val / np.sqrt(n)) if n > 0 else 0.0
    min_val = float(np.min(arr)) if n > 0 else 0.0
    max_val = float(np.max(arr)) if n > 0 else 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "sem": sem_val,
        "min": min_val,
        "max": max_val,
        "n_seeds": n,
    }


def evaluate_falsification_criteria(
    contrast_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Formally evaluate Take-Home Assessment Section 7.D decision rules:
    "What experimental result would convince you the temporal approach is not useful?"

    Checks:
      1. Order Dependency Falsification: Model C vs Model B (Delta_order <= 0 or p >= 0.05)
      2. Capacity Parity Falsification: Model D vs Model B (Delta_parity <= 0 or p >= 0.05)
      3. Chronological Order Sensitivity Falsification: Model C Canonical vs Shuffled (Delta_shuffle <= 0 or p >= 0.05)
      4. Reversal Directional Falsification: Model C Forward vs Reversed (Delta_rev <= 0 or p >= 0.05)

    Returns:
        Structured dict with pass/fail certification and scientific verdicts.
    """
    def _find_contrast(patterns: List[str]) -> Optional[Dict[str, Any]]:
        for pat in patterns:
            for k, v in contrast_results.items():
                if pat.lower() in k.lower():
                    return v
        return None

    def _get_val(entry: Dict[str, Any], keys: List[str], default: float) -> float:
        for k in keys:
            if k in entry:
                return float(entry[k])
        return default

    checks = {}

    # 1. Order Dependency Gain (Model C vs Model B)
    c_vs_b = _find_contrast(["Model C vs Model B", "Order Dependency Gain", "Order Gain"])
    if c_vs_b:
        delta = _get_val(c_vs_b, ["delta_mean", "delta_accuracy", "delta"], 0.0)
        p_val = _get_val(c_vs_b, ["p_holm_bonferroni", "mcnemar_p_mean", "mcnemar_p_value", "p_uncorrected", "p_value", "p"], 1.0)
        falsified = (delta <= 0.0) or (p_val >= 0.05)
        checks["order_dependency"] = {
            "hypothesis": "Model C temporal modeling significantly outperforms order-blind Model B",
            "delta_accuracy": delta,
            "p_value": p_val,
            "falsified": falsified,
            "verdict": "REJECTED (Temporal Gain Confirmed)" if not falsified else "FALSIFIED (No Order Value)",
        }

    # 2. Capacity Parity Gain (Model D vs Model B)
    d_vs_b = _find_contrast(["Model D vs Model B", "Capacity Parity"])
    if d_vs_b:
        delta = _get_val(d_vs_b, ["delta_mean", "delta_accuracy", "delta"], 0.0)
        p_val = _get_val(d_vs_b, ["p_holm_bonferroni", "mcnemar_p_mean", "mcnemar_p_value", "p_uncorrected", "p_value", "p"], 1.0)
        falsified = (delta <= 0.0) or (p_val >= 0.05)
        checks["capacity_parity"] = {
            "hypothesis": "Model D outperforms Model B at exact 103,700 parameter parity",
            "delta_accuracy": delta,
            "p_value": p_val,
            "falsified": falsified,
            "verdict": "REJECTED (Capacity Parity Gain Confirmed)" if not falsified else "FALSIFIED (Confound Re-emerged)",
        }

    # 3. Chronological Order Sensitivity (Model C Canonical vs Shuffled)
    c_shuf = _find_contrast(["Model C Canonical vs Shuffled", "Model C: Can vs Shuf"])
    if c_shuf:
        delta = _get_val(c_shuf, ["delta_mean", "delta_accuracy", "delta"], 0.0)
        p_val = _get_val(c_shuf, ["p_holm_bonferroni", "mcnemar_p_mean", "mcnemar_p_value", "p_uncorrected", "p_value", "p"], 1.0)
        falsified = (delta <= 0.0) or (p_val >= 0.05)
        checks["order_sensitivity"] = {
            "hypothesis": "Model C performance collapses under random temporal permutation",
            "delta_accuracy": delta,
            "p_value": p_val,
            "falsified": falsified,
            "verdict": "REJECTED (Order Sensitivity Confirmed)" if not falsified else "FALSIFIED (Order-Insensitive)",
        }

    # 4. Temporal Reversal Sensitivity (Model C Forward vs Reversed)
    c_rev = _find_contrast(["Model C Forward vs Reversed", "Model C: Fwd vs Rev"])
    if c_rev:
        delta = _get_val(c_rev, ["delta_mean", "delta_accuracy", "delta"], 0.0)
        p_val = _get_val(c_rev, ["p_holm_bonferroni", "mcnemar_p_mean", "mcnemar_p_value", "p_uncorrected", "p_value", "p"], 1.0)
        falsified = (delta <= 0.0) or (p_val >= 0.05)
        checks["reversal_sensitivity"] = {
            "hypothesis": "Model C performance collapses when sequence is flipped in time",
            "delta_accuracy": delta,
            "p_value": p_val,
            "falsified": falsified,
            "verdict": "REJECTED (Arrow of Time Confirmed)" if not falsified else "FALSIFIED (Direction-Blind)",
        }

    all_confirmed = all(not c["falsified"] for c in checks.values()) if len(checks) > 0 else False

    return {
        "all_hypotheses_robustly_confirmed": all_confirmed,
        "checks": checks,
        "overall_status": "CERTIFIED_ROBUST" if all_confirmed else "PARTIAL_FALSIFICATION",
    }


def build_comparison_suites(preds: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Define all 18 planned paired contrasts across the 5 experimental suites.
    Directly fulfills Take-Home Assessment Sections 5, 7, 10, & 11.
    """
    suites = [
        # Suite 1: Architectural Contrasts on Canonical Video
        {
            "category": "Architectural Contrasts",
            "name": "Model C vs Model A (Total Temporal Gain)",
            "short_name": "Model C vs Model A",
            "m1": ("baseline", "canonical"),
            "m2": ("gru", "canonical"),
            "hypothesis": "H1: Recurrent temporal modeling outperforms single-frame ceiling",
        },
        {
            "category": "Architectural Contrasts",
            "name": "Model B vs Model A (Observation Accumulation)",
            "short_name": "Model B vs Model A",
            "m1": ("baseline", "canonical"),
            "m2": ("meanpool", "canonical"),
            "hypothesis": "H1: Multi-frame spatial pooling outperforms single frame",
        },
        {
            "category": "Architectural Contrasts",
            "name": "Model C vs Model B (Order Dependency Gain)",
            "short_name": "Model C vs Model B",
            "m1": ("meanpool", "canonical"),
            "m2": ("gru", "canonical"),
            "hypothesis": "H2: Chronological order provides distinct gain over spatial pooling",
        },
        {
            "category": "Architectural Contrasts",
            "name": "Model D vs Model B (Capacity Parity)",
            "short_name": "Model D vs Model B",
            "m1": ("meanpool", "canonical"),
            "m2": ("compact_gru", "canonical"),
            "hypothesis": "Parity: Temporal gain persists at identical 103,700 capacity",
        },
        {
            "category": "Architectural Contrasts",
            "name": "Model C vs Model D (Parameter Scaling)",
            "short_name": "Model C vs Model D",
            "m1": ("compact_gru", "canonical"),
            "m2": ("gru", "canonical"),
            "hypothesis": "Scale: 249k GRU expands capacity beyond 103k compact head",
        },

        # Suite 2: Temporal Order Sensitivity (M4.3)
        {
            "category": "Temporal Order Sensitivity",
            "name": "Model C Canonical vs Shuffled",
            "short_name": "Model C: Can vs Shuf",
            "m1": ("gru", "shuffled"),
            "m2": ("gru", "canonical"),
            "hypothesis": "H2: Model C collapses under random temporal permutation",
        },
        {
            "category": "Temporal Order Sensitivity",
            "name": "Model C Canonical vs Repeated Midpoint",
            "short_name": "Model C: Can vs Rep",
            "m1": ("gru", "repeated"),
            "m2": ("gru", "canonical"),
            "hypothesis": "H2: Static repeated frames provide zero transition information",
        },
        {
            "category": "Temporal Order Sensitivity",
            "name": "Model D Canonical vs Shuffled",
            "short_name": "Model D: Can vs Shuf",
            "m1": ("compact_gru", "shuffled"),
            "m2": ("compact_gru", "canonical"),
            "hypothesis": "H2: Model D remains order-sensitive at 103.7k parameter parity",
        },
        {
            "category": "Temporal Order Sensitivity",
            "name": "Model B Canonical vs Shuffled (Control Invariance)",
            "short_name": "Model B: Can vs Shuf",
            "m1": ("meanpool", "shuffled"),
            "m2": ("meanpool", "canonical"),
            "hypothesis": "Control: Mean-pooling is strictly order-invariant (Delta == 0.00)",
        },

        # Suite 3: Directional Semantics & Arrow of Time (M4.5)
        {
            "category": "Directional Arrow of Time",
            "name": "Model C Forward vs Reversed",
            "short_name": "Model C: Fwd vs Rev",
            "m1": ("gru", "reversed"),
            "m2": ("gru", "canonical"),
            "hypothesis": "H5: Reversing time destroys forward action dynamics",
        },
        {
            "category": "Directional Arrow of Time",
            "name": "Model D Forward vs Reversed",
            "short_name": "Model D: Fwd vs Rev",
            "m1": ("compact_gru", "reversed"),
            "m2": ("compact_gru", "canonical"),
            "hypothesis": "H5: Compact recurrent head depends on arrow of time",
        },
        {
            "category": "Directional Arrow of Time",
            "name": "Model B Forward vs Reversed (Direction Invariance)",
            "short_name": "Model B: Fwd vs Rev",
            "m1": ("meanpool", "reversed"),
            "m2": ("meanpool", "canonical"),
            "hypothesis": "Control: Mean-pooling is strictly time-symmetric (Delta == 0.00)",
        },

        # Suite 4: Missing Observations & Robustness (M4.4)
        {
            "category": "Missing Observations & Robustness",
            "name": "Model C Clean vs 25% Frame Drop",
            "short_name": "Model C: Clean vs 25% Drop",
            "m1": ("gru", "drop25"),
            "m2": ("gru", "canonical"),
            "hypothesis": "H3: Graceful degradation under 25% missing frames",
        },
        {
            "category": "Missing Observations & Robustness",
            "name": "Model C Clean vs 50% Frame Drop",
            "short_name": "Model C: Clean vs 50% Drop",
            "m1": ("gru", "drop50"),
            "m2": ("gru", "canonical"),
            "hypothesis": "H3: Recurrent state retains accuracy under 50% dropout",
        },
        {
            "category": "Missing Observations & Robustness",
            "name": "Model C Clean vs Noise sigma=0.25",
            "short_name": "Model C: Clean vs Noise 0.25",
            "m1": ("gru", "noise025"),
            "m2": ("gru", "canonical"),
            "hypothesis": "H4: Continuous recurrent dynamics withstand moderate noise",
        },
        {
            "category": "Missing Observations & Robustness",
            "name": "Model D Clean vs 50% Frame Drop",
            "short_name": "Model D: Clean vs 50% Drop",
            "m1": ("compact_gru", "drop50"),
            "m2": ("compact_gru", "canonical"),
            "hypothesis": "H3: Capacity parity model degrades gracefully under 50% loss",
        },

        # Suite 5: Observation Horizon Margins (M4.2)
        {
            "category": "Observation Scaling Frontier",
            "name": "Model C Observation Horizon K=2 vs K=1",
            "short_name": "Model C: K=2 vs K=1",
            "m1": ("gru", "k1"),
            "m2": ("gru", "k2"),
            "hypothesis": "H1: Doubling observations from 1 to 2 frames yields marginal gain",
        },
        {
            "category": "Observation Scaling Frontier",
            "name": "Model C Observation Horizon K=4 vs K=2",
            "short_name": "Model C: K=4 vs K=2",
            "m1": ("gru", "k2"),
            "m2": ("gru", "k4"),
            "hypothesis": "H1: 4 frames resolve multi-state action transitions",
        },
        {
            "category": "Observation Scaling Frontier",
            "name": "Model C Observation Horizon K=8 vs K=4",
            "short_name": "Model C: K=8 vs K=4",
            "m1": ("gru", "k4"),
            "m2": ("gru", "k8"),
            "hypothesis": "H1 & H5: Full 8 frames achieve empirical saturation K*",
        },
    ]
    return suites


def run_full_statistical_battery(
    y_true: np.ndarray,
    seq_ids: List[str],
    preds: Dict[str, Any],
    suites: List[Dict[str, Any]],
    n_boot: int = 2000,
    seeds: List[int] = [42, 43, 44],
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Execute all pairwise paired tests, bootstrap CIs, dispersion calculations, and Holm-Bonferroni adjustments.

    Args:
        y_true: Ground truth class array [N]
        seq_ids: List of sequence identifiers [N]
        preds: Prediction lookup dict: preds[model][condition][seed] -> np.ndarray
        suites: List of contrast specifications from build_comparison_suites
        n_boot: Number of bootstrap iterations (default: 2000)
        seeds: Random seeds evaluated (default: [42, 43, 44])

    Returns:
        (results_summary, df_discordance)
    """
    results_summary = []
    discordance_matrix_records = []

    # Prepare per-sequence discordance table
    for i, seq_id in enumerate(seq_ids):
        discordance_matrix_records.append({
            "sequence_id": seq_id,
            "ground_truth": int(y_true[i]),
        })

    for contrast in suites:
        c_name = contrast["name"]
        cat = contrast["category"]
        (m1_name, cond1) = contrast["m1"]
        (m2_name, cond2) = contrast["m2"]

        per_seed_deltas = []
        per_seed_pvals = []
        per_seed_chi2 = []
        per_seed_n01 = []
        per_seed_n10 = []
        per_seed_ors = []
        per_seed_cohen_g = []
        per_seed_ci_low = []
        per_seed_ci_high = []

        # Store mean difference vector for discordance logging
        mean_diff_vec = np.zeros(len(y_true), dtype=float)

        for s in seeds:
            y_m1 = preds[m1_name][cond1][s]
            y_m2 = preds[m2_name][cond2][s]

            contingency = compute_paired_contingency(y_true, y_m1, y_m2, f"{m1_name}_{cond1}", f"{m2_name}_{cond2}")
            chi2_stat, p_val, test_type = mcnemar_paired_test(y_true, y_m1, y_m2)
            _, ci_lo, ci_hi = bootstrap_paired_margin_ci(y_true, y_m1, y_m2, n_boot=n_boot, seed=s)

            per_seed_deltas.append(contingency["delta_accuracy"])
            per_seed_pvals.append(p_val)
            per_seed_chi2.append(chi2_stat)
            per_seed_n01.append(contingency["n_01_rescued_by_m2"])
            per_seed_n10.append(contingency["n_10_degraded_by_m2"])
            per_seed_ors.append(contingency["odds_ratio"])
            per_seed_cohen_g.append(contingency["cohens_g"])
            per_seed_ci_low.append(ci_lo)
            per_seed_ci_high.append(ci_hi)

            # Difference vector: +1 if M2 right and M1 wrong; -1 if M1 right and M2 wrong; 0 if concordant
            diff_s = (y_m2 == y_true).astype(float) - (y_m1 == y_true).astype(float)
            mean_diff_vec += diff_s / len(seeds)

        # Record per-sequence discordance column for this contrast
        clean_key = contrast["short_name"].replace(" ", "_").replace(":", "").replace("/", "_")
        for i in range(len(y_true)):
            discordance_matrix_records[i][f"delta_{clean_key}"] = float(mean_diff_vec[i])

        # Compute cross-seed dispersion
        disp_delta = compute_multi_seed_dispersion(per_seed_deltas)
        disp_p = compute_multi_seed_dispersion(per_seed_pvals)
        disp_chi2 = compute_multi_seed_dispersion(per_seed_chi2)
        disp_n01 = compute_multi_seed_dispersion(per_seed_n01)
        disp_n10 = compute_multi_seed_dispersion(per_seed_n10)
        disp_or = compute_multi_seed_dispersion(per_seed_ors)
        disp_g = compute_multi_seed_dispersion(per_seed_cohen_g)

        results_summary.append({
            "category": cat,
            "contrast_name": c_name,
            "short_name": contrast["short_name"],
            "hypothesis": contrast["hypothesis"],
            "delta_mean": disp_delta["mean"],
            "delta_std": disp_delta["std"],
            "delta_sem": disp_delta["sem"],
            "ci_95_low": float(np.mean(per_seed_ci_low)),
            "ci_95_high": float(np.mean(per_seed_ci_high)),
            "mcnemar_chi2_mean": disp_chi2["mean"],
            "mcnemar_p_mean": disp_p["mean"],
            "p_uncorrected": float(np.mean(per_seed_pvals)),
            "n_rescued_mean": disp_n01["mean"],
            "n_degraded_mean": disp_n10["mean"],
            "odds_ratio_mean": disp_or["mean"],
            "cohens_g_mean": disp_g["mean"],
            "is_significant_05": bool(disp_p["mean"] < 0.05),
            "is_significant_001": bool(disp_p["mean"] < 0.001),
        })

    # Apply Holm-Bonferroni correction across all contrasts
    raw_pvals = [r["p_uncorrected"] for r in results_summary]
    adj_pvals = holm_bonferroni_correction(raw_pvals)

    for r, p_adj in zip(results_summary, adj_pvals):
        r["p_holm_bonferroni"] = p_adj
        r["is_significant_adj_05"] = bool(p_adj < 0.05)
        # Format string for display
        if r["p_holm_bonferroni"] < 0.001:
            r["significance_marker"] = "***"
        elif r["p_holm_bonferroni"] < 0.01:
            r["significance_marker"] = "**"
        elif r["p_holm_bonferroni"] < 0.05:
            r["significance_marker"] = "*"
        else:
            r["significance_marker"] = "NS"

    df_discordance = pd.DataFrame(discordance_matrix_records)
    return results_summary, df_discordance
