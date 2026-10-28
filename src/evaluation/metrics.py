"""
Evaluation metrics for temporal visual intelligence experiments.
(research.md §14, §26)

Always report:
  - Accuracy
  - Macro-F1 (especially important for class imbalance)
  - Per-class F1
  - Mean ± std across seeds

Never report a single-seed accuracy as the final result.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


# Canonical reversible action pairs in benchmark (M3.14 Directional Error Topology)
DEFAULT_DIRECTIONAL_PAIRS = [
    (0, 1, "Moving away vs Moving towards"),
    (2, 3, "Pushing left-to-right vs Pushing right-to-left"),
    (4, 5, "Picking up vs Putting down"),
    (6, 7, "Opening vs Closing"),
    (8, 9, "Turning camera left vs Turning camera right"),
    (10, 11, "Putting on top vs Taking off"),
    (12, 13, "Covering vs Uncovering"),
    (14, 15, "Putting into vs Taking out of"),
    (16, 17, "Rotating clockwise vs Rotating counter-clockwise"),
    (18, 19, "Tipping over vs Standing upright"),
]


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics (M3.14).
    Reports Accuracy, Macro-F1, Macro-Precision, Macro-Recall, Per-Class Breakdown,
    Support counts, and Confusion Matrix.

    Args:
        labels: ground truth class indices [N]
        predictions: predicted class indices [N]
        class_names: optional list of class name strings

    Returns:
        dict with accuracy, macro_f1, macro_precision, macro_recall,
        per_class_precision, per_class_recall, per_class_f1, per_class_support,
        confusion_matrix, and named breakdowns.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    n_classes = len(class_names) if class_names else int(max(np.max(labels), np.max(predictions)) + 1)
    class_indices = list(range(n_classes))

    acc = accuracy_score(labels, predictions)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        labels, predictions, labels=class_indices, average=None, zero_division=0
    )
    cm = confusion_matrix(labels, predictions, labels=class_indices)

    result = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'weighted_f1': float(weighted_f1),
        'weighted_precision': float(weighted_p),
        'weighted_recall': float(weighted_r),
        'per_class_precision': [float(p) for p in per_p],
        'per_class_recall': [float(r) for r in per_r],
        'per_class_f1': [float(f) for f in per_f1],
        'per_class_support': [int(s) for s in per_support],
        'confusion_matrix': cm.tolist(),
        'n_samples': int(len(labels)),
        'n_classes': n_classes,
    }

    if class_names:
        result['per_class_metrics_named'] = {
            name: {
                'precision': float(p),
                'recall': float(r),
                'f1': float(f),
                'support': int(s),
            }
            for name, p, r, f, s in zip(class_names, per_p, per_r, per_f1, per_support)
        }
        # Backward compatibility key
        result['per_class_f1_named'] = {
            name: float(f) for name, f in zip(class_names, per_f1)
        }

    return result


def generate_per_class_table(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    model_name: str = "Model",
) -> str:
    """
    Generate Markdown formatted per-class classification table (M3.14).
    Columns: Class ID, Action Name, Precision, Recall, F1-Score, Support.
    Includes Macro and Weighted average summary rows.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    metrics = compute_metrics(labels, predictions, class_names=class_names)

    lines = [
        f"# Per-Class Classification Performance ({model_name})",
        f"Detailed Precision, Recall, F1-Score, and Support breakdown (Take-Home Assessment §4 & §10)\n",
        "| Class ID | Action Name | Precision | Recall | F1-Score | Support |",
        "|:---:|---|---:|---:|---:|---:|",
    ]

    n_classes = metrics['n_classes']
    for c in range(n_classes):
        name = class_names[c] if class_names and c < len(class_names) else f"Class_{c}"
        p = metrics['per_class_precision'][c]
        r = metrics['per_class_recall'][c]
        f = metrics['per_class_f1'][c]
        s = metrics['per_class_support'][c]
        lines.append(f"| {c} | {name} | {p:.3f} | {r:.3f} | {f:.3f} | {s} |")

    # Add summary rows
    lines.append("|:---:|---|---:|---:|---:|---:|")
    lines.append(
        f"| — | **Macro Average** | **{metrics['macro_precision']:.3f}** | **{metrics['macro_recall']:.3f}** | **{metrics['macro_f1']:.3f}** | **{metrics['n_samples']}** |"
    )
    lines.append(
        f"| — | **Weighted Average** | **{metrics['weighted_precision']:.3f}** | **{metrics['weighted_recall']:.3f}** | **{metrics['weighted_f1']:.3f}** | **{metrics['n_samples']}** |"
    )
    lines.append(
        f"| — | **Overall Accuracy** | — | — | **{metrics['accuracy']:.3f}** | **{metrics['n_samples']}** |"
    )

    content = "\n".join(lines) + "\n"
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Saved per-class report table → {output_path}")

    return content


def analyze_directional_confusion(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Optional[List[str]] = None,
    pairs: Optional[List[Tuple[int, int, str]]] = None,
) -> Dict[str, Any]:
    """
    Quantify directional error rates E_dir across reversible action pairs (M3.14).
    Measures cross-confusion between forward and reverse action sequences:
      E_dir(A, B) = (M_{A, B} + M_{B, A}) / (N_A + N_B)

    Order-blind models exhibit high directional confusion because they cannot
    resolve chronological sequence direction.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    pairs = pairs or DEFAULT_DIRECTIONAL_PAIRS
    n_classes = len(class_names) if class_names else int(max(np.max(labels), np.max(predictions)) + 1)
    cm = confusion_matrix(labels, predictions, labels=list(range(n_classes)))

    pair_results = []
    total_cross_confusions = 0
    total_pair_support = 0

    for idx_a, idx_b, desc in pairs:
        if idx_a >= n_classes or idx_b >= n_classes:
            continue
        # True A predicted as B (forward confused as reverse)
        ab = int(cm[idx_a, idx_b])
        # True B predicted as A (reverse confused as forward)
        ba = int(cm[idx_b, idx_a])
        # Total support for both classes
        support_a = int(cm[idx_a].sum())
        support_b = int(cm[idx_b].sum())
        pair_support = support_a + support_b
        cross_err = (ab + ba) / pair_support if pair_support > 0 else 0.0

        total_cross_confusions += (ab + ba)
        total_pair_support += pair_support

        name_a = class_names[idx_a] if class_names else f"Class_{idx_a}"
        name_b = class_names[idx_b] if class_names else f"Class_{idx_b}"

        pair_results.append({
            'class_a': idx_a,
            'class_b': idx_b,
            'name_a': name_a,
            'name_b': name_b,
            'description': desc,
            'a_to_b_confusion': ab,
            'b_to_a_confusion': ba,
            'cross_confusion_count': ab + ba,
            'pair_support': pair_support,
            'directional_error_rate': float(cross_err),
        })

    overall_directional_error = (
        float(total_cross_confusions / total_pair_support) if total_pair_support > 0 else 0.0
    )

    return {
        'pair_results': pair_results,
        'total_directional_confusions': total_cross_confusions,
        'total_pair_support': total_pair_support,
        'overall_directional_error_rate': overall_directional_error,
        'formatted_overall_error': f"{overall_directional_error * 100:.1f}%",
    }



def aggregate_across_seeds(
    results_per_seed: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute mean ± std across multiple seed runs.

    Args:
        results_per_seed: list of metric dicts (one per seed)

    Returns:
        dict with mean, std, and 95% CI for each metric
    """
    accs = [r['accuracy'] for r in results_per_seed]
    f1s  = [r['macro_f1'] for r in results_per_seed]

    def bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95) -> tuple:
        """95% bootstrap confidence interval."""
        rng = np.random.RandomState(42)
        boots = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(n_boot)]
        alpha = (1 - ci) / 2
        return (float(np.percentile(boots, 100 * alpha)),
                float(np.percentile(boots, 100 * (1 - alpha))))

    acc_ci = bootstrap_ci(accs)
    f1_ci  = bootstrap_ci(f1s)

    result = {
        'accuracy': {
            'mean': float(np.mean(accs)),
            'std':  float(np.std(accs)),
            'ci_95': acc_ci,
            'formatted': f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
        },
        'macro_f1': {
            'mean': float(np.mean(f1s)),
            'std':  float(np.std(f1s)),
            'ci_95': f1_ci,
            'formatted': f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}",
        },
        'n_seeds': len(results_per_seed),
        'seeds_summary': [
            {
                'accuracy': r['accuracy'],
                'macro_f1': r['macro_f1'],
                'macro_precision': r.get('macro_precision'),
                'macro_recall': r.get('macro_recall'),
            }
            for r in results_per_seed
        ],
    }

    if any('macro_precision' in r for r in results_per_seed):
        ps = [r['macro_precision'] for r in results_per_seed if 'macro_precision' in r]
        result['macro_precision'] = {
            'mean': float(np.mean(ps)),
            'std': float(np.std(ps)),
            'formatted': f"{np.mean(ps):.3f} ± {np.std(ps):.3f}",
        }
    if any('macro_recall' in r for r in results_per_seed):
        rs = [r['macro_recall'] for r in results_per_seed if 'macro_recall' in r]
        result['macro_recall'] = {
            'mean': float(np.mean(rs)),
            'std': float(np.std(rs)),
            'formatted': f"{np.mean(rs):.3f} ± {np.std(rs):.3f}",
        }

    return result


def format_ablation_table(
    results: Dict[str, Dict],
    baseline_key: str = 'baseline_1frame',
) -> str:
    """
    Format results as a Markdown ablation table.
    (research.md §27)
    """
    baseline_acc = results.get(baseline_key, {}).get('accuracy', {}).get('mean', 0)

    rows = ["| Experiment | Accuracy | Macro-F1 | Δ Acc |",
            "|---|---:|---:|---:|"]

    for name, metrics in results.items():
        if 'accuracy' in metrics and isinstance(metrics['accuracy'], dict):
            acc = metrics['accuracy']['mean']
            f1  = metrics['macro_f1']['mean']
            std_acc = metrics['accuracy']['std']
            delta = acc - baseline_acc
            rows.append(
                f"| {name} | {acc:.3f} ± {std_acc:.3f} | {f1:.3f} | {delta:+.3f} |"
            )

    return "\n".join(rows)


def bootstrap_ci(metric_fn, y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, alpha: float = 0.95, seed: int = 42) -> tuple:
    """
    Compute 95% non-parametric bootstrap confidence interval for any scalar evaluation metric.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = float(np.percentile(scores, (1 - alpha) / 2 * 100))
    hi = float(np.percentile(scores, (1 + alpha) / 2 * 100))
    return float(np.mean(scores)), lo, hi


def mcnemar_test(y_base: np.ndarray, y_temp: np.ndarray, y_true: np.ndarray) -> float:
    """
    McNemar's test with continuity correction: evaluates if discordance between
    Model 1 and Model 2 on identical test sequences is statistically significant.
    Returns p-value.
    """
    from scipy.stats import chi2
    b = int(np.sum((y_base != y_true) & (y_temp == y_true)))  # base wrong, temp right
    c = int(np.sum((y_base == y_true) & (y_temp != y_true)))  # base right, temp wrong
    n = b + c
    if n == 0:
        return 1.0
    chi2_stat = (abs(b - c) - 1.0) ** 2 / n
    return float(1.0 - chi2.cdf(chi2_stat, df=1))


def compute_temporal_delta(
    single_metrics: Dict[str, Any],
    temporal_metrics: Dict[str, Any],
    y_single: Optional[np.ndarray] = None,
    y_temporal: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes Delta_temporal = Performance_temporal - Performance_single (M-3.md §M3.2).
    Quantifies the exact empirical gain provided by temporal modeling over the single-frame ceiling.
    """
    def _extract_acc(m):
        if isinstance(m.get('accuracy'), dict):
            return m['accuracy']['mean']
        return float(m['accuracy'])

    def _extract_f1(m):
        if isinstance(m.get('macro_f1'), dict):
            return m['macro_f1']['mean']
        return float(m['macro_f1'])

    single_acc = _extract_acc(single_metrics)
    temporal_acc = _extract_acc(temporal_metrics)
    single_f1 = _extract_f1(single_metrics)
    temporal_f1 = _extract_f1(temporal_metrics)

    delta_acc = temporal_acc - single_acc
    delta_f1 = temporal_f1 - single_f1

    result = {
        'single_accuracy': single_acc,
        'temporal_accuracy': temporal_acc,
        'delta_accuracy': delta_acc,
        'delta_macro_f1': delta_f1,
        'formatted_delta_acc': f"{delta_acc:+.3f}",
        'formatted_delta_f1': f"{delta_f1:+.3f}",
    }

    if y_single is not None and y_temporal is not None and y_true is not None:
        p_val = mcnemar_test(np.array(y_single), np.array(y_temporal), np.array(y_true))
        result['mcnemar_p_value'] = p_val
        result['statistically_significant'] = bool(p_val < 0.05)

    return result


def disentangle_temporal_gains(
    single_metrics: Dict[str, Any],
    meanpool_metrics: Dict[str, Any],
    temporal_metrics: Dict[str, Any],
    y_single: Optional[np.ndarray] = None,
    y_meanpool: Optional[np.ndarray] = None,
    y_temporal: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Decomposes total temporal gain into Observation Accumulation vs. Temporal Order (M-3.md §M3.5).
      Delta_obs = MeanPool - SingleFrame (Benefit of multiple observations)
      Delta_order = TemporalGRU - MeanPool (Benefit of chronological sequence order)
      Delta_total = TemporalGRU - SingleFrame = Delta_obs + Delta_order
      Order-Contribution Ratio: R_order = Delta_order / Delta_total
    """
    def _extract_acc(m):
        if isinstance(m.get('accuracy'), dict):
            return m['accuracy']['mean']
        return float(m['accuracy'])

    def _extract_f1(m):
        if isinstance(m.get('macro_f1'), dict):
            return m['macro_f1']['mean']
        return float(m['macro_f1'])

    acc_single = _extract_acc(single_metrics)
    acc_mean = _extract_acc(meanpool_metrics)
    acc_temporal = _extract_acc(temporal_metrics)

    f1_single = _extract_f1(single_metrics)
    f1_mean = _extract_f1(meanpool_metrics)
    f1_temporal = _extract_f1(temporal_metrics)

    delta_obs_acc = acc_mean - acc_single
    delta_order_acc = acc_temporal - acc_mean
    delta_total_acc = acc_temporal - acc_single

    delta_obs_f1 = f1_mean - f1_single
    delta_order_f1 = f1_temporal - f1_mean
    delta_total_f1 = f1_temporal - f1_single

    denom = delta_total_acc if abs(delta_total_acc) > 1e-6 else 1e-6
    order_ratio = float(delta_order_acc / denom)
    obs_ratio = float(delta_obs_acc / denom)

    if delta_order_acc > 0.03:
        interpretation = "Temporal order provides significant predictive information beyond pooled frames."
    elif delta_obs_acc > 0.03 and delta_order_acc <= 0.015:
        interpretation = "Gains stem primarily from multi-frame observation accumulation, not chronological order."
    elif delta_total_acc <= 0.02:
        interpretation = "Single-frame baseline is near-sufficient; task exhibits low temporal dependency."
    else:
        interpretation = "Mixed contribution from both observation accumulation and sequential ordering."

    result = {
        'single_accuracy': acc_single,
        'meanpool_accuracy': acc_mean,
        'temporal_accuracy': acc_temporal,
        'delta_obs_accuracy': delta_obs_acc,
        'delta_order_accuracy': delta_order_acc,
        'delta_total_accuracy': delta_total_acc,
        'order_ratio': float(order_ratio),
        'obs_ratio': float(obs_ratio),
        'delta_obs_macro_f1': delta_obs_f1,
        'delta_order_macro_f1': delta_order_f1,
        'delta_total_macro_f1': delta_total_f1,
        'formatted_delta_obs_acc': f"{delta_obs_acc:+.3f}",
        'formatted_delta_order_acc': f"{delta_order_acc:+.3f}",
        'formatted_delta_total_acc': f"{delta_total_acc:+.3f}",
        'formatted_order_ratio': f"{order_ratio * 100:.1f}%",
        'interpretation': interpretation,
    }

    # Hypothesis testing: H0: Model C <= Model B (Assessment §7.D)
    if y_meanpool is not None and y_temporal is not None and y_true is not None:
        p_val_order = mcnemar_test(np.array(y_meanpool), np.array(y_temporal), np.array(y_true))
        result['mcnemar_p_value_order'] = p_val_order
        result['order_significant'] = bool(p_val_order < 0.05 and delta_order_acc > 0)
        result['temporal_hypothesis_falsified'] = bool(delta_order_acc <= 0 or p_val_order >= 0.05)

    return result
