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

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        labels: ground truth class indices [N]
        predictions: predicted class indices [N]
        class_names: optional list of class name strings

    Returns:
        dict with accuracy, macro_f1, per_class_f1, confusion_matrix
    """
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    cm = confusion_matrix(labels, predictions)

    result = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'per_class_f1': per_class_f1.tolist(),
        'n_samples': int(len(labels)),
        'n_classes': int(len(np.unique(labels))),
    }

    if class_names:
        result['per_class_f1_named'] = {
            name: float(f1) for name, f1 in zip(class_names, per_class_f1)
        }

    return result


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

    return {
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
            {'accuracy': r['accuracy'], 'macro_f1': r['macro_f1']}
            for r in results_per_seed
        ],
    }


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

