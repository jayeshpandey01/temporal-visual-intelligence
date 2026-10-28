"""
Milestone M4.5: Experiment D — Temporal Reversal & Directional Semantics (H2 & H5).
Directly fulfills Take-Home Assessment M4.5:
  - Tests H2 & H5: Does the model depend on the forward direction of time (arrow of time),
    and does it recognize reversed action transitions?
  - Conditions evaluated:
      1. Forward (Canonical): F_1 -> F_2 -> ... -> F_8
      2. Reversed:            F_8 -> F_7 -> ... -> F_1
  - Scientific properties of temporal reversal:
      - Preserves all frames (identical visual content).
      - Preserves local frame adjacencies (F_t is adjacent to F_{t-1} and F_{t+1}).
      - Modifies exclusively the chronological direction of the temporal vector.
  - The Directional Inversion Hypothesis (P_inv):
      - On the 10 canonical reversible action pairs in Something-Something V2,
        reversal physically inverts the action (e.g., Opening -> Closing).
      - P_inv = P(y_hat_reversed = inv(y) | y) measures whether the model systematically
        recognizes the inverse action rather than making random errors.
  - Controls:
      - Model B (MeanPool Control): Mathematically blind to direction (Delta_reverse = 0.00 pp).
      - Model D (CompactGRU): Exact parameter match (103.7k) to verify temporal dependency
        is not an artifact of parameter capacity.
      - Model C (TemporalGRU): Full recurrent head (249.4k).
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.metrics import (
    compute_metrics,
    bootstrap_ci,
    mcnemar_test,
    analyze_directional_confusion,
    DEFAULT_DIRECTIONAL_PAIRS,
)
from src.experiments.controls import (
    SingleVariableIntervention,
    ExperimentalCondition,
)


def get_inverse_class(class_idx: int) -> int:
    """
    Return the physically inverted action class index for a given class.

    In the 20-class Something-Something V2 Directional Benchmark, classes are
    organized in 10 consecutive reversible pairs:
      0 <-> 1:  Moving away vs Moving towards
      2 <-> 3:  Pushing left-to-right vs Pushing right-to-left
      4 <-> 5:  Picking up vs Putting down
      6 <-> 7:  Opening vs Closing
      8 <-> 9:  Turning camera left vs Turning camera right
      10 <-> 11: Putting on top vs Taking off
      12 <-> 13: Covering vs Uncovering
      14 <-> 15: Putting into vs Taking out of
      16 <-> 17: Rotating clockwise vs Rotating counter-clockwise
      18 <-> 19: Tipping over vs Standing upright

    Formally: inv(c) = c ^ 1 (c + 1 if c is even, c - 1 if c is odd).
    """
    if not (0 <= class_idx < 20):
        raise ValueError(f"Class index {class_idx} out of range [0, 19].")
    return class_idx ^ 1


def apply_temporal_reversal(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Reverse the temporal sequence dimension (dim=1).

    Args:
        embeddings: [B, T, D] representation tensor

    Returns:
        [B, T, D] tensor with frames along dim=1 flipped in time
    """
    if embeddings.dim() < 2:
        raise ValueError(f"Expected at least 2D tensor [B, T, ...], got shape {embeddings.shape}")
    return embeddings.flip(dims=[1])


def bootstrap_paired_delta_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute non-parametric bootstrap confidence interval for paired accuracy delta:
        Delta = Acc(y_pred_a) - Acc(y_pred_b)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        acc_a = float(np.mean(y_pred_a[idx] == y_true[idx]))
        acc_b = float(np.mean(y_pred_b[idx] == y_true[idx]))
        deltas.append(acc_a - acc_b)
    lo = float(np.percentile(deltas, (1 - alpha) / 2 * 100))
    hi = float(np.percentile(deltas, (1 + alpha) / 2 * 100))
    return float(np.mean(deltas)), lo, hi


def run_reversal_experiment(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 32,
    compute_bootstrap: bool = True,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
    model_name: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run temporal reversal evaluation on a model and test dataset.

    Evaluates:
      1. Forward (canonical) order: F_1 -> ... -> F_8
      2. Reversed order:            F_8 -> ... -> F_1

    Calculates:
      - Forward & Reversed Accuracy and Macro-F1
      - Reversal Drop: Delta_reverse = Acc_fwd - Acc_rev
      - Prediction Flip Rate: P_flip
      - Pairwise Inversion Rate: P_inv = P(y_hat_rev = inv(y))
      - Forward Inversion Rate: P_inv_fwd = P(y_hat_fwd = inv(y))
      - Paired McNemar significance test
      - Directional confusion error rate E_dir
      - 95% bootstrap confidence intervals
      - Sample-level prediction records

    Returns:
        Structured dict with evaluation metrics, statistical tests, and sample predictions.
    """
    model.eval()
    all_sequence_ids = []

    if hasattr(dataset, 'get_all_sequence_ids'):
        all_sequence_ids = dataset.get_all_sequence_ids()
    elif hasattr(dataset, 'samples'):
        all_sequence_ids = [s[0] if isinstance(s, (list, tuple)) else str(i)
                            for i, s in enumerate(dataset.samples)]

    # Validate single-variable intervention rule if model_name is provided
    if model_name:
        intervention_engine = SingleVariableIntervention()
        cond_rev = ExperimentalCondition(
            model_name=model_name,
            seed=seed,
            frame_count=8,
            order="reversed",
            corruption="none",
            corruption_fraction=0.0,
        )
        intervention_engine.validate_intervention(cond_rev)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_labels = []
    all_preds_fwd = []
    all_preds_rev = []

    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)

            # Forward evaluation
            logits_fwd = model(embeddings)
            preds_fwd = logits_fwd.argmax(dim=-1)

            # Reversed evaluation
            rev_embeddings = apply_temporal_reversal(embeddings)
            logits_rev = model(rev_embeddings)
            preds_rev = logits_rev.argmax(dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds_fwd.extend(preds_fwd.cpu().numpy())
            all_preds_rev.extend(preds_rev.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred_fwd = np.array(all_preds_fwd)
    y_pred_rev = np.array(all_preds_rev)
    y_inv = np.array([get_inverse_class(int(y)) for y in y_true])

    # Core performance metrics
    metrics_fwd = compute_metrics(y_true, y_pred_fwd, class_names=class_names)
    metrics_rev = compute_metrics(y_true, y_pred_rev, class_names=class_names)

    # Reversal drops
    reversal_drop_acc = float(metrics_fwd['accuracy'] - metrics_rev['accuracy'])
    reversal_drop_f1 = float(metrics_fwd['macro_f1'] - metrics_rev['macro_f1'])

    # Prediction flip rate: how many predictions change under reversal
    p_flip = float(np.mean(y_pred_fwd != y_pred_rev))

    # Pairwise inversion rates
    # P_inv: when reversed, how often does the model predict the physical inverse action?
    is_pairwise_inverted = (y_pred_rev == y_inv)
    p_inv = float(np.mean(is_pairwise_inverted))

    # P_inv_fwd: baseline probability of predicting inverse action on clean forward sequence
    is_forward_inverted = (y_pred_fwd == y_inv)
    p_inv_fwd = float(np.mean(is_forward_inverted))

    # Inversion multiplier over uniform random chance (1/20 = 5.0%)
    chance_level = 1.0 / metrics_fwd['n_classes'] if metrics_fwd['n_classes'] > 0 else 0.05
    inversion_multiplier = float(p_inv / chance_level) if chance_level > 0 else 1.0

    # Directional confusion analysis
    dir_fwd = analyze_directional_confusion(y_true, y_pred_fwd, class_names=class_names)
    dir_rev = analyze_directional_confusion(y_true, y_pred_rev, class_names=class_names)

    # Paired McNemar discordance test (Forward vs Reversed accuracy discordance)
    mcnemar_p = mcnemar_test(y_pred_rev, y_pred_fwd, y_true)

    # Non-parametric bootstrap confidence intervals (95% CI)
    bootstrap_results = {}
    if compute_bootstrap:
        _, acc_fwd_lo, acc_fwd_hi = bootstrap_ci(
            lambda yt, yp: accuracy_score(yt, yp),
            y_true, y_pred_fwd,
            n_boot=bootstrap_resamples,
            seed=seed,
        )
        _, f1_fwd_lo, f1_fwd_hi = bootstrap_ci(
            lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
            y_true, y_pred_fwd,
            n_boot=bootstrap_resamples,
            seed=seed,
        )
        _, acc_rev_lo, acc_rev_hi = bootstrap_ci(
            lambda yt, yp: accuracy_score(yt, yp),
            y_true, y_pred_rev,
            n_boot=bootstrap_resamples,
            seed=seed,
        )
        _, f1_rev_lo, f1_rev_hi = bootstrap_ci(
            lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
            y_true, y_pred_rev,
            n_boot=bootstrap_resamples,
            seed=seed,
        )
        _, delta_lo, delta_hi = bootstrap_paired_delta_ci(
            y_true, y_pred_fwd, y_pred_rev,
            n_boot=bootstrap_resamples,
            seed=seed,
        )
        # CI for P_inv
        _, pinv_lo, pinv_hi = bootstrap_ci(
            lambda yt, yp: float(np.mean(yp == np.array([get_inverse_class(int(c)) for c in yt]))),
            y_true, y_pred_rev,
            n_boot=bootstrap_resamples,
            seed=seed,
        )

        metrics_fwd['ci_95_accuracy'] = (acc_fwd_lo, acc_fwd_hi)
        metrics_fwd['ci_95_macro_f1'] = (f1_fwd_lo, f1_fwd_hi)
        metrics_rev['ci_95_accuracy'] = (acc_rev_lo, acc_rev_hi)
        metrics_rev['ci_95_macro_f1'] = (f1_rev_lo, f1_rev_hi)

        bootstrap_results = {
            'ci_95_acc_fwd': (acc_fwd_lo, acc_fwd_hi),
            'ci_95_f1_fwd': (f1_fwd_lo, f1_fwd_hi),
            'ci_95_acc_rev': (acc_rev_lo, acc_rev_hi),
            'ci_95_f1_rev': (f1_rev_lo, f1_rev_hi),
            'ci_95_reversal_drop_acc': (delta_lo, delta_hi),
            'ci_95_p_inv': (pinv_lo, pinv_hi),
        }

    # Format sample-level records for inspection and win/loss mining
    sample_predictions = []
    for i in range(len(y_true)):
        seq_id = all_sequence_ids[i] if i < len(all_sequence_ids) else str(i)
        sample_predictions.append({
            'sequence_id': seq_id,
            'ground_truth': int(y_true[i]),
            'inverse_label': int(y_inv[i]),
            'pred_forward': int(y_pred_fwd[i]),
            'pred_reversed': int(y_pred_rev[i]),
            'forward_correct': bool(y_pred_fwd[i] == y_true[i]),
            'reversed_correct': bool(y_pred_rev[i] == y_true[i]),
            'is_prediction_flipped': bool(y_pred_fwd[i] != y_pred_rev[i]),
            'is_pairwise_inverted': bool(is_pairwise_inverted[i]),
            'is_forward_inverted': bool(is_forward_inverted[i]),
        })

    return {
        'forward': metrics_fwd,
        'reversed': metrics_rev,
        'reversal_drop_acc': reversal_drop_acc,
        'reversal_drop_f1': reversal_drop_f1,
        'p_flip': p_flip,
        'p_inv': p_inv,
        'p_inv_fwd': p_inv_fwd,
        'inversion_multiplier': inversion_multiplier,
        'chance_level': chance_level,
        'mcnemar_p_value': mcnemar_p,
        'directional_confusion_fwd': dir_fwd,
        'directional_confusion_rev': dir_rev,
        'bootstrap': bootstrap_results,
        'sample_predictions': sample_predictions,
    }

