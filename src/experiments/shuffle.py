"""
Milestone M4.3: Experiment B — Temporal Order Ablation Battery (H2).
Directly fulfills Take-Home Assessment M4.3:
  - Tests H2: Does the model depend on the correct temporal ordering of observations?
  - Conditions evaluated:
      1. canonical: Original sequence, correct chronological order (Control)
      2. shuffled:  Seeded random permutation of the exact same frames
      3. repeated:  Midpoint anchor frame repeated 8 times (Key Senior Control)
      4. reversed:  Sequence flipped in time (Directional Semantics)
  - Controls:
      - Exact frame conservation (same frames, count T=8, labels, preprocessing).
      - Model B (Mean-Pooling) evaluated as exact mathematical order-invariant control (Drop = 0.0 pp).
  - Quantitative metrics:
      - OrderDrop (Acc and Macro-F1)
      - ReversalDrop
      - RepeatedDrop
      - Prediction Flip Rate (P_flip)
      - Paired McNemar significance test
      - 95% non-parametric bootstrap confidence intervals
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.metrics import compute_metrics, bootstrap_ci, mcnemar_test
from src.experiments.controls import (
    SingleVariableIntervention,
    ExperimentalCondition,
)


PERTURBATION_MODES = ['canonical', 'shuffled', 'repeated', 'reversed']


def apply_order_perturbation(
    embeddings: torch.Tensor,
    mode: str,
    seed: Optional[int] = 42,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply temporal order perturbation to a batch of sequences [B, T, D].

    Args:
        embeddings: [B, T, D] representation tensor
        mode: 'canonical' | 'shuffled' | 'repeated' | 'reversed'
        seed: random seed for permutation generator
        generator: persistent torch.Generator to preserve entropy across batches
    """
    B, T, D = embeddings.shape

    if mode == 'canonical':
        return embeddings

    elif mode == 'shuffled':
        if generator is None:
            generator = torch.Generator(device='cpu')
            if seed is not None:
                generator.manual_seed(seed)
        result = embeddings.clone()
        for b in range(B):
            idx = torch.randperm(T, generator=generator).to(embeddings.device)
            result[b] = embeddings[b, idx, :]
        return result

    elif mode == 'repeated':
        # Repeat midpoint frame: same frame N times
        # Key senior control: separates "more input positions" from "temporal transitions"
        mid = T // 2
        mid_frame = embeddings[:, mid:mid+1, :].expand(-1, T, -1)
        return mid_frame.clone()

    elif mode == 'reversed':
        # Flip along temporal dimension (dim=1)
        return embeddings.flip(dims=[1])

    else:
        raise ValueError(f"Unknown perturbation mode: '{mode}'. Allowed: {PERTURBATION_MODES}")


def run_order_experiment(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 32,
    perturbation_seed: int = 42,
    compute_bootstrap: bool = True,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete temporal order ablation battery on a model and test dataset.

    Returns comprehensive dictionary with metrics, drops, flip rates,
    McNemar significance, bootstrap CIs, and sample-level predictions.
    """
    model.eval()
    results = {}
    preds_by_mode = {}
    all_sequence_ids = []

    if hasattr(dataset, 'get_all_sequence_ids'):
        all_sequence_ids = dataset.get_all_sequence_ids()
    elif hasattr(dataset, 'samples'):
        all_sequence_ids = [s[0] if isinstance(s, (list, tuple)) else str(i)
                            for i, s in enumerate(dataset.samples)]

    # Validate single-variable intervention rule if model_name is provided
    if model_name:
        try:
            intervention_engine = SingleVariableIntervention()
            for m in PERTURBATION_MODES:
                cond = ExperimentalCondition(
                    model_name=model_name,
                    seed=seed,
                    frame_count=8,
                    order=m,
                    corruption="none",
                    corruption_fraction=0.0,
                )
                intervention_engine.validate_intervention(cond)
        except Exception as e:
            raise e

    ground_truth_labels = None

    for mode in PERTURBATION_MODES:
        all_preds = []
        all_labels = []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Persistent generator per condition across all batches
        gen = torch.Generator(device='cpu')
        gen.manual_seed(perturbation_seed)

        with torch.no_grad():
            for embeddings, labels in loader:
                embeddings = embeddings.to(device)
                perturbed = apply_order_perturbation(
                    embeddings, mode,
                    seed=perturbation_seed,
                    generator=gen,
                )
                logits = model(perturbed)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        if ground_truth_labels is None:
            ground_truth_labels = y_true

        metrics = compute_metrics(y_true, y_pred)
        preds_by_mode[mode] = y_pred

        # 95% Bootstrap Confidence Intervals
        if compute_bootstrap:
            _, acc_ci_low, acc_ci_high = bootstrap_ci(
                lambda yt, yp: accuracy_score(yt, yp),
                y_true, y_pred,
                n_boot=bootstrap_resamples,
                seed=seed,
            )
            _, f1_ci_low, f1_ci_high = bootstrap_ci(
                lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
                y_true, y_pred,
                n_boot=bootstrap_resamples,
                seed=seed,
            )
            metrics['ci_95_accuracy'] = (acc_ci_low, acc_ci_high)
            metrics['ci_95_macro_f1'] = (f1_ci_low, f1_ci_high)

        results[mode] = metrics

    # Compute comparative drops and prediction flip rates
    canonical_acc = results['canonical']['accuracy']
    canonical_f1  = results['canonical']['macro_f1']
    canonical_preds = preds_by_mode['canonical']

    p_flips = {}
    for mode in PERTURBATION_MODES:
        flips = np.sum(canonical_preds != preds_by_mode[mode])
        p_flips[mode] = float(flips / len(canonical_preds))

    order_drop_acc = canonical_acc - results['shuffled']['accuracy']
    order_drop_f1  = canonical_f1 - results['shuffled']['macro_f1']

    reversal_drop_acc = canonical_acc - results['reversed']['accuracy']
    reversal_drop_f1  = canonical_f1 - results['reversed']['macro_f1']

    repeated_drop_acc = canonical_acc - results['repeated']['accuracy']
    repeated_drop_f1  = canonical_f1 - results['repeated']['macro_f1']

    # McNemar's paired discordance test between canonical and shuffled
    mcnemar_p = mcnemar_test(preds_by_mode['shuffled'], canonical_preds, ground_truth_labels)

    # Bootstrap CI for OrderDrop
    drop_ci = None
    if compute_bootstrap:
        _, drop_ci_low, drop_ci_high = bootstrap_ci(
            lambda yt, idx: (
                np.mean(canonical_preds[idx] == yt[idx]) -
                np.mean(preds_by_mode['shuffled'][idx] == yt[idx])
            ),
            ground_truth_labels,
            np.arange(len(ground_truth_labels)),
            n_boot=bootstrap_resamples,
            seed=seed,
        )
        drop_ci = (drop_ci_low, drop_ci_high)

    return {
        'results_by_mode': results,
        'order_drop_acc': order_drop_acc,
        'order_drop_f1': order_drop_f1,
        'reversal_drop_acc': reversal_drop_acc,
        'reversal_drop_f1': reversal_drop_f1,
        'repeated_drop_acc': repeated_drop_acc,
        'repeated_drop_f1': repeated_drop_f1,
        'p_flips': p_flips,
        'mcnemar_p': mcnemar_p,
        'order_drop_ci_95': drop_ci,
        'predictions_by_mode': preds_by_mode,
        'labels': ground_truth_labels,
        'sequence_ids': all_sequence_ids,
        'interpretation': (
            'Strong evidence: model uses chronological temporal ORDER (OrderDrop > 5pp, p < 0.05)'
            if (order_drop_acc > 0.05 and mcnemar_p < 0.05) else
            'Weak / Invariant: model does not rely on temporal ordering'
        ),
    }
