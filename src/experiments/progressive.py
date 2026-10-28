"""
Milestone M4.2: Experiment A — Progressive Observation Sweep (H1 & H5).
Directly fulfills Take-Home Assessment M4.2:
  - Tests H1: Additional sequential observations improve prediction.
  - Tests H5: Information saturates beyond a critical threshold K*.
  - Enforces Single-Variable Intervention: Only observation count K varies;
    order is canonical, corruption is none, encoder and model checkpoints are frozen.
  - Controls temporal coverage: Uniform duration sampling avoids position confounding.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.metrics import compute_metrics, bootstrap_ci
from src.data.progressive import sample_progressive_frames
from src.experiments.controls import (
    SingleVariableIntervention,
    ExperimentalCondition,
)


def run_progressive_sweep(
    model: torch.nn.Module,
    dataset,
    frame_counts: List[int],
    device: torch.device,
    batch_size: int = 32,
    sampling_mode: str = 'uniform',
    compute_bootstrap: bool = True,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate model across progressive observation counts K in frame_counts.

    Supports:
      1. Uniform duration coverage: Evenly spaces K observations across duration [0, T-1].
      2. Single-variable validation: Enforces that each condition differs from canonical baseline
         ONLY in frame_count.
      3. Non-parametric 95% bootstrap confidence intervals for Macro-F1 and Accuracy.
      4. Information saturation calculation (H5 threshold K*).
      5. Sample-level prediction collection.
    """
    model.eval()
    results = {}
    predictions_by_k = {}
    all_sequence_ids = []

    # Get sequence IDs if available from dataset
    if hasattr(dataset, 'get_all_sequence_ids'):
        all_sequence_ids = dataset.get_all_sequence_ids()
    elif hasattr(dataset, 'samples'):
        all_sequence_ids = [s[0] if isinstance(s, (list, tuple)) else str(i)
                            for i, s in enumerate(dataset.samples)]

    # Validate single-variable intervention rule if model_name is provided
    if model_name:
        try:
            intervention_engine = SingleVariableIntervention()
            for k in frame_counts:
                cond = ExperimentalCondition(
                    model_name=model_name,
                    seed=seed,
                    frame_count=k,
                    order="canonical",
                    corruption="none",
                    corruption_fraction=0.0,
                )
                intervention_engine.validate_intervention(cond)
        except Exception as e:
            # Re-raise to enforce research standards
            raise e

    ground_truth_labels = None

    for n_frames in frame_counts:
        all_preds = []
        all_labels = []

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for embeddings, labels in loader:
                sampled = sample_progressive_frames(embeddings, k=n_frames, mode=sampling_mode)
                sampled = sampled.to(device)
                labels = labels.to(device)

                logits = model(sampled)
                preds = logits.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        if ground_truth_labels is None:
            ground_truth_labels = y_true

        metrics = compute_metrics(y_true, y_pred)
        predictions_by_k[n_frames] = y_pred

        # 95% Non-Parametric Bootstrap Confidence Intervals (B=1000)
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

        results[n_frames] = metrics

        delta_acc = metrics['accuracy'] - results[frame_counts[0]]['accuracy']
        delta_f1 = metrics['macro_f1'] - results[frame_counts[0]]['macro_f1']
        ci_f1_str = (
            f" [95% CI: {metrics['ci_95_macro_f1'][0]:.3f}, {metrics['ci_95_macro_f1'][1]:.3f}]"
            if compute_bootstrap else ""
        )
        print(f"  K={n_frames:2d} frames | Acc={metrics['accuracy']:.3f} (Δ={delta_acc:+.3f}) | "
              f"Macro-F1={metrics['macro_f1']:.3f}{ci_f1_str} (Δ={delta_f1:+.3f})")

    accs = [results[n]['accuracy'] for n in frame_counts]
    f1s = [results[n]['macro_f1'] for n in frame_counts]
    gains_acc = [a - accs[0] for a in accs]
    gains_f1 = [f - f1s[0] for f in f1s]

    # Compute saturation threshold K* (H5: where subsequent marginal gain < 0.5 pp / 0.005)
    saturation_idx = 0
    for i in range(1, len(gains_f1)):
        marginal_gain = f1s[i] - f1s[i - 1]
        if marginal_gain < 0.005:
            saturation_idx = i - 1
            break
        saturation_idx = i

    saturation_k = frame_counts[saturation_idx]

    return {
        'results_by_frame_count': results,
        'frame_counts': frame_counts,
        'accuracies': accs,
        'macro_f1s': f1s,
        'gains_acc': gains_acc,
        'gains_f1': gains_f1,
        'saturation_at_n_frames': saturation_k,
        'max_gain_acc': gains_acc[-1],
        'max_gain_f1': gains_f1[-1],
        'predictions_by_k': predictions_by_k,
        'labels': ground_truth_labels,
        'sequence_ids': all_sequence_ids,
    }
