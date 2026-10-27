"""
Experiment 1: Progressive Observation Sweep (H1)
(research.md §13)

Tests whether more temporal frames → better performance.
Computes ΔN = Performance_N - Performance_1 (multi-frame gain).

Key result: information curve (accuracy vs. num_frames)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any

from src.evaluation.metrics import compute_metrics


def run_progressive_sweep(
    model: torch.nn.Module,
    dataset,
    frame_counts: List[int],
    device: torch.device,
    batch_size: int = 32,
    sampling_mode: str = 'prefix',  # 'prefix' (online early decision) or 'uniform' (duration coverage)
) -> Dict[str, Any]:
    """
    Evaluate model using N frames for each N in frame_counts.

    Supports two distinct scientific protocols:
      1. 'prefix': Uses first N frames (online decision readiness).
      2. 'uniform': Subsamples N frames evenly spaced across total duration [0, T-1].
    """
    model.eval()
    results = {}
    baseline_acc = None

    for n_frames in frame_counts:
        all_preds = []
        all_labels = []

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for embeddings, labels in loader:
                B, T, D = embeddings.shape
                if sampling_mode == 'prefix':
                    sampled = embeddings[:, :n_frames, :]
                elif sampling_mode == 'uniform':
                    idx = torch.linspace(0, T - 1, n_frames).long()
                    sampled = embeddings[:, idx, :]
                else:
                    raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

                sampled = sampled.to(device)
                labels = labels.to(device)

                logits = model(sampled)
                preds = logits.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
        )
        results[n_frames] = metrics

        if baseline_acc is None:
            baseline_acc = metrics['accuracy']

        delta = metrics['accuracy'] - baseline_acc
        print(f"  {n_frames:2d} frames | Acc={metrics['accuracy']:.3f} | "
              f"Macro-F1={metrics['macro_f1']:.3f} | ΔN={delta:+.3f}")

    # Compute saturation point (where gain < 0.5pp)
    accs = [results[n]['accuracy'] for n in frame_counts]
    gains = [a - accs[0] for a in accs]
    saturation_idx = next(
        (i for i in range(1, len(gains)) if gains[i] - gains[i-1] < 0.005),
        len(gains) - 1
    )

    return {
        'results_by_frame_count': results,
        'frame_counts': frame_counts,
        'accuracies': accs,
        'gains': gains,
        'saturation_at_n_frames': frame_counts[saturation_idx],
        'max_gain': max(gains),
    }
