"""
Experiment 5: Temporal Sparsity (H4)
(research.md §20)

Tests WHEN information becomes available by comparing:
  dense  — all frames (e.g., F1 F2 F3 F4 F5 F6 F7 F8)
  sparse — stride-2  (e.g., F1 F3 F5 F7)
  early  — first half (e.g., F1 F2 F3 F4)
  late   — last half  (e.g., F5 F6 F7 F8)

Answers: "When is the discriminative information available in the sequence?"
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

from src.evaluation.metrics import compute_metrics


def select_frames_by_mode(embeddings: torch.Tensor, mode: str, n: int) -> torch.Tensor:
    """Select n frames from sequence using named strategy."""
    B, T, D = embeddings.shape

    if mode == 'dense':
        idx = torch.linspace(0, T - 1, min(n, T)).long()
    elif mode == 'sparse':
        stride = max(1, T // n)
        idx = torch.arange(0, T, stride)[:n]
    elif mode == 'early':
        idx = torch.arange(min(n, T))
    elif mode == 'late':
        start = max(0, T - n)
        idx = torch.arange(start, T)
    else:
        raise ValueError(f"Unknown sparsity mode: {mode}")

    # Pad if shorter than n
    selected = embeddings[:, idx, :]
    if selected.shape[1] < n:
        pad = selected[:, -1:, :].expand(-1, n - selected.shape[1], -1)
        selected = torch.cat([selected, pad], dim=1)

    return selected


def run_sparsity_experiment(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    n_frames: int = 4,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Compare dense/sparse/early/late frame selection strategies.
    """
    model.eval()
    results = {}

    for mode in ['dense', 'sparse', 'early', 'late']:
        all_preds, all_labels = [], []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for embeddings, labels in loader:
                embeddings = embeddings.to(device)
                selected = select_frames_by_mode(embeddings, mode, n_frames)
                logits = model(selected)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        results[mode] = metrics
        print(f"  {mode:8s} | Acc={metrics['accuracy']:.3f} | Macro-F1={metrics['macro_f1']:.3f}")

    # Interpretation
    early_acc = results['early']['accuracy']
    late_acc  = results['late']['accuracy']
    if late_acc > early_acc + 0.03:
        interpretation = "Information accumulates over time (late frames more discriminative)"
    elif early_acc > late_acc + 0.03:
        interpretation = "Early frames are more discriminative (label determined quickly)"
    else:
        interpretation = "Information is distributed across the sequence"

    print(f"\n  → {interpretation}")
    return {'results_by_mode': results, 'interpretation': interpretation}
