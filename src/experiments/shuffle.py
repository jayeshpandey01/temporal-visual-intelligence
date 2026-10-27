"""
Experiment 2: Temporal Order Test (H2)
(research.md §15-16)

Three conditions:
  canonical  — correct temporal order (control)
  shuffled   — same frames, random order (destroys temporal relationships)
  repeated   — same frame N times (key control: separates "more positions" from "temporal info")

Key metrics:
  OrderDrop = Acc_canonical - Acc_shuffled
  Repeated should ≈ single-frame baseline

If OrderDrop is large → model exploits temporal ORDER, not just more visual info.
If repeated ≈ baseline → model isn't just benefiting from more input positions.
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional

from src.evaluation.metrics import compute_metrics


PERTURBATION_MODES = ['canonical', 'shuffled', 'repeated', 'reversed']


def apply_order_perturbation(
    embeddings: torch.Tensor,
    mode: str,
    seed: Optional[int] = 42,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply temporal order perturbation to a batch of sequences.

    Args:
        embeddings: [B, T, D]
        mode: 'canonical' | 'shuffled' | 'repeated' | 'reversed'
        seed: random seed for shuffled mode (used if generator is None)
        generator: persistent torch.Generator to prevent per-batch reset
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
        # Repeat middle frame — same frame N times
        # THIS IS THE KEY SENIOR-LEVEL CONTROL (research.md §16)
        # Expected: repeated ≈ single-frame baseline (no new temporal info)
        mid = T // 2
        mid_frame = embeddings[:, mid:mid+1, :].expand(-1, T, -1)
        return mid_frame.clone()

    elif mode == 'reversed':
        return embeddings.flip(dims=[1])

    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")


def run_order_experiment(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 32,
    perturbation_seed: int = 42,
) -> Dict[str, Any]:
    """
    Run all order perturbation conditions and compute comparative metrics.

    Returns dict with results per condition and computed gaps.
    """
    model.eval()
    results = {}

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
                    generator=gen
                )
                logits = model(perturbed)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        results[mode] = metrics
        print(f"  {mode:12s} | Acc={metrics['accuracy']:.3f} | Macro-F1={metrics['macro_f1']:.3f}")

    # Key metrics
    canonical_acc = results['canonical']['accuracy']
    shuffled_acc  = results['shuffled']['accuracy']
    repeated_acc  = results.get('repeated', {}).get('accuracy', None)

    order_drop = canonical_acc - shuffled_acc
    repeated_gap = (canonical_acc - repeated_acc) if repeated_acc is not None else None

    print(f"\n  OrderDrop (canonical - shuffled): {order_drop:+.3f}")
    if repeated_gap is not None:
        print(f"  Repeated gap (canonical - repeated): {repeated_gap:+.3f}")
        print(f"  → {'Temporal ORDER matters (not just more frames)' if order_drop > 0.05 else 'Order has limited impact'}")

    return {
        'results_by_mode': results,
        'order_drop': order_drop,
        'repeated_gap': repeated_gap,
        'interpretation': (
            'Strong evidence: model uses temporal ORDER (OrderDrop > 5pp)'
            if order_drop > 0.05 else
            'Weak evidence: model may not exploit temporal order'
        ),
    }
