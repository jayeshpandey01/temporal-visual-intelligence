"""
Experiment 3: Corruption Robustness (H3)
(research.md §17-18)

Tests whether temporal context helps recover from corrupted/occluded observations.

Corruption modes (applied to embedding space):
  blur_25pct  — zero out 25% of frames (simulates blur/noise in embedding space)
  mask_25pct  — replace with zero vector (hard mask)
  drop_25pct  — replace with mean embedding of sequence

Key metric:
  RobustnessDrop = Acc_clean - Acc_corrupted (per model)
  Temporal model should have smaller drop (H3).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional

from src.evaluation.metrics import compute_metrics


def apply_corruption(
    embeddings: torch.Tensor,
    mode: str,
    fraction: float = 0.25,
    seed: Optional[int] = 42,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply frame-level corruption to embedding sequence.

    Args:
        embeddings: [B, T, D]
        mode: 'zero_mask' | 'gaussian_noise' | 'mean_replace'
        fraction: fraction of frames to corrupt
        seed: for reproducibility (used if generator is None)
        generator: persistent torch.Generator
    """
    B, T, D = embeddings.shape
    if generator is None:
        generator = torch.Generator(device='cpu')
        if seed is not None:
            generator.manual_seed(seed)
    result = embeddings.clone()

    n_corrupt = max(1, int(T * fraction))

    for b in range(B):
        corrupt_idx = torch.randperm(T, generator=generator)[:n_corrupt].to(embeddings.device)

        if mode == 'zero_mask':
            result[b, corrupt_idx, :] = 0.0

        elif mode == 'gaussian_noise':
            noise_scale = embeddings[b].norm(dim=-1).mean() * 0.5
            noise = torch.randn(n_corrupt, D, generator=generator) * noise_scale
            result[b, corrupt_idx, :] = result[b, corrupt_idx, :] + noise.to(embeddings.device)

        elif mode == 'mean_replace':
            seq_mean = embeddings[b].mean(dim=0)
            result[b, corrupt_idx, :] = seq_mean

    return result


CORRUPTION_MODES = ['zero_mask', 'gaussian_noise', 'mean_replace']


def run_corruption_experiment(
    baseline_model: torch.nn.Module,
    temporal_model: torch.nn.Module,
    dataset,
    device: torch.device,
    corruption_fraction: float = 0.25,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compare baseline and temporal model robustness under frame corruption.

    Expected: temporal model has smaller RobustnessDrop (H3).
    """
    results = {}

    for model_name, model in [('baseline', baseline_model), ('temporal', temporal_model)]:
        model.eval()
        results[model_name] = {}

        # Clean performance
        all_preds, all_labels = [], []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for embeddings, labels in loader:
                embeddings = embeddings.to(device)
                logits = model(embeddings)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        clean_metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        results[model_name]['clean'] = clean_metrics
        print(f"\n  {model_name:10s} | Clean: Acc={clean_metrics['accuracy']:.3f}")

        # Corrupted performance
        for corrupt_mode in CORRUPTION_MODES:
            all_preds, all_labels = [], []
            # Persistent generator per corruption mode to preserve entropy across batches
            gen = torch.Generator(device='cpu')
            gen.manual_seed(seed)

            with torch.no_grad():
                for embeddings, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                    embeddings = embeddings.to(device)
                    corrupted = apply_corruption(
                        embeddings, corrupt_mode,
                        fraction=corruption_fraction,
                        generator=gen
                    )
                    logits = model(corrupted)
                    preds = logits.argmax(dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            corrupt_metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
            robustness_drop = clean_metrics['accuracy'] - corrupt_metrics['accuracy']
            results[model_name][corrupt_mode] = {
                **corrupt_metrics,
                'robustness_drop': robustness_drop,
            }
            print(f"  {model_name:10s} | {corrupt_mode:15s}: "
                  f"Acc={corrupt_metrics['accuracy']:.3f} | Drop={robustness_drop:+.3f}")

    # Summary comparison
    print("\n  Summary (RobustnessDrop comparison):")
    print(f"  {'Corruption':20s} | {'Baseline Drop':15s} | {'Temporal Drop':15s} | {'Improvement':10s}")
    print(f"  {'-'*65}")
    for mode in CORRUPTION_MODES:
        b_drop = results['baseline'][mode]['robustness_drop']
        t_drop = results['temporal'][mode]['robustness_drop']
        improvement = b_drop - t_drop
        print(f"  {mode:20s} | {b_drop:+13.3f}  | {t_drop:+13.3f}  | {improvement:+8.3f}")

    return results
