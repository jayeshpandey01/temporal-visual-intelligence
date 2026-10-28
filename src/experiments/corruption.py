"""
Milestone M4.4: Experiment C — Corruption / Missing Observations Battery (H3 & H4).
Directly fulfills Take-Home Assessment M4.4:
  - Tests H4 / H3: Can temporal context make the model more robust when observations are unreliable?
  - Compares graceful degradation between:
      1. Model A: Single-Frame Baseline (103.7k)
      2. Model B: MultiFrameMeanPool Control (103.7k)
      3. Model D: CompactTemporalGRU (103.7k Parity)
      4. Model C: TemporalGRU (249.4k Primary Recurrent Head)
  - Corruption axes:
      1. Structured Frame Dropping (Zero-Masking): 0/8 (Clean), 2/8 (25%), 4/8 (50%)
      2. Latent Gaussian Noise Injection: σ ∈ {0.0, 0.1, 0.25, 0.50}
      3. Sequence Mean Replacement: 2/8 (25% neutral non-informative frames)
  - Enforces Single-Variable Intervention rules: only corruption parameters vary.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.metrics import compute_metrics, bootstrap_ci
from src.experiments.controls import (
    SingleVariableIntervention,
    ExperimentalCondition,
)


CORRUPTION_MODES = ['zero_mask', 'gaussian_noise', 'mean_replace']


def apply_corruption(
    embeddings: torch.Tensor,
    mode: str,
    fraction: float = 0.25,
    sigma: Optional[float] = None,
    seed: Optional[int] = 42,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply structured corruption or noise to a batch of sequences [B, T, D].

    Args:
        embeddings: [B, T, D] representation tensor
        mode: 'clean' | 'zero_mask' | 'gaussian_noise' | 'mean_replace'
        fraction: fraction of frames to corrupt (for zero_mask and mean_replace)
        sigma: noise standard deviation (for gaussian_noise)
        seed: random seed for reproducibility
        generator: persistent torch.Generator
    """
    if mode == 'clean' or (mode == 'zero_mask' and fraction <= 0.0) or (mode == 'gaussian_noise' and sigma == 0.0):
        return embeddings

    B, T, D = embeddings.shape
    if generator is None:
        generator = torch.Generator(device='cpu')
        if seed is not None:
            generator.manual_seed(seed)

    result = embeddings.clone()

    if mode == 'zero_mask':
        n_corrupt = max(1, int(round(T * fraction)))
        for b in range(B):
            corrupt_idx = torch.randperm(T, generator=generator)[:n_corrupt].to(embeddings.device)
            result[b, corrupt_idx, :] = 0.0
        return result

    elif mode == 'gaussian_noise':
        noise_std = sigma if sigma is not None else 0.25
        # Standard latent Gaussian noise injection
        noise = torch.randn(embeddings.shape, generator=generator).to(embeddings.device) * noise_std
        return result + noise

    elif mode == 'mean_replace':
        n_corrupt = max(1, int(round(T * fraction)))
        for b in range(B):
            corrupt_idx = torch.randperm(T, generator=generator)[:n_corrupt].to(embeddings.device)
            seq_mean = embeddings[b].mean(dim=0)
            result[b, corrupt_idx, :] = seq_mean
        return result

    else:
        raise ValueError(f"Unknown corruption mode: '{mode}'. Allowed: {CORRUPTION_MODES + ['clean']}")


def run_corruption_experiment(
    models: Dict[str, torch.nn.Module],
    dataset,
    device: torch.device,
    drop_fractions: Optional[List[float]] = None,
    noise_sigmas: Optional[List[float]] = None,
    batch_size: int = 32,
    seed: int = 42,
    compute_bootstrap: bool = True,
    bootstrap_resamples: int = 1000,
) -> Dict[str, Any]:
    """
    Evaluate multiple models under structured frame drops and latent Gaussian noise.

    Returns comprehensive results with clean vs corrupted metrics,
    robustness drops, bootstrap CIs, and sample-level predictions.
    """
    drop_fractions = drop_fractions or [0.0, 0.25, 0.50]
    noise_sigmas = noise_sigmas or [0.0, 0.1, 0.25, 0.50]

    all_sequence_ids = []
    if hasattr(dataset, 'get_all_sequence_ids'):
        all_sequence_ids = dataset.get_all_sequence_ids()
    elif hasattr(dataset, 'samples'):
        all_sequence_ids = [s[0] if isinstance(s, (list, tuple)) else str(i)
                            for i, s in enumerate(dataset.samples)]

    results: Dict[str, Any] = {
        'drops': {},
        'noise': {},
        'predictions': {},
        'sequence_ids': all_sequence_ids,
    }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for m_name, model in models.items():
        model.eval()
        results['drops'][m_name] = {}
        results['noise'][m_name] = {}
        results['predictions'][m_name] = {}

        # -------------------------------------------------------------
        # 1. Frame Dropping Sweep (Zero-Masking)
        # -------------------------------------------------------------
        for frac in drop_fractions:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(seed + 200)

            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch_x, batch_y in loader:
                    corrupt_x = apply_corruption(batch_x, mode='zero_mask', fraction=frac, generator=gen).to(device)
                    logits = model(corrupt_x)
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_y.numpy())

            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            met = compute_metrics(y_true, y_pred)

            if compute_bootstrap:
                _, acc_lo, acc_hi = bootstrap_ci(
                    lambda yt, yp: accuracy_score(yt, yp),
                    y_true, y_pred, n_boot=bootstrap_resamples, seed=seed
                )
                _, f1_lo, f1_hi = bootstrap_ci(
                    lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
                    y_true, y_pred, n_boot=bootstrap_resamples, seed=seed
                )
                met['ci_95_accuracy'] = (acc_lo, acc_hi)
                met['ci_95_macro_f1'] = (f1_lo, f1_hi)

            results['drops'][m_name][frac] = met
            results['predictions'][m_name][f'drop_{int(frac*8)}'] = y_pred

        # -------------------------------------------------------------
        # 2. Latent Gaussian Noise Sweep
        # -------------------------------------------------------------
        for sigma in noise_sigmas:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(seed + 300)

            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch_x, batch_y in loader:
                    noisy_x = apply_corruption(batch_x, mode='gaussian_noise', sigma=sigma, generator=gen).to(device)
                    logits = model(noisy_x)
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_y.numpy())

            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            met = compute_metrics(y_true, y_pred)

            if compute_bootstrap:
                _, acc_lo, acc_hi = bootstrap_ci(
                    lambda yt, yp: accuracy_score(yt, yp),
                    y_true, y_pred, n_boot=bootstrap_resamples, seed=seed
                )
                _, f1_lo, f1_hi = bootstrap_ci(
                    lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
                    y_true, y_pred, n_boot=bootstrap_resamples, seed=seed
                )
                met['ci_95_accuracy'] = (acc_lo, acc_hi)
                met['ci_95_macro_f1'] = (f1_lo, f1_hi)

            results['noise'][m_name][sigma] = met
            results['predictions'][m_name][f'noise_{sigma}'] = y_pred

    return results
