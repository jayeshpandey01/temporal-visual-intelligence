"""
Adaptive Temporal Sampling via Latent Velocity & Feature Drift (Level 2 Upgrade).

Instead of uniform static stride sampling, adaptive sampling tracks
the rate of visual/representational change across the video timeline:
    v_t = ||E_{t+1} - E_t||_2

Samples densely (high FPS) across the peak action/interaction window,
and sparsely (low FPS) across quiescent lead-in/lead-out phases.
"""

from typing import List, Optional, Union
import math
import numpy as np
import torch


def compute_latent_velocities(embeddings: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Computes inter-frame Euclidean velocity vector:
        v_t = ||E_{t+1} - E_t||_2  for t in {0, ..., L-2}
    """
    if isinstance(embeddings, torch.Tensor):
        embs = embeddings.detach().cpu().numpy()
    else:
        embs = np.array(embeddings)

    if len(embs) <= 1:
        return np.array([0.0])

    diffs = embs[1:] - embs[:-1]
    velocities = np.linalg.norm(diffs, axis=-1)
    return velocities


def adaptive_feature_drift_sample(
    embeddings: Union[np.ndarray, torch.Tensor],
    target_frames: int = 8,
    peak_ratio: float = 0.6,
) -> List[int]:
    """
    Allocates target_frames non-uniformly based on latent velocity profile.

    Args:
        embeddings: [L, D] sequence of feature representations or frame stats
        target_frames: desired number of sampled frame indices (N)
        peak_ratio: fraction of budget allocated to peak motion window (0.5 to 0.75)
    Returns:
        indices: List of N sorted unique integer frame indices covering [0, L-1]
    """
    L = len(embeddings)
    N = target_frames

    if L <= 0 or N <= 0:
        return []
    if L <= N:
        return list(range(L))
    if N == 1:
        return [L // 2]

    # Compute velocity profile
    velocities = compute_latent_velocities(embeddings)  # Length L - 1

    # Detect peak velocity center
    peak_center = int(np.argmax(velocities))  # Index in 0 .. L-2

    # Allocate budget: dense frames in peak window, sparse in outer tails
    n_dense = max(2, int(round(N * peak_ratio)))
    n_sparse = N - n_dense

    # Define peak window around center
    half_window = max(2, int(L * 0.2))
    w_start = max(0, peak_center - half_window)
    w_end = min(L - 1, peak_center + half_window)

    # Dense indices within window
    if (w_end - w_start + 1) >= n_dense:
        dense_indices = [
            int(math.floor(w_start + k * (w_end - w_start) / (n_dense - 1)))
            for k in range(n_dense)
        ]
    else:
        dense_indices = list(range(w_start, w_end + 1))

    # Sparse indices outside window (lead-in and lead-out)
    n_left = n_sparse // 2
    n_right = n_sparse - n_left

    left_indices = []
    if n_left > 0 and w_start > 0:
        left_indices = [
            int(math.floor(k * (w_start - 1) / max(1, n_left - 1))) if n_left > 1 else 0
            for k in range(n_left)
        ]

    right_indices = []
    if n_right > 0 and (L - 1 > w_end):
        right_indices = [
            int(math.floor((w_end + 1) + k * (L - 1 - (w_end + 1)) / max(1, n_right - 1)))
            if n_right > 1 else L - 1
            for k in range(n_right)
        ]

    all_indices = sorted(list(set(left_indices + dense_indices + right_indices)))

    # Ensure exact N indices
    if len(all_indices) < N:
        remaining = [i for i in range(L) if i not in all_indices]
        all_indices.extend(remaining[: N - len(all_indices)])
        all_indices = sorted(all_indices)
    elif len(all_indices) > N:
        # Uniformly subsample down to N preserving endpoints
        indices_down = [
            all_indices[int(math.floor(k * (len(all_indices) - 1) / (N - 1)))]
            for k in range(N)
        ]
        all_indices = sorted(list(set(indices_down)))
        while len(all_indices) < N:
            for i in range(L):
                if i not in all_indices:
                    all_indices.append(i)
                    if len(all_indices) == N:
                        break
            all_indices = sorted(all_indices)

    return all_indices

