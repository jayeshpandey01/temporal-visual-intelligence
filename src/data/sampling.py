"""
M2.2 / M2.3: Frame Sampling Strategies for Temporal Sequence Datasets.

Defines deterministic and stochastic temporal sampling operators across sequence horizons.
Ensures identical visual observation subsets across baseline, mean-pooling, and sequential models.

Design Principles (docs/milestone/M-2.md §M2.2, §M2.3):
  - Initial Strategy: Uniform sampling via floor formula:
        i_k = floor(k * (L - 1) / (N - 1)) for k in 0 .. N-1
  - Deterministic Evaluation Rule: Validation and Test splits must NEVER use stochastic sampling.
  - Avoids systematically favoring sequence onset or termination.
"""

from __future__ import annotations

import math
from typing import List, Callable, Dict, Optional, Union
import numpy as np


def uniform_sample(total_frames: int, target_frames: int) -> List[int]:
    """
    Evenly spaced frames across the full sequence horizon using the M2.2 floor equation:
        i_k = floor(k * (L - 1) / (N - 1))  for k in {0, ..., N-1}

    Guarantees:
        - Covers both endpoints: i_0 = 0 and i_{N-1} = L - 1.
        - Preserves natural chronological order: i_0 < i_1 < ... < i_{N-1}.
        - Avoids onset/offset bias.
        - Strictly deterministic across evaluation splits.

    Examples:
        L=12, N=4 -> [0, 3, 7, 11] (Corresponding to F1, F4, F8, F12)
        L=10, N=5 -> [0, 2, 4, 6, 8] (Corresponding to F1, F3, F5, F7, F9)
        L=16, N=8 -> [0, 2, 4, 6, 8, 10, 12, 15]

    Args:
        total_frames (L): Total number of frames available in the source sequence.
        target_frames (N): Desired number of sampled observations.

    Returns:
        List of integer frame indices (0-indexed).
    """
    if total_frames <= 0 or target_frames <= 0:
        return []

    if target_frames == 1:
        # Single-frame anchor: choose sequence midpoint
        return [total_frames // 2]

    if total_frames <= target_frames:
        return list(range(total_frames))

    L = total_frames
    N = target_frames

    indices = [int(math.floor(k * (L - 1) / (N - 1))) for k in range(N)]
    return indices


def first_n(total_frames: int, target_frames: int) -> List[int]:
    """
    Take the first N frames: [0, 1, ..., N-1].
    Used to test early observation commitment and prefix ambiguity (Hypothesis H1).
    """
    if total_frames <= 0 or target_frames <= 0:
        return []
    return list(range(min(target_frames, total_frames)))


def last_n(total_frames: int, target_frames: int) -> List[int]:
    """
    Take the last N frames: [L-N, ..., L-1].
    Used to test late-stage outcome bias and terminal state recognition.
    """
    if total_frames <= 0 or target_frames <= 0:
        return []
    start = max(0, total_frames - target_frames)
    return list(range(start, total_frames))


def sparse_sample(total_frames: int, target_frames: int) -> List[int]:
    """
    Stride-based sampling across the timeline: i_k = k * stride.
    Creates temporal gaps; useful for testing Nyquist-Shannon aliasing limits.
    """
    if total_frames <= 0 or target_frames <= 0:
        return []
    stride = max(1, total_frames // target_frames)
    indices = list(range(0, total_frames, stride))[:target_frames]
    return indices


def segment_sample(
    total_frames: int,
    target_frames: int,
    mode: str = "center",
    seed: Optional[int] = None
) -> List[int]:
    """
    Temporal Segment Network (TSN; Wang et al., ECCV 2016) sampling.
    Divides sequence into N equal duration segments:
        segment_k = [k * L / N, (k + 1) * L / N)

    Args:
        total_frames (L): Available frames.
        target_frames (N): Target observations.
        mode: 'center' (deterministic midpoint) | 'random' (train-time augmentation only).
        seed: Optional random seed for reproducible stochastic jitter.

    Returns:
        List of integer frame indices.
    """
    if total_frames <= 0 or target_frames <= 0:
        return []

    if total_frames <= target_frames:
        return list(range(total_frames))

    L = total_frames
    N = target_frames
    ticks = np.linspace(0, L, N + 1)

    rng = np.random.RandomState(seed) if seed is not None else np.random

    indices = []
    for k in range(N):
        start = ticks[k]
        end = ticks[k + 1]
        if mode == "center":
            idx = int(math.floor((start + end) / 2.0))
        elif mode == "random":
            low = int(math.floor(start))
            high = max(low + 1, int(math.ceil(end)))
            idx = int(rng.randint(low, high))
        else:
            raise ValueError(f"Unknown segment mode: {mode}. Choose 'center' or 'random'.")
        indices.append(min(L - 1, max(0, idx)))

    return indices


def random_sample(total_frames: int, target_frames: int, seed: Optional[int] = 42) -> List[int]:
    """
    Random subset sampling without replacement.
    IMPORTANT (M2.3 Rule): Strictly restricted to training data augmentation.
    NEVER use random_sample for validation or test split evaluation.
    """
    if total_frames <= 0 or target_frames <= 0:
        return []
    rng = np.random.RandomState(seed) if seed is not None else np.random
    k = min(target_frames, total_frames)
    indices = sorted(rng.choice(total_frames, k, replace=False).tolist())
    return [int(x) for x in indices]


def pad_or_truncate(
    indices: List[int],
    target_frames: int,
    pad_mode: str = "edge"
) -> List[int]:
    """
    Ensure the sampled index list has exact length `target_frames`.
    If len(indices) < target_frames, pad using the specified mode:
        'edge': replicate the last frame index.
        'zero': pad with zeros.
    """
    if len(indices) == target_frames:
        return indices
    if len(indices) > target_frames:
        return indices[:target_frames]

    deficit = target_frames - len(indices)
    if pad_mode == "edge":
        last_val = indices[-1] if indices else 0
        return indices + [last_val] * deficit
    elif pad_mode == "zero":
        return indices + [0] * deficit
    else:
        raise ValueError(f"Unknown pad_mode: {pad_mode}")


def dense_sample(total_frames: int, target_frames: int) -> List[int]:
    """
    Take the first N frames: [0, 1, ..., N-1] (dense prefix window).
    Alias for first_n.
    """
    return first_n(total_frames, target_frames)


def jittered_sample(
    total_frames: int,
    target_frames: int,
    jitter: int = 1,
    seed: Optional[int] = 42,
) -> List[int]:
    """
    Uniform sampling with ±jitter frame displacement per anchor.
    Simulates realistic frame-timestamp imprecision while preserving monotonicity.
    """
    if total_frames <= 0 or target_frames <= 0:
        return []
    base = uniform_sample(total_frames, target_frames)
    rng = np.random.RandomState(seed) if seed is not None else np.random
    jittered = []
    for idx in base:
        disp = int(rng.randint(-jitter, jitter + 1))
        new_idx = max(0, min(total_frames - 1, idx + disp))
        jittered.append(new_idx)
    # Sort to preserve chronological monotonicity
    return sorted(jittered)


SAMPLING_STRATEGIES: Dict[str, Callable[[int, int], List[int]]] = {
    "uniform": uniform_sample,
    "dense": dense_sample,
    "first_n": first_n,
    "last_n": last_n,
    "sparse": sparse_sample,
    "jittered": jittered_sample,
    "segment_center": lambda L, N: segment_sample(L, N, mode="center"),
    "random": random_sample,
}


def get_sampler(name: str) -> Callable[[int, int], List[int]]:
    """Get sampling operator by name from configuration."""
    if name not in SAMPLING_STRATEGIES:
        raise ValueError(
            f"Unknown sampling strategy: '{name}'. Available: {list(SAMPLING_STRATEGIES.keys())}"
        )
    return SAMPLING_STRATEGIES[name]
