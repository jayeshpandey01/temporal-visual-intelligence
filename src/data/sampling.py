"""
Frame sampling strategies for temporal sequence datasets.
Each strategy returns a list of frame indices given total_frames and target T.
"""

import numpy as np
from typing import List


def uniform_sample(total_frames: int, target_frames: int) -> List[int]:
    """Evenly spaced frames across the full sequence. Best for general use."""
    if total_frames <= target_frames:
        return list(range(total_frames))
    return list(np.linspace(0, total_frames - 1, target_frames, dtype=int))


def first_n(total_frames: int, target_frames: int) -> List[int]:
    """Take the first N frames. Best for early-observation experiments."""
    return list(range(min(target_frames, total_frames)))


def last_n(total_frames: int, target_frames: int) -> List[int]:
    """Take the last N frames. Tests late-observation bias."""
    start = max(0, total_frames - target_frames)
    return list(range(start, total_frames))


def sparse_sample(total_frames: int, target_frames: int) -> List[int]:
    """Stride-based sampling. Creates temporal gaps."""
    stride = max(1, total_frames // target_frames)
    indices = list(range(0, total_frames, stride))[:target_frames]
    return indices


def random_sample(total_frames: int, target_frames: int, seed: int = 42) -> List[int]:
    """Random sampling (for training augmentation only — never for evaluation)."""
    rng = np.random.RandomState(seed)
    indices = sorted(rng.choice(total_frames, min(target_frames, total_frames), replace=False))
    return list(indices)


SAMPLING_STRATEGIES = {
    'uniform': uniform_sample,
    'first_n': first_n,
    'last_n':  last_n,
    'sparse':  sparse_sample,
    'random':  random_sample,
}


def get_sampler(name: str):
    """Get sampling function by name from config."""
    if name not in SAMPLING_STRATEGIES:
        raise ValueError(f"Unknown sampling strategy: {name}. Choose from {list(SAMPLING_STRATEGIES.keys())}")
    return SAMPLING_STRATEGIES[name]
