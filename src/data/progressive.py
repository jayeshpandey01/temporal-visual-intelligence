"""
Progressive Observation Interface for Temporal Sequence Modeling.
Fulfills Milestone M3.18, preparing M4 (Controlled Experiments & Hypothesis H1).

Enforces the architectural principle:
"We should design M3 so that M4 doesn't require rewriting the models.
The temporal model should accept: N = 1, N = 2, N = 4, N = 8, ...
This allows M4 to measure Performance(K) where K is the number of observations."
"""

import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from src.data.sampling import uniform_sample, first_n, last_n, sparse_sample


def get_progressive_frame_indices(
    total_frames: int,
    target_frames: int,
    mode: str = "uniform",
) -> List[int]:
    """
    Calculate deterministic frame index mapping for an arbitrary observation horizon K.

    Args:
        total_frames (T): Total number of frames in the raw / source sequence.
        target_frames (K): Desired number of sampled observations.
        mode: Sampling strategy ('uniform', 'prefix', 'first_n', 'last_n', 'suffix', 'sparse').

    Returns:
        List of integer indices in chronological order.
    """
    if total_frames <= 0 or target_frames <= 0:
        return []

    if target_frames == 1:
        # Canonical single-frame midpoint anchor (M3.2 / M3.3)
        return [total_frames // 2]

    if target_frames >= total_frames:
        return list(range(total_frames))

    mode_lower = mode.lower()
    if mode_lower == "uniform":
        return uniform_sample(total_frames, target_frames)
    elif mode_lower in ("prefix", "first_n"):
        return first_n(total_frames, target_frames)
    elif mode_lower in ("last_n", "suffix"):
        return last_n(total_frames, target_frames)
    elif mode_lower == "sparse":
        return sparse_sample(total_frames, target_frames)
    else:
        raise ValueError(
            f"Unknown progressive sampling mode: '{mode}'. "
            f"Supported: 'uniform', 'prefix', 'first_n', 'last_n', 'suffix', 'sparse'"
        )


def sample_progressive_frames(
    x: Union[torch.Tensor, np.ndarray],
    k: int,
    mode: str = "uniform",
) -> Union[torch.Tensor, np.ndarray]:
    """
    Subsample K progressive observations from a sequence tensor or array.

    Supports:
        - 3D PyTorch Tensor: [B, T, D] -> [B, K, D]
        - 2D PyTorch Tensor: [T, D] -> [K, D]
        - 3D NumPy array:    [B, T, D] -> [B, K, D]
        - 2D NumPy array:    [T, D] -> [K, D]

    Preserves autograd gradient tracking for torch.Tensor.

    Args:
        x: Input representations with temporal axis at dim 1 (if 3D) or dim 0 (if 2D).
        k: Target observation count.
        mode: Sampling strategy ('uniform', 'prefix', 'last_n', etc.).

    Returns:
        Subsampled representation with temporal dimension equal to K.
    """
    is_torch = isinstance(x, torch.Tensor)
    if not (is_torch or isinstance(x, np.ndarray)):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")

    ndim = x.ndim
    if ndim not in (2, 3):
        raise ValueError(f"Expected 2D [T, D] or 3D [B, T, D] input, got {ndim}D shape {x.shape}")

    total_frames = x.shape[1] if ndim == 3 else x.shape[0]
    indices = get_progressive_frame_indices(total_frames, k, mode=mode)

    if is_torch:
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=x.device)
        if ndim == 3:
            return x[:, idx_tensor, :]
        else:
            return x[idx_tensor, :]
    else:
        if ndim == 3:
            return x[:, indices, :]
        else:
            return x[indices, :]


class ProgressiveObservationSampler(nn.Module):
    """
    PyTorch module wrapper for the Progressive Observation Interface (M3.18).
    Can be composed in sequential models or used dynamically in evaluation loops.
    """

    def __init__(self, k: int = 8, mode: str = "uniform"):
        super().__init__()
        self.k = k
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,
        k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass applying progressive sampling along temporal dimension.

        Args:
            x: [B, T, D] or [T, D] tensor.
            k: Optional override for target frames count.
            mode: Optional override for sampling mode.

        Returns:
            Subsampled tensor with K observations.
        """
        target_k = k if k is not None else self.k
        target_mode = mode if mode is not None else self.mode
        return sample_progressive_frames(x, k=target_k, mode=target_mode)

    def extra_repr(self) -> str:
        return f"k={self.k}, mode={self.mode}"

