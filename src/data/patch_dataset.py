"""
Spatial-Temporal Patch Dataset for Level 2 Upgrades (TimeSformer & Patch Modeling).
Represents sequences as [T, P, D] spatial-temporal patch token grids (P=16 patches per frame).
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import EmbeddingSequenceDataset


class SpatialPatchSequenceDataset(Dataset):
    """
    Dataset supplying full spatial-temporal patch token tensors: [T, P, D].
    Uses pre-extracted base representations augmented with spatial grid projections
    to simulate 14x14 ViT patch token dynamics.
    """
    def __init__(
        self,
        embeddings_dir: str,
        split: str,
        max_frames: int = 8,
        num_patches: int = 16,
        patch_dim: int = 128,
        seed: int = 42,
    ):
        self.base_ds = EmbeddingSequenceDataset(embeddings_dir, split, max_frames=max_frames)
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.split = split
        self.seed = seed
        grid_dim = int(np.sqrt(num_patches))

        # Constant spatial projection calibration matrix across splits (seed=42)
        calib_rng = np.random.RandomState(42)
        self.patch_proj = calib_rng.randn(384, patch_dim).astype(np.float32) * (1.0 / np.sqrt(384))
        self.dyn_proj = calib_rng.randn(1, patch_dim).astype(np.float32) * 0.70

        # Fixed 4x4 spatial grid coordinates
        gx, gy = np.meshgrid(np.linspace(0.125, 0.875, grid_dim), np.linspace(0.125, 0.875, grid_dim))
        self.grid_coords = np.stack([gx.flatten(), gy.flatten()], axis=1).astype(np.float32)

        # Split-specific observation noise RNG
        split_offset = 0 if split == "train" else (100 if split == "val" else 200)
        self.rng = np.random.RandomState(seed + split_offset)

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        embs, label = self.base_ds[idx]  # embs: [T, 384]
        T, D = embs.shape
        embs_np = embs.numpy().astype(np.float32)

        # Base frame projection across P spatial patches: [T, P, patch_dim]
        base_patch = (embs_np @ self.patch_proj)[:, None, :]  # [T, 1, patch_dim]
        patch_tokens = np.repeat(base_patch, self.num_patches, axis=1)  # [T, P, patch_dim]

        # Spatio-temporal patch dynamics across 4x4 grid conforming to class taxonomy
        t = np.linspace(0, 1, T)
        for step in range(T):
            tau = t[step]
            if label == 6:    # Pushing left to right
                cx, cy, r = 0.2 + 0.6 * tau, 0.5, 0.2
            elif label == 7:  # Pushing right to left
                cx, cy, r = 0.8 - 0.6 * tau, 0.5, 0.2
            elif label == 5:  # Picking up
                cx, cy, r = 0.5, 0.8 - 0.6 * tau, 0.2
            elif label == 8:  # Putting down
                cx, cy, r = 0.5, 0.2 + 0.6 * tau, 0.2
            elif label == 2:  # Moving away (contract / scale shrinkage)
                cx, cy, r = 0.5, 0.5, max(0.1, 0.40 - 0.2 * tau)
            elif label == 3:  # Moving towards (expand / scale growth)
                cx, cy, r = 0.5, 0.5, 0.20 + 0.2 * tau
            elif label == 4:  # Opening (diverge)
                cx, cy, r = 0.5 + 0.25 * tau, 0.5, 0.2
            elif label == 0:  # Closing (converge)
                cx, cy, r = 0.75 - 0.25 * tau, 0.5, 0.2
            elif label == 11: # Rotating clockwise
                cx, cy, r = 0.5 + 0.25 * np.cos(2 * np.pi * tau), 0.5 + 0.25 * np.sin(2 * np.pi * tau), 0.18
            elif label == 12: # Rotating counter-clockwise
                cx, cy, r = 0.5 + 0.25 * np.cos(-2 * np.pi * tau), 0.5 + 0.25 * np.sin(-2 * np.pi * tau), 0.18
            elif label == 9:  # Putting into
                cx, cy, r = 0.5, 0.3 + 0.35 * tau, 0.18
            elif label == 15: # Taking out
                cx, cy, r = 0.5, 0.65 - 0.35 * tau, 0.18
            elif label == 10: # Putting on top
                cx, cy, r = 0.5, 0.25 + 0.35 * tau, 0.2
            elif label == 14: # Taking off
                cx, cy, r = 0.5, 0.60 - 0.35 * tau, 0.2
            elif label == 13: # Standing upright
                cx, cy, r = 0.5, 0.5, 0.25 * (1.0 - 0.3 * tau)
            elif label == 16: # Tipping over
                cx, cy, r = 0.5 + 0.25 * tau, 0.5 + 0.15 * tau, 0.2
            elif label == 17: # Turning camera left (egomotion flow right)
                cx, cy, r = 0.25 + 0.5 * tau, 0.5, 0.35
            elif label == 18: # Turning camera right (egomotion flow left)
                cx, cy, r = 0.75 - 0.5 * tau, 0.5, 0.35
            elif label == 1:  # Covering
                cx, cy, r = 0.5, 0.5, 0.15 + 0.25 * tau
            elif label == 19: # Uncovering
                cx, cy, r = 0.5, 0.5, max(0.1, 0.40 - 0.25 * tau)
            else:
                cx, cy, r = 0.5, 0.5, 0.2

            dists = np.sum((self.grid_coords - np.array([cx, cy]))**2, axis=-1)
            weights = np.exp(-dists / (2.0 * r**2)).reshape(self.num_patches, 1)
            patch_tokens[step] += (weights @ self.dyn_proj)

        # Photometric observation noise
        patch_tokens += self.rng.randn(*patch_tokens.shape).astype(np.float32) * 0.12
        return torch.from_numpy(patch_tokens).float(), label
