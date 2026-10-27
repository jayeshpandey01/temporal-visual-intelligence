"""
Temporal sequence dataset with video-level GroupKFold splits.

Design principle (research.md §5):
  Split at video level — ALL frames from a video go to ONE split only.
  Never split individual frames across train/test (causes leakage).
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import GroupKFold
import torchvision.transforms as T

from src.data.sampling import uniform_sample, first_n, last_n, sparse_sample, segment_sample
from src.data.transforms import build_dinov2_transform, IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# ImageNet normalization (required for DINOv2 / CLIP)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform(frame_size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """Build official DINOv2 torchvision transform with ImageNet normalization."""
    return build_dinov2_transform(crop_size=frame_size[0], direct_resize=True)


def verify_split_isolation(split_dict: Dict[str, List[str]]) -> bool:
    """
    Verify zero sequence/video overlap across train, val, and test splits (Leakage Guard).
    Kapoor & Narayanan (2023).
    """
    train_ids = set(split_dict.get('train', []))
    val_ids   = set(split_dict.get('val', []))
    test_ids  = set(split_dict.get('test', []))

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids

    if tv or tt or vt:
        print(f"  [LEAKAGE DETECTED] train-val={len(tv)}, train-test={len(tt)}, val-test={len(vt)}")
        return False
    return True


def create_and_save_split_manifest(
    root: str,
    config: Dict[str, Any],
    metadata_dir: str = 'data/metadata'
) -> Dict[str, List[str]]:
    """
    Deterministic Stratified Group-Aware split partitioner.
    Guarantees:
      1. Zero group leakage (video capture IDs never cross train/val/test).
      2. Strict class balance across splits (each class has identical test representation).
    Generates data/metadata/split_manifest.json and data/metadata/sequences.csv.
    """
    root_path = Path(root)
    meta_path = Path(metadata_dir)
    meta_path.mkdir(parents=True, exist_ok=True)

    seed = config["dataset"]["seed"]
    val_frac  = config["dataset"]["val_split"]
    test_frac = config["dataset"]["test_split"]

    # Discover benchmark action classes (exclude metadata and tier2 exploratory directories)
    class_dirs = sorted(
        [d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith(('metadata', 'tier2', 'real'))],
        key=lambda d: d.name
    )
    train_set, val_set, test_set = [], [], []
    video_records = []

    for class_idx, class_dir in enumerate(class_dirs):
        class_videos = []
        for video_dir in sorted(class_dir.iterdir(), key=lambda d: d.name):
            if not video_dir.is_dir():
                continue
            frames = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            if len(frames) >= 2:
                vid = video_dir.name
                class_videos.append((vid, class_dir.name, class_idx, len(frames)))

        if not class_videos:
            continue

        # Deterministic per-class shuffle to enforce exact stratified balance
        rng = random.Random(seed + class_idx)
        rng.shuffle(class_videos)

        n_c = len(class_videos)
        n_test_c = max(1, int(n_c * test_frac))
        n_val_c = max(1, int(n_c * val_frac))

        test_c = class_videos[:n_test_c]
        val_c = class_videos[n_test_c:n_test_c + n_val_c]
        train_c = class_videos[n_test_c + n_val_c:]

        test_set.extend([v[0] for v in test_c])
        val_set.extend([v[0] for v in val_c])
        train_set.extend([v[0] for v in train_c])
        video_records.extend(class_videos)

    split_manifest = {
        'seed': seed,
        'train': sorted(train_set),
        'val': sorted(val_set),
        'test': sorted(test_set),
        'counts': {
            'train': len(train_set),
            'val': len(val_set),
            'test': len(test_set),
            'total': len(train_set) + len(val_set) + len(test_set)
        }
    }

    assert verify_split_isolation(split_manifest), "Cross-split leakage detected!"

    # Save JSON manifest
    manifest_file = meta_path / 'split_manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(split_manifest, f, indent=2)

    # Save sequences CSV
    csv_file = meta_path / 'sequences.csv'
    with open(csv_file, 'w') as f:
        f.write("video_id,class_name,class_idx,n_frames,split\n")
        split_lookup = {}
        for s, vids in [('train', train_set), ('val', val_set), ('test', test_set)]:
            for v in vids:
                split_lookup[v] = s
        for vid, cname, cidx, n_fr in video_records:
            f.write(f"{vid},{cname},{cidx},{n_fr},{split_lookup.get(vid, 'unknown')}\n")

    print(f"  [Leakage Guard] Saved stratified split manifest → {manifest_file} (Zero frame overlap verified)")
    print(f"  [Leakage Guard] Saved sequence registry → {csv_file}")
    return split_manifest


# ---------------------------------------------------------------------------
# Generic frame-sequence dataset
# ---------------------------------------------------------------------------
class TemporalSequenceDataset(Dataset):
    """
    Loads ordered frame sequences from a directory structure:

        data/
          class_0/
            video_001/
              frame_0001.jpg
              frame_0002.jpg
              ...
          class_1/
            video_002/
              ...

    Enforces video-level train/val/test splits using GroupKFold.
    """

    def __init__(
        self,
        root: str,
        split: str,                    # 'train' | 'val' | 'test'
        config: Dict[str, Any],
        transform: Optional[T.Compose] = None,
        precomputed_splits: Optional[Dict] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.config = config
        self.max_frames = config["sequence"]["max_frames"]
        self.frame_size = tuple(config["sequence"]["frame_size"])
        self.transform = transform or build_transform(self.frame_size)

        # Discover sequences
        self.class_names, self.samples = self._discover_sequences()

        # Apply split
        if precomputed_splits is not None:
            split_ids = set(precomputed_splits[split])
            self.samples = [(vid, frames, lbl) for vid, frames, lbl in self.samples
                            if vid in split_ids]
        else:
            self.samples = self._apply_group_split(config["dataset"]["seed"])

    def _discover_sequences(self) -> Tuple[List[str], List]:
        """Walk directory tree to find all video sequences."""
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        class_names = [d.name for d in class_dirs]
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        samples = []
        for class_dir in class_dirs:
            label = class_to_idx[class_dir.name]
            for video_dir in sorted(class_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                frames = sorted(
                    [f for f in video_dir.iterdir()
                     if f.suffix.lower() in ('.jpg', '.jpeg', '.png')],
                    key=lambda x: x.name  # Alphabetical = temporal order
                )
                if len(frames) >= 2:  # Skip degenerate sequences
                    samples.append((video_dir.name, frames, label))

        return class_names, samples

    def _apply_group_split(self, seed: int) -> List:
        """
        Video-level GroupKFold split.
        Groups = video_ids → ensures no video appears in >1 split.
        """
        video_ids = [s[0] for s in self.samples]
        labels    = [s[2] for s in self.samples]

        # Use hash-based deterministic split (avoids sklearn dependency at inference)
        rng = random.Random(seed)
        unique_videos = sorted(list(set(video_ids)))
        rng.shuffle(unique_videos)

        n = len(unique_videos)
        val_frac  = self.config["dataset"]["val_split"]
        test_frac = self.config["dataset"]["test_split"]
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))

        test_set  = set(unique_videos[:n_test])
        val_set   = set(unique_videos[n_test:n_test + n_val])
        train_set = set(unique_videos[n_test + n_val:])

        split_map = {'train': train_set, 'val': val_set, 'test': test_set}
        return [s for s in self.samples if s[0] in split_map[self.split]]

    def _sample_frames(self, frames: List[Path], mode: str = 'uniform') -> List[Path]:
        """
        Sample max_frames from the full sequence.
        Modes:
          - uniform: evenly spaced via M2.2 floor formula i_k = floor(k*(L-1)/(N-1))
          - first_n: take first N frames
          - last_n:  take last N frames
          - sparse:  stride-based sampling
          - segment_center: TSN segment center sampling
        """
        n = len(frames)
        T = self.max_frames

        if n <= T:
            return frames

        if mode == 'uniform':
            indices = uniform_sample(n, T)
        elif mode == 'first_n':
            indices = first_n(n, T)
        elif mode == 'last_n':
            indices = last_n(n, T)
        elif mode == 'sparse':
            indices = sparse_sample(n, T)
        elif mode == 'segment_center':
            indices = segment_sample(n, T, mode='center')
        else:
            indices = uniform_sample(n, T)

        return [frames[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_id, all_frames, label = self.samples[idx]

        # M2.3 Rule: Validation and Test splits must ALWAYS be deterministic.
        # Random / stochastic sampling is strictly confined to training data augmentation.
        if self.split in ('val', 'test'):
            sampling_mode = 'uniform'
        else:
            sampling_mode = self.config["sequence"].get("sampling", "uniform")
        frames = self._sample_frames(all_frames, mode=sampling_mode)

        # Load and transform
        tensors = []
        for frame_path in frames:
            img = Image.open(frame_path).convert("RGB")
            tensors.append(self.transform(img))

        # Pad if sequence is shorter than max_frames
        while len(tensors) < self.max_frames:
            tensors.append(tensors[-1].clone())  # Repeat last frame

        return torch.stack(tensors), label  # [T, C, H, W], int

    def get_video_id(self, idx: int) -> str:
        return self.samples[idx][0]

    def save_splits(self, output_path: str):
        """Save split assignment for reproducibility."""
        splits = {'train': [], 'val': [], 'test': []}
        for vid, _, _ in self.samples:
            splits[self.split].append(vid)
        with open(output_path, 'w') as f:
            json.dump(splits, f, indent=2)


# ---------------------------------------------------------------------------
# Embedding-based dataset (used after extraction)
# ---------------------------------------------------------------------------
class EmbeddingSequenceDataset(Dataset):
    """
    Fast dataset that loads pre-extracted embeddings from .npy files.
    Used for all training and evaluation after the extraction step.

    Directory structure:
        embeddings/
          {split}/
            sequence_{id}.npy   # shape [N_frames, embed_dim]
          metadata.json
    """

    def __init__(
        self,
        embeddings_dir: str,
        split: str,
        max_frames: int = 8,
        perturbation: str = 'canonical',
        perturbation_seed: int = 42,
    ):
        self.embeddings_dir = Path(embeddings_dir) / split
        self.max_frames = max_frames
        self.perturbation = perturbation
        self.perturbation_seed = perturbation_seed

        # Load metadata
        meta_path = Path(embeddings_dir) / 'metadata.json'
        with open(meta_path) as f:
            metadata = json.load(f)

        self.samples = [
            (entry['sequence_id'], entry['label'])
            for entry in metadata
            if entry['split'] == split
        ]

    def _apply_perturbation(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply temporal perturbation to embedding sequence."""
        rng = np.random.RandomState(self.perturbation_seed)
        T = len(embeddings)

        if self.perturbation == 'canonical':
            return embeddings

        elif self.perturbation == 'shuffled':
            idx = rng.permutation(T)
            return embeddings[idx]

        elif self.perturbation == 'reversed':
            return embeddings[::-1].copy()

        elif self.perturbation == 'repeated':
            # Key control: same frame repeated N times
            # Expected to perform ~same as single-frame baseline
            mid_frame = embeddings[T // 2]
            return np.stack([mid_frame] * T)

        elif 'corrupted' in self.perturbation:
            # Replace fraction of frames with zero vector (drop simulation)
            fraction = float(self.perturbation.split('_')[1].replace('pct', '')) / 100
            result = embeddings.copy()
            n_corrupt = max(1, int(T * fraction))
            corrupt_idx = rng.choice(T, n_corrupt, replace=False)
            result[corrupt_idx] = 0.0
            return result

        return embeddings

    def _select_frames(self, embeddings: np.ndarray, n: int) -> np.ndarray:
        """Select n frames uniformly from the sequence (for progressive eval)."""
        T = len(embeddings)
        if T <= n:
            # Pad with last frame
            pad = np.stack([embeddings[-1]] * (n - T))
            return np.concatenate([embeddings, pad], axis=0)
        indices = np.linspace(0, T - 1, n, dtype=int)
        return embeddings[indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sequence_id, label = self.samples[idx]
        emb_path = self.embeddings_dir / f"{sequence_id}.npy"
        embeddings = np.load(str(emb_path))  # [N, D]

        # Select frames
        embeddings = self._select_frames(embeddings, self.max_frames)
        # Apply perturbation
        embeddings = self._apply_perturbation(embeddings)

        return torch.from_numpy(embeddings).float(), label  # [T, D], int
