"""
M2.8 / M2.9: Embedding Cache Manager and Metadata Manifest.

Provides high-throughput persistent storage and indexed retrieval of frozen frame embeddings.
Ensures M3/M4 models load directly from validated numpy arrays without re-running vision models.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.embeddings.contract import SequenceEmbeddingOutput


class EmbeddingCache:
    """
    Manages persistent caching of frozen sequence embeddings partitioned by split.

    Directory structure:
        {cache_root}/
            manifest.json
            train/
                {seq_id}.npy
                {seq_id}_meta.json
            val/
                ...
            test/
                ...
    """

    def __init__(self, cache_root: Union[str, Path] = "embeddings"):
        self.root = Path(cache_root)
        self.manifest_path = self.root / "manifest.json"
        self._ensure_directories()
        self.manifest = self._load_manifest()

    def _ensure_directories(self) -> None:
        """Create split directories."""
        for split in ["train", "val", "test", "real"]:
            (self.root / split).mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> Dict[str, Any]:
        """Load or initialize master cache manifest."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "version": "2.0.0",
            "splits": {"train": {}, "val": {}, "test": {}, "real": {}},
            "total_sequences": 0,
            "encoder_version": None,
            "preprocessing_version": None,
        }

    def save_manifest(self) -> None:
        """Write manifest atomically."""
        self.manifest["total_sequences"] = sum(
            len(s) for s in self.manifest["splits"].values()
        )
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def put(
        self,
        output: SequenceEmbeddingOutput,
        split: str = "train",
        save_manifest: bool = True
    ) -> Tuple[Path, Path]:
        """
        Store a validated sequence embedding in the cache under the specified split.
        """
        if split not in self.manifest["splits"]:
            self.manifest["splits"][split] = {}
            (self.root / split).mkdir(parents=True, exist_ok=True)

        split_dir = self.root / split
        npy_path, json_path = output.save(split_dir)

        # Update manifest record
        meta_dict = output.to_metadata_dict()
        meta_dict["npy_relpath"] = str(npy_path.relative_to(self.root))
        meta_dict["json_relpath"] = str(json_path.relative_to(self.root))

        self.manifest["splits"][split][output.sequence_id] = meta_dict
        if not self.manifest.get("encoder"):
            self.manifest["encoder"] = output.encoder
        if not self.manifest.get("encoder_version"):
            self.manifest["encoder_version"] = output.encoder_version
        if not self.manifest.get("embedding_dimension"):
            self.manifest["embedding_dimension"] = output.embed_dim
        if not self.manifest.get("preprocessing_version"):
            self.manifest["preprocessing_version"] = output.preprocessing_version
        if not self.manifest.get("sampling_strategy"):
            self.manifest["sampling_strategy"] = output.sampling_strategy

        if save_manifest:
            self.save_manifest()
        return npy_path, json_path

    def get(self, sequence_id: str, split: str = "train") -> Optional[SequenceEmbeddingOutput]:
        """
        Retrieve a sequence embedding from cache. Returns None if not found.
        """
        if split not in self.manifest["splits"] or sequence_id not in self.manifest["splits"][split]:
            return None

        meta_dict = self.manifest["splits"][split][sequence_id]
        npy_path = self.root / meta_dict["npy_relpath"]
        json_path = self.root / meta_dict["json_relpath"]

        return SequenceEmbeddingOutput.load(npy_path, json_path)

    def exists(self, sequence_id: str, split: str = "train") -> bool:
        """Check if sequence embedding is present and valid in cache."""
        if split not in self.manifest["splits"] or sequence_id not in self.manifest["splits"][split]:
            return False
        meta_dict = self.manifest["splits"][split][sequence_id]
        npy_p = self.root / meta_dict.get("npy_relpath", "")
        return npy_p.exists()

    def get_split_tensor_dataset(
        self,
        split: str,
        class_to_idx: Optional[Dict[str, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all cached sequences in a split into stacked numpy arrays:
            X: [N_sequences, T, D]
            Y: [N_sequences] integer class indices
            sequence_ids: List of N_sequences string IDs

        Guarantees deterministic order by natural sorting of sequence_ids.
        """
        if split not in self.manifest["splits"]:
            raise ValueError(f"Unknown split: {split}. Available: {list(self.manifest['splits'].keys())}")

        split_dict = self.manifest["splits"][split]
        if not split_dict:
            raise ValueError(f"Cache split '{split}' is empty.")

        sorted_seq_ids = sorted(list(split_dict.keys()))

        if class_to_idx is None:
            # Build natural class index mapping
            unique_labels = sorted(list(set(split_dict[s]["label"] for s in sorted_seq_ids)))
            class_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

        embeddings_list = []
        labels_list = []
        valid_ids = []

        for s_id in sorted_seq_ids:
            item = self.get(s_id, split=split)
            if item is None:
                continue
            embeddings_list.append(item.embeddings)
            labels_list.append(class_to_idx[item.label])
            valid_ids.append(s_id)

        X = np.stack(embeddings_list, axis=0).astype(np.float32)
        Y = np.array(labels_list, dtype=np.int64)

        return X, Y, valid_ids

    def get_array(
        self,
        sequence_id: str,
        split: str = "train",
        mmap: bool = False
    ) -> Optional[np.ndarray]:
        """
        Direct high-throughput array access skipping JSON metadata parsing.
        Supports zero-copy memory mapping via mmap_mode='r'.
        """
        if split not in self.manifest["splits"] or sequence_id not in self.manifest["splits"][split]:
            return None
        meta = self.manifest["splits"][split][sequence_id]
        npy_path = self.root / meta.get("npy_relpath", "")
        if not npy_path.exists():
            return None
        return np.load(npy_path, mmap_mode="r" if mmap else None)

    def as_torch_dataset(
        self,
        split: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        mmap: bool = False
    ) -> "CachedEmbeddingDataset":
        """
        Wrap cached split directly as a PyTorch Dataset for DataLoader ingestion.
        M3/M4 models load directly from cached tensors with zero vision encoder overhead.
        """
        return CachedEmbeddingDataset(self, split=split, class_to_idx=class_to_idx, mmap=mmap)

    def summarize(self) -> Dict[str, Any]:
        """Return high-level cache summary statistics."""
        summary = {
            "cache_root": str(self.root),
            "encoder_version": self.manifest.get("encoder_version"),
            "preprocessing_version": self.manifest.get("preprocessing_version"),
            "splits": {}
        }
        for split, records in self.manifest["splits"].items():
            summary["splits"][split] = {
                "count": len(records),
                "labels": len(set(r["label"] for r in records.values())) if records else 0
            }
        summary["total_sequences"] = sum(s["count"] for s in summary["splits"].values())
        return summary


class CachedEmbeddingDataset(Dataset):
    """
    High-performance PyTorch Dataset streaming directly from cached numpy embeddings.
    Fulfills M2.8: M3/M4 models train exclusively on numerical tensors.
    """

    def __init__(
        self,
        cache: EmbeddingCache,
        split: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        mmap: bool = False
    ):
        self.cache = cache
        self.split = split
        self.mmap = mmap
        if split not in cache.manifest["splits"]:
            raise ValueError(f"Unknown split: {split}. Available: {list(cache.manifest['splits'].keys())}")
        self.split_records = cache.manifest["splits"][split]
        self.sequence_ids = sorted(list(self.split_records.keys()))

        if class_to_idx is None:
            unique_labels = sorted(list(set(self.split_records[s]["label"] for s in self.sequence_ids)))
            self.class_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq_id = self.sequence_ids[idx]
        record = self.split_records[seq_id]
        npy_path = self.cache.root / record["npy_relpath"]
        arr = np.load(npy_path, mmap_mode="r" if self.mmap else None)
        tensor = torch.from_numpy(np.array(arr, copy=False)).float()
        label_idx = self.class_to_idx[record["label"]]
        return tensor, label_idx

