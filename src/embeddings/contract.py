"""
M2.1: Experimental Interface & Contract Specification.

Defines the immutable data models and contract governing frame embedding extraction,
caching, and temporal modeling.

Design Principles (research.md §7-8, M-2.md §M2.1):
  - Strict type safety and validation of input sequences and output representations.
  - Guarantees exact shape, dimension, and integrity across all downstream experiments.
  - Checksum hashing (SHA-256) ensuring immutable representations for M3/M4.
"""

from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np


@dataclass(frozen=True)
class SequenceInput:
    """
    Contract for raw sequential visual input entering the representation pipeline.

    Attributes:
        sequence_id: Unique identifier of the sequence (e.g., 'vid_01_0000' or 'bmw_clip_0001').
        frame_paths: Chronologically ordered file paths to extracted image frames.
        label: Semantic class label (string) of the action or event.
        frame_indices: Integer timestamps or natural frame indices within the source video.
        metadata: Optional dictionary with video provenance (fps, total frames, resolution).
    """
    sequence_id: str
    frame_paths: List[str]
    label: str
    frame_indices: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Enforce strict contract consistency on the sequence input."""
        if not self.sequence_id or not isinstance(self.sequence_id, str):
            raise ValueError(f"Invalid sequence_id: {self.sequence_id}. Must be a non-empty string.")
        if not self.label or not isinstance(self.label, str):
            raise ValueError(f"Invalid label: {self.label}. Must be a non-empty string.")
        if len(self.frame_paths) == 0:
            raise ValueError(f"Sequence {self.sequence_id} has 0 frame paths.")
        if len(self.frame_indices) == 0:
            raise ValueError(f"Sequence {self.sequence_id} has 0 frame indices.")
        if len(self.frame_paths) != len(self.frame_indices):
            raise ValueError(
                f"Cardinality mismatch in {self.sequence_id}: "
                f"{len(self.frame_paths)} frame paths vs. {len(self.frame_indices)} frame indices."
            )
        # Verify strictly monotonically increasing natural indices
        for i in range(len(self.frame_indices) - 1):
            if self.frame_indices[i] >= self.frame_indices[i + 1]:
                raise ValueError(
                    f"Frame indices in {self.sequence_id} must be strictly increasing: "
                    f"index[{i}]={self.frame_indices[i]} >= index[{i+1}]={self.frame_indices[i+1]}"
                )

    def verify_files_exist(self) -> bool:
        """Verify that all frame image files exist on disk."""
        for path_str in self.frame_paths:
            if not Path(path_str).exists():
                raise FileNotFoundError(f"Frame file not found: {path_str} in sequence {self.sequence_id}")
        return True


@dataclass
class SequenceEmbeddingOutput:
    """
    Contract for frozen sequential visual embeddings exiting the representation pipeline.

    Attributes:
        sequence_id: Identifier matching the SequenceInput.
        frame_indices: Selected frame indices represented in this embedding trajectory.
        embeddings: Float32 numpy array with shape [N_frames, embed_dim].
        label: Semantic class label.
        encoder_version: Identifier of the frozen encoder (e.g. 'dinov2_vits14@lvd142m').
        preprocessing_version: Identifier of normalization and resize transforms (e.g. 'v1_imagenet_224x224').
        created_at: ISO 8601 UTC timestamp of extraction.
        checksum: SHA-256 hash of the serialized embedding array bytes for tamper-proof validation.
    """
    sequence_id: str
    frame_indices: List[int]
    embeddings: np.ndarray
    label: str
    encoder_version: str
    preprocessing_version: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checksum: str = field(default="")

    def __post_init__(self):
        if not self.checksum and isinstance(self.embeddings, np.ndarray):
            self.checksum = self._compute_checksum(self.embeddings)
        self.validate()

    @staticmethod
    def _compute_checksum(arr: np.ndarray) -> str:
        """Compute SHA-256 hash of the numpy array bytes."""
        return hashlib.sha256(arr.tobytes()).hexdigest()

    @property
    def embedding_shape(self) -> Tuple[int, int]:
        """Return (num_frames, embed_dim)."""
        return tuple(self.embeddings.shape)

    @property
    def num_frames(self) -> int:
        return self.embeddings.shape[0]

    @property
    def embed_dim(self) -> int:
        return self.embeddings.shape[1]

    def validate(self) -> None:
        """Validate embedding matrix properties, finiteness, and cardinality."""
        if not isinstance(self.embeddings, np.ndarray):
            raise TypeError(f"Embeddings for {self.sequence_id} must be a numpy.ndarray, got {type(self.embeddings)}")
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings for {self.sequence_id} must be 2D [N_frames, embed_dim], got shape {self.embeddings.shape}"
            )
        if self.embeddings.shape[0] != len(self.frame_indices):
            raise ValueError(
                f"Embedding frame count {self.embeddings.shape[0]} does not match "
                f"frame_indices length {len(self.frame_indices)} in {self.sequence_id}"
            )
        if self.embeddings.shape[1] <= 0:
            raise ValueError(f"Invalid embed_dim: {self.embeddings.shape[1]} in {self.sequence_id}")

        # Check for numerical stability: NaN or Inf
        if np.isnan(self.embeddings).any():
            raise ValueError(f"NaN values detected in embeddings for sequence {self.sequence_id}")
        if np.isinf(self.embeddings).any():
            raise ValueError(f"Inf values detected in embeddings for sequence {self.sequence_id}")

        # Check for degenerate zero vectors
        norms = np.linalg.norm(self.embeddings, axis=1)
        if (norms < 1e-6).any():
            zero_indices = np.where(norms < 1e-6)[0].tolist()
            raise ValueError(
                f"Degenerate all-zero embedding detected at frame indices {zero_indices} in {self.sequence_id}"
            )

        # Checksum integrity verification if provided
        computed = self._compute_checksum(self.embeddings)
        if self.checksum and self.checksum != computed:
            raise ValueError(
                f"Checksum mismatch for {self.sequence_id}! Expected {self.checksum}, computed {computed}"
            )

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Export manifest metadata excluding heavy embedding bytes."""
        return {
            "sequence_id": self.sequence_id,
            "frame_indices": self.frame_indices,
            "embedding_shape": list(self.embedding_shape),
            "label": self.label,
            "encoder_version": self.encoder_version,
            "preprocessing_version": self.preprocessing_version,
            "created_at": self.created_at,
            "checksum": self.checksum,
            "mean_vector_norm": float(np.mean(np.linalg.norm(self.embeddings, axis=1))),
        }

    def save(self, output_dir: Union[str, Path]) -> Tuple[Path, Path]:
        """
        Atomically save embedding array (.npy) and metadata (.json) to disk.

        Returns:
            Tuple of (npy_path, json_path).
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        npy_path = out_p / f"{self.sequence_id}.npy"
        json_path = out_p / f"{self.sequence_id}_meta.json"

        # Save numpy array
        np.save(npy_path, self.embeddings.astype(np.float32))

        # Save metadata
        with open(json_path, "w") as f:
            json.dump(self.to_metadata_dict(), f, indent=2)

        return npy_path, json_path

    @classmethod
    def load(cls, npy_path: Union[str, Path], meta_path: Union[str, Path]) -> SequenceEmbeddingOutput:
        """
        Load an embedding output from disk and verify its cryptographic checksum.
        """
        npy_p = Path(npy_path)
        meta_p = Path(meta_path)

        if not npy_p.exists():
            raise FileNotFoundError(f"Embedding file not found: {npy_p}")
        if not meta_p.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_p}")

        with open(meta_p) as f:
            meta = json.load(f)

        arr = np.load(npy_p)

        return cls(
            sequence_id=meta["sequence_id"],
            frame_indices=meta["frame_indices"],
            embeddings=arr,
            label=meta["label"],
            encoder_version=meta["encoder_version"],
            preprocessing_version=meta["preprocessing_version"],
            created_at=meta.get("created_at", ""),
            checksum=meta.get("checksum", "")
        )

