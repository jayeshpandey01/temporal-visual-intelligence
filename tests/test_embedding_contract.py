"""
Unit tests for M2.1 / M2.8 / M2.10: Experimental Interface, Caching, and Validation.
"""

import os
import json
import tempfile
import pytest
import numpy as np

from src.embeddings.contract import SequenceInput, SequenceEmbeddingOutput
from src.embeddings.cache import EmbeddingCache
from src.embeddings.validate import validate_embedding_cache


def test_sequence_input_valid():
    inp = SequenceInput(
        sequence_id="seq_001",
        frame_paths=["/tmp/f1.jpg", "/tmp/f2.jpg", "/tmp/f3.jpg"],
        label="Opening something",
        frame_indices=[1, 2, 3],
        metadata={"fps": 30}
    )
    assert inp.sequence_id == "seq_001"
    assert inp.label == "Opening something"
    assert len(inp.frame_paths) == 3
    assert inp.frame_indices == [1, 2, 3]


def test_sequence_input_invalid_length_mismatch():
    with pytest.raises(ValueError, match="Cardinality mismatch"):
        SequenceInput(
            sequence_id="seq_bad",
            frame_paths=["/tmp/f1.jpg", "/tmp/f2.jpg"],
            label="Closing something",
            frame_indices=[1, 2, 3]  # 3 indices vs 2 paths
        )


def test_sequence_input_invalid_non_monotonic():
    with pytest.raises(ValueError, match="strictly increasing"):
        SequenceInput(
            sequence_id="seq_bad",
            frame_paths=["/tmp/f1.jpg", "/tmp/f2.jpg"],
            label="Closing something",
            frame_indices=[5, 3]  # decreasing
        )


def test_sequence_embedding_output_valid():
    N, D = 4, 384
    arr = np.random.randn(N, D).astype(np.float32)
    out = SequenceEmbeddingOutput(
        sequence_id="seq_001",
        frame_indices=[4, 12, 20, 28],
        embeddings=arr,
        label="class_A",
        encoder_version="dinov2_vits14@lvd142m",
        preprocessing_version="v1_imagenet_224x224_center"
    )
    assert out.embedding_shape == (4, 384)
    assert out.num_frames == 4
    assert out.embed_dim == 384
    assert len(out.checksum) == 64  # SHA-256


def test_sequence_embedding_output_rejection_nan():
    arr = np.ones((4, 384), dtype=np.float32)
    arr[1, 10] = np.nan
    with pytest.raises(ValueError, match="NaN values detected"):
        SequenceEmbeddingOutput(
            sequence_id="seq_nan",
            frame_indices=[1, 2, 3, 4],
            embeddings=arr,
            label="class_A",
            encoder_version="dinov2_vits14",
            preprocessing_version="v1"
        )


def test_sequence_embedding_output_rejection_zero_vector():
    arr = np.ones((4, 384), dtype=np.float32)
    arr[2, :] = 0.0  # degenerate zero vector
    with pytest.raises(ValueError, match="Degenerate all-zero embedding"):
        SequenceEmbeddingOutput(
            sequence_id="seq_zero",
            frame_indices=[1, 2, 3, 4],
            embeddings=arr,
            label="class_A",
            encoder_version="dinov2_vits14",
            preprocessing_version="v1"
        )


def test_sequence_embedding_output_save_and_load(tmp_path):
    arr = np.random.randn(8, 384).astype(np.float32)
    orig = SequenceEmbeddingOutput(
        sequence_id="seq_persist",
        frame_indices=[1, 2, 3, 4, 5, 6, 7, 8],
        embeddings=arr,
        label="Pushing something from left to right",
        encoder_version="dinov2_vits14",
        preprocessing_version="v1"
    )

    npy_p, json_p = orig.save(tmp_path)
    assert npy_p.exists()
    assert json_p.exists()

    loaded = SequenceEmbeddingOutput.load(npy_p, json_p)
    assert loaded.sequence_id == orig.sequence_id
    assert loaded.label == orig.label
    assert loaded.frame_indices == orig.frame_indices
    assert loaded.checksum == orig.checksum
    assert np.allclose(loaded.embeddings, orig.embeddings)


def test_embedding_cache_put_get(tmp_path):
    cache = EmbeddingCache(cache_root=tmp_path)
    arr = np.random.randn(8, 384).astype(np.float32)
    item = SequenceEmbeddingOutput(
        sequence_id="seq_cached",
        frame_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        embeddings=arr,
        label="Turning camera left",
        encoder_version="dinov2_vits14",
        preprocessing_version="v1"
    )

    cache.put(item, split="train")
    assert cache.exists("seq_cached", split="train")
    assert not cache.exists("seq_cached", split="val")

    retrieved = cache.get("seq_cached", split="train")
    assert retrieved is not None
    assert retrieved.sequence_id == "seq_cached"
    assert np.allclose(retrieved.embeddings, arr)

    summary = cache.summarize()
    assert summary["splits"]["train"]["count"] == 1
    assert summary["total_sequences"] == 1


def test_embedding_cache_tensor_dataset(tmp_path):
    cache = EmbeddingCache(cache_root=tmp_path)
    labels = ["class_A", "class_B", "class_A"]
    for i, lbl in enumerate(labels):
        arr = np.random.randn(8, 384).astype(np.float32)
        item = SequenceEmbeddingOutput(
            sequence_id=f"seq_{i:03d}",
            frame_indices=list(range(8)),
            embeddings=arr,
            label=lbl,
            encoder_version="dinov2_vits14",
            preprocessing_version="v1"
        )
        cache.put(item, split="test")

    X, Y, ids = cache.get_split_tensor_dataset("test")
    assert X.shape == (3, 8, 384)
    assert Y.shape == (3,)
    assert len(ids) == 3
    assert ids == ["seq_000", "seq_001", "seq_002"]


def test_validate_embedding_cache(tmp_path):
    cache = EmbeddingCache(cache_root=tmp_path)
    for i in range(5):
        arr = np.random.randn(8, 384).astype(np.float32)
        item = SequenceEmbeddingOutput(
            sequence_id=f"vid_train_{i:04d}",
            frame_indices=list(range(8)),
            embeddings=arr,
            label="class_test",
            encoder_version="dinov2_vits14",
            preprocessing_version="v1"
        )
        cache.put(item, split="train")

    report_p = tmp_path / "audit_report.json"
    rep = validate_embedding_cache(
        cache_root=str(tmp_path),
        expected_dim=384,
        expected_frames=8,
        output_report=str(report_p)
    )
    assert rep["global_status"] == "PASS"
    assert rep["leakage_check"]["status"] == "PASS"
    assert rep["splits_audited"]["train"]["error_count"] == 0
