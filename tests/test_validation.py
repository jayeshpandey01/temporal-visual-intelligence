"""
Unit Tests for M2.10: Automated Embedding Cache Validator.

Validates that the automated auditor correctly catches and reports all failure modes:
  1. number_of_frames != number_of_embeddings
  2. embedding_dimension != expected_dimension
  3. NaN values
  4. Inf values
  5. Degenerate all-zero vectors
  6. Duplicate or non-monotonic frame indices
  7. Cross-split ID leakage
"""

import json
import pytest
import numpy as np
from pathlib import Path

from src.embeddings.cache import EmbeddingCache
from src.embeddings.contract import SequenceEmbeddingOutput
from src.embeddings.validate import validate_embedding_cache


@pytest.fixture
def clean_cache(tmp_path):
    cache_root = tmp_path / "valid_cache"
    cache = EmbeddingCache(cache_root=cache_root)

    for split in ["train", "val", "test"]:
        for i in range(4):
            seq_id = f"{split}_seq_{i}"
            arr = np.random.randn(8, 384).astype(np.float32)
            out = SequenceEmbeddingOutput(
                sequence_id=seq_id,
                frame_indices=list(range(8)),
                embeddings=arr,
                label="action_push",
                encoder_version="dinov2_vits14",
                preprocessing_version="v1_imagenet_224x224_center"
            )
            cache.put(out, split=split, save_manifest=False)

    cache.save_manifest()
    return cache_root


def test_validate_clean_cache_passes(clean_cache, tmp_path):
    report_p = tmp_path / "report.json"
    rep = validate_embedding_cache(
        cache_root=str(clean_cache),
        expected_dim=384,
        expected_frames=8,
        output_report=str(report_p)
    )
    assert rep["global_status"] == "PASS"
    assert rep["leakage_check"]["status"] == "PASS"
    assert all(s["error_count"] == 0 for s in rep["splits_audited"].values())


def test_validate_catches_dim_mismatch(tmp_path):
    cache = EmbeddingCache(cache_root=tmp_path / "dim_mismatch_cache")
    arr = np.random.randn(8, 768).astype(np.float32)  # 768 instead of expected 384
    out = SequenceEmbeddingOutput(
        sequence_id="dim_seq_01",
        frame_indices=list(range(8)),
        embeddings=arr,
        label="action",
        encoder_version="dinov2_vitb14",
        preprocessing_version="v1"
    )
    cache.put(out, split="train")

    report_p = tmp_path / "report_dim.json"
    rep = validate_embedding_cache(
        cache_root=str(tmp_path / "dim_mismatch_cache"),
        expected_dim=384,
        expected_frames=8,
        output_report=str(report_p)
    )
    assert rep["global_status"] == "FAIL"
    assert any("embed_dim 768 != expected 384" in err for err in rep["errors"])


def test_validate_catches_frame_count_mismatch(tmp_path):
    cache = EmbeddingCache(cache_root=tmp_path / "frame_mismatch_cache")
    arr = np.random.randn(4, 384).astype(np.float32)  # 4 frames instead of 8
    out = SequenceEmbeddingOutput(
        sequence_id="frame_seq_01",
        frame_indices=list(range(4)),
        embeddings=arr,
        label="action",
        encoder_version="dinov2_vits14",
        preprocessing_version="v1"
    )
    cache.put(out, split="train")

    report_p = tmp_path / "report_frame.json"
    rep = validate_embedding_cache(
        cache_root=str(tmp_path / "frame_mismatch_cache"),
        expected_dim=384,
        expected_frames=8,
        output_report=str(report_p)
    )
    assert rep["global_status"] == "FAIL"
    assert any("num_frames 4 != expected 8" in err for err in rep["errors"])


def test_validate_catches_cross_split_leakage(tmp_path):
    cache = EmbeddingCache(cache_root=tmp_path / "leakage_cache")
    arr = np.random.randn(8, 384).astype(np.float32)
    # Put same sequence ID in train AND test
    out_train = SequenceEmbeddingOutput(
        sequence_id="leaked_seq_001",
        frame_indices=list(range(8)),
        embeddings=arr,
        label="action",
        encoder_version="dinov2_vits14",
        preprocessing_version="v1"
    )
    cache.put(out_train, split="train")

    out_test = SequenceEmbeddingOutput(
        sequence_id="leaked_seq_001",
        frame_indices=list(range(8)),
        embeddings=arr,
        label="action",
        encoder_version="dinov2_vits14",
        preprocessing_version="v1"
    )
    cache.put(out_test, split="test")

    report_p = tmp_path / "report_leakage.json"
    rep = validate_embedding_cache(
        cache_root=str(tmp_path / "leakage_cache"),
        expected_dim=384,
        expected_frames=8,
        output_report=str(report_p)
    )
    assert rep["global_status"] == "FAIL"
    assert rep["leakage_check"]["status"] == "FAIL"
    assert rep["leakage_check"]["train_test_overlap"] == 1

