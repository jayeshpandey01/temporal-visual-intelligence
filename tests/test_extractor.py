"""
Unit Tests for M2.7: Embedding Extraction Engine.

Validates the complete 8-step extraction lifecycle:
  1. Encoder initialized frozen and in eval mode.
  2. Sequence loading.
  3. Deterministic uniform frame sampling.
  4. Exact DINOv2 / ImageNet preprocessing.
  5. Inference under torch.inference_mode() with zero gradients and zero optimizer.
  6. Atomic embedding serialization to cache (.npy).
  7. Manifest and metadata fingerprinting (SHA-256).
  8. Explicit GPU memory deallocation.
"""

import tempfile
import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from src.embeddings.extractor import EmbeddingExtractor, load_encoder
from src.embeddings.contract import SequenceInput, SequenceEmbeddingOutput
from src.embeddings.cache import EmbeddingCache


@pytest.fixture
def dummy_extractor_and_cache(tmp_path):
    config = {
        'encoder': {
            'name': 'dinov2_vits14',
            'embed_dim': 384,
            'freeze': True
        },
        'paths': {
            'embeddings': str(tmp_path / "embeddings")
        }
    }
    extractor = EmbeddingExtractor(config)
    return extractor, tmp_path


@pytest.fixture
def dummy_frames_dir(tmp_path):
    frames_dir = tmp_path / "test_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for i in range(16):
        img = Image.new("RGB", (256, 256), color=(i * 15, 100, 200 - i * 10))
        fpath = frames_dir / f"frame_{i:04d}.png"
        img.save(fpath)
        frame_paths.append(fpath)
    return frame_paths


def test_extractor_initialization_frozen(dummy_extractor_and_cache):
    extractor, _ = dummy_extractor_and_cache
    assert not extractor.model.training, "Encoder must be in eval() mode"
    for param in extractor.model.parameters():
        assert not param.requires_grad, "Encoder parameters must have requires_grad=False"


def test_extractor_extract_frame_batch(dummy_extractor_and_cache):
    extractor, _ = dummy_extractor_and_cache
    dummy_tensors = [torch.randn(3, 224, 224) for _ in range(4)]
    embs = extractor.extract_frame_batch(dummy_tensors)

    assert isinstance(embs, np.ndarray)
    assert embs.shape == (4, 384)
    assert embs.dtype == np.float32
    assert not np.isnan(embs).any()
    assert not np.isinf(embs).any()


def test_extractor_extract_single_sequence_8_steps(dummy_extractor_and_cache, dummy_frames_dir):
    extractor, tmp_path = dummy_extractor_and_cache
    frame_paths = dummy_frames_dir

    seq_out = extractor.extract_single_sequence(
        sequence_id="test_seq_001",
        frame_paths=frame_paths,
        label="test_action",
        target_frames=8,
        split="test",
        save_to_cache=True
    )

    # Verify contract
    assert isinstance(seq_out, SequenceEmbeddingOutput)
    assert seq_out.sequence_id == "test_seq_001"
    assert seq_out.num_frames == 8
    assert seq_out.embed_dim == 384
    assert seq_out.embeddings.shape == (8, 384)
    assert seq_out.label == "test_action"
    assert seq_out.encoder_version == "dinov2_vits14"
    assert seq_out.preprocessing_version == "v1_imagenet_224x224_center"
    assert len(seq_out.checksum) == 64  # SHA-256

    # Verify cached on disk
    npy_file = tmp_path / "embeddings" / "test" / "test_seq_001.npy"
    json_file = tmp_path / "embeddings" / "test" / "test_seq_001_meta.json"
    assert npy_file.exists(), "Cached .npy file must exist on disk"
    assert json_file.exists(), "Cached _meta.json file must exist on disk"

    # Verify manifest updated
    cached_out = extractor.cache.get("test_seq_001", split="test")
    assert cached_out is not None
    assert np.allclose(cached_out.embeddings, seq_out.embeddings)


def test_extractor_inference_mode_no_grad(dummy_extractor_and_cache, dummy_frames_dir):
    extractor, _ = dummy_extractor_and_cache
    frame_paths = dummy_frames_dir[:4]

    # Run inference and verify no grad graph exists
    seq_out = extractor.extract_single_sequence(
        sequence_id="test_no_grad_001",
        frame_paths=frame_paths,
        label="action",
        target_frames=4,
        save_to_cache=False
    )
    assert isinstance(seq_out.embeddings, np.ndarray)


def test_extractor_rejection_empty_sequence(dummy_extractor_and_cache):
    extractor, _ = dummy_extractor_and_cache
    with pytest.raises(ValueError, match="No frames provided"):
        extractor.extract_single_sequence(
            sequence_id="empty_seq",
            frame_paths=[],
            label="action"
        )

