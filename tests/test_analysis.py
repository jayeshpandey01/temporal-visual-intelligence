"""
Unit Tests for M2.11: Representation Sanity Checks.

Validates:
  1. Consecutive cosine similarity S_t = cos(E_t, E_{t+1}) exact calculations.
  2. Identity invariance (identical frames yield S_t = 1.0).
  3. Orthogonal invariance (orthogonal latent vectors yield S_t = 0.0).
  4. Cumulative drift Delta(t) = 1 - cos(E_0, E_t) endpoint bounds.
  5. Curve visualization generator functions and outputs.
"""

import pytest
import numpy as np
from pathlib import Path

from src.embeddings.analysis import (
    consecutive_cosine_similarity,
    cumulative_drift,
    temporal_variance,
    plot_sequence_similarity_curve,
    plot_representative_sanity_comparisons
)


def test_consecutive_cosine_similarity_identity():
    """Identical frames produce S_t = 1.0."""
    v = np.random.randn(384).astype(np.float32)
    embs = np.stack([v, v, v, v, v])  # 5 identical frames
    s_t = consecutive_cosine_similarity(embs)
    assert len(s_t) == 4
    assert np.allclose(s_t, 1.0, atol=1e-5)


def test_consecutive_cosine_similarity_orthogonal():
    """Orthogonal frames produce S_t = 0.0."""
    e1 = np.zeros(384, dtype=np.float32)
    e2 = np.zeros(384, dtype=np.float32)
    e3 = np.zeros(384, dtype=np.float32)
    e1[0] = 1.0
    e2[1] = 1.0
    e3[2] = 1.0
    embs = np.stack([e1, e2, e3])
    s_t = consecutive_cosine_similarity(embs)
    assert len(s_t) == 2
    assert np.allclose(s_t, 0.0, atol=1e-5)


def test_cumulative_drift_bounds():
    """Cumulative drift starts at 0.0 at t=0 and is bounded in [0, 2]."""
    embs = np.random.randn(8, 384).astype(np.float32)
    drift = cumulative_drift(embs)
    assert len(drift) == 8
    assert np.isclose(drift[0], 0.0, atol=1e-5)
    assert (drift >= -1e-5).all()
    assert (drift <= 2.0 + 1e-5).all()


def test_temporal_variance():
    """Temporal variance of constant sequence is 0.0."""
    v = np.random.randn(384).astype(np.float32)
    embs = np.stack([v] * 6)
    var = temporal_variance(embs)
    assert np.isclose(var, 0.0, atol=1e-5)


def test_plot_sequence_similarity_curve(tmp_path):
    """M2.11 plot generator outputs valid figure file."""
    embs = np.random.randn(5, 384).astype(np.float32)
    out_img = tmp_path / "sim_curve.png"
    s_t, path = plot_sequence_similarity_curve(
        embeddings=embs,
        sequence_id="test_seq_m2_11",
        frame_indices=[1, 2, 3, 4, 5],
        output_path=out_img
    )
    assert len(s_t) == 4
    assert path.exists()
    assert path.stat().st_size > 1000
