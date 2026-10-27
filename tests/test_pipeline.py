"""
Unit tests for the Temporal Visual Intelligence research pipeline.
Tests model architectures, perturbations, metrics, and data structures.
"""

import os
import sys
import yaml
import pytest
import numpy as np
import torch

from src.models.baseline import SingleFrameMLP, MajorityClassBaseline
from src.models.temporal import TemporalGRU, TemporalTransformer
from src.experiments.shuffle import apply_order_perturbation
from src.experiments.corruption import apply_corruption
from src.experiments.sparsity import select_frames_by_mode
from src.evaluation.metrics import compute_metrics, aggregate_across_seeds
from src.evaluation.analysis import compute_win_loss_quadrants


@pytest.fixture
def sample_config():
    with open('configs/experiment.yaml') as f:
        return yaml.safe_load(f)


def test_baseline_mlp_forward():
    B, T, D, C = 4, 8, 384, 10
    model = SingleFrameMLP(embed_dim=D, hidden_dim=64, num_classes=C)
    
    # Test with 2D tensor [B, D]
    x_2d = torch.randn(B, D)
    out_2d = model(x_2d)
    assert out_2d.shape == (B, C)

    # Test with 3D tensor [B, T, D] (should use middle frame)
    x_3d = torch.randn(B, T, D)
    out_3d = model(x_3d)
    assert out_3d.shape == (B, C)


def test_multiframe_meanpool_forward():
    from src.models.baseline import MultiFrameMeanPool
    B, T, D, C = 4, 8, 384, 10
    model = MultiFrameMeanPool(embed_dim=D, hidden_dim=64, num_classes=C)
    base_mlp = SingleFrameMLP(embed_dim=D, hidden_dim=64, num_classes=C)

    # Parameter capacity parity
    assert model.n_parameters() == base_mlp.n_parameters()

    # Forward pass [B, T, D]
    x = torch.randn(B, T, D)
    out = model(x)
    assert out.shape == (B, C)

    # Order invariance proof: permuting T dimension must produce exact identical output in eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
        perm = torch.randperm(T)
        out_perm = model(x[:, perm, :])
        assert torch.allclose(out_eval, out_perm, atol=1e-5)


def test_temporal_gru_forward():
    B, T, D, C = 4, 8, 384, 10
    model = TemporalGRU(embed_dim=D, hidden_dim=64, num_layers=1, num_classes=C)
    
    x = torch.randn(B, T, D)
    out = model(x)
    assert out.shape == (B, C)


def test_temporal_transformer_forward():
    B, T, D, C = 4, 8, 384, 10
    model = TemporalTransformer(embed_dim=D, hidden_dim=64, num_heads=4, num_layers=1, num_classes=C)
    
    x = torch.randn(B, T, D)
    out = model(x)
    assert out.shape == (B, C)


def test_order_perturbations():
    B, T, D = 2, 6, 16
    x = torch.arange(B * T * D, dtype=torch.float32).reshape(B, T, D)

    # Canonical
    canon = apply_order_perturbation(x, 'canonical')
    assert torch.equal(canon, x)

    # Shuffled
    shuffled = apply_order_perturbation(x, 'shuffled', seed=42)
    assert shuffled.shape == x.shape

    # Repeated
    repeated = apply_order_perturbation(x, 'repeated')
    assert repeated.shape == x.shape
    mid = T // 2
    for t in range(T):
        assert torch.equal(repeated[:, t, :], x[:, mid, :])

    # Reversed
    reversed_x = apply_order_perturbation(x, 'reversed')
    assert torch.equal(reversed_x[:, 0, :], x[:, -1, :])


def test_corruption():
    B, T, D = 2, 8, 16
    x = torch.ones(B, T, D)

    # Zero mask
    masked = apply_corruption(x, 'zero_mask', fraction=0.25)
    assert masked.shape == x.shape
    # At least some frames should be zero
    assert (masked == 0.0).any()


def test_sparsity():
    B, T, D = 2, 8, 16
    x = torch.arange(B * T * D, dtype=torch.float32).reshape(B, T, D)

    dense = select_frames_by_mode(x, 'dense', 4)
    assert dense.shape == (B, 4, D)

    sparse = select_frames_by_mode(x, 'sparse', 4)
    assert sparse.shape == (B, 4, D)

    early = select_frames_by_mode(x, 'early', 4)
    assert early.shape == (B, 4, D)
    assert torch.equal(early, x[:, :4, :])


def test_metrics():
    labels = np.array([0, 1, 2, 1, 0])
    preds = np.array([0, 1, 2, 0, 0])
    
    metrics = compute_metrics(labels, preds)
    assert metrics['accuracy'] == 0.8
    assert 'macro_f1' in metrics
    assert len(metrics['per_class_f1']) == 3


def test_win_loss_quadrants():
    labels = np.array([0, 1, 2, 3])
    base_preds = np.array([0, 0, 2, 0])  # correct at 0, 2 (idx 0, 2)
    temp_preds = np.array([0, 1, 0, 0])  # correct at 0, 1 (idx 0, 1)

    # idx 0: both correct -> A
    # idx 1: base wrong, temp correct -> C (temporal rescued)
    # idx 2: base correct, temp wrong -> B (baseline only)
    # idx 3: both wrong -> D
    quad = compute_win_loss_quadrants(base_preds, temp_preds, labels)
    counts = quad['quadrant_counts']
    assert counts['A_both_correct'] == 1
    assert counts['B_baseline_only'] == 1
    assert counts['C_temporal_rescued'] == 1
    assert counts['D_both_wrong'] == 1


def test_split_isolation():
    from src.data.dataset import verify_split_isolation
    valid_split = {
        'train': ['v1', 'v2', 'v3'],
        'val': ['v4'],
        'test': ['v5', 'v6']
    }
    assert verify_split_isolation(valid_split) is True

    leaking_split = {
        'train': ['v1', 'v2', 'v3'],
        'val': ['v2', 'v4'],  # v2 is in both train and val!
        'test': ['v5']
    }
    assert verify_split_isolation(leaking_split) is False


def test_temporal_structure_metrics():
    from src.data.temporal_structure import (
        compute_sequence_length_statistics,
        estimate_informative_horizon,
        audit_temporal_ambiguity_progression,
    )
    # Mock sequences
    mock_seqs = {
        'class_a': [{'n_frames': 8}, {'n_frames': 8}],
        'class_b': [{'n_frames': 8}, {'n_frames': 10}],
    }
    stats = compute_sequence_length_statistics(mock_seqs)
    assert stats['total_sequences'] == 4
    assert stats['min'] == 8
    assert stats['max'] == 10
    assert 'p50' in stats['percentiles']

    # Test horizon estimation
    mock_curves = {
        'class_a': [0.01, 0.02, 0.05, 0.04, 0.02, 0.01, 0.01]
    }
    horizon = estimate_informative_horizon(mock_curves)
    assert 'mean_t_informative' in horizon
    assert horizon['mean_t_informative'] >= 1.0

    # Test ambiguity audit
    ambiguity = audit_temporal_ambiguity_progression()
    assert 'K_1_frame' in ambiguity
    assert 'K_8_frames' in ambiguity


def test_two_frame_micro_analysis(tmp_path):
    from PIL import Image
    from src.embeddings.micro_analysis import (
        compute_pixel_metrics,
        compute_representation_metrics,
        evaluate_controlled_micro_probes,
        analyze_two_frame_dyad,
    )

    # 1. Pixel metrics unit test
    img1 = Image.new('RGB', (224, 224), color=(50, 50, 50))
    img2 = Image.new('RGB', (224, 224), color=(100, 100, 100))
    pix = compute_pixel_metrics(img1, img2)
    assert pix['pixel_mse'] > 0
    assert pix['pixel_l1'] > 0
    assert pix['diff_map'].shape == (224, 224)

    # 2. Representation metrics unit test
    E1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    E2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    rep = compute_representation_metrics(E1, E2)
    assert np.isclose(rep['cosine_similarity'], 0.0, atol=1e-5)
    assert np.isclose(rep['temporal_drift'], 1.0, atol=1e-5)
    assert np.isclose(rep['l2_distance'], np.sqrt(2.0), atol=1e-5)
    assert np.isclose(rep['kinetic_energy'], 2.0, atol=1e-5)

    # 3. Controlled micro-probes unit test
    probes = evaluate_controlled_micro_probes(E1, E2)
    assert probes['arrow_inversion_symmetric'] is True
    assert probes['mean_pooling_order_invariant'] is True
    assert np.isclose(probes['static_control_1_displacement'], 0.0)

    # 4. End-to-end dyad test on temporary images
    p1 = tmp_path / "frame_0001.jpg"
    p2 = tmp_path / "frame_0002.jpg"
    img1.save(p1)
    img2.save(p2)

    report = analyze_two_frame_dyad(
        f1_path=p1,
        f2_path=p2,
        output_dir=str(tmp_path / "out"),
        figures_dir=str(tmp_path / "figs"),
        output_prefix="test_dyad"
    )
    assert 'pixel_metrics' in report
    assert 'latent_representation_metrics' in report
    assert 'controlled_micro_probes' in report
    assert (tmp_path / "figs" / "test_dyad_pair.png").exists()
    assert (tmp_path / "out" / "test_dyad_report.json").exists()




