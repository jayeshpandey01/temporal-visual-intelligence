import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.baseline import SingleFrameMLP, MultiFrameMeanPool
from src.models.temporal import TemporalGRU
from src.embeddings.extractor import load_encoder


def test_m2_12_representation_change_vs_semantic_boundary():
    # Representation change != semantic understanding
    # A spatial translation alters embeddings without changing semantic class
    e1 = np.ones(384, dtype=np.float32)
    e1 = e1 / np.linalg.norm(e1)
    
    # Simulate a spatial shift (e.g., object moving right)
    e2 = e1.copy()
    e2[:100] += 0.5
    e2 = e2 / np.linalg.norm(e2)
    
    cos_sim = float(np.dot(e1, e2))
    delta_rep = 1.0 - cos_sim
    
    assert delta_rep > 0.01
    assert cos_sim < 1.0
    # The representation changed, but the encoder alone cannot classify the temporal verb


def test_m2_13_visual_modalities_integrity():
    # Verify all 6 visual artifacts exist and are non-empty
    visual_artifacts = [
        "results/representation/similarity_plots/synthetic_action_similarity_curves.png",
        "results/representation/similarity_plots/real_50_frames_similarity_and_drift.png",
        "results/representation/similarity_plots/representative_sanity_curves.png",
        "results/representation/embedding_pca/embedding_pca_trajectories.png",
        "results/representation/temporal_strips/horse_gallop_strip_aligned.png",
        "results/representation/temporal_strips/bmw_drift_strip_aligned.png",
    ]
    for rel_path in visual_artifacts:
        path = Path(rel_path)
        assert path.exists(), f"Missing artifact: {rel_path}"
        assert path.stat().st_size > 1000, f"Artifact file too small: {rel_path}"


def test_m2_14_midpoint_ambiguity_and_velocity_resolution():
    # Test that midpoint states of opposing trajectories have high static similarity
    # while their temporal derivatives invert
    t_steps = 8
    d = 384
    
    # Static background component (representing table/room)
    bg = np.ones(d, dtype=np.float32) * 2.0
    
    # Left-to-right trajectory
    traj_fwd = np.zeros((t_steps, d), dtype=np.float32)
    for t in range(t_steps):
        traj_fwd[t] = bg.copy()
        traj_fwd[t, 0] += (t - 3.5) * 0.5  # moving right
    traj_fwd = traj_fwd / np.linalg.norm(traj_fwd, axis=1, keepdims=True)
    
    # Right-to-left trajectory (moving left)
    traj_rev = np.zeros((t_steps, d), dtype=np.float32)
    for t in range(t_steps):
        traj_rev[t] = bg.copy()
        traj_rev[t, 0] -= (t - 3.5) * 0.5  # moving left
    traj_rev = traj_rev / np.linalg.norm(traj_rev, axis=1, keepdims=True)
    
    # Midpoints (t=3 vs t=4) have high cosine similarity due to background and central position
    mid_fwd = traj_fwd[3]
    mid_rev = traj_rev[3]
    mid_sim = float(np.dot(mid_fwd, mid_rev))
    assert mid_sim > 0.99, f"Midpoint similarity should be >0.99, got {mid_sim}"
    
    # But velocity vector v_t = e_{t+1} - e_t inverts sign
    v_fwd = traj_fwd[4] - traj_fwd[3]
    v_rev = traj_rev[4] - traj_rev[3]
    
    # Check that velocity vectors are opposing
    v_sim = float(np.dot(v_fwd, v_rev) / (np.linalg.norm(v_fwd) * np.linalg.norm(v_rev) + 1e-6))
    assert v_sim < -0.90, f"Velocity vectors must oppose each other across reverse trajectories, got {v_sim}"


def test_m2_15_confound_free_parity():
    # Verify Model A and Model B parameter parity
    num_classes = 20
    embed_dim = 384
    hidden_dim = 256
    
    model_a = SingleFrameMLP(embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model_b = MultiFrameMeanPool(embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model_c = TemporalGRU(embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    
    params_a = sum(p.numel() for p in model_a.parameters())
    params_b = sum(p.numel() for p in model_b.parameters())
    
    assert params_a == params_b == 103700, f"Expected 103700 params, got A={params_a}, B={params_b}"
    
    # Verify all models accept the same shape tensor
    batch_size = 4
    n_frames = 8
    x_seq = torch.randn(batch_size, n_frames, embed_dim)
    
    out_a = model_a(x_seq[:, 0, :])
    out_b = model_b(x_seq)
    out_c = model_c(x_seq)
    
    assert out_a.shape == (batch_size, num_classes)
    assert out_b.shape == (batch_size, num_classes)
    assert out_c.shape == (batch_size, num_classes)


def test_m2_16_negative_constraints():
    # Check that encoder is frozen and requires_grad is strictly False
    device = torch.device("cpu")
    encoder = load_encoder(name="dinov2_vits14", device=device)
    for p in encoder.parameters():
        assert not p.requires_grad, "Encoder parameter must have requires_grad=False"
    assert not encoder.training, "Encoder must be in eval() mode"


def test_m2_acceptance_gate_deliverables_complete():
    # Programmatic verification of M2 acceptance deliverables
    required_files = [
        "src/embeddings/extractor.py",
        "src/embeddings/cache.py",
        "src/embeddings/validate.py",
        "src/embeddings/analysis.py",
        "configs/embedding.yaml",
        "docs/representation_analysis.md",
        "results/representation/cosine_similarity.csv",
        "results/representation/embedding_statistics.csv",
        "results/representation/cache_validation_report.json",
    ]
    for rel_path in required_files:
        path = Path(rel_path)
        assert path.exists(), f"Missing acceptance deliverable: {rel_path}"
