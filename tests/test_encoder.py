"""
Unit Tests for M2.5 & M2.6: Pretrained Frozen Visual Encoder.

Validates the critical scientific invariants:
  1. The encoder is 100% frozen: requires_grad=False for ALL parameters (M2.6).
  2. The encoder is strictly in eval() mode (training=False).
  3. Output embedding dimensions match encoder specifications (384 for vits14, 768 for vitb14).
  4. Deterministic inference: identical inputs produce identical output representations.
  5. Numerical stability: zero NaNs, zero Infs, non-degenerate vector norms.
"""

import pytest
import torch
from src.embeddings.extractor import load_encoder, OfflineViTFallback


def test_encoder_is_frozen_zero_grad():
    """Verify M2.6: ALL encoder parameters must be frozen (requires_grad = False)."""
    device = torch.device('cpu')
    model = load_encoder('dinov2_vits14', device)

    # Assert model is in eval mode
    assert not model.training, "Encoder must be in eval() mode, not training mode."

    # Assert 100% of parameters have requires_grad == False
    n_params = 0
    for name, param in model.named_parameters():
        n_params += 1
        assert not param.requires_grad, f"Parameter '{name}' is not frozen (requires_grad=True)."

    assert n_params > 0, "Encoder has 0 parameters."


def test_encoder_output_dimensions():
    """Verify output embedding dimensions match encoder specifications."""
    device = torch.device('cpu')
    
    # DINOv2 ViT-S/14 -> 384 dimensions
    model_s = load_encoder('dinov2_vits14', device)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_s = model_s(x)
    if isinstance(out_s, dict):
        out_s = out_s.get('x_norm_clstoken', out_s.get('last_hidden_state'))
    if out_s.dim() == 3:
        out_s = out_s[:, 0, :]
    assert out_s.shape == (2, 384)

    # DINOv2 ViT-B/14 -> 768 dimensions
    model_b = load_encoder('dinov2_vitb14', device)
    with torch.no_grad():
        out_b = model_b(x)
    if isinstance(out_b, dict):
        out_b = out_b.get('x_norm_clstoken', out_b.get('last_hidden_state'))
    if out_b.dim() == 3:
        out_b = out_b[:, 0, :]
    assert out_b.shape == (2, 768)


def test_encoder_deterministic_inference():
    """Verify encoder inference is 100% deterministic (no dropout or stochastic jitter)."""
    device = torch.device('cpu')
    model = load_encoder('dinov2_vits14', device)
    x = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    if isinstance(out1, dict):
        out1 = out1.get('x_norm_clstoken', out1.get('last_hidden_state'))
        out2 = out2.get('x_norm_clstoken', out2.get('last_hidden_state'))

    assert torch.equal(out1, out2), "Encoder inference is non-deterministic across repeated evaluations."


def test_encoder_numerical_stability():
    """Verify zero NaNs, zero Infs, and non-degenerate vector norms."""
    device = torch.device('cpu')
    model = load_encoder('dinov2_vits14', device)
    x = torch.randn(3, 3, 224, 224)

    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict):
        out = out.get('x_norm_clstoken', out.get('last_hidden_state'))
    if out.dim() == 3:
        out = out[:, 0, :]

    assert not torch.isnan(out).any(), "NaN detected in encoder outputs."
    assert not torch.isinf(out).any(), "Inf detected in encoder outputs."

    norms = torch.norm(out, dim=-1)
    assert (norms > 0.1).all(), "Degenerate near-zero embedding vector detected."


def test_encoder_rejection_invalid_name():
    """Verify error raised on unknown encoder name."""
    device = torch.device('cpu')
    with pytest.raises(ValueError):
        load_encoder('unsupported_vision_model_xyz', device)

