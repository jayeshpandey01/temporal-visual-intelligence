"""
Temporal models — GRU and Transformer encoder heads (research.md §11).

CRITICAL: Fair comparison with baseline requires IDENTICAL:
  - embed_dim (from same frozen encoder)
  - hidden_dim
  - optimizer (AdamW)
  - learning rate, weight decay
  - training seeds
  - dataset splits

The ONLY difference from baseline: input is [B, T, D] instead of [B, D].
"""

import math
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseSequenceModel, SequencePrediction


# ---------------------------------------------------------------------------
# GRU Temporal Head (primary model)
# ---------------------------------------------------------------------------
class TemporalGRU(BaseSequenceModel):
    """
    Sequence-to-label GRU classifier (Model C, M-3.md §M3.6-M3.9, research.md §11).

    Receives sequence of frozen frame representations in chronological order:
        E = [E_1, E_2, ..., E_N], E_t in R^D

    Input:  [B, T, D]  — T frame embeddings per sequence
    Output: [B, C]     — class logits

    Architecture:
      Input projection (D -> d_proj)
      -> GRU recurrence: h_t = GRU(E_t, h_{t-1})
      -> Terminal hidden state h_N = h_n[-1]
      -> Dropout
      -> Linear classification head (H -> C)

    Mathematical Formulation:
      - Affine projection: \\tilde{E}_t = W_proj * E_t + b_proj in R^{d_proj}
      - Reset gate:      r_t = \\sigma(W_ir * \\tilde{E}_t + b_ir + W_hr * h_{t-1} + b_hr)
      - Update gate:     z_t = \\sigma(W_iz * \\tilde{E}_t + b_iz + W_hz * h_{t-1} + b_hz)
      - Candidate state: \\tilde{h}_t = \\tanh(W_in * \\tilde{E}_t + b_in + r_t \\odot (W_hn * h_{t-1} + b_hn))
      - State update:    h_t = (1 - z_t) \\odot \\tilde{h}_t + z_t \\odot h_{t-1}
      - Terminal state:  h_N
      - Classification:  logits = W_cls * Dropout(h_N) + b_cls
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 256,
        projection_dim: Optional[int] = None,
        num_layers: int = 1,
        num_classes: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__(embed_dim=embed_dim, num_classes=num_classes)
        self.projection_dim = projection_dim or hidden_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Input projection (aligns embedding dimension with recurrent cell)
        self.input_proj = nn.Linear(embed_dim, self.projection_dim)

        # GRU temporal encoder
        self.gru = nn.GRU(
            input_size=self.projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classifier head (terminal state to class logits)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def extract_hidden_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the full sequence of intermediate hidden states [B, T, H].
        Used for diagnostic trajectory analysis (M1.5, M5).

        Args:
            x: [B, T, D] sequence of frame embeddings
        Returns:
            trajectory: [B, T, H] hidden states across all timesteps
        """
        if x.dim() != 3:
            raise ValueError(f"TemporalGRU requires 3D tensor [B, T, D], got shape {x.shape}")
        x_proj = self.input_proj(x)
        out, _ = self.gru(x_proj)
        return out

    def verify_causal_encapsulation(self, x: torch.Tensor, t: int = 3, atol: float = 1e-6) -> bool:
        """
        Verify the causal encapsulation property of the recurrent architecture (M3.7 Pillar 1).

        Asserts that hidden state h_t is strictly a function of past/present observations
        {E_1, ..., E_t} and has zero dependence on future timesteps {E_{t+1}, ..., E_T}.
        Mutating future frames E_{>t} must yield identical h_t.

        Args:
            x: [B, T, D] sequence of frame embeddings
            t: Pivot timestep to verify (0 <= t < T-1)
            atol: Absolute tolerance
        Returns:
            bool: True if h_t is strictly invariant to future frame alterations
        """
        if x.dim() != 3:
            raise ValueError(f"TemporalGRU requires 3D tensor [B, T, D], got shape {x.shape}")
        B, T, _ = x.shape
        if not (0 <= t < T - 1):
            raise ValueError(f"Timestep t={t} must satisfy 0 <= t < T-1={T-1}")

        self.eval()
        with torch.no_grad():
            orig_traj = self.extract_hidden_trajectory(x)
            h_t_orig = orig_traj[:, t, :]

            x_mutated = x.clone()
            x_mutated[:, t + 1 :, :] = torch.randn_like(x_mutated[:, t + 1 :, :]) * 10.0

            mutated_traj = self.extract_hidden_trajectory(x_mutated)
            h_t_mutated = mutated_traj[:, t, :]

            return bool(torch.allclose(h_t_orig, h_t_mutated, atol=atol))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sequential forward pass through GRU (M3.18 Progressive Observation Interface).
        Naturally accepts any observation count K >= 1: [B, K, D] or [B, D].

        Args:
            x: [B, K, D] or [B, D] — sequence of K frame embeddings
        Returns:
            logits: [B, C]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] -> [B, 1, D] for single-frame ingestion
        elif x.dim() != 3:
            raise ValueError(f"TemporalGRU requires 2D [B, D] or 3D [B, K, D] tensor, got shape {x.shape}")

        # Project input embeddings
        x_proj = self.input_proj(x)        # [B, K, d_proj]

        # GRU: process chronological sequence
        _, h_n = self.gru(x_proj)          # h_n: [num_layers, B, H]
        h_last = h_n[-1]                   # [B, H] — terminal hidden state

        return self.classifier(h_last)     # [B, C]

    def predict_sequence(
        self,
        x: torch.Tensor,
        ground_truth: int = 0,
        sequence_id: str = "unknown",
        seed: int = 42,
        class_name: Optional[str] = None,
    ) -> SequencePrediction:
        """
        High-level sequence prediction contract returning SequencePrediction dataclass.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            raw_logits = self(x)[0].cpu().numpy()
            probs = self.predict_proba(x)[0].cpu().numpy()
            # Normalize to guarantee float sum == 1.0 within tolerance
            probs = probs / np.sum(probs)
            pred_class = int(np.argmax(probs))
            conf = float(probs[pred_class])

        return SequencePrediction(
            sequence_id=sequence_id,
            ground_truth=ground_truth,
            predicted_class=pred_class,
            confidence=conf,
            probabilities=[float(p) for p in probs],
            model_type="gru",
            seed=seed,
            logits=[float(z) for z in raw_logits],
            class_name=class_name,
            metadata={
                "model": "TemporalGRU",
                "hidden_dim": self.hidden_dim,
                "projection_dim": self.projection_dim,
                "num_frames": x.shape[1],
            },
        )

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_parameters_breakdown(self) -> Dict[str, int]:
        """Detailed parameter audit breakdown (M3.10)."""
        proj = sum(p.numel() for p in self.input_proj.parameters() if p.requires_grad)
        temporal_core = sum(p.numel() for p in self.gru.parameters() if p.requires_grad)
        cls = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        return {
            "input_projection": proj,
            "temporal_core_gru": temporal_core,
            "classifier": cls,
            "total_trainable": proj + temporal_core + cls,
        }


# ---------------------------------------------------------------------------
# Transformer Temporal Head (alternative)
# ---------------------------------------------------------------------------
class TemporalTransformer(BaseSequenceModel):
    """
    Transformer encoder for temporal sequence modeling.

    Input:  [B, T, D]  — T frame embeddings per sample
    Output: [B, C]     — class logits

    Architecture:
      Input projection → Positional Encoding → TransformerEncoder
      → Mean pooling (or CLS token) → Classifier
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 20,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__(embed_dim=embed_dim, num_classes=num_classes)
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        # CLS token for sequence-level representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] — sequence of T frame embeddings
        Returns:
            logits: [B, C]
        """
        B, T, _ = x.shape

        # Project
        x = self.input_proj(x)           # [B, T, H]

        # Add positional encoding
        x = x + self.pos_embedding[:, :T, :]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, H]
        x = torch.cat([cls, x], dim=1)           # [B, T+1, H]

        # Transformer
        x = self.transformer(x)          # [B, T+1, H]

        # Use CLS token for classification
        cls_output = x[:, 0, :]          # [B, H]

        return self.classifier(cls_output)  # [B, C]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_temporal_model(config: Dict[str, Any]) -> nn.Module:
    """Build temporal model from config."""
    model_name = config['temporal']['model']
    kwargs = dict(
        embed_dim=config['encoder']['embed_dim'],
        hidden_dim=config['temporal']['hidden_dim'],
        num_classes=config['dataset']['subset_classes'],
        dropout=config['temporal']['dropout'],
    )

    if model_name == 'gru':
        projection_dim = config['temporal'].get('projection_dim', None)
        return TemporalGRU(
            num_layers=config['temporal']['num_layers'],
            projection_dim=projection_dim,
            **kwargs,
        )
    elif model_name == 'compact_gru':
        projection_dim = config['temporal'].get('projection_dim', 64)
        hidden_dim = config['temporal'].get('hidden_dim', 130)
        return TemporalGRU(
            embed_dim=config['encoder']['embed_dim'],
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            num_layers=config['temporal'].get('num_layers', 1),
            num_classes=config['dataset']['subset_classes'],
            dropout=config['temporal']['dropout'],
        )
    elif model_name == 'transformer':
        return TemporalTransformer(
            num_heads=config['temporal']['num_heads'],
            num_layers=config['temporal']['num_layers'],
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown temporal model: {model_name}. Choose 'gru' or 'transformer'.")

