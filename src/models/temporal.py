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
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseSequenceModel, SequencePrediction


# ---------------------------------------------------------------------------
# Temporal Attention Pooling (Level 1 Win)
# ---------------------------------------------------------------------------
class TemporalAttentionPooling(nn.Module):
    """
    Temporal Attention Pooling layer.
    Computes learnable scalar attention scores across the sequence of hidden states:
        a_t = w_a^T tanh(W_h * h_t + b_h)
        alpha = softmax(a, dim=1)
        h_pooled = sum_{t=1}^K alpha_t * h_t

    Addresses recency forgetting by dynamically weighting key action frames.
    """
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        attn_dim = hidden_dim or max(32, in_dim // 2)
        self.proj = nn.Linear(in_dim, attn_dim)
        self.tanh = nn.Tanh()
        self.attn_vec = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            h: [B, T, D] hidden states across timesteps
            return_weights: whether to return attention weights [B, T]
        Returns:
            pooled: [B, D]
            (optional) weights: [B, T]
        """
        # h: [B, T, D]
        scores = self.attn_vec(self.tanh(self.proj(h)))  # [B, T, 1]
        weights = torch.softmax(scores, dim=1)           # [B, T, 1]
        pooled = torch.sum(h * weights, dim=1)           # [B, D]
        if return_weights:
            return pooled, weights.squeeze(-1)           # [B, D], [B, T]
        return pooled


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
        pooling: str = "last",
        bidirectional: bool = False,
    ):
        super().__init__(embed_dim=embed_dim, num_classes=num_classes)
        self.projection_dim = projection_dim or hidden_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pooling = pooling
        self.bidirectional = bidirectional

        # Input projection (aligns embedding dimension with recurrent cell)
        self.input_proj = nn.Linear(embed_dim, self.projection_dim)

        # GRU temporal encoder
        self.gru = nn.GRU(
            input_size=self.projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        if pooling == "attention":
            self.attention_pool = TemporalAttentionPooling(in_dim=gru_out_dim, hidden_dim=hidden_dim)
        else:
            self.attention_pool = None

        # Classifier head (terminal or pooled state to class logits)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, num_classes),
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

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sequential forward pass through GRU (M3.18 Progressive Observation Interface).
        Naturally accepts any observation count K >= 1: [B, K, D] or [B, D].

        Args:
            x: [B, K, D] or [B, D] — sequence of K frame embeddings
            return_attention: whether to return attention weights when pooling='attention'
        Returns:
            logits: [B, C] (or tuple (logits, weights) if return_attention=True)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] -> [B, 1, D] for single-frame ingestion
        elif x.dim() != 3:
            raise ValueError(f"TemporalGRU requires 2D [B, D] or 3D [B, K, D] tensor, got shape {x.shape}")

        # Project input embeddings
        x_proj = self.input_proj(x)        # [B, K, d_proj]

        # GRU: process chronological sequence
        out, h_n = self.gru(x_proj)        # out: [B, K, H*num_dir], h_n: [num_layers*num_dir, B, H]

        if self.pooling == "attention" and self.attention_pool is not None:
            pooled, weights = self.attention_pool(out, return_weights=True)
            logits = self.classifier(pooled)
            if return_attention:
                return logits, weights
            return logits
        elif self.pooling == "mean":
            pooled = out.mean(dim=1)
            return self.classifier(pooled)
        else:
            if self.bidirectional:
                h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                h_last = h_n[-1]                   # [B, H] — terminal hidden state
            return self.classifier(h_last)         # [B, C]

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
        breakdown = {
            "input_projection": proj,
            "temporal_core_gru": temporal_core,
            "classifier": cls,
        }
        if self.attention_pool is not None:
            attn = sum(p.numel() for p in self.attention_pool.parameters() if p.requires_grad)
            breakdown["attention_pooling"] = attn
            breakdown["total_trainable"] = proj + temporal_core + cls + attn
        else:
            breakdown["total_trainable"] = proj + temporal_core + cls
        return breakdown


# ---------------------------------------------------------------------------
# Attentive Bidirectional GRU with Contrastive Head (Level 1 Win)
# ---------------------------------------------------------------------------
class AttentiveBiGRU(BaseSequenceModel):
    """
    Level 1 Model: Bidirectional GRU with Temporal Attention Pooling & Contrastive Projection.
    Addresses:
      1. Recency forgetting: Attends across all timesteps to capture mid-sequence interactions.
      2. Arrow-of-time awareness: Bidirectional GRU reads forward (t_1 -> t_K) and backward (t_K -> t_1).
      3. Directional discriminability: Features contrastive projection head for InfoNCE training.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 128,
        projection_dim: Optional[int] = 256,
        num_layers: int = 1,
        num_classes: int = 20,
        dropout: float = 0.2,
        proj_head_dim: int = 128,
    ):
        super().__init__(embed_dim=embed_dim, num_classes=num_classes)
        self.projection_dim = projection_dim or hidden_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.proj_head_dim = proj_head_dim

        # Input projection
        self.input_proj = nn.Linear(embed_dim, self.projection_dim)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=self.projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Temporal Attention Pooling over [B, T, 2 * hidden_dim]
        self.attention_pool = TemporalAttentionPooling(in_dim=hidden_dim * 2, hidden_dim=hidden_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

        # Contrastive projection head (for InfoNCE directional training)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, proj_head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_head_dim, proj_head_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def extract_hidden_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"AttentiveBiGRU requires 3D tensor [B, T, D], got shape {x.shape}")
        x_proj = self.input_proj(x)
        out, _ = self.gru(x_proj)
        return out

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"AttentiveBiGRU requires 2D [B, D] or 3D [B, K, D] tensor, got shape {x.shape}")

        x_proj = self.input_proj(x)
        out, _ = self.gru(x_proj)          # [B, K, 2*H]
        pooled, weights = self.attention_pool(out, return_weights=True)  # pooled: [B, 2*H], weights: [B, K]
        logits = self.classifier(pooled)   # [B, C]

        if return_features:
            norm_feat = nn.functional.normalize(self.contrastive_proj(pooled), dim=-1)
            if return_attention:
                return logits, norm_feat, weights
            return logits, norm_feat

        if return_attention:
            return logits, weights

        return logits

    def predict_sequence(
        self,
        x: torch.Tensor,
        ground_truth: int = 0,
        sequence_id: str = "unknown",
        seed: int = 42,
        class_name: Optional[str] = None,
    ) -> SequencePrediction:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            raw_logits = self(x)[0].cpu().numpy()
            probs = self.predict_proba(x)[0].cpu().numpy()
            probs = probs / np.sum(probs)
            pred_class = int(np.argmax(probs))
            conf = float(probs[pred_class])

        return SequencePrediction(
            sequence_id=sequence_id,
            ground_truth=ground_truth,
            predicted_class=pred_class,
            confidence=conf,
            probabilities=[float(p) for p in probs],
            model_type="attentive_bigru",
            seed=seed,
            logits=[float(z) for z in raw_logits],
            class_name=class_name,
            metadata={
                "model": "AttentiveBiGRU",
                "hidden_dim": self.hidden_dim,
                "projection_dim": self.projection_dim,
                "num_frames": x.shape[1],
            },
        )

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_parameters_breakdown(self) -> Dict[str, int]:
        proj = sum(p.numel() for p in self.input_proj.parameters() if p.requires_grad)
        gru = sum(p.numel() for p in self.gru.parameters() if p.requires_grad)
        attn = sum(p.numel() for p in self.attention_pool.parameters() if p.requires_grad)
        cls = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        contra = sum(p.numel() for p in self.contrastive_proj.parameters() if p.requires_grad)
        return {
            "input_projection": proj,
            "temporal_core_bigru": gru,
            "temporal_attention": attn,
            "classifier": cls,
            "contrastive_head": contra,
            "total_trainable": proj + gru + attn + cls + contra,
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
        pooling = config['temporal'].get('pooling', 'last')
        bidirectional = config['temporal'].get('bidirectional', False)
        return TemporalGRU(
            num_layers=config['temporal']['num_layers'],
            projection_dim=projection_dim,
            pooling=pooling,
            bidirectional=bidirectional,
            **kwargs,
        )
    elif model_name in ('attentive_bigru', 'attentive_gru', 'bi_gru'):
        projection_dim = config['temporal'].get('projection_dim', None)
        proj_head_dim = config['temporal'].get('proj_head_dim', 128)
        return AttentiveBiGRU(
            projection_dim=projection_dim,
            num_layers=config['temporal'].get('num_layers', 1),
            proj_head_dim=proj_head_dim,
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

