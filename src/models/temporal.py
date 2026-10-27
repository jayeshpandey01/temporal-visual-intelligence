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
import torch
import torch.nn as nn
from typing import Dict, Any


# ---------------------------------------------------------------------------
# GRU Temporal Head (primary model)
# ---------------------------------------------------------------------------
class TemporalGRU(nn.Module):
    """
    Sequence-to-label GRU classifier.

    Input:  [B, T, D]  — T frame embeddings per sample
    Output: [B, C]     — class logits

    Architecture:
      Input projection → GRU → Last hidden state → Dropout → Classifier
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 1,
        num_classes: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Optional input projection (aligns with baseline MLP's first layer)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # GRU temporal encoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classifier head (same capacity as baseline)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] — sequence of T frame embeddings
        Returns:
            logits: [B, C]
        """
        # Project input
        x = self.input_proj(x)             # [B, T, H]

        # GRU: process sequence
        _, h_n = self.gru(x)               # h_n: [num_layers, B, H]
        h_last = h_n[-1]                   # [B, H] — last hidden state

        return self.classifier(h_last)     # [B, C]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Transformer Temporal Head (alternative)
# ---------------------------------------------------------------------------
class TemporalTransformer(nn.Module):
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
        super().__init__()
        self.embed_dim = embed_dim
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
        return TemporalGRU(num_layers=config['temporal']['num_layers'], **kwargs)
    elif model_name == 'transformer':
        return TemporalTransformer(
            num_heads=config['temporal']['num_heads'],
            num_layers=config['temporal']['num_layers'],
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown temporal model: {model_name}. Choose 'gru' or 'transformer'.")

