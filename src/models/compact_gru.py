"""
Capacity-Matched Compact Temporal GRU Module (Milestone M4).
Engineered to achieve EXACT parameter parity (103,700 parameters)
with Model A (SingleFrameMLP) and Model B (MultiFrameMeanPool).

Architecture:
- embed_dim: 384
- projection_dim: 64
- hidden_dim: 130
- num_classes: 20
Parameters:
- input_proj: Linear(384, 64) -> 384 * 64 + 64 = 24,640
- gru: GRU(64, 130) -> 3 * (64*130 + 130*130 + 130 + 130) = 76,440
- classifier: Linear(130, 20) -> 130 * 20 + 20 = 2,620
Total: 24,640 + 76,440 + 2,620 = 103,700 parameters (exact match)
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from src.models.temporal import TemporalGRU


class CompactTemporalGRU(TemporalGRU):
    """
    Capacity-matched order-aware temporal model.
    Inherits TemporalGRU and locks projection_dim=64 and hidden_dim=130
    to guarantee exact 103,700 parameter parity with Model A and Model B.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 130,
        projection_dim: int = 64,
        num_layers: int = 1,
        num_classes: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )


def build_compact_gru_model(config: Dict[str, Any]) -> CompactTemporalGRU:
    """Build CompactTemporalGRU from configuration dictionary."""
    embed_dim = config["encoder"]["embed_dim"]
    num_classes = config["dataset"]["subset_classes"]
    dropout = config["temporal"].get("dropout", 0.1)

    return CompactTemporalGRU(
        embed_dim=embed_dim,
        hidden_dim=130,
        projection_dim=64,
        num_layers=1,
        num_classes=num_classes,
        dropout=dropout,
    )


__all__ = ["CompactTemporalGRU", "build_compact_gru_model"]

