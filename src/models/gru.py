"""
Temporal GRU Model Module (Milestone M3.21).
Fulfills modular engineering structure: src/models/gru.py.
"""

from typing import Dict, Any
from src.models.temporal import TemporalGRU


def build_gru_model(config: Dict[str, Any]) -> TemporalGRU:
    """Build TemporalGRU from config dictionary."""
    embed_dim = config["encoder"]["embed_dim"]
    num_classes = config["dataset"]["subset_classes"]
    hidden_dim = config["temporal"]["hidden_dim"]
    proj_dim = config["temporal"].get("projection_dim", hidden_dim)
    num_layers = config["temporal"].get("num_layers", 1)
    dropout = config["temporal"].get("dropout", 0.2)

    return TemporalGRU(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        projection_dim=proj_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )


__all__ = ["TemporalGRU", "build_gru_model"]

