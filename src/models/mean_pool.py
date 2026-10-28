"""
Multi-Frame Mean-Pooling Control Model Module (Milestone M3.21).
Fulfills modular engineering structure: src/models/mean_pool.py.
"""

from typing import Dict, Any
from src.models.baseline import MultiFrameMeanPool


def build_mean_pool_model(config: Dict[str, Any]) -> MultiFrameMeanPool:
    """Build MultiFrameMeanPool from config dictionary."""
    embed_dim = config["encoder"]["embed_dim"]
    num_classes = config["dataset"]["subset_classes"]
    hidden_dim = config.get("baseline", {}).get("hidden_dim", 256)
    dropout = config.get("baseline", {}).get("dropout", 0.1)

    return MultiFrameMeanPool(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )


__all__ = ["MultiFrameMeanPool", "build_mean_pool_model"]

