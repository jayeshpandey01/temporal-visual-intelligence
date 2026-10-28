"""
Single-Frame Baseline Model Module (Milestone M3.21).
Fulfills modular engineering structure: src/models/single_frame.py.
"""

from typing import Dict, Any
from src.models.baseline import SingleFrameMLP


def build_single_frame_model(config: Dict[str, Any]) -> SingleFrameMLP:
    """Build SingleFrameMLP from config dictionary."""
    embed_dim = config["encoder"]["embed_dim"]
    num_classes = config["dataset"]["subset_classes"]
    hidden_dim = config.get("baseline", {}).get("hidden_dim", 256)
    dropout = config.get("baseline", {}).get("dropout", 0.1)
    anchor_mode = config.get("baseline", {}).get("eval_anchor", "middle")
    train_sampling = config.get("baseline", {}).get("train_sampling", "random")

    return SingleFrameMLP(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        anchor_mode=anchor_mode,
        train_sampling=train_sampling,
    )


__all__ = ["SingleFrameMLP", "build_single_frame_model"]

