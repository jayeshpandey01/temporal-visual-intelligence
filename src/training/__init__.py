"""
Training infrastructure package for Temporal Visual Intelligence (M3.11, M3.12, M3.13).
Exports:
  - build_optimizer
  - build_criterion
  - compute_class_weights
  - audit_class_distribution
  - EarlyStopping
  - RunTracker
  - build_run_config
"""

from src.training.optimizer import build_optimizer
from src.training.losses import (
    build_criterion,
    compute_class_weights,
    audit_class_distribution,
    DirectionalContrastiveLoss,
    CombinedClassificationLoss,
)
from src.training.early_stopping import EarlyStopping
from src.training.reproducibility import RunTracker, build_run_config

__all__ = [
    "build_optimizer",
    "build_criterion",
    "compute_class_weights",
    "audit_class_distribution",
    "DirectionalContrastiveLoss",
    "CombinedClassificationLoss",
    "EarlyStopping",
    "RunTracker",
    "build_run_config",
]
