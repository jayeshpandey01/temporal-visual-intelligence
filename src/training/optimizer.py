"""
Optimizer and learning rate scheduler construction (M3.11).
(research.md §12.5)

Configures AdamW with decoupled weight decay and CosineAnnealingLR.
Identical optimizer parameters are enforced across all model families (sequence-model-trainer).
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler


def build_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> Tuple[torch.optim.Optimizer, _LRScheduler]:
    """
    Builds AdamW optimizer with CosineAnnealingLR scheduler.
    Identical across Model A (MLP), Model B (MeanPool), and Model C (GRU).

    Args:
        model: PyTorch model with trainable parameters
        config: Experiment configuration dictionary
    Returns:
        (optimizer, scheduler)
    """
    lr = float(config['training']['lr'])
    weight_decay = float(config['training']['weight_decay'])
    epochs = int(config['training']['epochs'])

    # Filter trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0.0,
    )

    return optimizer, scheduler

