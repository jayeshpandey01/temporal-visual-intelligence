"""
Early stopping and model checkpointing infrastructure (M3.12).

Enforces strict test split isolation:
- Monitors validation split metrics only (e.g. val_accuracy, val_loss, val_macro_f1).
- Test set remains untouched during training and model selection.
- Restores best validation checkpoint before final test set evaluation.
"""

import os
from typing import Dict, Any, Optional
import copy
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping handler with validation checkpoint selection.

    Tracks validation performance, saves the best model checkpoint, halts
    training when validation performance stagnates beyond `patience` epochs,
    and restores best checkpoint parameters for final evaluation.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        monitor: str = "val_accuracy",
        mode: str = "auto",
        checkpoint_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            patience: Number of epochs with no improvement after which training stops.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            monitor: Name of the metric to monitor (must start with 'val_' to prevent test leakage).
            mode: 'min' (for loss), 'max' (for accuracy/f1), or 'auto' (inferred).
            checkpoint_path: File path to save the best model weights.
            verbose: If True, prints messages when checkpoint is updated or early stop triggered.
        """
        # Strict validation leakage guard: early stopping must never monitor test metrics
        if not monitor.startswith("val_") and monitor not in ("val_loss", "val_accuracy", "val_macro_f1"):
            raise ValueError(
                f"Early stopping monitor must target validation metrics (e.g. 'val_accuracy', 'val_loss'). "
                f"Got '{monitor}'. Test split metrics must never be monitored during training."
            )

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

        # Determine mode
        if mode == "auto":
            if "loss" in monitor:
                self.mode = "min"
            else:
                self.mode = "max"
        elif mode in ("min", "max"):
            self.mode = mode
        else:
            raise ValueError(f"mode must be 'min', 'max', or 'auto', got '{mode}'")

        # Tracking state
        self.counter = 0
        self.best_epoch = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.stopped_epoch = 0
        self.total_epochs_seen = 0

    def is_better(self, current: float, best: float) -> bool:
        """Check if current score improves upon best score by at least min_delta."""
        if self.mode == "max":
            return (current - best) > self.min_delta
        else:
            return (best - current) > self.min_delta

    def __call__(
        self,
        current_score: float,
        model: nn.Module,
        epoch: int,
        checkpoint_path: Optional[str] = None,
    ) -> bool:
        """
        Evaluate current epoch score and determine whether to save checkpoint or stop.

        Args:
            current_score: Current validation metric value.
            model: PyTorch model being trained.
            epoch: Current epoch number.
            checkpoint_path: Optional override for checkpoint saving path.

        Returns:
            bool: True if training should halt (patience exhausted), False otherwise.
        """
        self.total_epochs_seen = epoch
        save_path = checkpoint_path or self.checkpoint_path

        if self.best_score is None:
            # First epoch initialization
            self.best_score = float(current_score)
            self.best_epoch = epoch
            self._save_checkpoint(model, save_path)
            if self.verbose:
                print(f"  [EarlyStopping] Epoch {epoch}: Initial best {self.monitor} = {self.best_score:.4f}")
            return False

        if self.is_better(current_score, self.best_score):
            # Significant improvement observed
            delta = abs(current_score - self.best_score)
            if self.verbose:
                print(
                    f"  [EarlyStopping] Epoch {epoch}: {self.monitor} improved from "
                    f"{self.best_score:.4f} to {current_score:.4f} (Δ={delta:.4f} > {self.min_delta}). Checkpoint saved."
                )
            self.best_score = float(current_score)
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model, save_path)
        else:
            # Stagnation
            self.counter += 1
            if self.verbose:
                print(
                    f"  [EarlyStopping] Epoch {epoch}: {self.monitor} did not improve ({current_score:.4f} vs best {self.best_score:.4f}). "
                    f"Patience: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch
                if self.verbose:
                    print(
                        f"  [EarlyStopping] Early stopping triggered at epoch {epoch}! "
                        f"Best epoch was {self.best_epoch} with {self.monitor}={self.best_score:.4f}."
                    )
                return True

        return False

    def _save_checkpoint(self, model: nn.Module, save_path: Optional[str]):
        """Save deep copy of model state in memory and optionally to disk."""
        self.best_state_dict = copy.deepcopy(model.state_dict())
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save(self.best_state_dict, save_path)

    def restore_best_weights(self, model: nn.Module, device: Optional[torch.device] = None) -> nn.Module:
        """
        Restore the model weights to the best validation checkpoint.

        Args:
            model: PyTorch model to restore.
            device: Target device (cpu, cuda).

        Returns:
            nn.Module: Restored model.
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            if device is not None:
                model.to(device)
            if self.verbose:
                print(f"  [EarlyStopping] Restored model weights from best epoch {self.best_epoch} ({self.monitor}={self.best_score:.4f}).")
            return model
        elif self.checkpoint_path and os.path.exists(self.checkpoint_path):
            state_dict = torch.load(self.checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            if device is not None:
                model.to(device)
            if self.verbose:
                print(f"  [EarlyStopping] Restored model weights from disk checkpoint {self.checkpoint_path}.")
            return model
        else:
            raise RuntimeError("No checkpoint or state dict available to restore best weights.")

    def get_summary(self) -> Dict[str, Any]:
        """Return diagnostic dictionary of the early stopping state."""
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "stopped_epoch": self.stopped_epoch if self.early_stop else self.total_epochs_seen,
            "early_stop_triggered": self.early_stop,
            "total_epochs_seen": self.total_epochs_seen,
        }

