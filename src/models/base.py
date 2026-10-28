"""
Base abstraction and prediction contract for sequence models (M-3.md §M3.1, research.md §9.5).

Freezes the unit of prediction:
    one sequence -> one label
    y_hat = f(sequence)
    output: probability vector p in Delta^(C-1)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn


@dataclass
class SequencePrediction:
    """
    Standardized per-sequence prediction record (M3.1 & M3.16).

    Guarantees consistent data representation across:
      - Single-frame baseline (Model A)
      - Mean-pooling control (Model B)
      - Temporal GRU (Model C)
      - Downstream Win/Loss Quadrant Miner (temporal-win-loss-miner)
      - Qualitative VLM probes (vlm-semantic-change-probe)

    Attributes:
        sequence_id: Identifier of the sequence (e.g. 'vid_01_0000').
        ground_truth: Ground truth integer class index in {0, ..., C-1}.
        predicted_class: Predicted integer class index in {0, ..., C-1}.
        confidence: Maximum class probability max_c p(class_c) in [0.0, 1.0].
        probabilities: Full probability simplex vector [p_1, ..., p_C], sum(p) == 1.
        model_type: Name of the model ('baseline', 'meanpool', 'gru', 'transformer').
        seed: Random seed used during training or evaluation.
        logits: Raw unnormalized logits [z_1, ..., z_C].
        class_name: Human-readable name of ground truth class (optional).
        predicted_class_name: Human-readable name of predicted class (optional).
        metadata: Arbitrary run metadata (e.g., num_frames, perturbation mode).
    """
    sequence_id: str
    ground_truth: int
    predicted_class: int
    confidence: float
    probabilities: List[float]
    model_type: str
    seed: int
    logits: Optional[List[float]] = None
    class_name: Optional[str] = None
    predicted_class_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate probability simplex axioms and consistency."""
        if not self.sequence_id or not isinstance(self.sequence_id, str):
            raise ValueError(f"Invalid sequence_id: {self.sequence_id}")
        if self.ground_truth < 0:
            raise ValueError(f"Ground truth label must be non-negative, got {self.ground_truth}")
        if self.predicted_class < 0:
            raise ValueError(f"Predicted class must be non-negative, got {self.predicted_class}")
        if not (0.0 <= self.confidence <= 1.0001):
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")
        if not self.probabilities:
            raise ValueError("Probabilities list cannot be empty.")

        # Non-negativity check
        for i, p in enumerate(self.probabilities):
            if p < -1e-6:
                raise ValueError(f"Negative probability at index {i}: {p}")

        # Normalization check: sum(p) == 1.0
        prob_sum = float(sum(self.probabilities))
        if abs(prob_sum - 1.0) > 1e-3:
            raise ValueError(f"Probabilities must sum to 1.0, got {prob_sum:.6f}")

        # Argmax check
        max_idx = int(np.argmax(self.probabilities))
        if max_idx != self.predicted_class:
            raise ValueError(
                f"predicted_class ({self.predicted_class}) does not match argmax(probabilities) ({max_idx})"
            )

    @property
    def is_correct(self) -> bool:
        """Binary correctness indicator."""
        return self.predicted_class == self.ground_truth

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "ground_truth": self.ground_truth,
            "predicted_class": self.predicted_class,
            "confidence": round(float(self.confidence), 6),
            "correct": self.is_correct,
            "probabilities": [round(float(p), 6) for p in self.probabilities],
            "logits": [round(float(z), 6) for z in self.logits] if self.logits else None,
            "model_type": self.model_type,
            "seed": self.seed,
            "class_name": self.class_name,
            "predicted_class_name": self.predicted_class_name,
            "metadata": self.metadata,
        }


class BaseSequenceModel(nn.Module, ABC):
    """
    Abstract base class for all sequence models in M3.

    Enforces the frozen prediction task contract (M3.1):
      Inputs:
        x: Tensor of shape [B, T, D] (or [B, D] for single-frame representations)
      Outputs:
        forward(x): Logits of shape [B, C]
        predict_proba(x): Class probability distributions on the simplex Delta^(C-1)
        predict(x): Discrete class predictions in {0, ..., C-1}
    """

    def __init__(self, embed_dim: int = 384, num_classes: int = 20):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute unnormalized class logits for the sequence batch.

        Args:
            x: Tensor of shape [B, T, D] (or [B, D])
        Returns:
            logits: Tensor of shape [B, C]
        """
        raise NotImplementedError

    def predict_proba(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute class probability vector p in Delta^(C-1) via softmax.

        Args:
            x: Tensor of shape [B, T, D] or [B, D]
        Returns:
            probabilities: Tensor of shape [B, C] with sum_c p_c = 1 and p_c >= 0
        """
        logits = self.forward(x, **kwargs)
        return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute discrete class prediction y_hat = argmax_c p_c.

        Args:
            x: Tensor of shape [B, T, D] or [B, D]
        Returns:
            predictions: LongTensor of shape [B]
        """
        logits = self.forward(x, **kwargs)
        return torch.argmax(logits, dim=-1)

    def forward_progressive(
        self,
        x: torch.Tensor,
        k: int,
        mode: str = "uniform",
        **kwargs,
    ) -> torch.Tensor:
        """
        Progressive observation forward interface (M3.18).
        Dynamically subsamples K observations from sequence x [B, T, D]
        and computes class logits without model modifications.

        Args:
            x: Input tensor of shape [B, T, D] or [T, D].
            k: Number of temporal observations (N = 1, 2, 4, 8, ...).
            mode: Progressive sampling mode ('uniform', 'prefix', 'last_n', etc.).

        Returns:
            logits: Output logits of shape [B, C].
        """
        from src.data.progressive import sample_progressive_frames
        x_k = sample_progressive_frames(x, k=k, mode=mode)
        return self.forward(x_k, **kwargs)

    def n_parameters(self) -> int:
        """Return total count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> Dict[str, int]:
        """
        Report parameter distribution for research fairness comparisons (M3.10).
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": trainable + frozen,
        }

