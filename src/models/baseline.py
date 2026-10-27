"""
Single-frame baseline model (research.md §10).

Architecture:
  Embedding(D) → Linear(D→hidden) → ReLU → Dropout → Linear(hidden→C)

Frame selection strategy:
  - During training: random frame from sequence
  - During evaluation: middle frame (deterministic)
  This is documented and consistent.

IMPORTANT: This is the comparison baseline. All temporal models must use
the SAME optimizer, LR, weight decay, and seeds.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class SingleFrameMLP(nn.Module):
    """
    Single-frame classification head.

    Input: embedding of shape [B, D] (one frame per sample)
    Output: logits of shape [B, C]
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 256,
        num_classes: int = 20,
        dropout: float = 0.1,
        anchor_mode: str = 'middle',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.anchor_mode = anchor_mode

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, anchor: Optional[str] = None) -> torch.Tensor:
        """
        Args:
            x: [B, D] — single frame embedding, OR
               [B, T, D] — sequence (random frame during train, anchor frame during eval)
            anchor: optional override ('first' | 'middle' | 'last' | 'random')
        Returns:
            logits: [B, C]
        """
        if x.dim() == 3:
            active_anchor = anchor or self.anchor_mode
            if self.training and anchor is None:
                t_idx = torch.randint(0, x.shape[1], (1,)).item()
                x = x[:, t_idx, :]
            elif active_anchor == 'first':
                x = x[:, 0, :]
            elif active_anchor == 'last':
                x = x[:, -1, :]
            elif active_anchor == 'random':
                t_idx = torch.randint(0, x.shape[1], (1,)).item()
                x = x[:, t_idx, :]
            else:  # 'middle' (default)
                mid = x.shape[1] // 2
                x = x[:, mid, :]
        return self.classifier(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiFrameMeanPool(nn.Module):
    """
    Model B: Order-Invariant Multi-Frame Mean-Pooling Baseline (research.md §10).
    Averages frame embeddings across time: e_mean = (1/T) * sum(E_t).
    Shares identical parameter capacity with SingleFrameMLP.
    Isolates observation count gains from chronological ordering gains.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 256,
        num_classes: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] or [B, T, D]
        Returns:
            logits: [B, C]
        """
        if x.dim() == 3:
            x = x.mean(dim=1)  # [B, T, D] -> [B, D] (Order-invariant temporal mean)
        return self.classifier(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MajorityClassBaseline:
    """
    Trivial baseline that always predicts the most frequent class.
    Establishes the floor — temporal model must beat this convincingly.
    """

    def __init__(self):
        self.majority_class = None

    def fit(self, labels):
        from collections import Counter
        self.majority_class = Counter(labels).most_common(1)[0][0]

    def predict(self, n_samples: int):
        return [self.majority_class] * n_samples

    def accuracy(self, labels) -> float:
        preds = self.predict(len(labels))
        return sum(p == l for p, l in zip(preds, labels)) / len(labels)


def build_baseline(config: Dict[str, Any], model_type: str = 'baseline') -> nn.Module:
    """Build baseline model from config (SingleFrameMLP or MultiFrameMeanPool)."""
    if model_type == 'meanpool':
        return MultiFrameMeanPool(
            embed_dim=config['encoder']['embed_dim'],
            hidden_dim=config['temporal']['hidden_dim'],
            num_classes=config['dataset']['subset_classes'],
            dropout=config['temporal']['dropout'],
        )
    return SingleFrameMLP(
        embed_dim=config['encoder']['embed_dim'],
        hidden_dim=config['temporal']['hidden_dim'],
        num_classes=config['dataset']['subset_classes'],
        dropout=config['temporal']['dropout'],
    )

