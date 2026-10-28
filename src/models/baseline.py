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
from typing import Dict, Any, Optional, List, Union
import numpy as np

from src.models.base import BaseSequenceModel


class SingleFrameMLP(BaseSequenceModel):
    """
    Single-frame classification head (Model A, research.md §10, M-3.md §M3.2).

    Input: embedding of shape [B, D] (one frame per sample), OR
           [B, T, D] sequence (per-sample random frame during train, anchor frame during eval)
    Output: logits of shape [B, C]
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 256,
        num_classes: int = 20,
        dropout: float = 0.1,
        anchor_mode: Union[str, int] = 'middle',
        train_sampling: str = 'random',
    ):
        super().__init__(embed_dim=embed_dim, num_classes=num_classes)
        self.hidden_dim = hidden_dim
        self.anchor_mode = anchor_mode
        self.train_sampling = train_sampling

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

    def forward(self, x: torch.Tensor, anchor: Optional[Union[str, int]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, D] — single frame embedding, OR
               [B, T, D] — sequence (random frame during train, anchor frame during eval)
            anchor: optional override ('first' | 'middle' | 'last' | 'random' | int timestep)
        Returns:
            logits: [B, C]
        """
        if x.dim() not in (2, 3):
            raise ValueError(f"SingleFrameMLP requires 2D [B, D] or 3D [B, T, D] tensor, got shape {x.shape}")

        if x.dim() == 3:
            B, T, _ = x.shape
            active_anchor = anchor if anchor is not None else self.anchor_mode

            if self.training and anchor is None:
                if self.train_sampling == 'random':
                    # Approach B: Per-sample independent random frame sampling (M3.3)
                    t_idx = torch.randint(0, T, (B,), device=x.device)
                    x = x[torch.arange(B, device=x.device), t_idx, :]
                else:
                    # Approach A: Fixed frame during training
                    if isinstance(active_anchor, int):
                        idx = max(0, min(active_anchor, T - 1))
                        x = x[:, idx, :]
                    elif active_anchor == 'first':
                        x = x[:, 0, :]
                    elif active_anchor == 'last':
                        x = x[:, -1, :]
                    else:  # 'middle' (default)
                        mid = T // 2
                        x = x[:, mid, :]
            elif isinstance(active_anchor, int):
                # Integer anchor index clamped to valid timestep range [0, T-1]
                idx = max(0, min(active_anchor, T - 1))
                x = x[:, idx, :]
            elif active_anchor == 'first':
                x = x[:, 0, :]
            elif active_anchor == 'last':
                x = x[:, -1, :]
            elif active_anchor == 'random':
                t_idx = torch.randint(0, T, (B,), device=x.device)
                x = x[torch.arange(B, device=x.device), t_idx, :]
            else:  # 'middle' (default)
                mid = T // 2
                x = x[:, mid, :]
        return self.classifier(x)


class MultiFrameMeanPool(BaseSequenceModel):
    """
    Model B: Order-Invariant Multi-Frame Mean-Pooling Baseline (research.md §10, M-3.md §M3.4).
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
        super().__init__(embed_dim=embed_dim, num_classes=num_classes)
        self.hidden_dim = hidden_dim

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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, D] or [B, T, D]
        Returns:
            logits: [B, C]
        """
        if x.dim() not in (2, 3):
            raise ValueError(f"MultiFrameMeanPool requires 2D [B, D] or 3D [B, T, D] tensor, got shape {x.shape}")
        if x.dim() == 3:
            x = x.mean(dim=1)  # [B, T, D] -> [B, D] (Order-invariant temporal mean)
        return self.classifier(x)


class MajorityClassBaseline:
    """
    Trivial baseline that always predicts the most frequent class (M3.23).
    Establishes the floor — temporal model must beat this convincingly.
    """

    def __init__(self, num_classes: int = 20):
        self.num_classes = num_classes
        self.majority_class = None
        self.class_frequencies = None

    def fit(self, labels: List[int]):
        from collections import Counter
        counts = Counter(labels)
        self.majority_class = counts.most_common(1)[0][0]
        total = len(labels)
        self.class_frequencies = np.zeros(self.num_classes, dtype=np.float32)
        for c, count in counts.items():
            if c < self.num_classes:
                self.class_frequencies[c] = count / total

    def predict(self, n_samples: int) -> List[int]:
        if self.majority_class is None:
            raise ValueError("Baseline must be fit before predict().")
        return [self.majority_class] * n_samples

    def predict_proba(self, n_samples: int) -> np.ndarray:
        """Output empirical class distribution vector on simplex Delta^(C-1)."""
        if self.class_frequencies is None:
            raise ValueError("Baseline must be fit before predict_proba().")
        return np.tile(self.class_frequencies, (n_samples, 1))

    def accuracy(self, labels: List[int]) -> float:
        preds = self.predict(len(labels))
        return sum(p == l for p, l in zip(preds, labels)) / len(labels)


def build_baseline(config: Dict[str, Any], model_type: str = 'baseline') -> nn.Module:
    """Build baseline model from config (SingleFrameMLP or MultiFrameMeanPool)."""
    baseline_cfg = config.get('baseline', {})
    hidden_dim = baseline_cfg.get('hidden_dim', 256)
    dropout = baseline_cfg.get('dropout', 0.1)

    if model_type == 'meanpool':
        return MultiFrameMeanPool(
            embed_dim=config['encoder']['embed_dim'],
            hidden_dim=hidden_dim,
            num_classes=config['dataset']['subset_classes'],
            dropout=dropout,
        )
    train_sampling = baseline_cfg.get('train_sampling', 'random')
    eval_anchor = baseline_cfg.get('eval_anchor', 'middle')
    return SingleFrameMLP(
        embed_dim=config['encoder']['embed_dim'],
        hidden_dim=hidden_dim,
        num_classes=config['dataset']['subset_classes'],
        dropout=dropout,
        anchor_mode=eval_anchor,
        train_sampling=train_sampling,
    )


@torch.no_grad()
def evaluate_anchor_sweep(
    model: SingleFrameMLP,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    seq_len: int = 8,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluates SingleFrameMLP across all discrete frame anchor indices t in {0, ..., seq_len-1}.
    Provides empirical proof of whether task information is concentrated at early, middle, or late timesteps.
    (M-3.md §M3.3)
    """
    model.eval()
    results = {}
    for t in range(seq_len):
        all_preds, all_labels = [], []
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            logits = model(embeddings, anchor=t)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)
        acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        results[t] = {"accuracy": acc, "n_samples": len(all_labels)}
    return results

