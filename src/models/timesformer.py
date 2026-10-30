"""
Divided Space-Time Attention (TimeSformer) for Spatio-Temporal Patch Modeling (Level 2 Upgrade).
Paper: Bertasius et al., "Is Space-Time Attention All You Need for Video Understanding?" (ICML 2021).

Processes full [B, T, P, D] spatial patch token grids from DINOv2 (14x14 = 196 patches):
  1. Temporal Attention across T frames for each spatial patch (tracks motion/trajectories).
  2. Spatial Attention across P patches for each frame (locates hand/object micro-interactions).
"""

from typing import Optional, Dict, Any, Union, Tuple
import math
import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseSequenceModel, SequencePrediction
from src.models.temporal import TemporalAttentionPooling


class DividedSpaceTimeBlock(nn.Module):
    """
    Divided Space-Time Attention Block.
    Applies decoupled temporal attention across T frames followed by
    spatial attention across P patches within each frame.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 16,
        max_patches: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Positional encodings
        self.time_pos = nn.Parameter(torch.zeros(1, max_frames, 1, embed_dim))
        self.space_pos = nn.Parameter(torch.zeros(1, 1, max_patches, embed_dim))

        # Temporal attention
        self.norm_time = nn.LayerNorm(embed_dim)
        self.attn_time = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Spatial attention
        self.norm_space = nn.LayerNorm(embed_dim)
        self.attn_space = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.norm_mlp = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.space_pos, std=0.02)

    def forward(self, x: torch.Tensor, add_pos: bool = True) -> torch.Tensor:
        """
        Args:
            x: [B, T, P, D] spatial-temporal patch token tensor
            add_pos: whether to add space-time positional embeddings
        Returns:
            out: [B, T, P, D]
        """
        B, T, P, D = x.shape

        if add_pos:
            x = x + self.time_pos[:, :T, :, :] + self.space_pos[:, :, :P, :]

        # 1. Temporal Attention: across T frames for each spatial patch
        x_t = x.permute(0, 2, 1, 3).reshape(B * P, T, D)
        norm_x_t = self.norm_time(x_t)
        attn_t, _ = self.attn_time(norm_x_t, norm_x_t, norm_x_t)
        x_t = x_t + attn_t
        x = x_t.reshape(B, P, T, D).permute(0, 2, 1, 3)

        # 2. Spatial Attention: across P patches within each temporal frame
        x_s = x.reshape(B * T, P, D)
        norm_x_s = self.norm_space(x_s)
        attn_s, _ = self.attn_space(norm_x_s, norm_x_s, norm_x_s)
        x_s = x_s + attn_s
        x = x_s.reshape(B, T, P, D)

        # 3. Feed-forward network
        x = x + self.mlp(self.norm_mlp(x))
        return x


class TimeSformerModel(BaseSequenceModel):
    """
    TimeSformer model for Video Action Recognition.
    Processes full [B, T, P, D] patch tokens with Divided Space-Time Attention
    and Temporal Attention Pooling over space-time representations.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 20,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_frames: int = 16,
        max_patches: int = 256,
    ):
        super().__init__(embed_dim=embed_dim, num_classes=num_classes)
        self.max_frames = max_frames
        self.max_patches = max_patches

        self.blocks = nn.ModuleList([
            DividedSpaceTimeBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_frames=max_frames,
                max_patches=max_patches,
            )
            for _ in range(num_layers)
        ])

        # Recurrent temporal head with attention pooling over spatial representations
        self.temporal_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_pool = TemporalAttentionPooling(in_dim=embed_dim * 2, hidden_dim=embed_dim)

        self.norm = nn.LayerNorm(embed_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, P, D] 4D patch tensor, or [B, T, D] 3D sequence tensor
               (if 3D, treats P=1 as sequence of global tokens)
        Returns:
            logits: [B, C]
        """
        if x.dim() == 3:
            # [B, T, D] -> [B, T, 1, D]
            x = x.unsqueeze(2)
        elif x.dim() != 4:
            raise ValueError(f"TimeSformer requires 3D [B, T, D] or 4D [B, T, P, D], got {x.shape}")

        x = x.float()

        for i, block in enumerate(self.blocks):
            x = block(x, add_pos=(i == 0))

        # Spatial pooling across P patches: [B, T, P, D] -> [B, T, D]
        z_t = x.mean(dim=2)

        # Temporal bidirectional recurrence & attention pooling
        out, _ = self.temporal_gru(z_t)   # [B, T, 2 * D]
        pooled = self.temporal_pool(out)  # [B, 2 * D]

        return self.classifier(self.norm(pooled))  # [B, C]

    def predict_sequence(
        self,
        x: torch.Tensor,
        ground_truth: int = 0,
        sequence_id: str = "unknown",
        seed: int = 42,
        class_name: Optional[str] = None,
    ) -> SequencePrediction:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 3 and x.size(0) != 1:
            x = x.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            raw_logits = self(x)[0].cpu().numpy()
            probs = self.predict_proba(x)[0].cpu().numpy()
            probs = probs / np.sum(probs)
            pred_class = int(np.argmax(probs))
            conf = float(probs[pred_class])

        return SequencePrediction(
            sequence_id=sequence_id,
            ground_truth=ground_truth,
            predicted_class=pred_class,
            confidence=conf,
            probabilities=[float(p) for p in probs],
            model_type="timesformer",
            seed=seed,
            logits=[float(z) for z in raw_logits],
            class_name=class_name,
            metadata={
                "model": "TimeSformer",
                "embed_dim": self.embed_dim,
                "num_classes": self.num_classes,
            },
        )

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

