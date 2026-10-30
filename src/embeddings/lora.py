"""
Low-Rank Adaptation (LoRA) for Pretrained Visual Backbones (Level 2 Upgrade).
Paper: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022).

Enables parameter-efficient fine-tuning of frozen DINOv2 / ViT encoders on
egocentric video dynamics (motion blur, hand occlusions, rapid viewpoint shifts).
Trainable parameters: <0.5% of backbone (~110k params).
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA adapter wrapped around a frozen nn.Linear layer.
        h = W_0 * x + (alpha / r) * B * A * x
    where:
        W_0 in R^{d_out x d_in} is frozen
        A in R^{r x d_in} is initialized with Kaiming uniform
        B in R^{d_out x r} is initialized to zeros
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False

        # Freeze the base layer
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        # Low-rank adapter matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        # Kaiming uniform for A, zero for B ensures delta W is initially 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_linear(x)

        base_out = self.base_linear(x)
        # Low rank path: (x @ A.T) @ B.T * scaling
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        """Merge LoRA weights into base linear layer for zero-latency inference."""
        if self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base_linear.weight.data += delta
        self.merged = True

    def unmerge_weights(self):
        """Unmerge LoRA weights back out of base linear layer."""
        if not self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base_linear.weight.data -= delta
        self.merged = False


def apply_lora_to_dinov2(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_keywords: Tuple[str, ...] = ("qkv", "proj", "q_proj", "v_proj"),
) -> Dict[str, Any]:
    """
    Attaches LoRA adapters to Query/Value/Projection linear layers of DINOv2 / ViT.
    Freezes all base parameters and marks only LoRA parameters as trainable.

    Args:
        model: Pretrained visual backbone (e.g. dinov2_vits14 or OfflineViTFallback)
        rank: Low-rank bottleneck dimension (default: 8)
        alpha: LoRA scaling factor (default: 16.0)
        dropout: LoRA input dropout (default: 0.0)
        target_keywords: Substrings identifying target linear projections
    Returns:
        audit_dict: Parameter breakdown of frozen vs trainable parameters
    """
    # 1. Freeze entire model
    for p in model.parameters():
        p.requires_grad = False

    replaced_count = 0

    # 2. Walk named submodules
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                # Check if matches target keywords
                matches = any(kw in child_name.lower() or kw in name.lower() for kw in target_keywords)
                if matches:
                    lora_layer = LoRALinear(
                        base_linear=child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    setattr(module, child_name, lora_layer)
                    replaced_count += 1

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0.0

    return {
        "replaced_layers": replaced_count,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "total_parameters": total_params,
        "trainable_percent": round(trainable_pct, 4),
        "rank": rank,
        "alpha": alpha,
    }


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Retrieve list of trainable LoRA parameters for optimizer."""
    return [p for p in model.parameters() if p.requires_grad]

