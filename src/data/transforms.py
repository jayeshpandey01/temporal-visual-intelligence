"""
M2.4: Official Image Preprocessing Pipeline for Foundational Visual Encoders.

Enforces exact encoder-specific transforms without inventing parameters.
Supports official DINOv2 (Oquab et al., 2024) and CLIP (Radford et al., 2021) pipelines:
    Image (PIL or path)
      ↓
    RGB conversion (strict 3-channel check)
      ↓
    Resize (bicubic with antialias)
      ↓
    Center crop to 224x224
      ↓
    ToTensor (float32, [0, 1])
      ↓
    Normalize (exact official ImageNet statistics)
      ↓
    Tensor [3, 224, 224]
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union, Optional, List, Dict, Any

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np


# Official ImageNet statistics used by DINOv2 (LVD-142M) and CLIP ViT
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Version string for contract serialization (docs/milestone/M-2.md §M2.1, §M2.4)
PREPROCESSING_VERSION = "v1_dinov2_official_224x224_bicubic_center"


def build_dinov2_transform(
    crop_size: int = 224,
    resize_size: int = 256,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    direct_resize: bool = False
) -> T.Compose:
    """
    Build official DINOv2 evaluation transform (Oquab et al., 2024).

    DINOv2 uses bicubic interpolation with antialiasing followed by center crop:
        1. Resize shortest edge to 256 using bicubic antialiasing.
        2. Center crop 224x224 patch.
        3. Convert to tensor [3, 224, 224] in [0.0, 1.0].
        4. Normalize with ImageNet mean/std.

    If direct_resize is True, directly resizes to (crop_size, crop_size) without cropping.
    """
    if direct_resize:
        return T.Compose([
            T.Resize((crop_size, crop_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    return T.Compose([
        T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def build_clip_transform(crop_size: int = 224) -> T.Compose:
    """
    Build official OpenAI CLIP evaluation transform (Radford et al., 2021).
    """
    return T.Compose([
        T.Resize(crop_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(
    image_input: Union[str, Path, Image.Image],
    transform: Optional[T.Compose] = None,
    crop_size: int = 224
) -> torch.Tensor:
    """
    Load and preprocess a single image according to official M2.4 contract.

    Args:
        image_input: File path or PIL Image.
        transform: Optional torchvision Compose transform (defaults to official DINOv2 transform).
        crop_size: Target spatial dimension (default 224).

    Returns:
        Tensor of shape [3, crop_size, crop_size] of type float32.
    """
    if transform is None:
        transform = build_dinov2_transform(crop_size=crop_size)

    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")
        img = Image.open(img_path).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError(f"Expected file path or PIL Image, got {type(image_input)}")

    tensor = transform(img)

    # Validate output tensor integrity
    if tensor.shape != (3, crop_size, crop_size):
        raise ValueError(f"Unexpected tensor shape {tensor.shape}, expected (3, {crop_size}, {crop_size})")
    if torch.isnan(tensor).any():
        raise ValueError("NaN values detected in preprocessed image tensor")
    if torch.isinf(tensor).any():
        raise ValueError("Inf values detected in preprocessed image tensor")

    return tensor


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> torch.Tensor:
    """
    Invert ImageNet normalization for qualitative inspection and plotting.
    Input: [3, H, W] or [B, 3, H, W] normalized tensor.
    Output: Clamped tensor in [0.0, 1.0].
    """
    t = tensor.clone()
    if t.ndim == 3:
        for c in range(3):
            t[c] = t[c] * std[c] + mean[c]
    elif t.ndim == 4:
        for c in range(3):
            t[:, c] = t[:, c] * std[c] + mean[c]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {t.shape}")
    return torch.clamp(t, 0.0, 1.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized [3, H, W] float tensor back to a PIL Image."""
    denorm = denormalize_tensor(tensor)
    arr = (denorm.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)
