"""
Unit Tests for M2.4: Image Preprocessing Pipeline.

Validates official DINOv2 / ImageNet transform specification:
    RGB conversion -> Bicubic Resize -> Center Crop / Direct Resize -> ToTensor -> Normalize
Ensures exact normalization values (mu=[0.485, 0.456, 0.406], sigma=[0.229, 0.224, 0.225]),
shape consistency [3, 224, 224], invertibility via denormalization, and error handling.
"""

import pytest
import torch
import numpy as np
from PIL import Image

from src.data.transforms import (
    build_dinov2_transform,
    build_clip_transform,
    preprocess_image,
    denormalize_tensor,
    tensor_to_pil,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROCESSING_VERSION
)


def test_dinov2_transform_output_shape_and_dtype():
    """Verify official transform produces float32 tensor of exact shape [3, 224, 224]."""
    # Create arbitrary non-square test image (e.g., 640x360 landscape)
    img = Image.new("RGB", (640, 360), color=(128, 64, 200))
    transform = build_dinov2_transform(crop_size=224, resize_size=256)
    
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_dinov2_normalization_exact_values():
    """Verify exact mathematical normalization: x_norm = (x - mu) / sigma."""
    # Create pure red image: R=255, G=0, B=0 -> in [0, 1]: R=1.0, G=0.0, B=0.0
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    transform = build_dinov2_transform(crop_size=224, direct_resize=True)
    tensor = transform(img)

    expected_r = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]  # (1 - 0.485) / 0.229 = 2.248908
    expected_g = (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]  # (0 - 0.456) / 0.224 = -2.035714
    expected_b = (0.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]  # (0 - 0.406) / 0.225 = -1.804444

    assert torch.isclose(tensor[0].mean(), torch.tensor(expected_r, dtype=torch.float32), atol=1e-4)
    assert torch.isclose(tensor[1].mean(), torch.tensor(expected_g, dtype=torch.float32), atol=1e-4)
    assert torch.isclose(tensor[2].mean(), torch.tensor(expected_b, dtype=torch.float32), atol=1e-4)


def test_denormalize_tensor_inversion():
    """Verify that denormalize_tensor inverts ImageNet normalization accurately."""
    original_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    original_img = Image.fromarray(original_arr)

    transform = build_dinov2_transform(crop_size=224, direct_resize=True)
    norm_tensor = transform(original_img)

    denorm = denormalize_tensor(norm_tensor)
    assert denorm.min() >= 0.0
    assert denorm.max() <= 1.0

    restored_pil = tensor_to_pil(norm_tensor)
    restored_arr = np.array(restored_pil)

    # Discrepancy between original and restored must be within numerical rounding tolerance (<= 2 pixel values)
    diff = np.abs(original_arr.astype(int) - restored_arr.astype(int))
    assert np.percentile(diff, 99) <= 2


def test_preprocess_image_helper():
    """Verify preprocess_image wrapper handles both PIL Image and validates tensor integrity."""
    img = Image.new("RGB", (300, 300), color=(50, 100, 150))
    tensor = preprocess_image(img, crop_size=224)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

    # Verify rejection of invalid types
    with pytest.raises(TypeError):
        preprocess_image(12345)

    # Verify rejection of non-existent path
    with pytest.raises(FileNotFoundError):
        preprocess_image("non_existent_image_path_12345.png")


def test_clip_transform_shape():
    """Verify OpenAI CLIP transform compatibility."""
    img = Image.new("RGB", (256, 256), color=(10, 20, 30))
    clip_transform = build_clip_transform(crop_size=224)
    tensor = clip_transform(img)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
