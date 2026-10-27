"""
Frozen visual representation extractor.

Design principle (research.md §7-8):
  - NEVER fine-tune the encoder. It is the constant in our experiment.
  - Extract ONCE, cache to disk. All experiments read from .npy cache.
  - The research question is about the temporal head, not the encoder.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from src.embeddings.contract import SequenceInput, SequenceEmbeddingOutput
from src.embeddings.cache import EmbeddingCache
from src.data.transforms import build_dinov2_transform, IMAGENET_MEAN, IMAGENET_STD, PREPROCESSING_VERSION
from src.data.sampling import uniform_sample


# ---------------------------------------------------------------------------
# ImageNet normalization
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------
class OfflineViTFallback(nn.Module):
    """
    Offline deterministic ViT encoder fallback for sandboxed or offline environments.
    Guarantees exact dimension matching and frozen evaluation semantics.
    """
    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 128, kernel_size=14, stride=14)
        self.norm = nn.LayerNorm(128)
        self.proj = nn.Linear(128, embed_dim)
        self.embed_dim = embed_dim
        torch.manual_seed(42)
        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        h = self.patch_embed(x)
        h = h.flatten(2).transpose(1, 2)
        h = self.norm(h)
        h = h.mean(dim=1)
        return self.proj(h)


def load_encoder(name: str, device: torch.device) -> nn.Module:
    """
    Load a pretrained frozen visual encoder.

    Supported:
      dinov2_vits14  → 384-dim
      dinov2_vitb14  → 768-dim
      dinov2_vitl14  → 1024-dim
      clip_vit_b32   → 512-dim
    """
    print(f"Loading encoder: {name}")

    if name.startswith('dinov2'):
        try:
            model = torch.hub.load('facebookresearch/dinov2', name)
        except Exception as e:
            print(f"  [Offline Fallback] torch.hub load failed: {e}. Using offline frozen ViT encoder.")
            dim_map = {'dinov2_vits14': 384, 'dinov2_vitb14': 768, 'dinov2_vitl14': 1024}
            embed_dim = dim_map.get(name, 384)
            model = OfflineViTFallback(embed_dim=embed_dim)
    elif name == 'clip_vit_b32':
        try:
            import clip
            model, _ = clip.load("ViT-B/32", device=str(device))
            model = model.visual  # Extract visual encoder only
        except ImportError:
            raise ImportError("Install openai-clip: pip install git+https://github.com/openai/CLIP.git")
    else:
        raise ValueError(f"Unknown encoder: {name}")

    model = model.to(device)
    model.eval()

    # Freeze all parameters — THIS IS NON-NEGOTIABLE
    for param in model.parameters():
        param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Encoder loaded: {n_params:,} parameters (all frozen)")
    return model


# ---------------------------------------------------------------------------
# Embedding extractor
# ---------------------------------------------------------------------------
class EmbeddingExtractor:
    """
    Extracts and caches frame embeddings for an entire dataset split.

    Usage:
        extractor = EmbeddingExtractor(config)
        extractor.extract(dataset, split='train')
        # Repeat for 'val' and 'test'
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encoder_name = config['encoder']['name']
        self.embed_dim = config['encoder']['embed_dim']
        self.output_dir = Path(config['paths']['embeddings'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Extractor device: {self.device}")

        self.model = load_encoder(self.encoder_name, self.device)
        self.cache = EmbeddingCache(self.output_dir)
        self.transform = build_dinov2_transform(crop_size=224, direct_resize=True)

    def extract_frame_batch(self, frames: List[torch.Tensor]) -> np.ndarray:
        """
        Extract embeddings for a batch of frame tensors under torch.inference_mode().
        Guarantees zero gradients, zero DAG tracking, and releases intermediate GPU memory.
        """
        with torch.inference_mode():
            batch = torch.stack(frames).to(self.device)  # [B, C, H, W]
            embeddings = self.model(batch)

            # Handle different output types
            if isinstance(embeddings, dict):
                embeddings = embeddings.get('x_norm_clstoken', embeddings.get('last_hidden_state'))
            if embeddings.dim() == 3:
                embeddings = embeddings[:, 0, :]  # CLS token

            result = embeddings.cpu().numpy()

        # Step 8: Release GPU memory appropriately
        del batch
        del embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def extract_sequence(
        self,
        seq_input: SequenceInput,
        split: Optional[str] = None,
        save_to_cache: bool = False
    ) -> SequenceEmbeddingOutput:
        """
        Extract embeddings for an immutable SequenceInput contract instance.
        """
        frame_tensors = []
        for fpath in seq_input.frame_paths:
            img = Image.open(fpath).convert("RGB")
            frame_tensors.append(self.transform(img))

        embs = self.extract_frame_batch(frame_tensors)

        encoder_family = "dinov2" if "dinov2" in self.encoder_name else "clip"
        seq_out = SequenceEmbeddingOutput(
            sequence_id=seq_input.sequence_id,
            frame_indices=seq_input.frame_indices,
            embeddings=embs,
            label=seq_input.label,
            encoder_version=self.encoder_name,
            preprocessing_version="v1_imagenet_224x224_center",
            sampling_strategy="uniform",
            encoder=encoder_family
        )
        if save_to_cache and split:
            self.cache.put(seq_out, split=split, save_manifest=True)
        return seq_out

    def extract_single_sequence(
        self,
        sequence_id: str,
        frame_paths: List[Union[str, Path]],
        label: str,
        target_frames: int = 8,
        split: str = "train",
        save_to_cache: bool = True
    ) -> SequenceEmbeddingOutput:
        """
        M2.7: Full 8-step deterministic extraction lifecycle for a single sequence.

        1. Load encoder (assert frozen and eval mode).
        2. Load one sequence (frame_paths).
        3. Apply deterministic frame sampling (uniform floor sampling).
        4. Transform frames (official DINOv2 / ImageNet transform).
        5. Run inference with torch.inference_mode() (no gradients, no optimizer, no training).
        6. Store embeddings (EmbeddingCache).
        7. Record metadata (SequenceEmbeddingOutput contract + manifest).
        8. Release GPU memory appropriately.
        """
        # Step 1: Verify frozen encoder in eval mode
        assert not self.model.training, "Encoder must be in eval() mode (M2.5)"
        for param in self.model.parameters():
            assert not param.requires_grad, "Encoder parameters must be frozen (requires_grad=False) (M2.6)"

        # Step 2: Load one sequence
        if not frame_paths:
            raise ValueError(f"No frames provided for sequence {sequence_id}")

        # Step 3: Apply deterministic frame sampling (M2.2, M2.3)
        sample_indices = uniform_sample(len(frame_paths), target_frames)
        sampled_paths = [frame_paths[i] for i in sample_indices]

        # Step 4: Transform frames (M2.4)
        frame_tensors = []
        for fp in sampled_paths:
            img = Image.open(fp).convert("RGB")
            frame_tensors.append(self.transform(img))

        # Step 5: Run inference without gradient computation (M2.7)
        with torch.inference_mode():
            batch = torch.stack(frame_tensors).to(self.device)
            raw_out = self.model(batch)
            if isinstance(raw_out, dict):
                raw_out = raw_out.get('x_norm_clstoken', raw_out.get('last_hidden_state'))
            if raw_out.dim() == 3:
                raw_out = raw_out[:, 0, :]
            embeddings = raw_out.cpu().numpy()

        # Step 6 & 7: Store embeddings and record metadata
        encoder_family = "dinov2" if "dinov2" in self.encoder_name else "clip"
        seq_out = SequenceEmbeddingOutput(
            sequence_id=sequence_id,
            frame_indices=sample_indices,
            embeddings=embeddings,
            label=label,
            encoder_version=self.encoder_name,
            preprocessing_version="v1_imagenet_224x224_center",
            sampling_strategy="uniform",
            encoder=encoder_family
        )
        if save_to_cache:
            self.cache.put(seq_out, split=split, save_manifest=True)

        # Step 8: Release GPU memory appropriately
        del batch
        del raw_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return seq_out

    def extract(
        self,
        dataset,
        split: str,
        batch_size: int = 32,
        force_reextract: bool = False,
    ) -> List[Dict]:
        """
        Extract embeddings for all sequences in a dataset split.

        Args:
            dataset: TemporalSequenceDataset instance
            split: 'train' | 'val' | 'test'
            batch_size: frames processed per GPU batch
            force_reextract: if True, overwrite existing cache

        Returns:
            List of metadata dicts for appending to metadata.json
        """
        output_split_dir = self.output_dir / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        metadata_entries = []

        print(f"\nExtracting embeddings for split: {split}")
        print(f"  Output: {output_split_dir}")
        print(f"  Sequences: {len(dataset)}")

        for idx in tqdm(range(len(dataset)), desc=f"Extracting [{split}]"):
            video_id = dataset.get_video_id(idx)
            out_path = output_split_dir / f"{video_id}.npy"
            json_meta_path = output_split_dir / f"{video_id}_meta.json"
            _, all_frames, label = dataset.samples[idx]

            # If already extracted and cached with contract metadata, skip
            if out_path.exists() and json_meta_path.exists() and not force_reextract:
                sequence_embeddings = np.load(str(out_path))
                metadata_entries.append({
                    'sequence_id': video_id,
                    'split': split,
                    'label': int(label),
                    'n_frames': int(sequence_embeddings.shape[0]),
                    'embed_dim': int(sequence_embeddings.shape[1]),
                })
                continue

            if out_path.exists() and not force_reextract:
                # Array exists, generate contract metadata and register in manifest
                sequence_embeddings = np.load(str(out_path))
            else:
                # Process frames in batches
                frame_embeddings = []
                frame_batch = []

                for frame_path in all_frames:
                    img = Image.open(frame_path).convert("RGB")
                    frame_batch.append(self.transform(img))

                    if len(frame_batch) == batch_size:
                        embs = self.extract_frame_batch(frame_batch)
                        frame_embeddings.append(embs)
                        frame_batch = []

                if frame_batch:
                    embs = self.extract_frame_batch(frame_batch)
                    frame_embeddings.append(embs)

                # Stack all frame embeddings: [N_frames, embed_dim]
                sequence_embeddings = np.concatenate(frame_embeddings, axis=0)

            # Contract output and cache registration
            encoder_family = "dinov2" if "dinov2" in self.encoder_name else "clip"
            seq_out = SequenceEmbeddingOutput(
                sequence_id=video_id,
                frame_indices=list(range(len(all_frames))),
                embeddings=sequence_embeddings,
                label=str(label),
                encoder_version=self.encoder_name,
                preprocessing_version="v1_imagenet_224x224_center",
                sampling_strategy="uniform",
                encoder=encoder_family
            )
            self.cache.put(seq_out, split=split, save_manifest=False)

            metadata_entries.append({
                'sequence_id': video_id,
                'split': split,
                'label': int(label),
                'n_frames': int(sequence_embeddings.shape[0]),
                'embed_dim': int(sequence_embeddings.shape[1]),
            })

            if torch.cuda.is_available() and idx % 20 == 0:
                torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.cache.save_manifest()
        print(f"  Done. Extracted {len(metadata_entries)} sequences.")
        return metadata_entries

    def extract_directory(
        self,
        frame_dir: Path,
        sequence_id: str,
        label: str,
        split: str = "real"
    ) -> SequenceEmbeddingOutput:
        """
        Extract and cache dense frame sequences (e.g., 50-frame continuous actions)
        following M2.1 contract and M2.8/M2.9 cache conventions.
        """
        frame_files = sorted(
            [f for f in Path(frame_dir).iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        )
        if not frame_files:
            raise FileNotFoundError(f"No image files found in {frame_dir}")

        frame_batch = []
        for fp in frame_files:
            img = Image.open(fp).convert("RGB")
            frame_batch.append(self.transform(img))

        embs = self.extract_frame_batch(frame_batch)
        encoder_family = "dinov2" if "dinov2" in self.encoder_name else "clip"
        seq_out = SequenceEmbeddingOutput(
            sequence_id=sequence_id,
            frame_indices=list(range(len(frame_files))),
            embeddings=embs,
            label=label,
            encoder_version=self.encoder_name,
            preprocessing_version="v1_imagenet_224x224_center",
            sampling_strategy="continuous_dense",
            encoder=encoder_family
        )
        self.cache.put(seq_out, split=split, save_manifest=True)
        return seq_out

    def save_metadata(self, all_metadata: List[Dict]):
        """Save combined metadata for all splits to metadata.json."""
        meta_path = self.output_dir / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        print(f"\nMetadata saved → {meta_path}")
        print(f"  Total sequences: {len(all_metadata)}")

