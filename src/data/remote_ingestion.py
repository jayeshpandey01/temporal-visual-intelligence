"""
Remote Video & Image Ingestion Pipeline (Milestone 1.5).
Automates downloading, container decoding, and frame extraction for real action
video sequences from Hugging Face Hub and Kaggle.

Features:
  - Hugging Face Hub dataset download (via huggingface_hub)
  - Kaggle dataset acquisition (via kaggle CLI / API)
  - Universal video container decoding (MP4, AVI, WebM, MOV) via ffmpeg
  - Uniform temporal frame sampling and ImageNet spatial normalization (224x224)
  - Seamless integration with 7-check data integrity audit and leakage guard
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from PIL import Image


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg executable is available in PATH."""
    return shutil.which("ffmpeg") is not None


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    n_frames: int = 8,
    frame_size: Tuple[int, int] = (224, 224),
) -> List[Path]:
    """
    Extract n_frames uniformly spaced across video duration using ffmpeg or torchvision.
    Saves frames as frame_0001.jpg, frame_0002.jpg, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    W, H = frame_size

    # Check existing frames
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if len(existing) >= n_frames:
        return existing[:n_frames]

    if is_ffmpeg_available():
        # First extract all raw frames to temporary subfolder
        temp_dir = output_dir / ".temp_raw"
        temp_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
            "-q:v", "2",
            str(temp_dir / "raw_%05d.jpg")
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_frames = sorted(temp_dir.glob("raw_*.jpg"))

        if raw_frames:
            # Subsample uniformly to exactly n_frames
            import numpy as np
            indices = np.linspace(0, len(raw_frames) - 1, n_frames, dtype=int)
            final_frames = []
            for out_idx, raw_idx in enumerate(indices, 1):
                dst = output_dir / f"frame_{out_idx:04d}.jpg"
                shutil.copy(raw_frames[raw_idx], dst)
                final_frames.append(dst)

            shutil.rmtree(temp_dir, ignore_errors=True)
            return final_frames
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Fallback to python PIL / image reading if video is an image sequence or animated gif
    try:
        img = Image.open(video_path)
        frames = []
        n_frames_in_file = getattr(img, 'n_frames', 1)
        import numpy as np
        indices = np.linspace(0, n_frames_in_file - 1, n_frames, dtype=int)

        for out_idx, frame_idx in enumerate(indices, 1):
            if n_frames_in_file > 1:
                img.seek(frame_idx)
            frame_resized = img.convert("RGB").resize((W, H), Image.Resampling.BICUBIC)
            dst = output_dir / f"frame_{out_idx:04d}.jpg"
            frame_resized.save(dst, quality=95)
            frames.append(dst)
        return frames
    except Exception as e:
        print(f"  [Warning] Failed to decode {video_path}: {e}")
        return []


def download_from_huggingface(
    repo_id: str,
    filename: Optional[str] = None,
    target_dir: str = "data/raw_hf",
    repo_type: str = "dataset",
    token: Optional[str] = None,
) -> Path:
    """
    Download a video dataset archive or specific video file from Hugging Face Hub.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    hf_token = token or os.environ.get("HF_TOKEN")

    print(f"\n[HuggingFace Ingestion] Fetching from repo: {repo_id}...")
    if filename:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(target_path),
            token=hf_token,
        )
        print(f"  Downloaded file → {downloaded}")
        return Path(downloaded)
    else:
        # Download snapshot / entire repo or pattern
        downloaded_dir = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(target_path),
            token=hf_token,
            ignore_patterns=["*.git*", "*.md"],
        )
        print(f"  Downloaded repository snapshot → {downloaded_dir}")
        return Path(downloaded_dir)


def download_from_kaggle(
    dataset_slug: str,
    target_dir: str = "data/raw_kaggle",
) -> Path:
    """
    Download and extract a video dataset from Kaggle using the Kaggle API.
    Expects ~/.kaggle/kaggle.json or KAGGLE_USERNAME and KAGGLE_KEY environment variables.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[Kaggle Ingestion] Downloading dataset: {dataset_slug}...")
    cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(target_path), "--unzip"]

    try:
        subprocess.run(cmd, check=True)
        print(f"  Successfully extracted Kaggle dataset → {target_path}")
        return target_path
    except FileNotFoundError:
        raise RuntimeError(
            "Kaggle CLI not found in PATH. Install with `pip install kaggle` and setup ~/.kaggle/kaggle.json"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Kaggle download failed: {e}")


def ingest_and_organize_videos(
    source_dir: Path,
    output_root: Path,
    n_frames: int = 8,
    frame_size: Tuple[int, int] = (224, 224),
    max_per_class: int = 100,
) -> Dict[str, int]:
    """
    Walk a directory of downloaded raw videos, organize by class, decode frames,
    and output to data/{class_name}/{video_id}/frame_{index:04d}.jpg.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    video_exts = {'.mp4', '.avi', '.webm', '.mov', '.mkv'}

    # Discover videos
    class_counts = {}
    for video_file in sorted(source_dir.rglob("*")):
        if video_file.suffix.lower() not in video_exts:
            continue

        # Class is parent folder name or extracted from filename
        class_name = video_file.parent.name
        if class_name in ("raw_hf", "raw_kaggle", "data", "raw"):
            # Try splitting by filename if parent is root
            class_name = video_file.stem.split("_")[0]

        class_counts[class_name] = class_counts.get(class_name, 0)
        if class_counts[class_name] >= max_per_class:
            continue

        video_id = video_file.stem
        dest_video_dir = output_root / class_name / video_id
        extract_frames_from_video(video_file, dest_video_dir, n_frames=n_frames, frame_size=frame_size)
        class_counts[class_name] += 1

    print(f"\n[Ingestion Complete] Extracted sequences across {len(class_counts)} classes:")
    for c, count in class_counts.items():
        print(f"  {c:30s} : {count} videos processed")

    return class_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Milestone 1.5: Real Video Ingestion Pipeline")
    parser.add_argument("--source", choices=["hf", "kaggle", "local"], default="local")
    parser.add_argument("--repo-id", type=str, help="Hugging Face repo_id or Kaggle dataset slug (e.g., username/dataset-name)")
    parser.add_argument("--filename", type=str, default=None, help="Specific archive filename in HF repo")
    parser.add_argument("--input-dir", type=str, default="data/raw_videos", help="Local directory of raw videos")
    parser.add_argument("--output-root", type=str, default="data", help="Target processed dataset root")
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--max-per-class", type=int, default=100)
    args = parser.parse_args()

    if args.source == "hf":
        if not args.repo_id:
            print("Error: --repo-id is required for HuggingFace downloads.")
            sys.exit(1)
        raw_dir = download_from_huggingface(args.repo_id, filename=args.filename)
        ingest_and_organize_videos(raw_dir, Path(args.output_root), n_frames=args.n_frames, max_per_class=args.max_per_class)

    elif args.source == "kaggle":
        if not args.repo_id:
            print("Error: --repo-id is required for Kaggle downloads (e.g. 'pevogam/something-something-v2').")
            sys.exit(1)
        raw_dir = download_from_kaggle(args.repo_id)
        ingest_and_organize_videos(raw_dir, Path(args.output_root), n_frames=args.n_frames, max_per_class=args.max_per_class)

    else:
        in_path = Path(args.input_dir)
        if not in_path.exists():
            print(f"Local input directory {in_path} does not exist.")
            sys.exit(1)
        ingest_and_organize_videos(in_path, Path(args.output_root), n_frames=args.n_frames, max_per_class=args.max_per_class)

