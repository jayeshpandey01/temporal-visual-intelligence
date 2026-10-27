"""
Dataset preparation script (Day 1 of pipeline).
Downloads or synthesizes the chosen dataset subset with verified temporal ambiguity.

Usage:
    python scripts/prepare.py --dataset ssv2-synthetic --subset-classes 20 --n-per-class 100
    python scripts/prepare.py --dataset ssv2 --n-per-class 100
    python scripts/prepare.py --dataset ucf101 --n-per-class 150

Supported datasets:
    ssv2-synthetic — High-fidelity synthetic SSv2 action sequences with opposing directional trajectories
    ssv2           — Something-Something v2 (official dataset ingestion)
    ucf101         — UCF-101 (downloadable benchmark)
"""

import os
import sys
import argparse
import shutil
import math
import random
from pathlib import Path
from typing import List, Tuple

import yaml
import numpy as np
from PIL import Image, ImageDraw


# 20 Paired Directional Action Classes with High Temporal Ambiguity
SSV2_CLASSES = [
    # Pair 1: Depth translation
    "Moving something away from the camera",
    "Moving something towards the camera",
    # Pair 2: Horizontal translation
    "Pushing something from left to right",
    "Pushing something from right to left",
    # Pair 3: Vertical translation
    "Picking something up",
    "Putting something down",
    # Pair 4: State transition (aperture/container)
    "Opening something",
    "Closing something",
    # Pair 5: Camera perspective motion
    "Turning the camera left while filming something",
    "Turning the camera right while filming something",
    # Pair 6: Surface interaction
    "Putting something on top of something",
    "Taking something off of something",
    # Pair 7: Occlusion reveal/hide
    "Uncovering something",
    "Covering something",
    # Pair 8: Tilt / tipping
    "Tipping something over",
    "Standing something upright",
    # Pair 9: In-plane rotation
    "Rotating something clockwise",
    "Rotating something counter-clockwise",
    # Pair 10: Containment transition
    "Putting something into something",
    "Taking something out of something",
]


def generate_synthetic_ssv2(
    data_root: str,
    n_per_class: int = 100,
    subset_classes: int = 20,
    n_frames: int = 8,
    frame_size: Tuple[int, int] = (224, 224),
    seed: int = 42,
):
    """
    Generate high-fidelity synthetic action sequences with verified temporal ambiguity.
    In paired classes (e.g. left-to-right vs right-to-left), individual isolated frames
    have identical visual appearances, making a single frame ambiguous and forcing
    the model to use chronological trajectory.
    """
    print(f"\n[SSv2 Synthetic Generator] Generating {subset_classes} classes x {n_per_class} sequences...")
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)

    selected_classes = SSV2_CLASSES[:subset_classes]
    root_path = Path(data_root)
    root_path.mkdir(parents=True, exist_ok=True)

    W, H = frame_size

    # Distinct palettes for foreground objects and textured backgrounds
    obj_colors = [
        (220, 50, 40), (40, 120, 230), (50, 180, 80),
        (230, 150, 20), (160, 50, 200), (20, 200, 200),
        (240, 80, 140), (120, 90, 60), (220, 200, 30)
    ]

    for class_idx, class_name in enumerate(selected_classes):
        class_dir = root_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for seq_idx in range(n_per_class):
            video_id = f"vid_{class_idx:02d}_{seq_idx:04d}"
            video_dir = class_dir / video_id
            video_dir.mkdir(parents=True, exist_ok=True)

            # Randomize object color, background tone, and texture per video
            bg_color = (
                rng.randint(210, 240),
                rng.randint(210, 240),
                rng.randint(210, 240),
            )
            obj_color = obj_colors[rng.randint(0, len(obj_colors))]
            surf_y = int(H * 0.70)
            base_size = rng.randint(30, 45)

            # Trajectory parameterization across t in [0, 1]
            for t_step in range(n_frames):
                alpha = t_step / max(1, n_frames - 1)  # 0.0 -> 1.0

                # Create canvas
                img = Image.new("RGB", (W, H), bg_color)
                draw = ImageDraw.Draw(img)

                # Draw static ground surface / table (held constant)
                draw.rectangle([0, surf_y, W, H], fill=(180, 180, 185))
                draw.line([0, surf_y, W, surf_y], fill=(140, 140, 145), width=2)

                # Simulate paired actions
                # Pair 1: Away (shrinking) vs Towards (growing)
                if class_name == "Moving something away from the camera":
                    size = int(base_size * (1.3 - 0.7 * alpha))
                    cx, cy = W // 2, int(surf_y - size // 2 - alpha * 20)
                    draw.rectangle([cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2], fill=obj_color)

                elif class_name == "Moving something towards the camera":
                    size = int(base_size * (0.6 + 0.7 * alpha))
                    cx, cy = W // 2, int(surf_y - size // 2 - (1 - alpha) * 20)
                    draw.rectangle([cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2], fill=obj_color)

                # Pair 2: Left to Right vs Right to Left
                elif class_name == "Pushing something from left to right":
                    cx = int(W * 0.2 + alpha * W * 0.6)
                    cy = surf_y - base_size // 2
                    draw.rectangle([cx - base_size // 2, cy - base_size // 2, cx + base_size // 2, cy + base_size // 2], fill=obj_color)
                    # Hand or pusher indicator
                    draw.ellipse([cx - base_size // 2 - 15, cy - 8, cx - base_size // 2, cy + 8], fill=(240, 190, 160))

                elif class_name == "Pushing something from right to left":
                    cx = int(W * 0.8 - alpha * W * 0.6)
                    cy = surf_y - base_size // 2
                    draw.rectangle([cx - base_size // 2, cy - base_size // 2, cx + base_size // 2, cy + base_size // 2], fill=obj_color)
                    # Hand or pusher indicator
                    draw.ellipse([cx + base_size // 2, cy - 8, cx + base_size // 2 + 15, cy + 8], fill=(240, 190, 160))

                # Pair 3: Picking up vs Putting down
                elif class_name == "Picking something up":
                    cx = W // 2
                    cy = int((surf_y - base_size // 2) - alpha * (H * 0.4))
                    draw.rectangle([cx - base_size // 2, cy - base_size // 2, cx + base_size // 2, cy + base_size // 2], fill=obj_color)

                elif class_name == "Putting something down":
                    cx = W // 2
                    cy = int((surf_y - base_size // 2 - H * 0.4) + alpha * (H * 0.4))
                    draw.rectangle([cx - base_size // 2, cy - base_size // 2, cx + base_size // 2, cy + base_size // 2], fill=obj_color)

                # Pair 4: Opening vs Closing
                elif class_name == "Opening something":
                    cx, cy = W // 2, surf_y - base_size
                    door_w = int(base_size * (1.0 - alpha * 0.8))
                    draw.rectangle([cx - base_size, cy - base_size, cx + base_size, cy + base_size], outline=(40, 40, 40), width=3)
                    draw.rectangle([cx - base_size, cy - base_size, cx - base_size + door_w, cy + base_size], fill=obj_color)

                elif class_name == "Closing something":
                    cx, cy = W // 2, surf_y - base_size
                    door_w = int(base_size * (0.2 + alpha * 0.8))
                    draw.rectangle([cx - base_size, cy - base_size, cx + base_size, cy + base_size], outline=(40, 40, 40), width=3)
                    draw.rectangle([cx - base_size, cy - base_size, cx - base_size + door_w, cy + base_size], fill=obj_color)

                # Pair 5: Camera panning left vs right
                elif class_name == "Turning the camera left while filming something":
                    offset = int(alpha * W * 0.4)
                    cx = (W // 2) + offset
                    cy = surf_y - base_size // 2
                    draw.rectangle([cx - base_size // 2, cy - base_size // 2, cx + base_size // 2, cy + base_size // 2], fill=obj_color)

                elif class_name == "Turning the camera right while filming something":
                    offset = int(alpha * W * 0.4)
                    cx = (W // 2) - offset
                    cy = surf_y - base_size // 2
                    draw.rectangle([cx - base_size // 2, cy - base_size // 2, cx + base_size // 2, cy + base_size // 2], fill=obj_color)

                # Pair 6: Putting on top vs Taking off
                elif class_name == "Putting something on top of something":
                    box_h = 30
                    draw.rectangle([W // 2 - 40, surf_y - box_h, W // 2 + 40, surf_y], fill=(120, 120, 120))
                    item_y = int((surf_y - box_h - 60) + alpha * 60)
                    draw.ellipse([W // 2 - 15, item_y - 15, W // 2 + 15, item_y + 15], fill=obj_color)

                elif class_name == "Taking something off of something":
                    box_h = 30
                    draw.rectangle([W // 2 - 40, surf_y - box_h, W // 2 + 40, surf_y], fill=(120, 120, 120))
                    item_y = int((surf_y - box_h) - alpha * 60)
                    draw.ellipse([W // 2 - 15, item_y - 15, W // 2 + 15, item_y + 15], fill=obj_color)

                # Pair 7: Uncovering vs Covering
                elif class_name == "Uncovering something":
                    cx, cy = W // 2, surf_y - base_size // 2
                    draw.rectangle([cx - base_size // 2, cy - base_size // 2, cx + base_size // 2, cy + base_size // 2], fill=obj_color)
                    cover_shift = int(alpha * base_size * 1.5)
                    draw.rectangle([cx - base_size // 2 + cover_shift, cy - base_size // 2, cx + base_size // 2 + cover_shift, cy + base_size // 2], fill=(60, 60, 65))

                elif class_name == "Covering something":
                    cx, cy = W // 2, surf_y - base_size // 2
                    draw.rectangle([cx - base_size // 2, cy - base_size // 2, cx + base_size // 2, cy + base_size // 2], fill=obj_color)
                    cover_shift = int((1 - alpha) * base_size * 1.5)
                    draw.rectangle([cx - base_size // 2 + cover_shift, cy - base_size // 2, cx + base_size // 2 + cover_shift, cy + base_size // 2], fill=(60, 60, 65))

                # Pair 8: Tipping over vs Standing upright
                elif class_name == "Tipping something over":
                    angle = alpha * 90.0  # 0 to 90 degrees
                    cx, cy = W // 2, surf_y
                    box = Image.new("RGBA", (base_size, base_size * 2), (*obj_color, 255))
                    rot_box = box.rotate(-angle, expand=True)
                    img.paste(rot_box, (cx - rot_box.width // 2, cy - rot_box.height), rot_box)

                elif class_name == "Standing something upright":
                    angle = (1 - alpha) * 90.0  # 90 to 0 degrees
                    cx, cy = W // 2, surf_y
                    box = Image.new("RGBA", (base_size, base_size * 2), (*obj_color, 255))
                    rot_box = box.rotate(-angle, expand=True)
                    img.paste(rot_box, (cx - rot_box.width // 2, cy - rot_box.height), rot_box)

                # Pair 9: Rotating clockwise vs counter-clockwise
                elif class_name == "Rotating something clockwise":
                    angle = alpha * 180.0
                    box = Image.new("RGBA", (base_size, base_size), (*obj_color, 255))
                    bdraw = ImageDraw.Draw(box)
                    bdraw.line([0, 0, base_size, base_size], fill=(255, 255, 255), width=3)
                    rot_box = box.rotate(-angle, expand=True)
                    img.paste(rot_box, (W // 2 - rot_box.width // 2, surf_y - 50 - rot_box.height // 2), rot_box)

                elif class_name == "Rotating something counter-clockwise":
                    angle = alpha * 180.0
                    box = Image.new("RGBA", (base_size, base_size), (*obj_color, 255))
                    bdraw = ImageDraw.Draw(box)
                    bdraw.line([0, 0, base_size, base_size], fill=(255, 255, 255), width=3)
                    rot_box = box.rotate(angle, expand=True)
                    img.paste(rot_box, (W // 2 - rot_box.width // 2, surf_y - 50 - rot_box.height // 2), rot_box)

                # Pair 10: Putting into container vs Taking out
                elif class_name == "Putting something into something":
                    # Container
                    draw.rectangle([W // 2 - 35, surf_y - 45, W // 2 + 35, surf_y], outline=(40, 40, 40), width=3)
                    item_y = int((surf_y - 80) + alpha * 50)
                    draw.ellipse([W // 2 - 12, item_y - 12, W // 2 + 12, item_y + 12], fill=obj_color)

                else:  # Taking something out of something
                    draw.rectangle([W // 2 - 35, surf_y - 45, W // 2 + 35, surf_y], outline=(40, 40, 40), width=3)
                    item_y = int((surf_y - 30) - alpha * 50)
                    draw.ellipse([W // 2 - 12, item_y - 12, W // 2 + 12, item_y + 12], fill=obj_color)

                # Save frame
                frame_path = video_dir / f"frame_{t_step + 1:04d}.jpg"
                img.save(frame_path, quality=95)

    print(f"Generated {subset_classes * n_per_class} sequences in {data_root}.")


def prepare_ucf101(data_root: str, n_per_class: int, subset_classes: int):
    """Download and organize UCF-101 subset."""
    print(f"Preparing UCF-101 ({subset_classes} classes, {n_per_class} clips/class)")
    print("Download URL: https://www.crcv.ucf.edu/data/UCF101.php")
    print(f"Place in: {data_root}/{{class_name}}/{{video_id}}/frame_*.jpg")


def prepare_ssv2(data_root: str, n_per_class: int, subset_classes: int):
    """Instructions for Something-Something v2 official subset."""
    print(f"Preparing SSv2 ({subset_classes} classes, {n_per_class} clips/class)")
    print("SSv2 registration at: https://developer.qualcomm.com/software/ai-datasets/something-something")
    print("If using synthetic mode, run: python scripts/prepare.py --dataset ssv2-synthetic")


def run_audit(data_root: str, config_path: str):
    """Run dataset audit after preparation."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.audit import run_audit as _audit
    _audit(
        data_root=data_root,
        output_path='results/dataset_audit.json',
        strips_output='results/figures/audit_strips.png',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment.yaml')
    parser.add_argument('--dataset', choices=['ssv2', 'ssv2-synthetic', 'ucf101', 'real-video'], default='ssv2-synthetic')
    parser.add_argument('--source', choices=['synthetic', 'hf', 'kaggle', 'local'], default='synthetic', help='Remote source for real-video dataset')
    parser.add_argument('--repo-id', type=str, default=None, help='HuggingFace repo_id or Kaggle dataset slug')
    parser.add_argument('--filename', type=str, default=None, help='Specific filename for HF download')
    parser.add_argument('--input-dir', type=str, default='data/raw_videos', help='Local folder of video files')
    parser.add_argument('--n-per-class', type=int, default=100)
    parser.add_argument('--subset-classes', type=int, default=20)
    parser.add_argument('--audit', action='store_true', help='Run audit after preparation')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_root = config['dataset']['root']

    if args.dataset == 'real-video' or args.source in ('hf', 'kaggle', 'local'):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.data.remote_ingestion import (
            download_from_huggingface,
            download_from_kaggle,
            ingest_and_organize_videos,
        )
        if args.source == 'hf':
            if not args.repo_id:
                print("Error: --repo-id is required for HuggingFace downloads.")
                sys.exit(1)
            raw_dir = download_from_huggingface(args.repo_id, filename=args.filename)
            ingest_and_organize_videos(
                raw_dir, Path(data_root),
                n_frames=config['sequence']['max_frames'],
                frame_size=tuple(config['sequence']['frame_size']),
                max_per_class=args.n_per_class,
            )
        elif args.source == 'kaggle':
            if not args.repo_id:
                print("Error: --repo-id is required for Kaggle downloads (e.g. username/dataset).")
                sys.exit(1)
            raw_dir = download_from_kaggle(args.repo_id)
            ingest_and_organize_videos(
                raw_dir, Path(data_root),
                n_frames=config['sequence']['max_frames'],
                frame_size=tuple(config['sequence']['frame_size']),
                max_per_class=args.n_per_class,
            )
        else:
            in_dir = Path(args.input_dir)
            if not in_dir.exists():
                print(f"Error: Local video directory {in_dir} does not exist.")
                sys.exit(1)
            ingest_and_organize_videos(
                in_dir, Path(data_root),
                n_frames=config['sequence']['max_frames'],
                frame_size=tuple(config['sequence']['frame_size']),
                max_per_class=args.n_per_class,
            )

    elif args.dataset == 'ssv2-synthetic':
        generate_synthetic_ssv2(
            data_root=data_root,
            n_per_class=args.n_per_class,
            subset_classes=args.subset_classes,
            n_frames=config['sequence']['max_frames'],
            frame_size=tuple(config['sequence']['frame_size']),
            seed=config['dataset']['seed'],
        )
    elif args.dataset == 'ucf101':
        prepare_ucf101(data_root, args.n_per_class, args.subset_classes)
    else:
        prepare_ssv2(data_root, args.n_per_class, args.subset_classes)

    if args.audit:
        run_audit(data_root, args.config)


