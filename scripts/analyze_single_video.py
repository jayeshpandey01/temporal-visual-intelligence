"""
CLI runner for Minimalist Single-Video & Two-Frame Dyad Micro-Analysis.

Allows isolating a single video transition pair (f_1, f_2), evaluating:
1. Pixel dynamics (MSE, L1, visual motion delta heatmap)
2. Frozen DINOv2 latent trajectory geometry (Cosine similarity S, Temporal drift Δ, Kinetic displacement v, Kinetic energy E_k)
3. 4 Controlled micro-probes (Canonical forward, Reversed arrow, Static repetition, Order-invariant mean pooling)
4. Diagnostic 4-panel figure generation and JSON reporting.

Usage:
    # Explicit frame pair:
    python3 scripts/analyze_single_video.py \
        --frame1 "data/Pushing something from left to right/vid_02_0000/frame_0001.jpg" \
        --frame2 "data/Pushing something from left to right/vid_02_0000/frame_0008.jpg"

    # From video/sequence directory:
    python3 scripts/analyze_single_video.py \
        --video-dir "data/Opening something/vid_04_0000" \
        --output-prefix "opening_micro"
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.micro_analysis import analyze_two_frame_dyad
from src.data.remote_ingestion import extract_frames_from_video


def main():
    parser = argparse.ArgumentParser(
        description="Minimalist Single-Video / Two-Frame Dyad Micro-Analysis CLI"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to an actual video container file (.mp4, .webm, .mov, .avi)"
    )
    parser.add_argument(
        "--frame1", type=str, default=None,
        help="Path to frame 1 (early observation)"
    )
    parser.add_argument(
        "--frame2", type=str, default=None,
        help="Path to frame 2 (late observation)"
    )
    parser.add_argument(
        "--video-dir", type=str, default=None,
        help="Path to sequence directory containing frame_*.jpg files"
    )
    parser.add_argument(
        "--start-idx", type=int, default=1,
        help="Starting frame index when using --video-dir or --video (default: 1)"
    )
    parser.add_argument(
        "--end-idx", type=int, default=-1,
        help="Ending frame index when using --video-dir or --video (-1 for last frame)"
    )
    parser.add_argument(
        "--encoder", type=str, default="dinov2_vits14",
        help="Frozen ViT encoder (default: dinov2_vits14)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save JSON report (default: results)"
    )
    parser.add_argument(
        "--figures-dir", type=str, default="results/figures",
        help="Directory to save diagnostic figure (default: results/figures)"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="single_video_micro",
        help="Prefix for saved output files"
    )

    args = parser.parse_args()

    # Determine frame paths
    if args.video is not None:
        vpath = Path(args.video)
        if not vpath.is_file():
            print(f"Error: video file does not exist: {vpath}", file=sys.stderr)
            sys.exit(1)
        temp_dir = Path("results/.temp_video_frames") / vpath.stem
        frames = extract_frames_from_video(vpath, temp_dir, n_frames=8)
        if len(frames) < 2:
            print(f"Error: failed to extract at least 2 frames from {vpath}", file=sys.stderr)
            sys.exit(1)
        f1_path = frames[args.start_idx - 1] if args.start_idx <= len(frames) else frames[0]
        f2_path = frames[args.end_idx] if args.end_idx != -1 else frames[-1]
        if args.output_prefix == "single_video_micro":
            args.output_prefix = f"{vpath.stem}_micro"
    elif args.video_dir is not None:
        vdir = Path(args.video_dir)
        if not vdir.is_dir():
            print(f"Error: video directory does not exist: {vdir}", file=sys.stderr)
            sys.exit(1)
        frames = sorted(list(vdir.glob("*.jpg")) + list(vdir.glob("*.png")))
        if len(frames) < 2:
            print(f"Error: directory {vdir} contains fewer than 2 frames.", file=sys.stderr)
            sys.exit(1)

        f1_path = frames[args.start_idx - 1] if args.start_idx <= len(frames) else frames[0]
        f2_path = frames[args.end_idx] if args.end_idx != -1 else frames[-1]
    elif args.frame1 is not None and args.frame2 is not None:
        f1_path = Path(args.frame1)
        f2_path = Path(args.frame2)
    else:
        # Default fallback to first synthesized sequence if available
        default_dir = Path("data/Pushing something from left to right/vid_02_0000")
        if default_dir.is_dir():
            print("No frames specified. Falling back to default sequence:", default_dir)
            frames = sorted(list(default_dir.glob("*.jpg")))
            f1_path, f2_path = frames[0], frames[-1]
        else:
            print("Error: Must provide either --frame1 and --frame2 or --video-dir", file=sys.stderr)
            sys.exit(1)

    if not f1_path.exists() or not f2_path.exists():
        print(f"Error: Frame files not found:\n  F1: {f1_path}\n  F2: {f2_path}", file=sys.stderr)
        sys.exit(1)

    # Execute micro-analysis
    report = analyze_two_frame_dyad(
        f1_path=f1_path,
        f2_path=f2_path,
        config_or_encoder_name=args.encoder,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        output_prefix=args.output_prefix,
    )

    # Display clean scientific summary
    rep = report['latent_representation_metrics']
    pix = report['pixel_metrics']
    probes = report['controlled_micro_probes']

    print("\n" + "=" * 65)
    print("      ATOMIC TWO-FRAME DYAD MICRO-ANALYSIS RESULTS")
    print("=" * 65)
    print(f"Frame 1: {f1_path.name} | Frame 2: {f2_path.name}")
    print(f"Sequence Context: {f1_path.parent}")
    print("-" * 65)
    print(f"Pixel Dynamics:")
    print(f"  • MSE:                       {pix['pixel_mse']:.6f}")
    print(f"  • Mean L1 Difference:        {pix['pixel_l1']:.6f}")
    print(f"  • Max Pixel Delta:           {pix['max_pixel_delta']:.4f}")
    print("-" * 65)
    print(f"Latent Trajectory Geometry (Frozen DINOv2):")
    print(f"  • Pairwise Cosine Similarity: S = {rep['cosine_similarity']:.4f}")
    print(f"  • Temporal Drift (1 - S):    Δ = {rep['temporal_drift']:.4f}")
    print(f"  • Kinetic Displacement ||v||:    {rep['l2_distance']:.4f}")
    print(f"  • Kinetic Energy ||v||^2:    E_k = {rep['kinetic_energy']:.4f}")
    print("-" * 65)
    print(f"Controlled Micro-Probes:")
    print(f"  • Arrow Reversal (v == -v_rev):      {'PASS' if probes['arrow_inversion_symmetric'] else 'FAIL'}")
    print(f"  • Order-Invariant Mean Pooling:      {'PASS' if probes['mean_pooling_order_invariant'] else 'FAIL'}")
    print(f"  • Static Repeat Displacement (0.0):  {probes['static_control_1_displacement']:.6f}")
    print("=" * 65)
    print(f"Diagnostic visual figure saved to: {args.figures_dir}/{args.output_prefix}_pair.png")
    print(f"Structured JSON report saved to:   {args.output_dir}/{args.output_prefix}_report.json\n")


if __name__ == '__main__':
    main()

