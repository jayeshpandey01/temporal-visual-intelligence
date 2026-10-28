#!/usr/bin/env python3
"""
Dedicated Multi-Frame Mean-Pooling Control Training Script (Model B, M3.21).
Usage:
    python3 scripts/train_mean.py --config configs/mean_pool.yaml --seeds 42 43 44
"""

import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp')
import sys
import argparse
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train import run_training


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Frame Mean-Pooling Control (Model B)")
    parser.add_argument("--config", default="configs/mean_pool.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke_test:
        config["training"]["epochs"] = 3
        args.seeds = [42]

    ckpt_dir = config.get("paths", {}).get("checkpoints", "results/checkpoints/meanpool")
    run_training(config, "meanpool", args.seeds, ckpt_dir)


if __name__ == "__main__":
    main()

