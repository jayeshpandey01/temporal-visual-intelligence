#!/usr/bin/env python3
"""
Dedicated Temporal GRU Sequence Model Training Script (Model C, M3.21).
Usage:
    python3 scripts/train_gru.py --config configs/gru.yaml --seeds 42 43 44
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
    parser = argparse.ArgumentParser(description="Train Temporal GRU (Model C)")
    parser.add_argument("--config", default="configs/gru.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke_test:
        config["training"]["epochs"] = 3
        args.seeds = [42]

    ckpt_dir = config.get("paths", {}).get("checkpoints", "results/checkpoints/gru")
    run_training(config, "gru", args.seeds, ckpt_dir)


if __name__ == "__main__":
    main()

