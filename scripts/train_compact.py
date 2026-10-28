#!/usr/bin/env python3
"""
Dedicated Capacity-Matched Compact GRU Training Script (Model D, Milestone 4).
Usage:
    python3 scripts/train_compact.py --config configs/compact_gru.yaml --seeds 42 43 44
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
    parser = argparse.ArgumentParser(description="Train Capacity-Matched Compact GRU (Model D)")
    parser.add_argument("--config", default="configs/compact_gru.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke_test:
        config["training"]["epochs"] = 3
        args.seeds = [42]

    ckpt_dir = config.get("paths", {}).get("checkpoints", "results/checkpoints/compact_gru")
    run_training(config, "compact_gru", args.seeds, ckpt_dir)


if __name__ == "__main__":
    main()

