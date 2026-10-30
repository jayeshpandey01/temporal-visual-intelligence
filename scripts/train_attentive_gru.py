#!/usr/bin/env python3
"""
Dedicated Level 1 Attentive Bi-Directional GRU Training Script.
Features:
  - Temporal Attention Pooling (soft weighting over all 8 frames)
  - Bidirectional Recurrence (forward + backward sequence reads)
  - Directional Contrastive Loss (InfoNCE against opposites & sequence reversals)

Usage:
    python3 scripts/train_attentive_gru.py --config configs/gru_attentive.yaml --seeds 42 43 44
    python3 scripts/train_attentive_gru.py --smoke-test
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
    parser = argparse.ArgumentParser(description="Train Level 1 Attentive Bi-Directional GRU")
    parser.add_argument("--config", default="configs/gru_attentive.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke_test:
        config["training"]["epochs"] = 3
        args.seeds = [42]

    ckpt_dir = config.get("paths", {}).get("checkpoints", "results/checkpoints/gru_attentive")
    run_training(config, "attentive_bigru", args.seeds, ckpt_dir)


if __name__ == "__main__":
    main()

