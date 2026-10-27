"""
Evaluation script — runs all controlled experiments.
(research.md §13-20)

Usage:
    # Run all experiments (3 seeds)
    python scripts/evaluate.py --config configs/experiment.yaml --all-experiments --seeds 42 43 44

    # Quick smoke test (1 seed, 50 samples)
    python scripts/evaluate.py --config configs/experiment.yaml --smoke-test

    # Single experiment
    python scripts/evaluate.py --config configs/experiment.yaml --exp progressive
"""

import sys
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import EmbeddingSequenceDataset
from src.models.baseline import build_baseline
from src.models.temporal import build_temporal_model
from src.experiments.progressive import run_progressive_sweep
from src.experiments.shuffle import run_order_experiment
from src.experiments.corruption import run_corruption_experiment
from src.experiments.sparsity import run_sparsity_experiment
from src.evaluation.analysis import compute_win_loss_quadrants, plot_quadrant_matrix
from src.evaluation.metrics import compute_metrics, aggregate_across_seeds


def load_model(model_type: str, config: dict, checkpoint_path: str, device):
    if model_type in ('baseline', 'meanpool'):
        model = build_baseline(config, model_type=model_type)
    else:
        config['temporal']['model'] = model_type
        model = build_temporal_model(config)

    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"  WARNING: No checkpoint at {checkpoint_path}. Using initialized weights.")

    return model.to(device)


def run_all_experiments(config: dict, seeds: list, smoke_test: bool = False):
    """Run the complete experimental battery."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("EXPERIMENTAL EVALUATION SUITE")
    print(f"Device: {device} | Seeds: {seeds}")
    print(f"{'='*60}")

    # ----------------------------------------------------------------
    # Load test dataset
    # ----------------------------------------------------------------
    test_ds = EmbeddingSequenceDataset(
        config['paths']['embeddings'], 'test',
        max_frames=config['sequence']['max_frames'],
    )
    print(f"\nTest set: {len(test_ds)} sequences")

    checkpoint_dir = config['paths']['checkpoints']

    # ----------------------------------------------------------------
    # Experiment 1: Progressive observation sweep (H1)
    # ----------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("EXPERIMENT 1: Progressive Observation Sweep (H1)")
    print(f"{'─'*50}")

    temporal_results_per_seed = []
    for seed in seeds:
        ckpt = f"{checkpoint_dir}/gru/best_seed{seed}.pt"
        temporal = load_model('gru', config, ckpt, device)
        frame_counts = config['evaluation']['frame_counts']
        result = run_progressive_sweep(temporal, test_ds, frame_counts, device)
        temporal_results_per_seed.append(result)

    # ----------------------------------------------------------------
    # Experiment 2: Temporal order test (H2)
    # ----------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("EXPERIMENT 2: Temporal Order Test (H2)")
    print(f"{'─'*50}")

    for seed in seeds:
        ckpt = f"{checkpoint_dir}/gru/best_seed{seed}.pt"
        temporal = load_model('gru', config, ckpt, device)
        order_result = run_order_experiment(temporal, test_ds, device)
        print(f"  Seed {seed}: OrderDrop = {order_result['order_drop']:.3f}")

    # ----------------------------------------------------------------
    # Experiment 3: Corruption robustness (H3) across seeds
    # ----------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("EXPERIMENT 3: Corruption Robustness (H3) Across Seeds")
    print(f"{'─'*50}")

    corruption_results = []
    for seed in seeds:
        ckpt_base = f"{checkpoint_dir}/baseline/best_seed{seed}.pt"
        ckpt_temp = f"{checkpoint_dir}/gru/best_seed{seed}.pt"
        baseline = load_model('baseline', config, ckpt_base, device)
        temporal = load_model('gru', config, ckpt_temp, device)

        res = run_corruption_experiment(
            baseline, temporal, test_ds, device,
            corruption_fraction=config['evaluation']['corruption_fraction'],
            seed=seed,
        )
        corruption_results.append(res)
    corruption_result = corruption_results[0]

    # ----------------------------------------------------------------
    # Experiment 4: Temporal sparsity (H4) across seeds
    # ----------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("EXPERIMENT 4: Temporal Sparsity (H4) Across Seeds")
    print(f"{'─'*50}")

    sparsity_results = []
    for seed in seeds:
        ckpt_temp = f"{checkpoint_dir}/gru/best_seed{seed}.pt"
        temporal = load_model('gru', config, ckpt_temp, device)
        res = run_sparsity_experiment(temporal, test_ds, device)
        sparsity_results.append(res)
    sparsity_result = sparsity_results[0]

    # ----------------------------------------------------------------
    # Win/Loss Quadrant Analysis
    # ----------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("WIN/LOSS QUADRANT ANALYSIS")
    print(f"{'─'*50}")

    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    baseline_preds, temporal_preds, all_labels = [], [], []
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            b_logits = baseline(embeddings)
            t_logits = temporal(embeddings)
            baseline_preds.extend(b_logits.argmax(-1).cpu().numpy())
            temporal_preds.extend(t_logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.numpy())

    quadrant_result = compute_win_loss_quadrants(
        np.array(baseline_preds),
        np.array(temporal_preds),
        np.array(all_labels),
    )
    plot_quadrant_matrix(
        quadrant_result['quadrant_counts'],
        output_path=f"{config['paths']['figures']}/win_loss_quadrant.png",
    )

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"Results saved to {config['paths']['experiments_csv']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment.yaml')
    parser.add_argument('--all-experiments', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke_test:
        print("SMOKE TEST MODE")
        args.seeds = [42]

    if args.all_experiments:
        run_all_experiments(config, args.seeds, args.smoke_test)

