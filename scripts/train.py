"""
Training script for baseline and temporal models.
(research.md §12, §26, §33)

Usage:
    python scripts/train.py --config configs/experiment.yaml --model baseline --seeds 42 43 44
    python scripts/train.py --config configs/experiment.yaml --model gru --seeds 42 43 44
    python scripts/train.py --config configs/experiment.yaml --model transformer --seeds 42 43 44

IMPORTANT: Always run with at least 3 seeds. Report mean ± std.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import sys
import json
import argparse
import random
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import EmbeddingSequenceDataset
from src.models.baseline import SingleFrameMLP, MajorityClassBaseline, build_baseline
from src.models.temporal import build_temporal_model
from src.training import build_optimizer, build_criterion, EarlyStopping, RunTracker, build_run_config
from src.evaluation.metrics import compute_metrics, aggregate_across_seeds, analyze_directional_confusion

# Load benchmark class names if manifest is available
MANIFEST_PATH = Path("data/metadata/dataset_manifest.json")
if MANIFEST_PATH.exists():
    with open(MANIFEST_PATH) as f:
        CLASS_NAMES = json.load(f).get("classes", [f"Class_{i}" for i in range(20)])
else:
    CLASS_NAMES = [f"Class_{i}" for i in range(20)]


def set_seed(seed: int):
    """Deterministic training setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping (research.md: important for GRU stability)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return total_loss / len(loader), metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_probs=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_probs = []

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        logits = model(embeddings)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        if return_probs:
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    if return_probs:
        return total_loss / len(loader), metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)
    return total_loss / len(loader), metrics


def append_to_ledger(csv_path: str, row: dict):
    """Append a single experiment run to the ledger CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_seed(model, config, seed, device, checkpoint_dir, model_type='baseline'):
    """Train a single model with one seed. Returns test metrics and records reproducible run artifacts."""
    set_seed(seed)

    # Datasets
    train_ds = EmbeddingSequenceDataset(
        config['paths']['embeddings'], 'train',
        max_frames=config['sequence']['max_frames'],
    )
    val_ds = EmbeddingSequenceDataset(
        config['paths']['embeddings'], 'val',
        max_frames=config['sequence']['max_frames'],
    )
    test_ds = EmbeddingSequenceDataset(
        config['paths']['embeddings'], 'test',
        max_frames=config['sequence']['max_frames'],
    )

    num_workers = config['training'].get('num_workers', 0)
    pin_memory = (device.type == 'cuda')

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=config['training']['batch_size'],
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=config['training']['batch_size'],
                              shuffle=False, num_workers=num_workers)

    model = model.to(device)
    optimizer, scheduler = build_optimizer(model, config)
    criterion = build_criterion(config, train_ds)

    # Initialize reproducible RunTracker (M3.13)
    num_frames = config['sequence']['max_frames']
    run_id = f"{model_type}_seed{seed}_frames{num_frames}"
    run_config = build_run_config(config, model, model_type, seed, device)
    run_tracker = RunTracker(output_root="runs", run_id=run_id, run_config=run_config)

    best_path = os.path.join(checkpoint_dir, f"best_seed{seed}.pt")
    es_cfg = config['training'].get('early_stopping', {})
    monitor = es_cfg.get('monitor', 'val_accuracy')
    mode = es_cfg.get('mode', 'auto')
    min_delta = float(es_cfg.get('min_delta', 1e-4))
    patience = int(config['training'].get('patience', 5))

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        monitor=monitor,
        mode=mode,
        checkpoint_path=best_path,
        verbose=(seed == config['training']['seeds'][0]),
    )

    print(f"\n  Seed {seed} | Model params: {model.n_parameters():,} | Run ID: {run_id} | EarlyStopping(patience={patience}, monitor={monitor})")

    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=config['training']['gradient_clip'],
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Extract metric to monitor
        if monitor == 'val_loss':
            score = val_loss
        elif monitor in ('val_accuracy', 'val_acc'):
            score = val_metrics['accuracy']
        elif monitor in ('val_macro_f1', 'val_f1'):
            score = val_metrics['macro_f1']
        else:
            score = val_metrics.get(monitor.replace('val_', ''), val_metrics['accuracy'])

        prev_best = early_stopping.best_score
        is_best = (prev_best is None) or early_stopping.is_better(score, prev_best)

        # Log epoch dynamics (M3.13)
        run_tracker.log_train_epoch(epoch, train_loss, train_metrics, current_lr)
        run_tracker.log_val_epoch(epoch, val_loss, val_metrics, is_best=is_best, patience_counter=early_stopping.counter)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"train_acc={train_metrics['accuracy']:.3f} | "
                  f"val_acc={val_metrics['accuracy']:.3f} | "
                  f"val_f1={val_metrics['macro_f1']:.3f}")

        # Check early stopping & checkpoint saving on validation split
        should_stop = early_stopping(score, model, epoch, best_path)
        if is_best:
            run_tracker.save_checkpoint(model)

        if should_stop:
            print(f"  Seed {seed}: Early stopping triggered at epoch {epoch} (best val epoch: {early_stopping.best_epoch})")
            break

    # Strictly untouched test set: restore best validation checkpoint before test evaluation
    early_stopping.restore_best_weights(model, device)
    _, test_metrics, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, criterion, device, return_probs=True
    )
    dir_analysis = analyze_directional_confusion(test_labels, test_preds, class_names=CLASS_NAMES)
    print(f"  → Test Acc={test_metrics['accuracy']:.3f} | "
          f"Macro-F1={test_metrics['macro_f1']:.3f} | "
          f"Macro-P={test_metrics['macro_precision']:.3f} | "
          f"Macro-R={test_metrics['macro_recall']:.3f} | "
          f"DirErr={dir_analysis['formatted_overall_error']} | "
          f"(Restored from best val epoch {early_stopping.best_epoch})")

    # Serialize test predictions, per-class table, confusion matrix, & finalize run metadata (M3.13, M3.14)
    run_tracker.save_predictions(test_ds, test_labels, test_preds, test_probs, class_names=CLASS_NAMES)
    run_tracker.finalize(test_metrics, early_stopping.get_summary(), directional_confusion=dir_analysis)

    test_metrics['best_val_epoch'] = early_stopping.best_epoch
    test_metrics['best_val_score'] = early_stopping.best_score
    test_metrics['stopped_epoch'] = early_stopping.get_summary()['stopped_epoch']
    test_metrics['directional_error_rate'] = dir_analysis['overall_directional_error_rate']
    test_metrics['run_id'] = run_id
    return test_metrics


def run_training(config, model_type, seeds, checkpoint_dir):
    """Train model across multiple seeds and aggregate results."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_type.upper()} | Seeds: {seeds} | Device: {device}")
    print(f"{'='*60}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    results_per_seed = []
    experiment_id_base = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    for seed in seeds:
        # Build fresh model for each seed
        if model_type in ('baseline', 'meanpool'):
            model = build_baseline(config, model_type=model_type)
        else:
            config['temporal']['model'] = model_type
            model = build_temporal_model(config)

        test_metrics = train_one_seed(model, config, seed, device, checkpoint_dir, model_type=model_type)
        results_per_seed.append(test_metrics)

        # Compose experiment notes
        if model_type == 'baseline':
            approach = config.get('baseline', {}).get('approach', 'B')
            sampling = config.get('baseline', {}).get('train_sampling', 'random')
            anchor = config.get('baseline', {}).get('eval_anchor', 'middle')
            notes = f"Approach {approach}: {sampling} train, {anchor} eval"
        elif model_type == 'meanpool':
            notes = "Model B: Order-invariant multi-frame mean-pooling control"
        elif model_type == 'gru':
            h_dim = config['temporal']['hidden_dim']
            p_dim = config['temporal'].get('projection_dim', h_dim)
            notes = f"Model C: Chronological GRU (H={h_dim}, proj={p_dim})"
        elif model_type == 'compact_gru':
            notes = "Model D: Capacity-matched CompactGRU (103,700 params, exact parity)"
        else:
            notes = ''

        # Append to experiment ledger
        append_to_ledger(config['paths']['experiments_csv'], {
            'experiment_id': f"{experiment_id_base}_s{seed}",
            'dataset': config['dataset']['name'],
            'model': model_type,
            'num_frames': config['sequence']['max_frames'],
            'order': 'canonical',
            'corruption': 'none',
            'seed': seed,
            'accuracy': round(test_metrics['accuracy'], 4),
            'macro_f1': round(test_metrics['macro_f1'], 4),
            'notes': notes,
        })

    # Aggregate
    aggregated = aggregate_across_seeds(results_per_seed)
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({model_type}) across {len(seeds)} seeds:")
    print(f"  Accuracy : {aggregated['accuracy']['formatted']}")
    print(f"  Macro-F1 : {aggregated['macro_f1']['formatted']}")
    if 'macro_precision' in aggregated:
        print(f"  Macro-P  : {aggregated['macro_precision']['formatted']}")
    if 'macro_recall' in aggregated:
        print(f"  Macro-R  : {aggregated['macro_recall']['formatted']}")
    print(f"{'='*60}")

    return aggregated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment.yaml')
    parser.add_argument('--model', choices=['baseline', 'meanpool', 'gru', 'transformer'], required=True)
    parser.add_argument('--baseline-approach', choices=['A', 'B'], default=None,
                        help='M3.3 single-frame baseline approach: A (fixed frame) or B (random train frame)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run 3 epochs for quick sanity check')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if 'baseline' not in config:
        config['baseline'] = {}
    if args.baseline_approach == 'A':
        config['baseline']['approach'] = 'A'
        config['baseline']['train_sampling'] = 'fixed'
        print("BASELINE APPROACH A: Fixed frame training & evaluation")
    elif args.baseline_approach == 'B':
        config['baseline']['approach'] = 'B'
        config['baseline']['train_sampling'] = 'random'
        print("BASELINE APPROACH B: Random frame training, deterministic midpoint evaluation")

    if args.smoke_test:
        config['training']['epochs'] = 3
        args.seeds = [42]
        print("SMOKE TEST MODE: 3 epochs, seed=42 only")

    checkpoint_dir = os.path.join(config['paths']['checkpoints'], args.model)
    run_training(config, args.model, args.seeds, checkpoint_dir)
