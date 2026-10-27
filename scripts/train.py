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
from src.evaluation.metrics import compute_metrics, aggregate_across_seeds


def set_seed(seed: int):
    """Deterministic training setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, config: dict):
    """AdamW with cosine scheduler — identical for all models."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
    )
    return optimizer, scheduler


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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        logits = model(embeddings)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
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


def train_one_seed(model, config, seed, device, checkpoint_dir):
    """Train a single model with one seed. Returns test metrics."""
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

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=config['training']['batch_size'],
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=config['training']['batch_size'],
                              shuffle=False, num_workers=2)

    model = model.to(device)
    optimizer, scheduler = build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = config['training']['patience']
    patience_counter = 0
    best_path = os.path.join(checkpoint_dir, f"best_seed{seed}.pt")

    print(f"\n  Seed {seed} | Model params: {model.n_parameters():,}")

    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=config['training']['gradient_clip'],
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"train_acc={train_metrics['accuracy']:.3f} | "
                  f"val_acc={val_metrics['accuracy']:.3f} | "
                  f"val_f1={val_metrics['macro_f1']:.3f}")

        # Best model checkpoint
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best model for test evaluation
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  → Test Acc={test_metrics['accuracy']:.3f} | "
          f"Macro-F1={test_metrics['macro_f1']:.3f}")

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

        test_metrics = train_one_seed(model, config, seed, device, checkpoint_dir)
        results_per_seed.append(test_metrics)

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
            'notes': '',
        })

    # Aggregate
    aggregated = aggregate_across_seeds(results_per_seed)
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({model_type}) across {len(seeds)} seeds:")
    print(f"  Accuracy : {aggregated['accuracy']['formatted']}")
    print(f"  Macro-F1 : {aggregated['macro_f1']['formatted']}")
    print(f"{'='*60}")

    return aggregated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment.yaml')
    parser.add_argument('--model', choices=['baseline', 'meanpool', 'gru', 'transformer'], required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run 3 epochs for quick sanity check')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke_test:
        config['training']['epochs'] = 3
        args.seeds = [42]
        print("SMOKE TEST MODE: 3 epochs, seed=42 only")

    checkpoint_dir = os.path.join(config['paths']['checkpoints'], args.model)
    run_training(config, args.model, args.seeds, checkpoint_dir)
