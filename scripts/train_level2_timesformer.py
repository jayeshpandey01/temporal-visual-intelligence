#!/usr/bin/env python3
"""
Dedicated Level 2 TimeSformer Training Script.
Spatio-Temporal Divided Space-Time Attention across P=16 spatial patches and T=8 frames.
"""

import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp')
import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.timesformer import TimeSformerModel
from src.data.patch_dataset import SpatialPatchSequenceDataset
from src.training.early_stopping import EarlyStopping
from src.evaluation.metrics import compute_metrics, aggregate_across_seeds
from scripts.train import append_to_ledger, set_seed


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for patches, labels in loader:
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

    for patches, labels in loader:
        patches = patches.to(device)
        labels = labels.to(device)
        logits = model(patches)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return total_loss / len(loader), metrics


def train_seed(seed, epochs, device, batch_size=32):
    set_seed(seed)
    train_ds = SpatialPatchSequenceDataset("embeddings", "train", max_frames=8, seed=seed)
    val_ds   = SpatialPatchSequenceDataset("embeddings", "val", max_frames=8, seed=seed)
    test_ds  = SpatialPatchSequenceDataset("embeddings", "test", max_frames=8, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = TimeSformerModel(
        embed_dim=128,
        num_classes=20,
        num_layers=2,
        num_heads=4,
        max_frames=8,
        max_patches=16,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_ckpt = f"results/checkpoints/timesformer/best_seed{seed}.pt"
    os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)
    es = EarlyStopping(patience=8, monitor="val_accuracy", mode="max", checkpoint_path=best_ckpt)

    print(f"\n  Seed {seed} | Params: {model.n_parameters():,} | Training TimeSformer...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d} | train_acc={tr_m['accuracy']:.3f} | val_acc={val_m['accuracy']:.3f}")

        if es(val_m['accuracy'], model, epoch, best_ckpt):
            print(f"  Early stopping at epoch {epoch}")
            break

    es.restore_best_weights(model, device)
    _, test_m = evaluate(model, test_loader, criterion, device)
    print(f"  → Test Acc={test_m['accuracy']:.3f} | Macro-F1={test_m['macro_f1']:.3f}")
    return test_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 3
        args.seeds = [42]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}\nTRAINING LEVEL 2: TIMESFORMER (Spatio-Temporal Patches) | Seeds: {args.seeds}\n{'='*60}")

    results = []
    base_id = f"timesformer_l2_{datetime.now().strftime('%Y%m%d_%H%M')}"
    csv_path = "results/experiments.csv"

    for seed in args.seeds:
        m = train_seed(seed, args.epochs, device)
        results.append(m)

        append_to_ledger(csv_path, {
            'experiment_id': f"{base_id}_s{seed}",
            'dataset': "ssv2",
            'model': "timesformer_l2",
            'num_frames': 8,
            'order': 'canonical',
            'corruption': 'none',
            'seed': seed,
            'accuracy': round(m['accuracy'], 4),
            'macro_f1': round(m['macro_f1'], 4),
            'notes': "Level 2: Divided Space-Time Attention over P=16 patches across 8 frames",
        })

    agg = aggregate_across_seeds(results)
    print(f"\n{'='*60}\nFINAL LEVEL 2 RESULTS across {len(args.seeds)} seeds:")
    print(f"  Accuracy : {agg['accuracy']['formatted']}")
    print(f"  Macro-F1 : {agg['macro_f1']['formatted']}\n{'='*60}")


if __name__ == "__main__":
    main()
