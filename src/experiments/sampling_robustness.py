"""
Milestone M6.2: Temporal Sampling Robustness Analysis.
Evaluates whether the chronological temporal advantage survives alternative frame-sampling patterns.

Evaluated Strategies:
  - uniform: full 8-frame timeline [0, 1, 2, 3, 4, 5, 6, 7] (Control)
  - sparse: stride-based decimation [0, 2, 4, 6] (Low Nyquist limit)
  - dense: consecutive prefix window [0, 1, 2, 3] (Early observation commitment)
  - random: stochastic 4-frame subset (seeded per sequence)
  - jittered: uniform anchors with +/- 1 frame displacement (Sensor timestamp noise)
"""

from __future__ import annotations

import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.baseline import SingleFrameMLP, MultiFrameMeanPool, build_baseline
from src.models.mean_pool import build_mean_pool_model
from src.models.temporal import build_temporal_model, TemporalGRU
from src.data.sampling import uniform_sample, sparse_sample, first_n, jittered_sample
from src.evaluation.metrics import compute_metrics, mcnemar_test, bootstrap_ci
from src.experiments.checkpoint_paths import resolve_checkpoint_dir


def sample_indices_for_strategy(
    strategy: str,
    total_frames: int = 8,
    seed: int = 42,
) -> List[int]:
    """Return indices for a sequence according to sampling strategy."""
    if strategy == "uniform":
        return list(range(total_frames))
    elif strategy == "sparse":
        return [0, 2, 4, 6]
    elif strategy == "dense":
        return [0, 1, 2, 3]
    elif strategy == "random":
        rng = np.random.RandomState(seed)
        return sorted(rng.choice(total_frames, 4, replace=False).tolist())
    elif strategy == "jittered":
        return jittered_sample(total_frames, total_frames, jitter=1, seed=seed)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


class SamplingRobustnessDataset(Dataset):
    """Loads test sequences and applies sampling strategy."""
    def __init__(
        self,
        strategy: str,
        metadata_path: str = "embeddings/metadata.json",
        embeddings_dir: str = "embeddings/test",
    ):
        self.strategy = strategy
        self.embeddings_dir = Path(embeddings_dir)
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        self.samples = [x for x in meta if x["split"] == "test"]
        self.samples = sorted(self.samples, key=lambda x: x["sequence_id"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        seq_id = item["sequence_id"]
        path = self.embeddings_dir / f"{seq_id}.npy"
        arr = np.load(str(path))  # [8, 384]

        indices = sample_indices_for_strategy(
            self.strategy,
            total_frames=len(arr),
            seed=42 + idx,
        )
        sampled = arr[indices]
        return torch.from_numpy(sampled).float(), item["label"]


def load_model_checkpoint(
    model_name: str,
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Load model from saved checkpoint."""
    if model_name == "baseline":
        model = build_baseline(config)
    elif model_name == "meanpool":
        model = build_mean_pool_model(config)
    elif model_name == "gru":
        model = build_temporal_model(config)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class M6SamplingRobustnessRunner:
    """Evaluates sampling robustness across models and seeds."""
    def __init__(
        self,
        config: Dict[str, Any],
        strategies: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
    ):
        self.config = config
        self.strategies = strategies or ["uniform", "sparse", "dense", "random", "jittered"]
        self.seeds = seeds or [42, 43, 44]
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    def run(self) -> Dict[str, Any]:
        results = []
        ckpts_base = resolve_checkpoint_dir(Path.cwd())

        print(f"\n[M6.2 Sampling Robustness] Evaluating {len(self.strategies)} strategies on {self.device}")

        for strat in self.strategies:
            print(f"\nEvaluating strategy: {strat.upper()}...")
            ds = SamplingRobustnessDataset(strategy=strat)
            loader = DataLoader(ds, batch_size=32, shuffle=False)

            model_metrics = {}
            model_preds = {}
            labels = None

            for model_name in ["baseline", "meanpool", "gru"]:
                seed_accs, seed_f1s = [], []
                last_preds = None

                for s in self.seeds:
                    ckpt_p = ckpts_base / model_name / f"best_seed{s}.pt"
                    if not ckpt_p.exists():
                        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

                    model = load_model_checkpoint(model_name, str(ckpt_p), self.config, self.device)
                    all_preds, all_labels = [], []
                    with torch.no_grad():
                        for x, y in loader:
                            x = x.to(self.device)
                            logits = model(x)
                            preds = logits.argmax(dim=-1).cpu().numpy()
                            all_preds.extend(preds)
                            all_labels.extend(y.numpy())

                    y_true = np.array(all_labels)
                    y_pred = np.array(all_preds)
                    if labels is None:
                        labels = y_true

                    m = compute_metrics(y_true, y_pred)
                    seed_accs.append(m["accuracy"])
                    seed_f1s.append(m["macro_f1"])
                    last_preds = y_pred

                model_metrics[f"{model_name}_acc_mean"] = float(np.mean(seed_accs))
                model_metrics[f"{model_name}_acc_std"] = float(np.std(seed_accs))
                model_metrics[f"{model_name}_f1_mean"] = float(np.mean(seed_f1s))
                model_preds[model_name] = last_preds

            acc_a = model_metrics["baseline_acc_mean"]
            acc_b = model_metrics["meanpool_acc_mean"]
            acc_c = model_metrics["gru_acc_mean"]

            delta_temp = acc_c - acc_a
            delta_order = acc_c - acc_b
            mcnemar_p = mcnemar_test(model_preds["meanpool"], model_preds["gru"], labels)

            entry = {
                "strategy": strat,
                "baseline_acc": acc_a,
                "baseline_acc_std": model_metrics["baseline_acc_std"],
                "meanpool_acc": acc_b,
                "meanpool_acc_std": model_metrics["meanpool_acc_std"],
                "gru_acc": acc_c,
                "gru_acc_std": model_metrics["gru_acc_std"],
                "delta_temporal": delta_temp,
                "delta_order": delta_order,
                "mcnemar_p": mcnemar_p,
                "survived": bool(delta_temp > 0.0),
            }

            print(f"  Single MLP:  {acc_a*100:.2f} ± {entry['baseline_acc_std']*100:.2f}%")
            print(f"  MeanPool:    {acc_b*100:.2f} ± {entry['meanpool_acc_std']*100:.2f}%")
            print(f"  TemporalGRU: {acc_c*100:.2f} ± {entry['gru_acc_std']*100:.2f}%")
            print(f"  Δ_temporal:  {delta_temp*100:+.2f} pp | Δ_order: {delta_order*100:+.2f} pp (p={mcnemar_p:.2e}) | Survives: {entry['survived']}")

            results.append(entry)

        return {
            "strategies": results,
            "all_survived": all(r["survived"] for r in results),
        }

