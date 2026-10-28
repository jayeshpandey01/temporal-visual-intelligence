"""
Milestone M6.3: Systematic Degradation Curves and Observation Robustness.
Evaluates graceful degradation under missing observations, frame drops, and latent feature noise.

Quantifies:
  - Degradation function: D(r) = Performance(0) - Performance(r)
  - Comparison: D_GRU(r) vs D_Mean(r)
  - Missing observation fractions: r in {0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75}
  - Latent Gaussian noise: sigma in {0.0, 0.1, 0.3, 0.5, 1.0}
  - Structural corruptions: frame duplication, temporal jitter, frame omission
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
from src.evaluation.metrics import compute_metrics, mcnemar_test, bootstrap_ci
from src.experiments.checkpoint_paths import resolve_checkpoint_dir


class TestEmbeddingsDataset(Dataset):
    """Loads clean test embeddings."""
    def __init__(self, metadata_path: str = "embeddings/metadata.json", embeddings_dir: str = "embeddings/test"):
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
        arr = np.load(str(path))
        return torch.from_numpy(arr).float(), item["label"]


def apply_degradation(
    x: torch.Tensor,
    mode: str,
    rate: float = 0.0,
    sigma: float = 0.0,
    seed: int = 42,
) -> torch.Tensor:
    """
    Apply corruption to sequence tensor [B, T, D].
    """
    B, T, D = x.shape
    out = x.clone()
    rng = np.random.RandomState(seed)

    if mode == "frame_omission":
        # Zero-mask k frames
        k = int(round(T * rate))
        if k > 0:
            for b in range(B):
                drop_idx = rng.choice(T, k, replace=False)
                out[b, drop_idx, :] = 0.0
    elif mode == "frame_duplication":
        # Randomly duplicate frames to simulate frame-stalls
        k = max(1, int(round(T * rate)))
        for b in range(B):
            target_idx = rng.choice(T - 1, min(k, T - 1), replace=False)
            for idx in target_idx:
                out[b, idx + 1] = out[b, idx]
    elif mode == "temporal_jitter":
        # Randomly swap adjacent frames
        for b in range(B):
            for t in range(T - 1):
                if rng.rand() < rate:
                    temp = out[b, t].clone()
                    out[b, t] = out[b, t + 1]
                    out[b, t + 1] = temp
    elif mode == "feature_noise":
        if sigma > 0.0:
            noise = torch.randn_like(out) * sigma
            out = out + noise
    return out


class M6DegradationCurveRunner:
    """Computes systematic D(r) degradation curves and noise sweeps."""
    def __init__(self, config: Dict[str, Any], seeds: Optional[List[int]] = None):
        self.config = config
        self.seeds = seeds or [42, 43, 44]
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.missing_rates = [0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75]
        self.noise_sigmas = [0.0, 0.1, 0.3, 0.5, 1.0]

    def _eval_condition(
        self,
        models: Dict[str, List[nn.Module]],
        loader: DataLoader,
        mode: str,
        rate: float = 0.0,
        sigma: float = 0.0,
    ) -> Dict[str, float]:
        scores = {m: [] for m in models}

        for model_name, model_list in models.items():
            for seed_idx, model in enumerate(model_list):
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for b_idx, (x, y) in enumerate(loader):
                        corrupted_x = apply_degradation(
                            x, mode=mode, rate=rate, sigma=sigma, seed=42 + b_idx * 10 + seed_idx
                        ).to(self.device)
                        logits = model(corrupted_x)
                        preds = logits.argmax(dim=-1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(y.numpy())
                acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
                scores[model_name].append(acc)

        return {m: float(np.mean(scores[m])) for m in models}

    def run(self) -> Dict[str, Any]:
        ckpts_base = resolve_checkpoint_dir(Path.cwd())
        models = {"baseline": [], "meanpool": [], "gru": []}

        for m_name in models:
            for s in self.seeds:
                ckpt_p = ckpts_base / m_name / f"best_seed{s}.pt"
                if m_name == "baseline":
                    m = build_baseline(self.config)
                elif m_name == "meanpool":
                    m = build_mean_pool_model(self.config)
                else:
                    m = build_temporal_model(self.config)
                m.load_state_dict(torch.load(str(ckpt_p), map_location=self.device))
                m.to(self.device).eval()
                models[m_name].append(m)

        ds = TestEmbeddingsDataset()
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        # 1. Clean reference performance (r=0)
        clean_scores = self._eval_condition(models, loader, mode="frame_omission", rate=0.0)
        clean_gru = clean_scores["gru"]
        clean_mean = clean_scores["meanpool"]
        clean_base = clean_scores["baseline"]

        print(f"\n[M6.3 Degradation] Clean Accuracies: Baseline={clean_base*100:.2f}%, MeanPool={clean_mean*100:.2f}%, GRU={clean_gru*100:.2f}%")

        # 2. Missing observation rate sweep
        print("\n--- Missing Observation Sweep r in [0.0 .. 0.75] ---")
        drop_results = []
        for r in self.missing_rates:
            scores = self._eval_condition(models, loader, mode="frame_omission", rate=r)
            d_gru = clean_gru - scores["gru"]
            d_mean = clean_mean - scores["meanpool"]
            delta = scores["gru"] - scores["meanpool"]
            entry = {
                "rate": r,
                "n_dropped": int(round(8 * r)),
                "gru_acc": scores["gru"],
                "meanpool_acc": scores["meanpool"],
                "baseline_acc": scores["baseline"],
                "d_gru": d_gru,
                "d_mean": d_mean,
                "delta_advantage": delta,
                "gru_superior": bool(delta > 0),
            }
            drop_results.append(entry)
            print(f"  r={r*100:4.1f}% ({entry['n_dropped']}/8 drops) | GRU={scores['gru']*100:.2f}% (D={d_gru*100:+.2f}pp) | Mean={scores['meanpool']*100:.2f}% (D={d_mean*100:+.2f}pp) | Margin={delta*100:+.2f}pp")

        # 3. Latent Gaussian noise sweep
        print("\n--- Latent Gaussian Feature Noise Sweep sigma in [0.0 .. 1.0] ---")
        noise_results = []
        for sig in self.noise_sigmas:
            scores = self._eval_condition(models, loader, mode="feature_noise", sigma=sig)
            d_gru = clean_gru - scores["gru"]
            d_mean = clean_mean - scores["meanpool"]
            delta = scores["gru"] - scores["meanpool"]
            entry = {
                "sigma": sig,
                "gru_acc": scores["gru"],
                "meanpool_acc": scores["meanpool"],
                "baseline_acc": scores["baseline"],
                "d_gru": d_gru,
                "d_mean": d_mean,
                "delta_advantage": delta,
            }
            noise_results.append(entry)
            print(f"  sigma={sig:.2f} | GRU={scores['gru']*100:.2f}% | Mean={scores['meanpool']*100:.2f}% | Margin={delta*100:+.2f}pp")

        # 4. Other structural corruptions (duplication, jitter)
        print("\n--- Structural Corruptions (Duplication, Jitter) ---")
        structural_results = {}
        for corr in ["frame_duplication", "temporal_jitter"]:
            scores = self._eval_condition(models, loader, mode=corr, rate=0.25)
            structural_results[corr] = {
                "gru_acc": scores["gru"],
                "meanpool_acc": scores["meanpool"],
                "delta": scores["gru"] - scores["meanpool"],
            }
            print(f"  {corr} (25%): GRU={scores['gru']*100:.2f}% | MeanPool={scores['meanpool']*100:.2f}% | Margin={structural_results[corr]['delta']*100:+.2f}pp")

        return {
            "clean_scores": clean_scores,
            "drop_curves": drop_results,
            "noise_curves": noise_results,
            "structural": structural_results,
        }

