#!/usr/bin/env python3
"""
Diagnostic headline-metric recomputation across frozen checkpoints.
Compares current checkpoint outputs with stored claims; does not establish that
the original reported run used the same protocol or provenance.

Usage:
    python3 scripts/reproduce_results.py
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import sys
import json
import csv
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

from src.data.dataset import EmbeddingSequenceDataset
from src.models.baseline import build_baseline
from src.models.mean_pool import build_mean_pool_model
from src.models.temporal import build_temporal_model
from src.models.compact_gru import build_compact_gru_model, CompactTemporalGRU
from src.experiments.shuffle import apply_order_perturbation
from src.experiments.checkpoint_paths import resolve_checkpoint_dir
from src.data.progressive import sample_progressive_frames
from src.evaluation.metrics import compute_metrics, mcnemar_test, bootstrap_ci


CLAIMED_HEADLINE_FINDINGS = [
    {
        "finding_id": "F1",
        "name": "Model C Canonical Accuracy (K=8)",
        "claimed_value": "34.78 ± 1.10%",
        "claimed_metric": 0.3478,
        "tolerance": 0.001,
    },
    {
        "finding_id": "F2",
        "name": "Model B MeanPool Accuracy (K=8)",
        "claimed_value": "8.22 ± 1.26%",
        "claimed_metric": 0.0822,
        "tolerance": 0.001,
    },
    {
        "finding_id": "F3",
        "name": "Model A SingleFrame Baseline Accuracy",
        "claimed_value": "8.78 ± 3.28%",
        "claimed_metric": 0.0878,
        "tolerance": 0.001,
    },
    {
        "finding_id": "F4",
        "name": "Capacity-Matched Parity Gain (Model D vs Model B)",
        "claimed_value": "Checkpoint means differ from M4 summary (parity protocol to reconcile)",
        "claimed_metric": 0.1534, # upper reference
        "tolerance": 0.001,
    },
    {
        "finding_id": "F5",
        "name": "Temporal Order Destruction Drop (OrderDrop under shuffle)",
        "claimed_value": "Historical M4 artifacts report 13.89 pp",
        "claimed_metric": 0.1389,
        "tolerance": 0.001,
    },
    {
        "finding_id": "F6",
        "name": "Observation Horizon Scaling: K=1 to K=8",
        "claimed_value": "Historical M4 artifacts report 11.56% to 34.78% (+23.22 pp)",
        "claimed_metric": 0.2322,
        "tolerance": 0.001,
    },
]


def eval_checkpoint(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    order: str = "canonical",
    k_frames: int = 8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    # run_order_experiment uses one persistent generator for a full evaluation
    # pass. Reset it once here so each batch continues the same deterministic
    # permutation stream; reseeding per batch changes the intervention.
    generator = None
    if order == "shuffled":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    with torch.no_grad():
        for b_idx, (x, y) in enumerate(loader):
            if k_frames < 8:
                # Match the progressive experiment: K=1 uses the midpoint
                # anchor and K>1 uniformly covers the sequence duration.
                x = sample_progressive_frames(x, k=k_frames, mode="uniform")
            if order != "canonical":
                x = apply_order_perturbation(
                    x, mode=order, seed=seed, generator=generator
                )
            x = x.to(device)
            preds = model(x).argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return np.array(all_preds), np.array(all_labels)


def main():
    print("\n" + "=" * 76)
    print("  DIAGNOSTIC HEADLINE METRIC RECOMPUTATION")
    print("=" * 76)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"  Execution device: {device}")

    with open("configs/experiment.yaml") as f:
        config = yaml.safe_load(f)

    ds = EmbeddingSequenceDataset("embeddings", split="test")
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    seeds = [42, 43, 44]
    ckpts_base = resolve_checkpoint_dir(PROJECT_ROOT)

    remeasured = {}

    # 1. Remeasure Model A, B, C, D on canonical test split
    print("\n[1/4] Remeasuring Canonical Models across Seeds [42, 43, 44]...")
    for m_name in ["baseline", "meanpool", "gru", "compact_gru"]:
        accs, f1s = [], []
        for s in seeds:
            ckpt_p = ckpts_base / m_name / f"best_seed{s}.pt"
            if m_name == "baseline":
                m = build_baseline(config)
            elif m_name == "meanpool":
                m = build_mean_pool_model(config)
            elif m_name == "gru":
                m = build_temporal_model(config)
            else:
                m = build_compact_gru_model(config)
            m.load_state_dict(torch.load(str(ckpt_p), map_location=device))
            m.to(device).eval()

            preds, targets = eval_checkpoint(m, loader, device, order="canonical")
            accs.append(float(np.mean(preds == targets)))
            f1s.append(float(compute_metrics(targets, preds)["macro_f1"]))

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))
        remeasured[m_name] = {
            "mean": mean_acc,
            "std": std_acc,
            "accs": accs,
            "macro_f1_mean": mean_f1,
            "macro_f1_std": std_f1,
            "macro_f1s": f1s,
        }
        print(
            f"  {m_name:<12} : {mean_acc*100:.2f} ± {std_acc*100:.2f}% | "
            f"Macro-F1 {mean_f1:.4f} ± {std_f1:.4f}"
        )

    # 2. Remeasure Shuffle & Reversal on Model C
    print("\n[2/4] Remeasuring Order & Reversal Ablations on Model C...")
    shuf_accs, rev_accs = [], []
    for s in seeds:
        m = build_temporal_model(config)
        m.load_state_dict(torch.load(str(ckpts_base / "gru" / f"best_seed{s}.pt"), map_location=device))
        m.to(device).eval()

        # The experiment holds the intervention seed fixed at 42 across model
        # training seeds (run_order_experiment's perturbation_seed default).
        p_shuf, y = eval_checkpoint(m, loader, device, order="shuffled", seed=42)
        p_rev, _ = eval_checkpoint(m, loader, device, order="reversed")
        shuf_accs.append(float(np.mean(p_shuf == y)))
        rev_accs.append(float(np.mean(p_rev == y)))

    mean_shuf = float(np.mean(shuf_accs))
    mean_rev = float(np.mean(rev_accs))
    order_drop = remeasured["gru"]["mean"] - mean_shuf
    rev_drop = remeasured["gru"]["mean"] - mean_rev
    remeasured["order_drop"] = order_drop
    remeasured["reversal_drop"] = rev_drop
    print(f"  Shuffled GRU : {mean_shuf*100:.2f}% -> OrderDrop = {order_drop*100:+.2f} pp")
    print(f"  Reversed GRU : {mean_rev*100:.2f}% -> ReversalDrop = {rev_drop*100:+.2f} pp")

    # 3. Remeasure Progressive Scaling K=1 vs K=8 on Model C
    print("\n[3/4] Remeasuring Progressive Horizon (K=1 vs K=8) on Model C...")
    k1_accs = []
    for s in seeds:
        m = build_temporal_model(config)
        m.load_state_dict(torch.load(str(ckpts_base / "gru" / f"best_seed{s}.pt"), map_location=device))
        m.to(device).eval()
        p1, y = eval_checkpoint(m, loader, device, k_frames=1)
        k1_accs.append(float(np.mean(p1 == y)))

    mean_k1 = float(np.mean(k1_accs))
    horizon_gain = remeasured["gru"]["mean"] - mean_k1
    remeasured["k1_accuracy"] = mean_k1
    remeasured["horizon_gain"] = horizon_gain
    print(f"  K=1 Accuracy : {mean_k1*100:.2f}% | K=8 Accuracy: {remeasured['gru']['mean']*100:.2f}% -> Gain: {horizon_gain*100:+.2f} pp")

    # 4. Capacity Parity Parity (Compact GRU vs MeanPool)
    capacity_gain = remeasured["compact_gru"]["mean"] - remeasured["meanpool"]["mean"]
    remeasured["capacity_parity_gain"] = capacity_gain
    print(f"  Capacity Gain: CompactGRU ({remeasured['compact_gru']['mean']*100:.2f}%) vs MeanPool ({remeasured['meanpool']['mean']*100:.2f}%) -> {capacity_gain*100:+.2f} pp")

    # Build reconciliation table
    print("\n[4/4] Generating Reconciliation Report...")
    reconciliation_records = []

    actual_map = {
        "F1": (f"{remeasured['gru']['mean']*100:.2f} ± {remeasured['gru']['std']*100:.2f}%", remeasured["gru"]["mean"]),
        "F2": (f"{remeasured['meanpool']['mean']*100:.2f} ± {remeasured['meanpool']['std']*100:.2f}%", remeasured["meanpool"]["mean"]),
        "F3": (f"{remeasured['baseline']['mean']*100:.2f} ± {remeasured['baseline']['std']*100:.2f}%", remeasured["baseline"]["mean"]),
        "F4": (f"+{capacity_gain*100:.2f} pp", capacity_gain),
        "F5": (f"{order_drop*100:.2f} pp", order_drop),
        "F6": (f"{mean_k1*100:.2f}% to {remeasured['gru']['mean']*100:.2f}% (+{horizon_gain*100:.2f} pp)", horizon_gain),
    }

    all_reconciled = True
    for c in CLAIMED_HEADLINE_FINDINGS:
        fid = c["finding_id"]
        act_str, act_val = actual_map[fid]
        claimed_val = c["claimed_metric"]
        tol = c["tolerance"]
        diff = abs(act_val - claimed_val)
        matched = diff <= tol
        if not matched:
            all_reconciled = False
        status = "RECOMPUTED" if matched else "DIFFERS FROM CLAIM"

        reconciliation_records.append({
            "finding_id": fid,
            "headline_finding": c["name"],
            "claimed_reference": c["claimed_value"],
            "remeasured_result": act_str,
            "reconciliation_status": status,
        })

    # Save reconciliation markdown table
    out_table_path = PROJECT_ROOT / "results/tables/m7_reconciliation_table.md"
    lines = [
        "# Milestone M7.3: Headline Findings Reproduction & Reconciliation Table",
        "",
        "> **Objective**: Record diagnostic recomputation from currently available checkpoints. This does not prove the original protocol or resolve conflicting historical artifacts.",
        "",
        "| ID | Headline Empirical Finding | Historical Reference | Current Checkpoint Recompute | Status |",
        "|:---:|---|---|:---:|:---:|",
    ]

    for r in reconciliation_records:
        lines.append(f"| **{r['finding_id']}** | {r['headline_finding']} | {r['claimed_reference']} | **{r['remeasured_result']}** | {r['reconciliation_status']} |")

    lines.append("")
    lines.append("**Summary**: This script recomputed values from current checkpoints. Conflicting historical values are reported as differences; matching a stored canonical score does not establish full protocol provenance or fresh independent replication.")

    with open(out_table_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Save JSON metrics
    final_dir = PROJECT_ROOT / "results/final"
    final_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hashes = {
        f"{model}_seed{seed}": sha256_file(ckpts_base / model / f"best_seed{seed}.pt")
        for model in ["baseline", "meanpool", "gru", "compact_gru"]
        for seed in seeds
    }
    remeasured["_metadata"] = {
        "status": "diagnostic_current_checkpoint_recomputation",
        "split": "test",
        "seeds": seeds,
        "accuracy_std": "population standard deviation across seeds",
        "progressive_sampling": "uniform coverage; K=1 uses midpoint anchor",
        "shuffle_protocol": "persistent torch.Generator seeded at 42 across batches and model seeds",
        "checkpoint_sha256": checkpoint_hashes,
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/experiment.yaml"),
        "prediction_artifact_sources": [
            "results/experiments/progressive/progressive_predictions.csv",
            "results/experiments/shuffle/order_predictions.csv",
            "results/experiments/reversal/reversal_predictions.csv",
        ],
        "limitations": [
            "Re-evaluates available checkpoints; does not independently reproduce training.",
            "Historical statistical significance artifacts are not validated by this script.",
            "Raw per-example predictions are not regenerated or saved by this diagnostic.",
        ],
    }
    with open(final_dir / "headline_metrics.json", "w") as f:
        json.dump(remeasured, f, indent=2)
    with open(final_dir / "reconciliation_manifest.json", "w") as f:
        json.dump({"timestamp": datetime.utcnow().isoformat() + "Z", "records": reconciliation_records}, f, indent=2)

    print("=" * 76)
    print(f"  RESULT: {len(reconciliation_records)} values recomputed; protocol/provenance review remains required")
    print(f"  Table saved to: {out_table_path.relative_to(PROJECT_ROOT)}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
