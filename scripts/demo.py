#!/usr/bin/env python3
"""
Milestone M7.4: Scientific Interactive Demonstration & Sequence Inspector.
Demonstrates model predictions, dynamic temporal interventions, and failure modes.

Usage:
    # Run interactive sequence inspector:
    python3 scripts/demo.py

    # Non-interactive CLI inspection:
    python3 scripts/demo.py --sequence-id vid_07_0001 --condition canonical
    python3 scripts/demo.py --sequence-id vid_07_0001 --condition shuffled
    python3 scripts/demo.py --mode rescue
    python3 scripts/demo.py --mode failure
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import build_baseline
from src.models.mean_pool import build_mean_pool_model
from src.models.temporal import build_temporal_model
from src.models.compact_gru import build_compact_gru_model
from src.experiments.shuffle import apply_order_perturbation
from src.experiments.checkpoint_paths import resolve_checkpoint_dir


PRESET_SEQUENCES = {
    "rescue": {
        "sequence_id": "vid_07_0001",
        "name": "Quadrant C: Pure Temporal Rescue (State Direction)",
        "description": "Baseline & MeanPool fail due to static ambiguity; Recurrent GRU successfully predicts action.",
    },
    "failure": {
        "sequence_id": "vid_05_0002",
        "name": "Quadrant B: Temporal Regression Failure Case",
        "description": "Single-frame anchor succeeds; GRU overfits to background parallax in egocentric camera motion.",
    },
}


class ScientificSequenceInspector:
    """Inspector executing inference and interventions on individual sequences."""
    def __init__(self, config_path: str = "configs/m7_final.yaml", seed: int = 42):
        with open(PROJECT_ROOT / config_path) as f:
            self.config = yaml.safe_load(f)

        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        # Load class names
        with open(PROJECT_ROOT / "data/metadata/dataset_manifest.json") as f:
            self.class_names = json.load(f)["classes"]

        # Load metadata
        with open(PROJECT_ROOT / "embeddings/metadata.json") as f:
            self.metadata = {x["sequence_id"]: x for x in json.load(f)}

        # Load models from seed checkpoints
        ckpts_base = resolve_checkpoint_dir(PROJECT_ROOT)
        self.models = {}

        # Baseline Model A
        mA = build_baseline(self.config)
        mA.load_state_dict(torch.load(ckpts_base / "baseline" / f"best_seed{seed}.pt", map_location=self.device))
        self.models["Model A (SingleFrame MLP)"] = mA.to(self.device).eval()

        # Model B MeanPool
        mB = build_mean_pool_model(self.config)
        mB.load_state_dict(torch.load(ckpts_base / "meanpool" / f"best_seed{seed}.pt", map_location=self.device))
        self.models["Model B (MultiFrame MeanPool)"] = mB.to(self.device).eval()

        # Model C Temporal GRU
        mC = build_temporal_model(self.config)
        mC.load_state_dict(torch.load(ckpts_base / "gru" / f"best_seed{seed}.pt", map_location=self.device))
        self.models["Model C (Temporal GRU)"] = mC.to(self.device).eval()

        # Model D Compact GRU
        mD = build_compact_gru_model(self.config)
        mD.load_state_dict(torch.load(ckpts_base / "compact_gru" / f"best_seed{seed}.pt", map_location=self.device))
        self.models["Model D (Compact GRU - Parity)"] = mD.to(self.device).eval()

    def inspect_sequence(
        self,
        sequence_id: str,
        condition: str = "canonical",
        seed: int = 42,
    ) -> Dict[str, Any]:
        meta = self.metadata.get(sequence_id)
        if not meta:
            raise ValueError(f"Sequence ID '{sequence_id}' not found in metadata manifest.")

        split = meta.get("split", "test")
        emb_path = PROJECT_ROOT / "embeddings" / split / f"{sequence_id}.npy"
        arr = np.load(str(emb_path))  # [8, 384]

        x_raw = torch.from_numpy(arr).float().unsqueeze(0)  # [1, 8, 384]

        # Apply intervention
        if condition == "canonical":
            x_intervened = x_raw.clone()
        elif condition == "shuffled":
            x_intervened = apply_order_perturbation(x_raw, mode="shuffled", seed=seed)
        elif condition == "reversed":
            x_intervened = apply_order_perturbation(x_raw, mode="reversed")
        elif condition == "missing":
            # 50% frame omission (zero-mask)
            x_intervened = x_raw.clone()
            x_intervened[:, [1, 3, 5, 7], :] = 0.0
        else:
            raise ValueError(f"Unknown condition: '{condition}'")

        gt_label = meta["label"]
        gt_class = self.class_names[gt_label]

        results = {}
        for m_name, model in self.models.items():
            with torch.no_grad():
                inp = x_intervened.to(self.device)
                logits = model(inp)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                pred_conf = float(probs[pred_idx])
                gt_conf = float(probs[gt_label])

            results[m_name] = {
                "predicted_label": pred_idx,
                "predicted_class": self.class_names[pred_idx],
                "confidence": pred_conf,
                "ground_truth_confidence": gt_conf,
                "correct": bool(pred_idx == gt_label),
            }

        return {
            "sequence_id": sequence_id,
            "split": split,
            "condition": condition,
            "ground_truth_id": gt_label,
            "ground_truth_class": gt_class,
            "predictions": results,
            "provenance": {
                "embedding_path": str(emb_path.relative_to(PROJECT_ROOT)),
                "num_frames": len(arr),
                "feature_dimension": arr.shape[-1],
                "checkpoint_seed": self.seed,
            }
        }

    def print_inspection(self, data: Dict[str, Any]):
        seq_id = data["sequence_id"]
        cond = data["condition"]
        gt = data["ground_truth_class"]

        print("\n" + "=" * 76)
        print(f"  SEQUENCE INSPECTOR: {seq_id} | Condition: {cond.upper()}")
        print("=" * 76)
        print(f"  Ground Truth Action : {gt} (ID: {data['ground_truth_id']})")
        print(f"  Sequence Partition  : {data['split']} (N=8 frames, 384-D frozen DINOv2)")

        # Timeline visualization
        if cond == "canonical":
            timeline = "[F0] ──> [F1] ──> [F2] ──> [F3] ──> [F4] ──> [F5] ──> [F6] ──> [F7] (Chronological)"
        elif cond == "shuffled":
            timeline = "[F3] ──> [F0] ──> [F6] ──> [F1] ──> [F7] ──> [F2] ──> [F5] ──> [F4] (Scrambled Order)"
        elif cond == "reversed":
            timeline = "[F7] ──> [F6] ──> [F5] ──> [F4] ──> [F3] ──> [F2] ──> [F1] ──> [F0] (Time-Reversed)"
        else:
            timeline = "[F0] ──> [Ø]  ──> [F2] ──> [Ø]  ──> [F4] ──> [Ø]  ──> [F6] ──> [Ø]  (50% Zero-Masked)"

        print(f"  Observation Stream  : {timeline}")
        print("-" * 76)
        print(f"  {'Model Architecture':<32} {'Prediction':<30} {'Conf':>6}  {'Status':>8}")
        print("  " + "-" * 72)

        for m_name, res in data["predictions"].items():
            status = "[PASS]" if res["correct"] else "[WRONG]"
            print(f"  {m_name:<32} {res['predicted_class'][:28]:<30} {res['confidence']*100:>5.1f}%  {status:>8}")

        print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Scientific Sequence Inspector & Intervention Demo")
    parser.add_argument("--sequence-id", type=str, default=None, help="Sequence ID to inspect")
    parser.add_argument("--condition", type=str, default="canonical", choices=["canonical", "shuffled", "reversed", "missing"], help="Observation condition")
    parser.add_argument("--mode", type=str, default=None, choices=["rescue", "failure", "random"], help="Preset demonstration mode")
    parser.add_argument("--seed", type=int, default=42, help="Checkpoint seed")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    inspector = ScientificSequenceInspector(seed=args.seed)

    seq_id = args.sequence_id
    if args.mode == "rescue":
        seq_id = PRESET_SEQUENCES["rescue"]["sequence_id"]
        print(f"\n[DEMO PRESET] {PRESET_SEQUENCES['rescue']['name']}")
        print(f"Description: {PRESET_SEQUENCES['rescue']['description']}")
    elif args.mode == "failure":
        seq_id = PRESET_SEQUENCES["failure"]["sequence_id"]
        print(f"\n[DEMO PRESET] {PRESET_SEQUENCES['failure']['name']}")
        print(f"Description: {PRESET_SEQUENCES['failure']['description']}")
    elif seq_id is None:
        # Default interactive preview: showcase rescue sequence in canonical vs shuffled
        seq_id = "vid_07_0001"

    data = inspector.inspect_sequence(seq_id, condition=args.condition, seed=args.seed)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        inspector.print_inspection(data)


if __name__ == "__main__":
    main()

