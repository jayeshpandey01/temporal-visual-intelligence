"""
Milestone M6.5: Class-Wise Generalization Analysis.
Decomposes the temporal advantage across all 20 benchmark action classes.

Computes:
  - Delta_c = F1_{temporal, c} - F1_{baseline, c}
  - Ranks classes from largest temporal gain to smallest/negative
  - Maps to M5 taxonomy (State-Transition, Directional, Spatial-Context, Ambiguous)
"""

from __future__ import annotations

import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EmbeddingSequenceDataset
from src.models.baseline import build_baseline
from src.models.mean_pool import build_mean_pool_model
from src.models.temporal import build_temporal_model
from src.evaluation.metrics import compute_metrics
from src.experiments.checkpoint_paths import resolve_checkpoint_dir


# Taxonomy mapping from Milestone 5
CLASS_TAXONOMY_MAP = {
    "Closing something": "State-Transition",
    "Opening something": "State-Transition",
    "Taking something out of something": "State-Transition",
    "Putting something into something": "State-Transition",
    "Taking something off of something": "State-Transition",
    "Putting something on top of something": "State-Transition",
    "Covering something": "State-Transition",
    "Uncovering something": "State-Transition",
    "Picking something up": "Directional / Dynamic",
    "Putting something down": "Directional / Dynamic",
    "Pushing something from left to right": "Directional / Dynamic",
    "Pushing something from right to left": "Directional / Dynamic",
    "Moving something away from the camera": "Directional / Dynamic",
    "Moving something towards the camera": "Directional / Dynamic",
    "Rotating something clockwise": "Directional / Dynamic",
    "Rotating something counter-clockwise": "Directional / Dynamic",
    "Turning the camera left while filming something": "Camera Motion / Ego",
    "Turning the camera right while filming something": "Camera Motion / Ego",
    "Standing something upright": "Spatial-Context / Ambiguous",
    "Tipping something over": "Spatial-Context / Ambiguous",
}


class M6ClassGeneralizationAnalyzer:
    """Computes ranked per-class temporal gains and taxonomy breakdown."""
    def __init__(self, config: Dict[str, Any], seeds: Optional[List[int]] = None):
        self.config = config
        self.seeds = seeds or [42, 43, 44]
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        # Load class names
        with open("data/metadata/dataset_manifest.json") as f:
            self.class_names = json.load(f)["classes"]

    def run(self) -> Dict[str, Any]:
        ckpts_base = resolve_checkpoint_dir(Path.cwd())
        ds = EmbeddingSequenceDataset("embeddings", split="test")
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        per_model_f1 = {"baseline": [], "meanpool": [], "gru": []}

        for m_name in per_model_f1:
            seed_f1s = []
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

                preds, targets = [], []
                with torch.no_grad():
                    for x, y in loader:
                        x = x.to(self.device)
                        p = m(x).argmax(dim=-1).cpu().numpy()
                        preds.extend(p)
                        targets.extend(y.numpy())

                met = compute_metrics(np.array(targets), np.array(preds), class_names=self.class_names)
                seed_f1s.append(met["per_class_f1"])

            # Mean F1 per class across seeds
            per_model_f1[m_name] = np.mean(seed_f1s, axis=0)

        # Build class breakdown table
        class_records = []
        for c_idx, c_name in enumerate(self.class_names):
            f1_a = per_model_f1["baseline"][c_idx]
            f1_b = per_model_f1["meanpool"][c_idx]
            f1_c = per_model_f1["gru"][c_idx]
            delta_temp = f1_c - f1_a
            delta_ord = f1_c - f1_b
            tax = CLASS_TAXONOMY_MAP.get(c_name, "General Action")

            class_records.append({
                "class_id": c_idx,
                "class_name": c_name,
                "f1_baseline": float(f1_a),
                "f1_meanpool": float(f1_b),
                "f1_gru": float(f1_c),
                "delta_temporal": float(delta_temp),
                "delta_order": float(delta_ord),
                "taxonomy": tax,
            })

        # Rank by delta_temporal descending
        class_records = sorted(class_records, key=lambda x: x["delta_temporal"], reverse=True)
        for rank, r in enumerate(class_records, start=1):
            r["rank"] = rank

        return {
            "classes": class_records,
            "top_gain_class": class_records[0]["class_name"],
            "positive_gain_count": sum(1 for r in class_records if r["delta_temporal"] > 0),
        }

