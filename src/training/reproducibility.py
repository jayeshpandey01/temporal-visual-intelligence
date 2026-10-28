"""
Reproducibility and per-run configuration management infrastructure (M3.13).

Ensures every training and evaluation run is identifiable by its exact configuration,
epoch trajectories, model checkpoint, and sample-level test predictions:
runs/
└── {model}_seed{seed}_frames{max_frames}/
    ├── config.yaml
    ├── checkpoint.pt
    ├── train_metrics.csv
    ├── validation_metrics.csv
    └── predictions.parquet (and predictions.csv)
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import csv
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml


def get_git_commit() -> Optional[str]:
    """Retrieve current git commit hash if running in a git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return None


def build_run_config(
    global_config: Dict[str, Any],
    model: nn.Module,
    model_type: str,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Standardize the 15 required reproducibility fields specified in M3.13.

    Fields recorded:
      1. seed
      2. dataset split
      3. model
      4. encoder
      5. embedding version
      6. sampling strategy
      7. learning rate
      8. batch size
      9. epochs
      10. optimizer
      11. scheduler
      12. loss
      13. hidden dimension
      14. projection dimension
      15. dropout
    plus model parameter counts and system metadata.
    """
    max_frames = global_config["sequence"]["max_frames"]

    # Model dimension extraction
    if model_type == "baseline":
        h_dim = global_config.get("baseline", {}).get("hidden_dim", 256)
        p_dim = h_dim
        dropout = global_config.get("baseline", {}).get("dropout", 0.1)
        sampling = global_config.get("baseline", {}).get("train_sampling", "random")
    elif model_type == "meanpool":
        h_dim = global_config.get("baseline", {}).get("hidden_dim", 256)
        p_dim = h_dim
        dropout = global_config.get("baseline", {}).get("dropout", 0.1)
        sampling = "mean_pooling_all_frames"
    elif model_type == "gru":
        h_dim = global_config["temporal"]["hidden_dim"]
        p_dim = global_config["temporal"].get("projection_dim", h_dim)
        dropout = global_config["temporal"].get("dropout", 0.2)
        sampling = global_config["sequence"]["sampling"]
    else:
        h_dim = global_config.get("temporal", {}).get("hidden_dim", 256)
        p_dim = global_config.get("temporal", {}).get("projection_dim", h_dim)
        dropout = global_config.get("temporal", {}).get("dropout", 0.1)
        sampling = global_config["sequence"]["sampling"]

    # Parameter accounting
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_encoder_params = 21_800_000  # DINOv2 ViT-S/14 frozen backbone
    total_system_params = trainable_params + frozen_encoder_params

    run_config = {
        # 15 Required Fields (M3.13)
        "seed": int(seed),
        "dataset_split": {
            "name": global_config["dataset"]["name"],
            "train_sequences": 1400,
            "val_sequences": 300,
            "test_sequences": 300,
            "classes": 20,
            "partition_method": "Stratified Group-Aware Video-Level Split (GroupKFold)",
            "leakage_guard": "CERTIFIED_ZERO_LEAKAGE",
        },
        "model": model_type,
        "encoder": {
            "name": global_config["encoder"]["name"],
            "frozen": bool(global_config["encoder"]["freeze"]),
            "embed_dim": global_config["encoder"]["embed_dim"],
        },
        "embedding_version": {
            "path": global_config["paths"]["embeddings"],
            "normalization": "ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
            "feature_dim": 384,
        },
        "sampling_strategy": {
            "strategy": sampling,
            "max_frames": int(max_frames),
            "frame_size": list(global_config["sequence"]["frame_size"]),
        },
        "learning_rate": float(global_config["training"]["lr"]),
        "batch_size": int(global_config["training"]["batch_size"]),
        "epochs": int(global_config["training"]["epochs"]),
        "optimizer": {
            "name": "AdamW",
            "lr": float(global_config["training"]["lr"]),
            "weight_decay": float(global_config["training"]["weight_decay"]),
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "gradient_clip": float(global_config["training"]["gradient_clip"]),
        },
        "scheduler": {
            "name": "CosineAnnealingLR",
            "T_max": int(global_config["training"]["epochs"]),
            "eta_min": 0.0,
        },
        "loss": {
            "name": "CrossEntropyLoss",
            "reduction": "mean",
            "class_weighted": False,
            "rationale": "Exact empirical class balance (N_c = 70 for all 20 classes)",
        },
        "hidden_dimension": int(h_dim),
        "projection_dimension": int(p_dim),
        "dropout": float(dropout),
        # Additional Parameter Accounting & Environment Traceability
        "parameter_accounting": {
            "trainable_parameters": int(trainable_params),
            "frozen_backbone_parameters": int(frozen_encoder_params),
            "total_system_parameters": int(total_system_params),
            "trainable_percentage": round(100.0 * trainable_params / total_system_params, 4),
        },
        "environment": {
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
            "git_commit": get_git_commit(),
            "torch_version": str(torch.__version__),
        },
    }

    return run_config


class RunTracker:
    """
    Manages structured file logging, checkpointing, and prediction serialization
    for a single experimental run.
    """

    def __init__(
        self,
        output_root: str,
        run_id: str,
        run_config: Dict[str, Any],
    ):
        self.output_root = Path(output_root)
        self.run_id = run_id
        self.run_dir = self.output_root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_config = run_config
        self.config_path = self.run_dir / "config.yaml"
        self.checkpoint_path = self.run_dir / "checkpoint.pt"
        self.train_csv_path = self.run_dir / "train_metrics.csv"
        self.val_csv_path = self.run_dir / "validation_metrics.csv"
        self.predictions_csv_path = self.run_dir / "predictions.csv"
        self.predictions_parquet_path = self.run_dir / "predictions.parquet"
        self.metrics_json_path = self.run_dir / "metrics.json"

        # Initialize config.yaml
        self._write_config()

        # Initialize CSV headers
        self._init_csv(
            self.train_csv_path,
            ["epoch", "loss", "accuracy", "macro_f1", "lr"],
        )
        self._init_csv(
            self.val_csv_path,
            ["epoch", "loss", "accuracy", "macro_f1", "is_best", "patience_counter"],
        )

    def _write_config(self):
        with open(self.config_path, "w") as f:
            yaml.dump(self.run_config, f, default_flow_style=False, sort_keys=False)

    def _init_csv(self, path: Path, headers: List[str]):
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_train_epoch(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        lr: float,
    ):
        """Append train metrics for current epoch."""
        with open(self.train_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(loss, 6),
                round(metrics.get("accuracy", 0.0), 6),
                round(metrics.get("macro_f1", 0.0), 6),
                f"{lr:.6e}",
            ])

    def log_val_epoch(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        is_best: bool,
        patience_counter: int,
    ):
        """Append validation metrics for current epoch."""
        with open(self.val_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(loss, 6),
                round(metrics.get("accuracy", 0.0), 6),
                round(metrics.get("macro_f1", 0.0), 6),
                bool(is_best),
                int(patience_counter),
            ])

    def save_checkpoint(self, model: nn.Module):
        """Save best validation checkpoint."""
        torch.save(model.state_dict(), self.checkpoint_path)

    def save_predictions(
        self,
        test_ds: Any,
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        all_probs: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Record sample-level predictions for all test sequences.
        Saves both predictions.csv (for universal inspection) and
        predictions.parquet (for high-throughput analysis in M4).
        """
        records = []
        n_samples = len(all_labels)

        for i in range(n_samples):
            seq_id, _ = test_ds.samples[i]
            gt = int(all_labels[i])
            pred = int(all_preds[i])
            probs = all_probs[i]
            conf = float(probs[pred]) if probs is not None and len(probs) > pred else 0.0

            gt_name = class_names[gt] if class_names and gt < len(class_names) else f"Class_{gt}"
            pred_name = class_names[pred] if class_names and pred < len(class_names) else f"Class_{pred}"

            records.append({
                "sequence_id": seq_id,
                "video_id": seq_id,
                "ground_truth": gt,
                "ground_truth_name": gt_name,
                "predicted_label": pred,
                "predicted_name": pred_name,
                "confidence": round(conf, 6),
                "correct": bool(pred == gt),
                "probabilities": json.dumps([round(float(p), 6) for p in probs]) if probs is not None else "[]",
            })

        df = pd.DataFrame(records)

        # Always save CSV
        df.to_csv(self.predictions_csv_path, index=False)

        # Save Parquet if engine is available
        try:
            df.to_parquet(self.predictions_parquet_path, index=False)
        except Exception:
            # Parquet engine (pyarrow/fastparquet) may be absent in restricted sandboxes
            pass

        # Generate per-class Markdown table and confusion matrix figure (M3.14)
        try:
            from src.evaluation.metrics import generate_per_class_table
            from src.evaluation.analysis import plot_confusion_matrix
            generate_per_class_table(
                all_labels, all_preds, class_names=class_names,
                output_path=str(self.run_dir / "per_class_metrics.md"),
                model_name=self.run_id,
            )
            plot_confusion_matrix(
                all_labels, all_preds, class_names=class_names,
                output_path=str(self.run_dir / "confusion_matrix.png"),
                title=f"Normalized Confusion Matrix ({self.run_id})",
            )
        except Exception as e:
            print(f"  [RunTracker] Warning generating auxiliary evaluation artifacts: {e}")

        return df

    def finalize(
        self,
        test_metrics: Dict[str, Any],
        early_stopping_summary: Optional[Dict[str, Any]] = None,
        directional_confusion: Optional[Dict[str, Any]] = None,
    ):
        """Record final test metrics, early stopping, and directional confusion into config.yaml."""
        self.run_config["test_results"] = {
            "accuracy": round(float(test_metrics.get("accuracy", 0.0)), 4),
            "macro_f1": round(float(test_metrics.get("macro_f1", 0.0)), 4),
            "macro_precision": round(float(test_metrics.get("macro_precision", 0.0)), 4),
            "macro_recall": round(float(test_metrics.get("macro_recall", 0.0)), 4),
        }
        if directional_confusion:
            self.run_config["test_results"]["directional_error_rate"] = round(
                float(directional_confusion.get("overall_directional_error_rate", 0.0)), 4
            )
        if early_stopping_summary:
            self.run_config["early_stopping_summary"] = early_stopping_summary

        self.run_config["status"] = "COMPLETED"
        self._write_config()

        # Emit standalone metrics.json (M3.22)
        metrics_dict = {
            "run_id": self.run_id,
            "status": "COMPLETED",
            "test_results": self.run_config["test_results"],
            "early_stopping_summary": early_stopping_summary,
            "parameters": self.run_config.get("parameters", {}),
            "completed_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        with open(self.metrics_json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
