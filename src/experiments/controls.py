"""
Milestone M4.1: Frozen Experimental Controls & Single-Variable Intervention Engine.

Enforces the core research rule:
    "Change exactly one experimental variable at a time."

Before running any experiments, freezes:
  1. Dataset (SSv2 Directional Benchmark, 20 classes, 2000 sequences)
  2. Encoder (Frozen DINOv2 ViT-S/14, 384-dim, 21.8M params)
  3. Preprocessing (224x224, ImageNet norm, center crop, 8 uniform frames)
  4. Train split (1,400 sequences, certified zero leakage)
  5. Validation split (300 sequences)
  6. Test split (300 sequences)
  7. Model checkpoints (Model A: 103.7k, Model B: 103.7k, Model C: 249.4k across seeds 42, 43, 44)
  8. Evaluation metrics (accuracy, macro_f1, per_class_f1, McNemar, 95% bootstrap CI)
  9. Random seeds ([42, 43, 44], fixed perturbation seed 42)
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import torch
import yaml


class ConfoundViolationError(ValueError):
    """Raised when an experiment violates the single-variable intervention rule."""
    pass


@dataclass
class ExperimentalCondition:
    """
    Encapsulates the exact state of an experimental evaluation run.
    """
    model_name: str
    seed: int
    frame_count: int = 8
    order: str = "canonical"
    corruption: str = "none"
    corruption_fraction: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def condition_tag(self) -> str:
        """Compact identifier for the condition."""
        parts = [f"{self.model_name}_s{self.seed}_k{self.frame_count}"]
        if self.order != "canonical":
            parts.append(self.order)
        if self.corruption != "none":
            pct = int(self.corruption_fraction * 100)
            parts.append(f"{self.corruption}_{pct}pct")
        return "_".join(parts)


@dataclass
class FrozenControlsSpec:
    """
    In-memory representation of the frozen experimental controls.
    """
    version: str
    milestone: str
    freeze_timestamp: str
    canonical_rule: str
    dataset: Dict[str, Any]
    encoder: Dict[str, Any]
    preprocessing: Dict[str, Any]
    splits: Dict[str, Any]
    models: Dict[str, Any]
    metrics: Dict[str, Any]
    seeds: Dict[str, Any]
    allowable_interventions: Dict[str, Any]

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "FrozenControlsSpec":
        """Load frozen controls from yaml file."""
        if config_path is None:
            # Default to standard path relative to project root
            root = Path(__file__).parent.parent.parent
            config_path = str(root / "configs" / "experimental_controls.yaml")

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Frozen controls specification not found at: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            version=data.get("version", "1.0.0"),
            milestone=data.get("milestone", "M4.1"),
            freeze_timestamp=data.get("freeze_timestamp", ""),
            canonical_rule=data.get("canonical_rule", "Change exactly one experimental variable at a time."),
            dataset=data.get("dataset", {}),
            encoder=data.get("encoder", {}),
            preprocessing=data.get("preprocessing", {}),
            splits=data.get("splits", {}),
            models=data.get("models", {}),
            metrics=data.get("metrics", {}),
            seeds=data.get("seeds", {}),
            allowable_interventions=data.get("allowable_interventions", {}),
        )

    def canonical_condition(self, model_name: str, seed: int) -> ExperimentalCondition:
        """Returns the canonical control condition for a given model and seed."""
        return ExperimentalCondition(
            model_name=model_name,
            seed=seed,
            frame_count=self.preprocessing.get("base_frames", 8),
            order="canonical",
            corruption="none",
            corruption_fraction=0.0,
        )


class SingleVariableIntervention:
    """
    Enforces the single-variable intervention principle:
    Change exactly one experimental variable at a time.
    """

    ALLOWED_VARIABLES = {"frame_count", "order", "corruption", "corruption_fraction"}

    def __init__(self, spec: Optional[FrozenControlsSpec] = None):
        self.spec = spec or FrozenControlsSpec.load()

    def detect_changes(
        self,
        candidate: ExperimentalCondition,
        baseline: Optional[ExperimentalCondition] = None,
    ) -> Dict[str, Tuple[Any, Any]]:
        """
        Detect differences between candidate condition and canonical baseline.
        Returns dict of {variable_name: (baseline_val, candidate_val)}.
        """
        if baseline is None:
            baseline = self.spec.canonical_condition(candidate.model_name, candidate.seed)

        changes = {}
        if candidate.frame_count != baseline.frame_count:
            changes["frame_count"] = (baseline.frame_count, candidate.frame_count)
        if candidate.order != baseline.order:
            changes["order"] = (baseline.order, candidate.order)
        if candidate.corruption != baseline.corruption:
            changes["corruption"] = (baseline.corruption, candidate.corruption)
        if candidate.corruption_fraction != baseline.corruption_fraction:
            changes["corruption_fraction"] = (baseline.corruption_fraction, candidate.corruption_fraction)

        return changes

    def validate_intervention(
        self,
        candidate: ExperimentalCondition,
        baseline: Optional[ExperimentalCondition] = None,
    ) -> List[str]:
        """
        Validates that candidate condition alters at most ONE experimental axis.

        Raises:
            ConfoundViolationError: If 2 or more variables change simultaneously,
                                    or if an invalid intervention value is used.
        Returns:
            List of modified variable names (length 0 or 1).
        """
        if baseline is None:
            baseline = self.spec.canonical_condition(candidate.model_name, candidate.seed)

        # Ensure model and seed are matched
        if candidate.model_name != baseline.model_name:
            raise ConfoundViolationError(
                f"Model mismatch in intervention comparison: {baseline.model_name} vs {candidate.model_name}. "
                f"Model must remain constant when evaluating interventions."
            )
        if candidate.seed != baseline.seed:
            raise ConfoundViolationError(
                f"Seed mismatch in intervention comparison: {baseline.seed} vs {candidate.seed}. "
                f"Seed must remain constant when evaluating interventions."
            )

        changes = self.detect_changes(candidate, baseline)

        # Note: changing corruption usually couples setting corruption_fraction > 0.
        # We treat (corruption, corruption_fraction) as a single composite axis: 'corruption'.
        if "corruption" in changes and "corruption_fraction" in changes:
            active_variables = set(changes.keys()) - {"corruption_fraction"}
        else:
            active_variables = set(changes.keys())

        if len(active_variables) > 1:
            details = ", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in changes.items())
            raise ConfoundViolationError(
                f"Confound Violation! Multiple experimental variables changed simultaneously: [{details}]. "
                f"The fundamental scientific rule of M4 mandates: Change exactly ONE experimental variable at a time."
            )

        # Validate allowed values if 1 variable changed
        if len(active_variables) == 1:
            var = list(active_variables)[0]
            if var == "frame_count":
                allowed = self.spec.allowable_interventions.get("progressive", {}).get(
                    "allowed_values", [1, 2, 3, 4, 6, 8]
                )
                if candidate.frame_count not in allowed:
                    raise ConfoundViolationError(
                        f"Frame count {candidate.frame_count} is not in allowed progressive values: {allowed}"
                    )
            elif var == "order":
                allowed_order = self.spec.allowable_interventions.get("order", {}).get(
                    "allowed_values", ["canonical", "shuffled", "repeated"]
                )
                allowed_rev = self.spec.allowable_interventions.get("reversal", {}).get(
                    "allowed_values", ["canonical", "reversed"]
                )
                all_allowed_order = set(allowed_order) | set(allowed_rev)
                if candidate.order not in all_allowed_order:
                    raise ConfoundViolationError(
                        f"Order '{candidate.order}' is not in allowed order values: {all_allowed_order}"
                    )
            elif var == "corruption":
                allowed_corrupt = self.spec.allowable_interventions.get("corruption", {}).get(
                    "allowed_values", ["none", "zero_mask", "gaussian_noise", "mean_replace"]
                )
                if candidate.corruption not in allowed_corrupt:
                    raise ConfoundViolationError(
                        f"Corruption '{candidate.corruption}' is not in allowed corruption values: {allowed_corrupt}"
                    )

        return list(active_variables)

    # -------------------------------------------------------------------------
    # Condition Generators for Downstream M4 Milestones
    # -------------------------------------------------------------------------
    def get_progressive_conditions(
        self,
        model_name: str = "gru",
        seeds: Optional[List[int]] = None,
    ) -> List[ExperimentalCondition]:
        """Generate strictly controlled progressive observation conditions (M4.2)."""
        seeds = seeds or self.spec.seeds.get("model_seeds", [42, 43, 44])
        frame_counts = self.spec.allowable_interventions.get("progressive", {}).get(
            "allowed_values", [1, 2, 3, 4, 6, 8]
        )
        conditions = []
        for seed in seeds:
            for k in frame_counts:
                cond = ExperimentalCondition(
                    model_name=model_name,
                    seed=seed,
                    frame_count=k,
                    order="canonical",
                    corruption="none",
                    corruption_fraction=0.0,
                )
                self.validate_intervention(cond)
                conditions.append(cond)
        return conditions

    def get_order_conditions(
        self,
        model_name: str = "gru",
        seeds: Optional[List[int]] = None,
    ) -> List[ExperimentalCondition]:
        """Generate strictly controlled order ablation conditions (M4.3)."""
        seeds = seeds or self.spec.seeds.get("model_seeds", [42, 43, 44])
        orders = self.spec.allowable_interventions.get("order", {}).get(
            "allowed_values", ["canonical", "shuffled", "repeated"]
        )
        conditions = []
        for seed in seeds:
            for ord_mode in orders:
                cond = ExperimentalCondition(
                    model_name=model_name,
                    seed=seed,
                    frame_count=8,
                    order=ord_mode,
                    corruption="none",
                    corruption_fraction=0.0,
                )
                self.validate_intervention(cond)
                conditions.append(cond)
        return conditions

    def get_corruption_conditions(
        self,
        model_name: str = "gru",
        seeds: Optional[List[int]] = None,
        fraction: float = 0.25,
    ) -> List[ExperimentalCondition]:
        """Generate strictly controlled corruption robustness conditions (M4.4)."""
        seeds = seeds or self.spec.seeds.get("model_seeds", [42, 43, 44])
        corruptions = self.spec.allowable_interventions.get("corruption", {}).get(
            "allowed_values", ["none", "zero_mask", "gaussian_noise", "mean_replace"]
        )
        conditions = []
        for seed in seeds:
            for corr in corruptions:
                f_frac = fraction if corr != "none" else 0.0
                cond = ExperimentalCondition(
                    model_name=model_name,
                    seed=seed,
                    frame_count=8,
                    order="canonical",
                    corruption=corr,
                    corruption_fraction=f_frac,
                )
                self.validate_intervention(cond)
                conditions.append(cond)
        return conditions

    def get_reversal_conditions(
        self,
        model_name: str = "gru",
        seeds: Optional[List[int]] = None,
    ) -> List[ExperimentalCondition]:
        """Generate strictly controlled temporal reversal conditions (M4.5)."""
        seeds = seeds or self.spec.seeds.get("model_seeds", [42, 43, 44])
        conditions = []
        for seed in seeds:
            for ord_mode in ["canonical", "reversed"]:
                cond = ExperimentalCondition(
                    model_name=model_name,
                    seed=seed,
                    frame_count=8,
                    order=ord_mode,
                    corruption="none",
                    corruption_fraction=0.0,
                )
                self.validate_intervention(cond)
                conditions.append(cond)
        return conditions


# -----------------------------------------------------------------------------
# Controls Verification & Integrity Auditor
# -----------------------------------------------------------------------------
def verify_controls_integrity(
    config_path: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Perform a complete programmatic audit of all 9 frozen experimental controls.
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent

    spec = FrozenControlsSpec.load(config_path)
    audit: Dict[str, Any] = {
        "status": "PASS",
        "timestamp": spec.freeze_timestamp,
        "controls": {},
    }

    def record_check(name: str, passed: bool, details: Dict[str, Any]):
        audit["controls"][name] = {"passed": passed, **details}
        if not passed:
            audit["status"] = "FAIL"

    # 1. Dataset Check
    manifest_p = project_root / spec.dataset.get("manifest_path", "data/metadata/dataset_manifest.json")
    seq_p = project_root / spec.dataset.get("sequences_csv", "data/metadata/sequences.csv")
    ds_pass = manifest_p.exists() and seq_p.exists()
    seq_count = 0
    if seq_p.exists():
        with open(seq_p, "r") as f:
            lines = [l for l in f if l.strip()]
            seq_count = len(lines) - 1  # Subtract header
    record_check(
        "Control 1: Dataset",
        ds_pass and (seq_count == spec.dataset.get("total_sequences", 2000)),
        {
            "manifest_exists": manifest_p.exists(),
            "sequences_csv_exists": seq_p.exists(),
            "sequence_count": seq_count,
            "expected_sequences": spec.dataset.get("total_sequences", 2000),
            "expected_classes": spec.dataset.get("class_count", 20),
        },
    )

    # 2. Visual Encoder Check
    emb_dir = project_root / spec.encoder.get("embeddings_dir", "embeddings/")
    emb_manifest = project_root / spec.encoder.get("manifest_path", "embeddings/manifest.json")
    enc_pass = (
        spec.encoder.get("freeze") is True
        and spec.encoder.get("embed_dim") == 384
        and emb_manifest.exists()
        and (emb_dir / "train").exists()
        and (emb_dir / "val").exists()
        and (emb_dir / "test").exists()
    )
    record_check(
        "Control 2: Visual Encoder",
        enc_pass,
        {
            "frozen": spec.encoder.get("freeze"),
            "embed_dim": spec.encoder.get("embed_dim"),
            "backbone": spec.encoder.get("name"),
            "manifest_exists": emb_manifest.exists(),
        },
    )

    # 3. Preprocessing Check
    prep = spec.preprocessing
    prep_pass = (
        prep.get("resolution") == [224, 224]
        and prep.get("base_frames") == 8
        and prep.get("color_space") == "RGB"
    )
    record_check(
        "Control 3: Preprocessing",
        prep_pass,
        {
            "version": prep.get("version"),
            "resolution": prep.get("resolution"),
            "base_frames": prep.get("base_frames"),
            "color_space": prep.get("color_space"),
        },
    )

    # 4, 5, 6. Splits & Leakage Check
    split_p = project_root / spec.splits.get("manifest_path", "data/metadata/split_manifest.json")
    splits_pass = False
    train_ids, val_ids, test_ids = set(), set(), set()
    train_val_overlap, train_test_overlap, val_test_overlap = 0, 0, 0
    if split_p.exists():
        with open(split_p, "r") as f:
            split_data = json.load(f)
            train_ids = set(split_data.get("train", []))
            val_ids = set(split_data.get("val", []))
            test_ids = set(split_data.get("test", []))
        train_val_overlap = len(train_ids & val_ids)
        train_test_overlap = len(train_ids & test_ids)
        val_test_overlap = len(val_ids & test_ids)
        splits_pass = (
            len(train_ids) == spec.splits["train"]["count"]
            and len(val_ids) == spec.splits["validation"]["count"]
            and len(test_ids) == spec.splits["test"]["count"]
            and train_val_overlap == 0
            and train_test_overlap == 0
            and val_test_overlap == 0
        )
    record_check(
        "Control 4, 5, 6: Splits & Leakage Guard",
        splits_pass,
        {
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "train_val_overlap": train_val_overlap,
            "train_test_overlap": train_test_overlap,
            "val_test_overlap": val_test_overlap,
            "leakage_guard_status": "ZERO_LEAKAGE" if (train_val_overlap == train_test_overlap == val_test_overlap == 0) else "LEAKAGE_DETECTED",
        },
    )

    # 7. Model Checkpoints & Parameter Accounting
    models_spec = spec.models
    model_results = {}
    models_all_passed = True
    for m_key, m_info in models_spec.items():
        exp_params = m_info["expected_parameters"]
        ckpts = m_info.get("checkpoints", {})
        ckpt_status = {}
        for s_key, c_path in ckpts.items():
            full_p = project_root / c_path
            exists = full_p.exists()
            param_match = False
            actual_params = 0
            if exists:
                state = torch.load(full_p, map_location="cpu")
                actual_params = sum(p.numel() for p in state.values())
                param_match = (actual_params == exp_params)
            ckpt_status[s_key] = {
                "path": c_path,
                "exists": exists,
                "param_count": actual_params,
                "param_match": param_match,
            }
            if not (exists and param_match):
                models_all_passed = False
        model_results[m_key] = {
            "name": m_info["name"],
            "expected_parameters": exp_params,
            "checkpoints": ckpt_status,
        }
    record_check(
        "Control 7: Model Checkpoints & Parameters",
        models_all_passed,
        model_results,
    )

    # 8. Evaluation Metrics Check
    primary_metrics = spec.metrics.get("primary", [])
    stat_metrics = spec.metrics.get("statistical", [])
    metrics_pass = (
        "accuracy" in primary_metrics
        and "macro_f1" in primary_metrics
        and "mcnemar_test" in stat_metrics
        and "bootstrap_ci_95" in stat_metrics
    )
    record_check(
        "Control 8: Evaluation Metrics",
        metrics_pass,
        {
            "primary": primary_metrics,
            "statistical": stat_metrics,
            "bootstrap_resamples": spec.metrics.get("bootstrap", {}).get("n_resamples", 1000),
        },
    )

    # 9. Random Seeds Protocol
    m_seeds = spec.seeds.get("model_seeds", [])
    p_seed = spec.seeds.get("perturbation_seed")
    seeds_pass = (m_seeds == [42, 43, 44] and p_seed == 42)
    record_check(
        "Control 9: Random Seeds",
        seeds_pass,
        {
            "model_seeds": m_seeds,
            "perturbation_seed": p_seed,
            "protocol": "Fixed perturbation seed 42 preserves paired sequence comparison across model seeds",
        },
    )

    return audit

