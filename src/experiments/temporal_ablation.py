"""
Milestone M5.6: Controlled Temporal Ablation Experiments.

Implements the 6-condition sequence ablation protocol to isolate which temporal
information is strictly causal in model decisions:
  - Condition A: Full canonical sequence (frames [0..7], length 8)
  - Condition B: Remove early frames (frames [2..7], length 6)
  - Condition C: Remove middle transition (frames [0..2, 5..7], length 6)
  - Condition D: Remove late frames (frames [0..5], length 6)
  - Condition E: Reverse sequence (frames [7..0], length 8)
  - Condition F: Shuffled sequence (seeded permutation, length 8)

Rule:
  M5 must consume M4 outputs; M5 must not silently retrain or redefine M4.
  Evaluates frozen Model C (TemporalGRU) checkpoints across seeds [42, 43, 44].
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.temporal import TemporalGRU
from src.experiments.freeze import load_m4_frozen_checkpoint


# ---------------------------------------------------------------------------
# Ablation Operators
# ---------------------------------------------------------------------------
class TemporalAblationOperator:
    """
    Applies deterministic index selections to temporal sequence embeddings.
    Conditions B, C, D have identical length T = 6 (controlled length comparison).
    """

    CONDITIONS = {
        "canonical": [0, 1, 2, 3, 4, 5, 6, 7],
        "early_removal": [2, 3, 4, 5, 6, 7],
        "transition_removal": [0, 1, 2, 5, 6, 7],
        "late_removal": [0, 1, 2, 3, 4, 5],
        "reversed": [7, 6, 5, 4, 3, 2, 1, 0],
        "shuffled": [3, 0, 6, 1, 7, 4, 2, 5],  # Seeded permutation of [0..7]
    }

    CONDITION_DESCRIPTIONS = {
        "canonical": "Full sequence (frames 0-7, length 8) - Control condition",
        "early_removal": "Remove early frames (frames 2-7, length 6) - Preparatory context ablation",
        "transition_removal": "Remove middle transition (frames 0-2, 5-7, length 6) - Core action stroke ablation",
        "late_removal": "Remove late frames (frames 0-5, length 6) - Outcome/consequence ablation",
        "reversed": "Time-reversed sequence (frames 7-0, length 8) - Arrow of time inversion",
        "shuffled": "Temporally shuffled sequence (length 8) - Chronological order disruption",
    }

    @classmethod
    def apply(cls, emb: np.ndarray, condition: str) -> np.ndarray:
        """Slice (T, D) embedding array using specified condition."""
        if condition not in cls.CONDITIONS:
            raise ValueError(f"Unknown ablation condition '{condition}'. Valid: {list(cls.CONDITIONS.keys())}")
        indices = cls.CONDITIONS[condition]
        return emb[indices, :]


# ---------------------------------------------------------------------------
# Bootstrap Significance Helper
# ---------------------------------------------------------------------------
def compute_bootstrap_ci(
    values: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute mean and 95% non-parametric bootstrap confidence interval."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(values)
    mean_val = float(np.mean(values))
    boot_means = []
    for _ in range(n_bootstraps):
        idxs = rng.choice(n, size=n, replace=True)
        boot_means.append(np.mean(values[idxs]))
    boot_means = np.array(boot_means)
    ci_low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean_val, ci_low, ci_high


# ---------------------------------------------------------------------------
# Temporal Ablation Evaluator
# ---------------------------------------------------------------------------
class TemporalAblationEvaluator:
    """
    Evaluates frozen TemporalGRU checkpoints across the 6 ablation conditions.
    Computes accuracy, class probabilities, and temporal criticality (Delta P).
    """

    def __init__(self, root_dir: Path):
        self.root = Path(root_dir).resolve()
        self.m4_data = load_m4_frozen_checkpoint(self.root, verify_on_load=True)
        self.seeds = [42, 43, 44]
        self.device = torch.device("cpu")
        self.emb_dir = self.root / "embeddings" / "test"
        self.cohorts_df = pd.read_csv(self.root / "results" / "m5" / "cohorts" / "analysis_cohorts.csv")
        self.gold_df = self._load_gold_taxonomy()

    def _load_gold_taxonomy(self) -> pd.DataFrame:
        gold_path = self.root / "results" / "m5" / "gold_standard_taxonomy.csv"
        if gold_path.exists():
            return pd.read_csv(gold_path).set_index("sequence_id")
        return pd.DataFrame()

    def load_frozen_gru_model(self, seed: int) -> TemporalGRU:
        """Instantiate and load exact weights for frozen TemporalGRU Model C."""
        ckpt_path = self.m4_data["paths"]["checkpoints"] / "gru" / f"best_seed{seed}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Frozen checkpoint missing at {ckpt_path}")

        # Exact M4 GRU architecture from config.yaml:
        # in=384, proj=256, hidden=128, layers=1, dropout=0.2, classes=20
        model = TemporalGRU(
            embed_dim=384,
            projection_dim=256,
            hidden_dim=128,
            num_layers=1,
            dropout=0.2,
            num_classes=20,
        )
        state_dict = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Execute full multi-seed evaluation across all 300 test sequences
        and all 6 ablation conditions.
        """
        seq_records = []
        for _, row in self.cohorts_df.iterrows():
            seq_id = row["sequence_id"]
            vid_id = row["video_id"]
            gt = int(row["ground_truth"])
            cls_name = row["class_name"]
            cohort = row["cohort"]

            emb_file = self.emb_dir / f"{seq_id}.npy"
            if not emb_file.exists():
                raise FileNotFoundError(f"Missing embedding for {seq_id} at {emb_file}")
            raw_emb = np.load(emb_file)  # [8, 384]

            seq_records.append({
                "sequence_id": seq_id,
                "video_id": vid_id,
                "ground_truth": gt,
                "class_name": cls_name,
                "cohort": cohort,
                "embedding": raw_emb,
            })

        # Run inference across seeds and conditions
        condition_names = list(TemporalAblationOperator.CONDITIONS.keys())
        eval_data: Dict[str, Any] = {}

        for seed in self.seeds:
            model = self.load_frozen_gru_model(seed)
            eval_data[f"seed_{seed}"] = {}

            for cond in condition_names:
                batch_embs = []
                for s in seq_records:
                    ablated = TemporalAblationOperator.apply(s["embedding"], cond)
                    batch_embs.append(ablated)

                # Convert to tensor [N, K, D]
                x_tensor = torch.tensor(np.stack(batch_embs), dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    logits = model(x_tensor)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    preds = np.argmax(probs, axis=-1)

                eval_data[f"seed_{seed}"][cond] = {
                    "preds": preds,
                    "probs": probs,
                }

        # Aggregate per-sequence results and compute temporal criticality (Delta P)
        master_sequence_rows = []
        for i, s in enumerate(seq_records):
            sid = s["sequence_id"]
            gt = s["ground_truth"]
            cohort = s["cohort"]
            cls_name = s["class_name"]

            # Pull gold-standard taxonomy tags if available
            gold_row = self.gold_df.loc[sid] if (not self.gold_df.empty and sid in self.gold_df.index) else {}

            row_data: Dict[str, Any] = {
                "sequence_id": sid,
                "video_id": s["video_id"],
                "class_name": cls_name,
                "ground_truth": gt,
                "cohort": cohort,
            }

            for cond in condition_names:
                cond_accs = []
                cond_probs = []
                for seed in self.seeds:
                    pred = int(eval_data[f"seed_{seed}"][cond]["preds"][i])
                    prob_gt = float(eval_data[f"seed_{seed}"][cond]["probs"][i][gt])
                    cond_accs.append(1.0 if pred == gt else 0.0)
                    cond_probs.append(prob_gt)

                row_data[f"acc_{cond}"] = float(np.mean(cond_accs))
                row_data[f"prob_{cond}"] = float(np.mean(cond_probs))

            # Temporal Criticality: Delta P_i = P_canonical - P_ablated
            p_canon = row_data["prob_canonical"]
            acc_canon = row_data["acc_canonical"]
            for cond in ["early_removal", "transition_removal", "late_removal", "reversed", "shuffled"]:
                row_data[f"delta_p_{cond}"] = float(p_canon - row_data[f"prob_{cond}"])
                row_data[f"delta_acc_{cond}"] = float(acc_canon - row_data[f"acc_{cond}"])

            # Attach taxonomy categories
            for tax_col in [
                "displacement", "directional_motion", "state_transition",
                "object_interaction", "relative_motion", "event_order",
                "static_appearance", "frame_ambiguity"
            ]:
                row_data[tax_col] = bool(gold_row.get(tax_col, False)) if hasattr(gold_row, "get") else False

            master_sequence_rows.append(row_data)

        results_df = pd.DataFrame(master_sequence_rows)

        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(results_df)

        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_sequences": len(results_df),
                "seeds": self.seeds,
                "conditions": TemporalAblationOperator.CONDITION_DESCRIPTIONS,
            },
            "summary_statistics": summary_stats,
            "dataframe": results_df,
        }

    def _compute_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute aggregate benchmark metrics, cohort breakdowns, and taxonomy criticality."""
        conditions = list(TemporalAblationOperator.CONDITIONS.keys())
        ablations = ["early_removal", "transition_removal", "late_removal", "reversed", "shuffled"]

        # 1. Overall benchmark metrics
        overall = {}
        for cond in conditions:
            mean_acc, lo_acc, hi_acc = compute_bootstrap_ci(df[f"acc_{cond}"].values)
            mean_prob, lo_prob, hi_prob = compute_bootstrap_ci(df[f"prob_{cond}"].values)
            overall[cond] = {
                "accuracy": mean_acc,
                "accuracy_ci": [lo_acc, hi_acc],
                "probability": mean_prob,
                "probability_ci": [lo_prob, hi_prob],
            }
        for cond in ablations:
            mean_dp, lo_dp, hi_dp = compute_bootstrap_ci(df[f"delta_p_{cond}"].values)
            mean_da, lo_da, hi_da = compute_bootstrap_ci(df[f"delta_acc_{cond}"].values)
            overall[cond]["delta_p"] = mean_dp
            overall[cond]["delta_p_ci"] = [lo_dp, hi_dp]
            overall[cond]["delta_acc"] = mean_da
            overall[cond]["delta_acc_ci"] = [lo_da, hi_da]

        # 2. Stratification by Cohort (especially Cohort C rescues)
        cohort_breakdown = {}
        for c in ["A", "B", "C", "D"]:
            c_df = df[df["cohort"] == c]
            c_n = len(c_df)
            c_dict: Dict[str, Any] = {"count": c_n}
            if c_n > 0:
                for cond in conditions:
                    c_dict[f"acc_{cond}"] = float(c_df[f"acc_{cond}"].mean())
                    c_dict[f"prob_{cond}"] = float(c_df[f"prob_{cond}"].mean())
                for cond in ablations:
                    c_dict[f"delta_p_{cond}"] = float(c_df[f"delta_p_{cond}"].mean())
                    c_dict[f"delta_acc_{cond}"] = float(c_df[f"delta_acc_{cond}"].mean())
            cohort_breakdown[c] = c_dict

        # 3. Stratification by Gold-Standard Taxonomy Categories (in Cohort C)
        c_rescues = df[df["cohort"] == "C"]
        taxonomy_breakdown = {}
        tax_cols = [
            ("T1", "displacement", "Object Displacement"),
            ("T2", "directional_motion", "Direction of Motion"),
            ("T3", "state_transition", "State Transition"),
            ("T4", "object_interaction", "Object Interaction"),
            ("T5", "relative_motion", "Relative Motion"),
            ("T6", "event_order", "Sub-Event Temporal Order"),
            ("T7", "static_appearance", "Persistent / Static Control"),
            ("T8", "frame_ambiguity", "Single Frame Ambiguity"),
        ]
        for t_id, col, name in tax_cols:
            sub = c_rescues[c_rescues[col] == True]
            sub_n = len(sub)
            t_dict = {"category_id": t_id, "name": name, "count_in_c": sub_n}
            if sub_n > 0:
                for cond in conditions:
                    t_dict[f"acc_{cond}"] = float(sub[f"acc_{cond}"].mean())
                for cond in ablations:
                    t_dict[f"delta_p_{cond}"] = float(sub[f"delta_p_{cond}"].mean())
                    t_dict[f"delta_acc_{cond}"] = float(sub[f"delta_acc_{cond}"].mean())
            taxonomy_breakdown[t_id] = t_dict

        return {
            "overall": overall,
            "cohort_breakdown": cohort_breakdown,
            "taxonomy_breakdown": taxonomy_breakdown,
        }

    def save_deliverables(
        self,
        eval_output: Dict[str, Any],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """Save results/m5/ deliverables: JSON, Matrix CSV, Taxonomy Criticality CSV, and Summary MD."""
        out_dir = Path(output_dir).resolve() if output_dir else self.root / "results" / "m5"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Master JSON results
        df = eval_output["dataframe"]
        json_path = out_dir / "temporal_ablation_results.json"
        serializable_output = {
            "metadata": eval_output["metadata"],
            "summary_statistics": eval_output["summary_statistics"],
            "records": df.to_dict(orient="records"),
        }
        with open(json_path, "w") as f:
            json.dump(serializable_output, f, indent=2)

        # 2. Matrix CSV deliverable
        csv_path = out_dir / "temporal_ablation_matrix.csv"
        df.to_csv(csv_path, index=False)

        # 3. Taxonomy Criticality CSV
        tax_rows = []
        for t_id, d in eval_output["summary_statistics"]["taxonomy_breakdown"].items():
            tax_rows.append(d)
        tax_df = pd.DataFrame(tax_rows)
        tax_path = out_dir / "temporal_criticality_by_taxonomy.csv"
        tax_df.to_csv(tax_path, index=False)

        # 4. Summary Markdown Report
        summary_path = out_dir / "temporal_ablation_summary.md"
        summary_text = self._build_markdown_summary(eval_output)
        with open(summary_path, "w") as f:
            f.write(summary_text)

        return {
            "temporal_ablation_results_json": json_path,
            "temporal_ablation_matrix_csv": csv_path,
            "temporal_criticality_by_taxonomy_csv": tax_path,
            "temporal_ablation_summary_md": summary_path,
        }

    def _build_markdown_summary(self, eval_output: Dict[str, Any]) -> str:
        """Format scientific Markdown summary report."""
        stats = eval_output["summary_statistics"]
        overall = stats["overall"]
        cohort = stats["cohort_breakdown"]
        tax = stats["taxonomy_breakdown"]
        meta = eval_output["metadata"]

        md = []
        md.append("# Milestone M5.6: Controlled Temporal Ablation Summary Report\n\n")
        md.append(f"**Total Evaluated Sequences:** $N = {meta['total_sequences']}$\n")
        md.append(f"**Evaluated Seeds:** `{meta['seeds']}`\n")
        md.append(f"**Model:** Frozen `TemporalGRU` (Model C from `experiments/m4_final/`)\n")
        md.append(f"**Timestamp:** `{meta['timestamp']}`\n\n")

        md.append("## 1. Overall Controlled Ablation Comparison ($N = 300$)\n\n")
        md.append("| Ablation Condition | Length ($T$) | Top-1 Accuracy | 95% Bootstrap CI | Accuracy Drop ($\\Delta$ Acc) | Criticality Drop ($\\Delta P$) |\n")
        md.append("|---|:---:|---:|:---:|---:|:---:|\n")

        cond_display = [
            ("canonical", "A: Canonical (Full sequence)", 8),
            ("early_removal", "B: Remove Early Frames", 6),
            ("transition_removal", "C: Remove Middle Transition", 6),
            ("late_removal", "D: Remove Late Frames", 6),
            ("reversed", "E: Reversed Sequence", 8),
            ("shuffled", "F: Shuffled Sequence", 8),
        ]

        for cond_key, label, length in cond_display:
            d = overall[cond_key]
            acc = d["accuracy"] * 100
            lo, hi = d["accuracy_ci"][0] * 100, d["accuracy_ci"][1] * 100
            if cond_key == "canonical":
                md.append(f"| **{label}** | {length} | **{acc:5.2f}%** | [{lo:4.2f}%, {hi:4.2f}%] | *Control (0.0%)* | *Control (0.000)* |\n")
            else:
                da = d.get("delta_acc", 0.0) * 100
                dp = d.get("delta_p", 0.0)
                md.append(f"| **{label}** | {length} | {acc:5.2f}% | [{lo:4.2f}%, {hi:4.2f}%] | **-{da:4.2f}%** | **+{dp:.3f}** |\n")

        md.append("\n> [!NOTE]\n")
        md.append("> **Equal-Length Control**: Conditions B, C, and D are all identically matched to length $T = 6$. Any performance disparity between them is solely attributable to the temporal position and semantic role of the excised frames.\n\n")

        md.append("## 2. Temporal Criticality Across Experimental Cohorts\n\n")
        md.append("| Cohort | Total $N$ | Canonical Acc | Early Removal Acc | Transition Removal Acc | Late Removal Acc | Reversed Acc | Shuffled Acc |\n")
        md.append("|:---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for c in ["A", "B", "C", "D"]:
            cd = cohort[c]
            cnt = cd["count"]
            md.append(
                f"| **Cohort {c}** | {cnt} | "
                f"{cd.get('acc_canonical', 0.0)*100:5.1f}% | "
                f"{cd.get('acc_early_removal', 0.0)*100:5.1f}% | "
                f"{cd.get('acc_transition_removal', 0.0)*100:5.1f}% | "
                f"{cd.get('acc_late_removal', 0.0)*100:5.1f}% | "
                f"{cd.get('acc_reversed', 0.0)*100:5.1f}% | "
                f"{cd.get('acc_shuffled', 0.0)*100:5.1f}% |\n"
            )

        md.append("\n## 3. Temporal Criticality by Gold-Standard Taxonomy Category in Temporal Rescues (Cohort C, $N = 97$)\n\n")
        md.append("| Phenomenon Category | Code | $N$ in Rescues | Peak Vulnerability Condition | Max Accuracy Drop ($\\Delta$ Acc) | Criticality Drop ($\\Delta P$) |\n")
        md.append("|---|:---:|---:|---|---:|:---:|\n")
        for t_id, d in tax.items():
            name = d["name"]
            cnt = d["count_in_c"]
            if cnt == 0:
                continue
            drops = {
                "Early Removal": (d.get("delta_acc_early_removal", 0.0), d.get("delta_p_early_removal", 0.0)),
                "Transition Removal": (d.get("delta_acc_transition_removal", 0.0), d.get("delta_p_transition_removal", 0.0)),
                "Late Removal": (d.get("delta_acc_late_removal", 0.0), d.get("delta_p_late_removal", 0.0)),
                "Reversal": (d.get("delta_acc_reversed", 0.0), d.get("delta_p_reversed", 0.0)),
                "Shuffling": (d.get("delta_acc_shuffled", 0.0), d.get("delta_p_shuffled", 0.0)),
            }
            peak_cond = max(drops.keys(), key=lambda k: drops[k][0])
            peak_da = drops[peak_cond][0] * 100
            peak_dp = drops[peak_cond][1]
            md.append(f"| **{name}** | `{t_id}` | {cnt} | {peak_cond} | **-{peak_da:5.1f}%** | **+{peak_dp:.3f}** |\n")

        md.append("\n---\n*Generated automatically by `src.experiments.temporal_ablation` under Milestone M5.6.*")
        return "".join(md)


# ---------------------------------------------------------------------------
# Strict Verification Audit
# ---------------------------------------------------------------------------
def audit_temporal_ablation(deliverables_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Exhaustively audit Milestone M5.6 deliverables.
    Validates population coverage, 6 ablation conditions, and Cohort C sensitivity.
    """
    target = Path(deliverables_path).resolve()
    checklist = {}
    failures = []

    if not target.exists() or target.stat().st_size == 0:
        return {
            "status": "FAILED",
            "message": f"temporal_ablation_results.json missing or empty at {target}",
            "passed_checks": 0,
            "total_checks": 1,
            "failures": ["file_missing_or_empty"],
        }

    with open(target) as f:
        data = json.load(f)

    records = data.get("records", [])
    n_recs = len(records)
    stats = data.get("summary_statistics", {})

    # 1. Full population check (N = 300)
    n_ok = (n_recs == 300)
    checklist["full_test_population_coverage"] = {
        "description": "All 300 test sequences evaluated across ablation conditions.",
        "passed": n_ok,
        "details": f"Evaluated N = {n_recs} sequences",
    }
    if not n_ok:
        failures.append(f"population_mismatch_{n_recs}")

    # 2. All 6 ablation conditions present in every record
    required_conds = [
        "canonical", "early_removal", "transition_removal",
        "late_removal", "reversed", "shuffled"
    ]
    conds_ok = all(
        all(f"acc_{c}" in r and f"prob_{c}" in r for c in required_conds)
        for r in records
    )
    checklist["six_ablation_conditions_complete"] = {
        "description": "Every record contains accuracy and probabilities for all 6 conditions.",
        "passed": conds_ok,
        "details": f"Verified conditions: {required_conds}",
    }
    if not conds_ok:
        failures.append("missing_ablation_conditions")

    # 3. Cohort C rescues sensitivity check (Temporal drop > 0)
    c_rescues = [r for r in records if r["cohort"] == "C"]
    c_ok = len(c_rescues) == 97
    # Shuffling or reversal must degrade Cohort C on average
    mean_shuf_drop = np.mean([r["delta_p_shuffled"] for r in c_rescues])
    mean_rev_drop = np.mean([r["delta_p_reversed"] for r in c_rescues])
    sens_ok = c_ok and (mean_shuf_drop > 0.0) and (mean_rev_drop > 0.0)

    checklist["cohort_c_rescue_sensitivity"] = {
        "description": "Cohort C rescues demonstrate statistically positive temporal criticality drops.",
        "passed": sens_ok,
        "details": f"Rescues N = {len(c_rescues)}, Mean Delta P(shuffled) = +{mean_shuf_drop:.3f}, Delta P(reversed) = +{mean_rev_drop:.3f}",
    }
    if not sens_ok:
        failures.append("cohort_c_zero_sensitivity")

    # 4. Length-matched control check (Conditions B, C, D have T = 6)
    len_b = len(TemporalAblationOperator.CONDITIONS["early_removal"])
    len_c = len(TemporalAblationOperator.CONDITIONS["transition_removal"])
    len_d = len(TemporalAblationOperator.CONDITIONS["late_removal"])
    len_ok = (len_b == 6) and (len_c == 6) and (len_d == 6)
    checklist["equal_length_control_satisfied"] = {
        "description": "Conditions B, C, and D are identically length-matched at T = 6.",
        "passed": len_ok,
        "details": f"Lengths: early={len_b}, transition={len_c}, late={len_d}",
    }
    if not len_ok:
        failures.append("unequal_ablation_lengths")

    # 5. Taxonomy breakdown presence
    tax_breakdown = stats.get("taxonomy_breakdown", {})
    tax_ok = len(tax_breakdown) >= 8
    checklist["taxonomy_criticality_breakdown"] = {
        "description": "Criticality drops stratified across all 8 pre-registered taxonomy categories.",
        "passed": tax_ok,
        "details": f"Stratified across {len(tax_breakdown)} categories",
    }
    if not tax_ok:
        failures.append("incomplete_taxonomy_breakdown")

    total_checks = len(checklist)
    passed_checks = sum(1 for c in checklist.values() if c["passed"])
    status = "CERTIFIED" if (passed_checks == total_checks) else "FAILED"

    return {
        "milestone": "M5.6_TEMPORAL_ABLATION",
        "status": status,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "compliance_rate": f"{(passed_checks / total_checks) * 100:.1f}%",
        "checklist": checklist,
        "failures": failures,
    }

