"""
Milestone M5.7: Minimum Sufficient Temporal Context (K_min) Evaluator.

Formulates and computes:
  1. K_min^prefix: Minimum causal prefix horizon [F_0 ... F_{K-1}], K in [1, 8],
     required for the online recurrent model to correctly predict the action.
  2. K_min^window: Minimum contiguous sub-sequence window length w in [1, 8]
     where at least one window [F_t ... F_{t+w-1}] achieves correct classification.
  3. Optimal window coordinates [t_start, t_end], directly connecting to
     critical-frame localization (t_crit) from M5.4 and M5.5.

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
# Minimum Temporal Context Evaluator
# ---------------------------------------------------------------------------
class MinimumTemporalContextEvaluator:
    """
    Searches for minimum sufficient temporal context K_min across prefix horizons
    and arbitrary contiguous sub-windows using frozen Model C checkpoints.
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

        # Exact M4 GRU architecture:
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
        Execute prefix and window searches for all 300 test sequences across all 3 seeds.
        """
        # Load all embeddings
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

        n_seqs = len(seq_records)
        seed_results = {s: [] for s in self.seeds}

        for seed in self.seeds:
            model = self.load_frozen_gru_model(seed)

            # 1. Prefix evaluations: K in 1..8
            prefix_probs_by_k = {}
            prefix_preds_by_k = {}
            for k in range(1, 9):
                batch = np.stack([s["embedding"][:k] for s in seq_records])  # [N, K, 384]
                t_batch = torch.tensor(batch, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    logits = model(t_batch)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    preds = np.argmax(probs, axis=-1)
                prefix_probs_by_k[k] = probs
                prefix_preds_by_k[k] = preds

            # 2. Arbitrary window evaluations: w in 1..8, start in 0..(8-w)
            window_evals_by_seq = [[] for _ in range(n_seqs)]
            for w in range(1, 9):
                n_win = 8 - w + 1
                for start in range(n_win):
                    end = start + w - 1
                    batch = np.stack([s["embedding"][start : end + 1] for s in seq_records])
                    t_batch = torch.tensor(batch, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        logits = model(t_batch)
                        probs = F.softmax(logits, dim=-1).cpu().numpy()
                        preds = np.argmax(probs, axis=-1)

                    for i in range(n_seqs):
                        gt = seq_records[i]["ground_truth"]
                        window_evals_by_seq[i].append({
                            "window_len": w,
                            "start": start,
                            "end": end,
                            "pred": int(preds[i]),
                            "is_correct": bool(preds[i] == gt),
                            "prob_gt": float(probs[i][gt]),
                        })

            # Process per-sequence metrics for this seed
            for i, s in enumerate(seq_records):
                gt = s["ground_truth"]

                # A. K_min^prefix: smallest k where prefix predicts correctly
                k_min_prefix = 8  # Fallback to full sequence
                for k in range(1, 9):
                    if prefix_preds_by_k[k][i] == gt:
                        k_min_prefix = k
                        break

                # B. K_min^window: smallest w where at least one window correctly classifies
                wins = window_evals_by_seq[i]
                k_min_window = 8
                best_window = (0, 7)
                best_prob = 0.0

                for w in range(1, 9):
                    correct_wins = [win for win in wins if win["window_len"] == w and win["is_correct"]]
                    if correct_wins:
                        k_min_window = w
                        # Pick window with highest prob_gt
                        top_win = max(correct_wins, key=lambda x: x["prob_gt"])
                        best_window = (top_win["start"], top_win["end"])
                        best_prob = top_win["prob_gt"]
                        break

                # If no sub-window succeeded, find window with max prob_gt
                if best_prob == 0.0:
                    top_win = max(wins, key=lambda x: x["prob_gt"])
                    best_window = (top_win["start"], top_win["end"])
                    k_min_window = top_win["window_len"]
                    best_prob = top_win["prob_gt"]

                seed_results[seed].append({
                    "k_min_prefix": k_min_prefix,
                    "k_min_window": k_min_window,
                    "window_start": best_window[0],
                    "window_end": best_window[1],
                    "prefix_acc": {k: float(prefix_preds_by_k[k][i] == gt) for k in range(1, 9)},
                    "prefix_probs": {k: float(prefix_probs_by_k[k][i][gt]) for k in range(1, 9)},
                })

        # Consensus and multi-seed aggregation across all sequences
        master_rows = []
        for i, s in enumerate(seq_records):
            sid = s["sequence_id"]
            gt = s["ground_truth"]
            cohort = s["cohort"]
            cls_name = s["class_name"]

            gold_row = self.gold_df.loc[sid] if (not self.gold_df.empty and sid in self.gold_df.index) else {}

            k_prefix_seeds = [seed_results[seed][i]["k_min_prefix"] for seed in self.seeds]
            k_window_seeds = [seed_results[seed][i]["k_min_window"] for seed in self.seeds]
            w_starts = [seed_results[seed][i]["window_start"] for seed in self.seeds]
            w_ends = [seed_results[seed][i]["window_end"] for seed in self.seeds]

            # Average and median
            k_prefix_mean = float(np.mean(k_prefix_seeds))
            k_prefix_median = int(np.median(k_prefix_seeds))
            k_window_mean = float(np.mean(k_window_seeds))
            k_window_median = int(np.median(k_window_seeds))

            # Optimal window consensus (median bounds)
            med_start = int(np.median(w_starts))
            med_end = int(np.median(w_ends))
            if med_end < med_start:
                med_end = med_start

            row: Dict[str, Any] = {
                "sequence_id": sid,
                "video_id": s["video_id"],
                "class_name": cls_name,
                "ground_truth": gt,
                "cohort": cohort,
                "k_min_prefix": k_prefix_median,
                "k_min_prefix_mean": k_prefix_mean,
                "k_min_window": k_window_median,
                "k_min_window_mean": k_window_mean,
                "optimal_window": f"[{med_start}, {med_end}]",
                "window_start": med_start,
                "window_end": med_end,
                "window_span": med_end - med_start + 1,
            }

            # Prefix progressive accuracies averaged across seeds
            for k in range(1, 9):
                row[f"acc_prefix_{k}"] = float(np.mean([seed_results[sd][i]["prefix_acc"][k] for sd in self.seeds]))
                row[f"prob_prefix_{k}"] = float(np.mean([seed_results[sd][i]["prefix_probs"][k] for sd in self.seeds]))

            # Attach taxonomy categories
            for tax_col in [
                "displacement", "directional_motion", "state_transition",
                "object_interaction", "relative_motion", "event_order",
                "static_appearance", "frame_ambiguity"
            ]:
                row[tax_col] = bool(gold_row.get(tax_col, False)) if hasattr(gold_row, "get") else False

            master_rows.append(row)

        df = pd.DataFrame(master_rows)
        stats = self._compute_summary_statistics(df)

        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_sequences": len(df),
                "seeds": self.seeds,
            },
            "summary_statistics": stats,
            "dataframe": df,
        }

    def _compute_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute aggregate benchmark metrics, cohort distributions, and taxonomy breakdowns."""
        # 1. Overall benchmark distribution
        kp_mean, kp_lo, kp_hi = compute_bootstrap_ci(df["k_min_prefix"].values)
        kw_mean, kw_lo, kw_hi = compute_bootstrap_ci(df["k_min_window"].values)

        overall = {
            "k_min_prefix_mean": kp_mean,
            "k_min_prefix_ci": [kp_lo, kp_hi],
            "k_min_prefix_median": float(df["k_min_prefix"].median()),
            "k_min_window_mean": kw_mean,
            "k_min_window_ci": [kw_lo, kw_hi],
            "k_min_window_median": float(df["k_min_window"].median()),
            "progressive_acc_curve": {
                k: float(df[f"acc_prefix_{k}"].mean()) for k in range(1, 9)
            },
            "progressive_prob_curve": {
                k: float(df[f"prob_prefix_{k}"].mean()) for k in range(1, 9)
            },
        }

        # 2. Stratification by Cohort
        cohort_stats = {}
        for c in ["A", "B", "C", "D"]:
            c_df = df[df["cohort"] == c]
            c_n = len(c_df)
            if c_n > 0:
                p_m, p_lo, p_hi = compute_bootstrap_ci(c_df["k_min_prefix"].values)
                w_m, w_lo, w_hi = compute_bootstrap_ci(c_df["k_min_window"].values)
                cohort_stats[c] = {
                    "count": c_n,
                    "k_min_prefix_mean": p_m,
                    "k_min_prefix_ci": [p_lo, p_hi],
                    "k_min_prefix_median": float(c_df["k_min_prefix"].median()),
                    "k_min_window_mean": w_m,
                    "k_min_window_ci": [w_lo, w_hi],
                    "k_min_window_median": float(c_df["k_min_window"].median()),
                    "pct_k_le_3": float((c_df["k_min_window"] <= 3).mean() * 100),
                    "pct_k_4_to_6": float(((c_df["k_min_window"] >= 4) & (c_df["k_min_window"] <= 6)).mean() * 100),
                    "pct_k_ge_7": float((c_df["k_min_window"] >= 7).mean() * 100),
                }

        # 3. Stratification by Taxonomy Categories in Cohort C Rescues
        c_rescues = df[df["cohort"] == "C"]
        tax_stats = {}
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
            if sub_n > 0:
                p_m, p_lo, p_hi = compute_bootstrap_ci(sub["k_min_prefix"].values)
                w_m, w_lo, w_hi = compute_bootstrap_ci(sub["k_min_window"].values)
                tax_stats[t_id] = {
                    "category_id": t_id,
                    "name": name,
                    "count": sub_n,
                    "k_min_prefix_mean": p_m,
                    "k_min_prefix_ci": [p_lo, p_hi],
                    "k_min_prefix_median": float(sub["k_min_prefix"].median()),
                    "k_min_window_mean": w_m,
                    "k_min_window_ci": [w_lo, w_hi],
                    "k_min_window_median": float(sub["k_min_window"].median()),
                    "typical_window_span": float(sub["window_span"].mean()),
                }

        return {
            "overall": overall,
            "cohort_breakdown": cohort_stats,
            "taxonomy_breakdown": tax_stats,
        }

    def save_deliverables(
        self,
        eval_output: Dict[str, Any],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """Save results/m5/ deliverables: JSON, Matrix CSV, Taxonomy CSV, and Summary MD."""
        out_dir = Path(output_dir).resolve() if output_dir else self.root / "results" / "m5"
        out_dir.mkdir(parents=True, exist_ok=True)

        df = eval_output["dataframe"]

        # 1. Master JSON deliverable
        json_path = out_dir / "minimum_temporal_context.json"
        serializable = {
            "metadata": eval_output["metadata"],
            "summary_statistics": eval_output["summary_statistics"],
            "records": df.to_dict(orient="records"),
        }
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)

        # 2. Tabular CSV
        csv_path = out_dir / "minimum_temporal_context.csv"
        df.to_csv(csv_path, index=False)

        # 3. Taxonomy CSV
        tax_rows = list(eval_output["summary_statistics"]["taxonomy_breakdown"].values())
        tax_df = pd.DataFrame(tax_rows)
        tax_path = out_dir / "minimum_temporal_context_by_taxonomy.csv"
        tax_df.to_csv(tax_path, index=False)

        # 4. Summary Markdown Report
        summary_path = out_dir / "minimum_temporal_context_summary.md"
        summary_text = self._build_markdown_summary(eval_output)
        with open(summary_path, "w") as f:
            f.write(summary_text)

        return {
            "minimum_temporal_context_json": json_path,
            "minimum_temporal_context_csv": csv_path,
            "minimum_temporal_context_by_taxonomy_csv": tax_path,
            "minimum_temporal_context_summary_md": summary_path,
        }

    def _build_markdown_summary(self, eval_output: Dict[str, Any]) -> str:
        """Format scientific Markdown summary report."""
        stats = eval_output["summary_statistics"]
        overall = stats["overall"]
        cohort = stats["cohort_breakdown"]
        tax = stats["taxonomy_breakdown"]
        meta = eval_output["metadata"]

        md = []
        md.append("# Milestone M5.7: Minimum Sufficient Temporal Context (K_min) Summary\n\n")
        md.append(f"**Total Evaluated Sequences:** $N = {meta['total_sequences']}$\n")
        md.append(f"**Evaluated Seeds:** `{meta['seeds']}`\n")
        md.append(f"**Model:** Frozen `TemporalGRU` (Model C from `experiments/m4_final/`)\n")
        md.append(f"**Timestamp:** `{meta['timestamp']}`\n\n")

        md.append("## 1. Overall Temporal Context Requirements\n\n")
        kp_m = overall['k_min_prefix_mean']
        kp_ci = overall['k_min_prefix_ci']
        kp_med = overall['k_min_prefix_median']
        kw_m = overall['k_min_window_mean']
        kw_ci = overall['k_min_window_ci']
        kw_med = overall['k_min_window_median']
        md.append(f"- **Prefix Horizon (K_min^prefix):** Mean = `{kp_m:.2f}` [{kp_ci[0]:.2f}, {kp_ci[1]:.2f}] (95% CI), Median = `{kp_med:.0f}` frames\n")
        md.append(f"- **Arbitrary Sub-Window (K_min^window):** Mean = `{kw_m:.2f}` [{kw_ci[0]:.2f}, {kw_ci[1]:.2f}] (95% CI), Median = `{kw_med:.0f}` frames\n\n")

        md.append("### Cumulative Prefix Observation Horizon Curve (K = 1 to 8)\n\n")
        md.append("| Context Budget K | 1 Frame | 2 Frames | 3 Frames | 4 Frames | 5 Frames | 6 Frames | 7 Frames | 8 Frames (Full) |\n")
        md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        accs = [overall["progressive_acc_curve"][k] * 100 for k in range(1, 9)]
        md.append(f"| **Benchmark Accuracy** | {accs[0]:5.1f}% | {accs[1]:5.1f}% | {accs[2]:5.1f}% | {accs[3]:5.1f}% | {accs[4]:5.1f}% | {accs[5]:5.1f}% | {accs[6]:5.1f}% | **{accs[7]:5.1f}%** |\n\n")

        md.append("## 2. K_min Distribution Across Experimental Cohorts\n\n")
        md.append("| Cohort | Total N | Mean K_min (Prefix) | Median K_min (Prefix) | Mean K_min (Window) | Median K_min (Window) | <= 3 Frames | 4-6 Frames | 7-8 Frames |\n")
        md.append("|:---:|---:|:---:|:---:|:---:|:---:|---:|---:|---:|\n")
        for c in ["A", "B", "C", "D"]:
            cd = cohort.get(c, {})
            cnt = cd.get("count", 0)
            p_m = cd.get('k_min_prefix_mean', 0.0)
            p_med = cd.get('k_min_prefix_median', 0.0)
            w_m = cd.get('k_min_window_mean', 0.0)
            w_med = cd.get('k_min_window_median', 0.0)
            pct_le3 = cd.get('pct_k_le_3', 0.0)
            pct_46 = cd.get('pct_k_4_to_6', 0.0)
            pct_ge7 = cd.get('pct_k_ge_7', 0.0)
            md.append(
                f"| **Cohort {c}** | {cnt} | "
                f"`{p_m:.2f}` | `{p_med:.0f}` | "
                f"`{w_m:.2f}` | `{w_med:.0f}` | "
                f"{pct_le3:5.1f}% | {pct_46:5.1f}% | {pct_ge7:5.1f}% |\n"
            )

        md.append("\n## 3. Sufficient Context by Gold-Standard Taxonomy Category in Rescues (Cohort C, N = 97)\n\n")
        md.append("| Phenomenon Category | Code | N in Rescues | Mean K_min (Window) | 95% Bootstrap CI | Median K_min (Window) | Mean Window Span |\n")
        md.append("|---|:---:|---:|:---:|:---:|:---:|:---:|\n")
        for t_id, d in tax.items():
            name = d["name"]
            cnt = d["count"]
            ci = d["k_min_window_ci"]
            mean_kw = d['k_min_window_mean']
            med_kw = d['k_min_window_median']
            span = d['typical_window_span']
            md.append(
                f"| **{name}** | `{t_id}` | {cnt} | "
                f"`{mean_kw:.2f}` | [{ci[0]:.2f}, {ci[1]:.2f}] | "
                f"`{med_kw:.0f}` | `{span:.1f}` frames |\n"
            )

        md.append("\n---\n*Generated automatically by `src.experiments.minimum_context` under Milestone M5.7.*")
        return "".join(md)


# ---------------------------------------------------------------------------
# Strict Verification Audit
# ---------------------------------------------------------------------------
def audit_minimum_temporal_context(deliverables_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Exhaustively audit Milestone M5.7 deliverables.
    Validates population coverage, K_min bounds [1, 8], window inequalities, and Cohort C coverage.
    """
    target = Path(deliverables_path).resolve()
    checklist = {}
    failures = []

    if not target.exists() or target.stat().st_size == 0:
        return {
            "status": "FAILED",
            "message": f"minimum_temporal_context.json missing or empty at {target}",
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
        "description": "All 300 test sequences evaluated for minimum temporal context.",
        "passed": n_ok,
        "details": f"Evaluated N = {n_recs} sequences",
    }
    if not n_ok:
        failures.append(f"population_mismatch_{n_recs}")

    # 2. Bounds check on K_min (strictly between 1 and 8)
    bounds_ok = all(
        1 <= r["k_min_prefix"] <= 8 and 1 <= r["k_min_window"] <= 8
        for r in records
    )
    checklist["k_min_strictly_bounded"] = {
        "description": "All K_min values are strictly bounded within [1, 8] frames.",
        "passed": bounds_ok,
        "details": "Validated prefix and window bounds across all 300 records",
    }
    if not bounds_ok:
        failures.append("k_min_out_of_bounds")

    # 3. Window inequality check: K_min^window <= K_min^prefix
    # (Unconstrained contiguous sub-window search can never need more frames than prefix constrained to start at 0)
    ineq_ok = all(r["k_min_window"] <= r["k_min_prefix"] for r in records)
    checklist["window_le_prefix_inequality"] = {
        "description": "K_min^window <= K_min^prefix holds across all sequences.",
        "passed": ineq_ok,
        "details": "Confirmed optimal window requires <= frames than fixed prefix",
    }
    if not ineq_ok:
        failures.append("window_greater_than_prefix")

    # 4. Cohort C rescues coverage (N = 97)
    c_recs = [r for r in records if r["cohort"] == "C"]
    c_ok = len(c_recs) == 97
    checklist["cohort_c_rescue_completeness"] = {
        "description": "All 97 Cohort C rescues have audited K_min parameters.",
        "passed": c_ok,
        "details": f"Audited {len(c_recs)} Cohort C rescues",
    }
    if not c_ok:
        failures.append(f"cohort_c_incomplete_{len(c_recs)}")

    # 5. Taxonomy breakdown presence
    tax_breakdown = stats.get("taxonomy_breakdown", {})
    tax_ok = len(tax_breakdown) >= 8
    checklist["taxonomy_kmin_stratification"] = {
        "description": "K_min distributions computed for all 8 pre-registered taxonomy categories.",
        "passed": tax_ok,
        "details": f"Stratified across {len(tax_breakdown)} categories",
    }
    if not tax_ok:
        failures.append("incomplete_taxonomy_stratification")

    total_checks = len(checklist)
    passed_checks = sum(1 for c in checklist.values() if c["passed"])
    status = "CERTIFIED" if (passed_checks == total_checks) else "FAILED"

    return {
        "milestone": "M5.7_MINIMUM_CONTEXT",
        "status": status,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "compliance_rate": f"{(passed_checks / total_checks) * 100:.1f}%",
        "checklist": checklist,
        "failures": failures,
    }
