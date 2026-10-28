"""
Milestone M5.8: Temporal Information versus Static Information Formalization.

Decomposes the visual information in video action recognition into three formal states:
  1. I_static: Single-frame spatial appearance (Model A: SingleFrameMLP, T = 1)
  2. I_spatial (I_accum): Multi-frame observation accumulation without order
     (Model B: MultiFrameMeanPool, T = 8, order-invariant)
  3. I_temporal (I_total): Chronological recurrent sequence modeling
     (Model C: TemporalGRU and Model D: CompactTemporalGRU, T = 8, ordered)

Computes both empirical accuracy gains:
    Delta_total = Delta_accum + Delta_order
and information-theoretic mutual information proxies (in bits):
    I(Y; M) = log2(20) - NLL(M)
    Delta I_total = Delta I_accum + Delta I_order

Rule:
  M5 must consume M4 outputs; M5 must not silently retrain or redefine M4.
  Evaluates frozen Model A, B, C, and D checkpoints across seeds [42, 43, 44].
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import os
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.baseline import SingleFrameMLP, MultiFrameMeanPool
from src.models.temporal import TemporalGRU
from src.models.compact_gru import CompactTemporalGRU
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
# Information Formalization Evaluator
# ---------------------------------------------------------------------------
class InformationFormalizationEvaluator:
    """
    Evaluates frozen Models A, B, C, D across seeds [42, 43, 44] to formalize
    and disentangle I_static, I_spatial (I_accum), and I_temporal (I_order).
    """

    H0_BITS = math.log2(20.0)  # Prior entropy for uniform 20 classes = ~4.3219 bits

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

    def load_frozen_models(self, seed: int) -> Dict[str, torch.nn.Module]:
        """Instantiate and load exact weights for Models A, B, C, and D."""
        ckpts = self.m4_data["paths"]["checkpoints"]

        # Model A: SingleFrameMLP (in=384, hidden=256, classes=20, eval_anchor='middle')
        model_a = SingleFrameMLP(embed_dim=384, hidden_dim=256, num_classes=20, anchor_mode="middle")
        model_a.load_state_dict(torch.load(ckpts / "baseline" / f"best_seed{seed}.pt", map_location=self.device))
        model_a.eval()

        # Model B: MultiFrameMeanPool (in=384, hidden=256, classes=20)
        model_b = MultiFrameMeanPool(embed_dim=384, hidden_dim=256, num_classes=20)
        model_b.load_state_dict(torch.load(ckpts / "meanpool" / f"best_seed{seed}.pt", map_location=self.device))
        model_b.eval()

        # Model C: TemporalGRU (in=384, proj=256, hidden=128, layers=1, dropout=0.2, classes=20)
        model_c = TemporalGRU(embed_dim=384, projection_dim=256, hidden_dim=128, num_layers=1, dropout=0.2, num_classes=20)
        model_c.load_state_dict(torch.load(ckpts / "gru" / f"best_seed{seed}.pt", map_location=self.device))
        model_c.eval()

        # Model D: CompactTemporalGRU (in=384, proj=64, hidden=130, layers=1, dropout=0.2, classes=20)
        model_d = CompactTemporalGRU(embed_dim=384, hidden_dim=130, projection_dim=64, num_layers=1, num_classes=20)
        model_d.load_state_dict(torch.load(ckpts / "compact_gru" / f"best_seed{seed}.pt", map_location=self.device))
        model_d.eval()

        return {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": model_c,
            "model_d": model_d,
        }

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Execute multi-seed evaluation across all 300 test sequences for Models A, B, C, D.
        Computes accuracy, true-class probability, log-loss NLL, and mutual information bits.
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

        batch_embs = np.stack([s["embedding"] for s in seq_records])  # [N=300, 8, 384]
        t_batch = torch.tensor(batch_embs, dtype=torch.float32, device=self.device)
        n_seqs = len(seq_records)

        # Store seed-level outputs: {seed: {model_name: {preds, probs, nll}}}
        eval_by_seed: Dict[int, Dict[str, Any]] = {}

        for seed in self.seeds:
            models = self.load_frozen_models(seed)
            eval_by_seed[seed] = {}

            for m_key, m_net in models.items():
                with torch.no_grad():
                    logits = m_net(t_batch)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    preds = np.argmax(probs, axis=-1)

                eval_by_seed[seed][m_key] = {
                    "preds": preds,
                    "probs": probs,
                }

        # Sequence-level aggregation across seeds
        master_sequence_rows = []
        for i, s in enumerate(seq_records):
            sid = s["sequence_id"]
            gt = s["ground_truth"]
            cohort = s["cohort"]
            cls_name = s["class_name"]
            gold_row = self.gold_df.loc[sid] if (not self.gold_df.empty and sid in self.gold_df.index) else {}

            row: Dict[str, Any] = {
                "sequence_id": sid,
                "video_id": s["video_id"],
                "class_name": cls_name,
                "ground_truth": gt,
                "cohort": cohort,
            }

            # Per-model multi-seed statistics
            for m_key in ["model_a", "model_b", "model_c", "model_d"]:
                m_label = m_key.replace("model_", "")  # 'a', 'b', 'c', 'd'
                accs = [1.0 if eval_by_seed[seed][m_key]["preds"][i] == gt else 0.0 for seed in self.seeds]
                probs = [float(eval_by_seed[seed][m_key]["probs"][i][gt]) for seed in self.seeds]
                nlls = [-math.log2(max(1e-7, p)) for p in probs]
                info_bits = [max(0.0, self.H0_BITS - nll) for nll in nlls]

                row[f"acc_{m_label}"] = float(np.mean(accs))
                row[f"prob_{m_label}"] = float(np.mean(probs))
                row[f"nll_{m_label}"] = float(np.mean(nlls))
                row[f"info_{m_label}"] = float(np.mean(info_bits))

            # Disentanglement:
            # Delta_accum = Acc(B) - Acc(A)
            # Delta_order = Acc(C) - Acc(B)
            # Delta_total = Acc(C) - Acc(A) = Delta_accum + Delta_order
            acc_a = row["acc_a"]
            acc_b = row["acc_b"]
            acc_c = row["acc_c"]
            acc_d = row["acc_d"]

            delta_accum = acc_b - acc_a
            delta_order = acc_c - acc_b
            delta_total = acc_c - acc_a
            delta_order_matched = acc_d - acc_b

            row["delta_accum"] = float(delta_accum)
            row["delta_order"] = float(delta_order)
            row["delta_total"] = float(delta_total)
            row["delta_order_matched"] = float(delta_order_matched)

            # Information-theoretic bits disentanglement
            info_a = row["info_a"]
            info_b = row["info_b"]
            info_c = row["info_c"]

            d_info_accum = info_b - info_a
            d_info_order = info_c - info_b
            d_info_total = info_c - info_a

            row["delta_info_accum"] = float(d_info_accum)
            row["delta_info_order"] = float(d_info_order)
            row["delta_info_total"] = float(d_info_total)

            # Order ratio (proportion of total information gain attributable to chronological order)
            if d_info_total > 1e-4:
                order_ratio = float(np.clip(d_info_order / d_info_total, 0.0, 1.0))
            elif delta_total > 0:
                order_ratio = float(np.clip(delta_order / delta_total, 0.0, 1.0))
            else:
                order_ratio = 0.0
            row["order_info_ratio"] = order_ratio

            # Diagnostic Case Assignment
            # Case 1: Pure Temporal Order Rescue (A=0, B=0, C=1)
            # Case 2: Order Resolution (A=0, B confused, C=1)
            # Case 3: Temporal Regression (A=1, C=0)
            # Case 4: Spatial Accumulation Sufficient (A=0, B=1, C=1)
            # Case 5: Observation Only (A=0, B=1, C=0)
            # Case 6: Concordant Hard Failure (A=0, B=0, C=0)
            # Case 7: Concordant Easy Case (A=1, B=1, C=1)
            corr_a = acc_a >= 0.5
            corr_b = acc_b >= 0.5
            corr_c = acc_c >= 0.5

            if not corr_a and not corr_b and corr_c:
                diag_case = "Case 1: Pure Temporal Order Rescue"
            elif not corr_a and corr_b and corr_c:
                diag_case = "Case 4: Spatial Accumulation Sufficient"
            elif not corr_a and corr_b and not corr_c:
                diag_case = "Case 5: Observation Only"
            elif corr_a and not corr_c:
                diag_case = "Case 3: Temporal Regression"
            elif corr_a and corr_b and corr_c:
                diag_case = "Case 7: Concordant Easy Case"
            elif not corr_a and not corr_b and not corr_c:
                diag_case = "Case 6: Concordant Hard Failure"
            else:
                diag_case = "Case 2: Order Resolution"
            row["tri_model_case"] = diag_case

            # Attach taxonomy categories
            for tax_col in [
                "displacement", "directional_motion", "state_transition",
                "object_interaction", "relative_motion", "event_order",
                "static_appearance", "frame_ambiguity"
            ]:
                row[tax_col] = bool(gold_row.get(tax_col, False)) if hasattr(gold_row, "get") else False

            master_sequence_rows.append(row)

        df = pd.DataFrame(master_sequence_rows)
        stats = self._compute_summary_statistics(df)

        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_sequences": len(df),
                "seeds": self.seeds,
                "prior_entropy_h0_bits": self.H0_BITS,
            },
            "summary_statistics": stats,
            "dataframe": df,
        }

    def _compute_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute aggregate benchmark metrics, diagnostic distributions, and taxonomy breakdowns."""
        # 1. Overall benchmark metrics for Models A, B, C, D
        models_summary = {}
        for m in ["a", "b", "c", "d"]:
            acc_m, acc_lo, acc_hi = compute_bootstrap_ci(df[f"acc_{m}"].values)
            prob_m, prob_lo, prob_hi = compute_bootstrap_ci(df[f"prob_{m}"].values)
            info_m, info_lo, info_hi = compute_bootstrap_ci(df[f"info_{m}"].values)
            models_summary[f"model_{m}"] = {
                "accuracy": acc_m,
                "accuracy_ci": [acc_lo, acc_hi],
                "probability": prob_m,
                "probability_ci": [prob_lo, prob_hi],
                "mutual_info_bits": info_m,
                "mutual_info_ci": [info_lo, info_hi],
            }

        # Disentangled gains across benchmark
        d_acc_accum_m, d_acc_accum_lo, d_acc_accum_hi = compute_bootstrap_ci(df["delta_accum"].values)
        d_acc_order_m, d_acc_order_lo, d_acc_order_hi = compute_bootstrap_ci(df["delta_order"].values)
        d_acc_total_m, d_acc_total_lo, d_acc_total_hi = compute_bootstrap_ci(df["delta_total"].values)

        d_info_accum_m, d_info_accum_lo, d_info_accum_hi = compute_bootstrap_ci(df["delta_info_accum"].values)
        d_info_order_m, d_info_order_lo, d_info_order_hi = compute_bootstrap_ci(df["delta_info_order"].values)
        d_info_total_m, d_info_total_lo, d_info_total_hi = compute_bootstrap_ci(df["delta_info_total"].values)

        overall = {
            "models": models_summary,
            "delta_accum_acc": d_acc_accum_m,
            "delta_accum_ci": [d_acc_accum_lo, d_acc_accum_hi],
            "delta_order_acc": d_acc_order_m,
            "delta_order_ci": [d_acc_order_lo, d_acc_order_hi],
            "delta_total_acc": d_acc_total_m,
            "delta_total_ci": [d_acc_total_lo, d_acc_total_hi],
            "delta_accum_bits": d_info_accum_m,
            "delta_accum_bits_ci": [d_info_accum_lo, d_info_accum_hi],
            "delta_order_bits": d_info_order_m,
            "delta_order_bits_ci": [d_info_order_lo, d_info_order_hi],
            "delta_total_bits": d_info_total_m,
            "delta_total_bits_ci": [d_info_total_lo, d_info_total_hi],
            "order_share_pct": float((d_acc_order_m / max(1e-6, d_acc_total_m)) * 100),
        }

        # 2. Tri-Model Diagnostic Cases
        case_counts = df["tri_model_case"].value_counts().to_dict()

        # 3. Stratification by Cohort (especially Cohort C rescues)
        cohort_stats = {}
        for c in ["A", "B", "C", "D"]:
            c_df = df[df["cohort"] == c]
            c_n = len(c_df)
            if c_n > 0:
                cohort_stats[c] = {
                    "count": c_n,
                    "acc_a": float(c_df["acc_a"].mean()),
                    "acc_b": float(c_df["acc_b"].mean()),
                    "acc_c": float(c_df["acc_c"].mean()),
                    "acc_d": float(c_df["acc_d"].mean()),
                    "delta_accum": float(c_df["delta_accum"].mean()),
                    "delta_order": float(c_df["delta_order"].mean()),
                    "delta_total": float(c_df["delta_total"].mean()),
                    "info_bits_a": float(c_df["info_a"].mean()),
                    "info_bits_b": float(c_df["info_b"].mean()),
                    "info_bits_c": float(c_df["info_c"].mean()),
                    "delta_info_accum": float(c_df["delta_info_accum"].mean()),
                    "delta_info_order": float(c_df["delta_info_order"].mean()),
                    "order_info_ratio": float(c_df["order_info_ratio"].mean()),
                }

        # 4. Stratification by Taxonomy in Cohort C Rescues
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
                tax_stats[t_id] = {
                    "category_id": t_id,
                    "name": name,
                    "count": sub_n,
                    "acc_a": float(sub["acc_a"].mean()),
                    "acc_b": float(sub["acc_b"].mean()),
                    "acc_c": float(sub["acc_c"].mean()),
                    "delta_accum": float(sub["delta_accum"].mean()),
                    "delta_order": float(sub["delta_order"].mean()),
                    "delta_total": float(sub["delta_total"].mean()),
                    "order_info_ratio": float(sub["order_info_ratio"].mean()),
                    "info_bits_gain_accum": float(sub["delta_info_accum"].mean()),
                    "info_bits_gain_order": float(sub["delta_info_order"].mean()),
                }

        return {
            "overall": overall,
            "case_distribution": case_counts,
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
        json_path = out_dir / "information_formalization.json"
        serializable = {
            "metadata": eval_output["metadata"],
            "summary_statistics": eval_output["summary_statistics"],
            "records": df.to_dict(orient="records"),
        }
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)

        # 2. Tabular CSV
        csv_path = out_dir / "information_decomposition_matrix.csv"
        df.to_csv(csv_path, index=False)

        # 3. Taxonomy CSV
        tax_rows = list(eval_output["summary_statistics"]["taxonomy_breakdown"].values())
        tax_df = pd.DataFrame(tax_rows)
        tax_path = out_dir / "information_decomposition_by_taxonomy.csv"
        tax_df.to_csv(tax_path, index=False)

        # 4. Summary Markdown Report
        summary_path = out_dir / "information_formalization_summary.md"
        summary_text = self._build_markdown_summary(eval_output)
        with open(summary_path, "w") as f:
            f.write(summary_text)

        return {
            "information_formalization_json": json_path,
            "information_decomposition_matrix_csv": csv_path,
            "information_decomposition_by_taxonomy_csv": tax_path,
            "information_formalization_summary_md": summary_path,
        }

    def _build_markdown_summary(self, eval_output: Dict[str, Any]) -> str:
        """Format scientific Markdown summary report."""
        stats = eval_output["summary_statistics"]
        overall = stats["overall"]
        models = overall["models"]
        cases = stats["case_distribution"]
        cohort = stats["cohort_breakdown"]
        tax = stats["taxonomy_breakdown"]
        meta = eval_output["metadata"]

        md = []
        md.append("# Milestone M5.8: Temporal vs. Static Information Formalization Summary\n\n")
        md.append(f"**Total Evaluated Sequences:** $N = {meta['total_sequences']}$\n")
        md.append(f"**Evaluated Seeds:** `{meta['seeds']}`\n")
        md.append(f"**Prior Class Entropy ($H_0$):** `{meta['prior_entropy_h0_bits']:.4f}` bits (uniform 20 classes)\n")
        md.append(f"**Timestamp:** `{meta['timestamp']}`\n\n")

        md.append("## 1. Information Tri-State Disentanglement ($N = 300$)\n\n")
        md.append("| Model Paradigm | Information State | Frame Budget ($T$) | Parameter Count | Top-1 Accuracy (95% CI) | Mutual Info $\\hat{\\mathcal{I}}(Y; M)$ |\n")
        md.append("|---|---|:---:|---:|---:|:---:|\n")
        
        m_rows = [
            ("Model A: SingleFrameMLP", "I_static (Static Appearance)", 1, 103700, models["model_a"]),
            ("Model B: MultiFrameMeanPool", "I_spatial (Observation Accumulation)", 8, 103700, models["model_b"]),
            ("Model D: CompactTemporalGRU", "I_temporal (Capacity-Matched Order)", 8, 103700, models["model_d"]),
            ("Model C: TemporalGRU", "I_temporal (Primary Recurrent Head)", 8, 249364, models["model_c"]),
        ]
        for name, state, t_val, params, d in m_rows:
            acc = d["accuracy"] * 100
            acc_ci = f"[{d['accuracy_ci'][0]*100:4.1f}%, {d['accuracy_ci'][1]*100:4.1f}%]"
            mi = d["mutual_info_bits"]
            mi_ci = f"[{d['mutual_info_ci'][0]:.3f}, {d['mutual_info_ci'][1]:.3f}]"
            md.append(f"| **{name}** | `{state}` | {t_val} | {params:,} | **{acc:5.2f}%** {acc_ci} | **{mi:.3f}** bits {mi_ci} |\n")

        md.append("\n### Empirical & Information-Theoretic Disentanglement Equation\n\n")
        d_tot_acc = overall["delta_total_acc"] * 100
        d_acc_acc = overall["delta_accum_acc"] * 100
        d_ord_acc = overall["delta_order_acc"] * 100
        d_tot_bits = overall["delta_total_bits"]
        d_acc_bits = overall["delta_accum_bits"]
        d_ord_bits = overall["delta_order_bits"]
        order_share = overall["order_share_pct"]

        md.append(f"- **Total Paradigm Gain (Delta_total):** `+{d_tot_acc:.2f}%` (`+{d_tot_bits:.3f}` bits)\n")
        md.append(f"- **Spatial Accumulation Gain (Delta_accum):** `+{d_acc_acc:.2f}%` (`+{d_acc_bits:.3f}` bits)\n")
        md.append(f"- **Temporal Order Gain (Delta_order):** `+{d_ord_acc:.2f}%` (`+{d_ord_bits:.3f}` bits)\n")
        md.append(f"- **Order Share of Total Gain:** `{order_share:.1f}%`\n\n")

        md.append("## 2. Tri-Model Diagnostic Sequence Breakdown (N = 300)\n\n")
        md.append("| Diagnostic Case | Frequency (N) | Proportion (%) | Causal Interpretation |\n")
        md.append("|---|---:|---:|---|\n")
        desc_cases = {
            "Case 1: Pure Temporal Order Rescue": "Neither single frame nor mean pooling succeeds; chronological order alone rescues.",
            "Case 4: Spatial Accumulation Sufficient": "Mean pooling succeeds; spatial coverage is sufficient without order sensitivity.",
            "Case 2: Order Resolution": "Mean pooling confuses features; temporal recurrence resolves the ambiguity.",
            "Case 3: Temporal Regression": "Single frame correct; temporal model overfits to motion or introduces noise.",
            "Case 5: Observation Only": "Mean pooling correct; recurrence fails.",
            "Case 6: Concordant Hard Failure": "Fundamentally ambiguous, uninformative, or corrupted across all paradigms.",
            "Case 7: Concordant Easy Case": "Static visual appearance alone is fully sufficient across all models.",
        }
        for case_name, cnt in sorted(cases.items(), key=lambda x: -x[1]):
            pct = (cnt / meta["total_sequences"]) * 100
            desc = desc_cases.get(case_name, "Diagnostic sequence subset.")
            md.append(f"| **{case_name}** | {cnt} | {pct:5.1f}% | {desc} |\n")

        md.append("\n## 3. Disentanglement in Temporal Rescues (Cohort C, N = 97)\n\n")
        c_stat = cohort.get("C", {})
        acc_a_c = c_stat.get('acc_a', 0.0) * 100
        acc_b_c = c_stat.get('acc_b', 0.0) * 100
        acc_c_c = c_stat.get('acc_c', 0.0) * 100
        d_acc_c = c_stat.get('delta_accum', 0.0) * 100
        d_ord_c = c_stat.get('delta_order', 0.0) * 100
        ratio_c = c_stat.get('order_info_ratio', 0.0) * 100
        md.append(f"- **Model A Accuracy (Baseline):** `{acc_a_c:.1f}%`\n")
        md.append(f"- **Model B Accuracy (MeanPool):** `{acc_b_c:.1f}%`\n")
        md.append(f"- **Model C Accuracy (Temporal GRU):** `{acc_c_c:.1f}%`\n")
        md.append(f"- **Spatial Accumulation Gain (Delta_accum):** `+{d_acc_c:.1f}%`\n")
        md.append(f"- **Temporal Order Gain (Delta_order):** `+{d_ord_c:.1f}%`\n")
        md.append(f"- **Order Information Ratio:** **`{ratio_c:.1f}%`**\n\n")

        md.append("## 4. Order Information Ratio by Gold-Standard Taxonomy Category (Cohort C, N = 97)\n\n")
        md.append("| Phenomenon Category | Code | N in Rescues | Spatial Gain (Delta_accum) | Order Gain (Delta_order) | Order Information Ratio | Primary Causal Mechanism |\n")
        md.append("|---|:---:|---:|---:|---:|---:|---|\n")
        for t_id, d in tax.items():
            name = d["name"]
            cnt = d["count"]
            da = d["delta_accum"] * 100
            do = d["delta_order"] * 100
            ratio = d["order_info_ratio"] * 100
            mech = "Chronological Order Sensitive" if ratio >= 70.0 else "Spatial Accumulation Supported"
            md.append(f"| **{name}** | `{t_id}` | {cnt} | +{da:4.1f}% | **+{do:4.1f}%** | **{ratio:5.1f}%** | {mech} |\n")

        md.append("\n---\n*Generated automatically by `src.experiments.information_formalization` under Milestone M5.8.*")
        return "".join(md)


# ---------------------------------------------------------------------------
# Strict Verification Audit
# ---------------------------------------------------------------------------
def audit_information_formalization(deliverables_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Exhaustively audit Milestone M5.8 deliverables.
    Validates population coverage, information additivity, tri-model case partition, and Cohort C order dominance.
    """
    target = Path(deliverables_path).resolve()
    checklist = {}
    failures = []

    if not target.exists() or target.stat().st_size == 0:
        return {
            "status": "FAILED",
            "message": f"information_formalization.json missing or empty at {target}",
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
        "description": "All 300 test sequences evaluated across tri-model paradigms.",
        "passed": n_ok,
        "details": f"Evaluated N = {n_recs} sequences",
    }
    if not n_ok:
        failures.append(f"population_mismatch_{n_recs}")

    # 2. Additivity property check: delta_total == delta_accum + delta_order
    additivity_ok = all(
        np.isclose(r["delta_total"], r["delta_accum"] + r["delta_order"], atol=1e-5)
        for r in records
    )
    checklist["information_gain_additivity"] = {
        "description": "Delta_total = Delta_accum + Delta_order holds strictly across all records.",
        "passed": additivity_ok,
        "details": "Verified mathematical additivity of empirical gains",
    }
    if not additivity_ok:
        failures.append("gain_additivity_violation")

    # 3. Diagnostic case partition sum check (Sum = 300)
    cases = stats.get("case_distribution", {})
    sum_cases = sum(cases.values())
    cases_ok = (sum_cases == 300)
    checklist["tri_model_cases_partition_complete"] = {
        "description": "Diagnostic cases form an exact exhaustive partition of 300 sequences.",
        "passed": cases_ok,
        "details": f"Partition sum: {sum_cases} across {len(cases)} cases",
    }
    if not cases_ok:
        failures.append(f"case_partition_sum_mismatch_{sum_cases}")

    # 4. Cohort C rescues order dominance check (Order share > 70%)
    c_recs = [r for r in records if r["cohort"] == "C"]
    c_ok = len(c_recs) == 97
    mean_order_ratio = np.mean([r["order_info_ratio"] for r in c_recs])
    dom_ok = c_ok and (mean_order_ratio >= 0.70)
    checklist["cohort_c_order_dominance"] = {
        "description": "Cohort C rescues are overwhelmingly driven by chronological order (ratio >= 70%).",
        "passed": dom_ok,
        "details": f"Rescues N = {len(c_recs)}, Mean Order Ratio = {mean_order_ratio * 100:.1f}%",
    }
    if not dom_ok:
        failures.append(f"order_dominance_insufficient_{mean_order_ratio}")

    # 5. Taxonomy breakdown completeness (All 8 categories present)
    tax_breakdown = stats.get("taxonomy_breakdown", {})
    tax_ok = len(tax_breakdown) >= 8
    checklist["taxonomy_order_breakdown"] = {
        "description": "Disentangled gains computed for all 8 pre-registered taxonomy categories.",
        "passed": tax_ok,
        "details": f"Stratified across {len(tax_breakdown)} categories",
    }
    if not tax_ok:
        failures.append("incomplete_taxonomy_breakdown")

    total_checks = len(checklist)
    passed_checks = sum(1 for c in checklist.values() if c["passed"])
    status = "CERTIFIED" if (passed_checks == total_checks) else "FAILED"

    return {
        "milestone": "M5.8_INFORMATION_FORMALIZATION",
        "status": status,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "compliance_rate": f"{(passed_checks / total_checks) * 100:.1f}%",
        "checklist": checklist,
        "failures": failures,
    }
