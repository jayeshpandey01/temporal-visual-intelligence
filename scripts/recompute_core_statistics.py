#!/usr/bin/env python3
"""Recompute core paired comparisons from saved per-sequence predictions.

Inference unit: sequence. Each sequence's correctness difference is averaged
over the three saved model seeds, then sequences are resampled/permuted. This
quantifies test-sequence uncertainty conditional on these checkpoints; it does
not capture training-seed variability or constitute independent retraining.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)
BOOTSTRAPS = 20_000
PERMUTATIONS = 100_000
RNG_SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correctness_delta(predictions: pd.DataFrame, truth_col: str, left_cols: list[str], right_cols: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    truth = predictions[truth_col].to_numpy()
    per_seed = {}
    values = []
    for seed, left_col, right_col in zip(SEEDS, left_cols, right_cols):
        left = (predictions[left_col].to_numpy() == truth).astype(float)
        right = (predictions[right_col].to_numpy() == truth).astype(float)
        values.append(left - right)
        per_seed[str(seed)] = float(np.mean(left - right))
    return np.mean(np.column_stack(values), axis=1), per_seed


def paired_bootstrap_ci(deltas: np.ndarray, rng: np.random.Generator) -> list[float]:
    n = len(deltas)
    means = np.empty(BOOTSTRAPS, dtype=float)
    chunk_size = 2_000
    offset = 0
    while offset < BOOTSTRAPS:
        count = min(chunk_size, BOOTSTRAPS - offset)
        indices = rng.integers(0, n, size=(count, n))
        means[offset : offset + count] = deltas[indices].mean(axis=1)
        offset += count
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(lo), float(hi)]


def sign_flip_pvalue(deltas: np.ndarray, rng: np.random.Generator) -> tuple[float, int]:
    observed = abs(float(np.mean(deltas)))
    n = len(deltas)
    extreme = 0
    chunk_size = 5_000
    remaining = PERMUTATIONS
    while remaining:
        count = min(chunk_size, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, n))
        permuted = np.abs((signs * deltas).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
        remaining -= count
    return float((extreme + 1) / (PERMUTATIONS + 1)), extreme


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    adjusted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    output = np.empty(m, dtype=float)
    output[order] = np.minimum(adjusted, 1.0)
    return output.tolist()


def main() -> None:
    progressive_path = ROOT / "results/experiments/progressive/progressive_predictions.csv"
    order_path = ROOT / "results/experiments/shuffle/order_predictions.csv"
    reversal_path = ROOT / "results/experiments/reversal/reversal_predictions.csv"
    statistical_path = ROOT / "results/experiments/statistical_predictions.csv"

    progressive = pd.read_csv(progressive_path).set_index("sequence_id").sort_index()
    order = pd.read_csv(order_path).set_index("sequence_id").sort_index()
    reversal = pd.read_csv(reversal_path)
    stored_deltas = pd.read_csv(statistical_path).set_index("sequence_id").sort_index()

    if not progressive.index.equals(order.index) or not order.index.equals(stored_deltas.index):
        raise ValueError("Prediction artifacts do not contain identical sequence IDs.")
    if not np.array_equal(progressive["ground_truth"], order["ground_truth"]):
        raise ValueError("Progressive and order prediction ground truths differ.")
    if not np.array_equal(order["ground_truth"], stored_deltas["ground_truth"]):
        raise ValueError("Saved statistical deltas do not share the same ground truth rows.")

    reverse_gru = reversal[reversal["model"] == "gru"]
    order_long = order.reset_index()
    for seed in SEEDS:
        seed_rows = reverse_gru[reverse_gru["seed"] == seed].set_index("sequence_id").sort_index()
        if not seed_rows.index.equals(order.index):
            raise ValueError(f"Reversal prediction IDs differ for seed {seed}.")
        if not np.array_equal(seed_rows["pred_forward"].to_numpy(), order_long[f"pred_gru_s{seed}_canonical"].to_numpy()):
            raise ValueError(f"Forward predictions disagree between reversal and order artifacts for seed {seed}.")
        if not np.array_equal(seed_rows["pred_reversed"].to_numpy(), order_long[f"pred_gru_s{seed}_reversed"].to_numpy()):
            raise ValueError(f"Reversed predictions disagree between artifacts for seed {seed}.")

    comparisons = [
        ("H1", "GRU vs single-frame baseline (K=8)",
         correctness_delta(progressive, "ground_truth",
                           [f"pred_gru_s{s}_k8" for s in SEEDS],
                           [f"pred_baseline_s{s}_k8" for s in SEEDS]),
         "delta_Model_C_vs_Model_A"),
        ("H2", "GRU vs mean-pooling control (K=8)",
         correctness_delta(progressive, "ground_truth",
                           [f"pred_gru_s{s}_k8" for s in SEEDS],
                           [f"pred_meanpool_s{s}_k8" for s in SEEDS]),
         "delta_Model_C_vs_Model_B"),
        ("H3", "GRU canonical vs shuffled order",
         correctness_delta(order, "ground_truth",
                           [f"pred_gru_s{s}_canonical" for s in SEEDS],
                           [f"pred_gru_s{s}_shuffled" for s in SEEDS]),
         "delta_Model_C_Can_vs_Shuf"),
        ("H4", "GRU canonical vs reversed order",
         correctness_delta(order, "ground_truth",
                           [f"pred_gru_s{s}_canonical" for s in SEEDS],
                           [f"pred_gru_s{s}_reversed" for s in SEEDS]),
         "delta_Model_C_Fwd_vs_Rev"),
        ("H14", "CompactGRU vs mean-pooling control (K=8)",
         correctness_delta(progressive, "ground_truth",
                           [f"pred_compact_gru_s{s}_k8" for s in SEEDS],
                           [f"pred_meanpool_s{s}_k8" for s in SEEDS]),
         "delta_Model_D_vs_Model_B"),
    ]

    rng = np.random.default_rng(RNG_SEED)
    records = []
    for hypothesis, name, (deltas, per_seed), historical_column in comparisons:
        stored = stored_deltas[historical_column].to_numpy(dtype=float)
        if not np.allclose(deltas, stored, atol=1e-12, rtol=0):
            raise ValueError(f"Recomputed per-sequence deltas do not match {historical_column}.")
        p_value, exceedances = sign_flip_pvalue(deltas, rng)
        records.append({
            "hypothesis_id": hypothesis,
            "comparison": name,
            "n_sequences": int(len(deltas)),
            "mean_delta_accuracy": float(np.mean(deltas)),
            "delta_pp": float(np.mean(deltas) * 100),
            "bootstrap_ci95": paired_bootstrap_ci(deltas, rng),
            "p_sign_flip_two_sided": p_value,
            "permutation_exceedances": exceedances,
            "permutation_resolution": 1 / (PERMUTATIONS + 1),
            "per_seed_mean_delta_accuracy": per_seed,
            "source_delta_column_validated": historical_column,
            "validation_status": "RECOMPUTED_FROM_PAIRED_PREDICTIONS",
        })

    q_values = benjamini_hochberg([r["p_sign_flip_two_sided"] for r in records])
    for record, q in zip(records, q_values):
        record["q_bh_verified_family"] = q
        record["reject_bh_0_05"] = q <= 0.05

    result = {
        "status": "RECOMPUTED_CORE_FAMILY",
        "valid_for_inference": True,
        "family_size": len(records),
        "family_scope": "Five core contrasts with complete per-sequence paired prediction artifacts (H1-H4 and capacity-parity H14).",
        "inference_unit": "sequence; each sequence's correctness difference is averaged over seeds 42, 43, 44",
        "uncertainty": f"Percentile paired bootstrap over sequences, B={BOOTSTRAPS}; conditional on the saved checkpoints.",
        "test": f"Two-sided Monte Carlo sign-flip permutation over sequence-level paired differences, B={PERMUTATIONS}; RNG seed {RNG_SEED}. Report lower bounds at the Monte Carlo resolution rather than treating zero observed exceedances as p=0.",
        "multiple_testing": "Benjamini-Hochberg across only this verified five-comparison family (alpha=0.05).",
        "limitations": [
            "Intervals and tests condition on the saved model checkpoints and do not capture training-seed variability.",
            "The family excludes M6 sampling, corruption, CLIP, and cross-validation comparisons because complete paired per-sequence predictions were not available in the audited files.",
            "This is a reanalysis of saved predictions, not independent training replication.",
        ],
        "input_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [progressive_path, order_path, reversal_path, statistical_path]
        },
        "comparisons": records,
    }

    output_path = ROOT / "results/final/paired_statistics.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    table = [
        "# Recomputed Paired Core Statistics",
        "",
        "> Recomputed from sequence-level prediction rows. The five-comparison family is limited to H1-H4 and H14 because the other M6 contrasts lack complete paired prediction artifacts in the audited tree.",
        "> Bootstrap intervals and sign-flip tests resample sequences and condition on the saved checkpoints; they do not capture training-seed variability.",
        "",
        "| ID | Paired comparison | N | Mean Δ accuracy (pp) | 95% paired bootstrap CI (pp) | Sign-flip p | BH q (5-test family) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lo, hi = r["bootstrap_ci95"]
        table.append(
            f"| {r['hypothesis_id']} | {r['comparison']} | {r['n_sequences']} | "
            f"{r['delta_pp']:+.2f} | [{lo*100:+.2f}, {hi*100:+.2f}] | "
            f"{r['p_sign_flip_two_sided']:.5g} | {r['q_bh_verified_family']:.5g} |"
        )
    (ROOT / "results/tables/paired_statistics.md").write_text("\n".join(table) + "\n")

    for record in records:
        p_display = f"<{1/(PERMUTATIONS+1):.2g}" if record["permutation_exceedances"] == 0 else f"{record['p_sign_flip_two_sided']:.5g}"
        q_display = f"<{1/(PERMUTATIONS+1):.2g}" if record["q_bh_verified_family"] <= 1/(PERMUTATIONS+1) else f"{record['q_bh_verified_family']:.5g}"
        print(
            f"{record['hypothesis_id']} {record['delta_pp']:+.2f} pp "
            f"CI=[{record['bootstrap_ci95'][0]*100:+.2f}, {record['bootstrap_ci95'][1]*100:+.2f}] "
            f"p{p_display} q{q_display}"
        )
    print(f"Wrote {output_path.relative_to(ROOT)} and results/tables/paired_statistics.md")


if __name__ == "__main__":
    main()
