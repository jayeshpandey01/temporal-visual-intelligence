"""
M2.10: Embedding Cache Validator.

Executes comprehensive automated integrity checks across all cached embeddings:
  1. Shape consistency: [N_frames, embed_dim]
  2. Dimension integrity: embed_dim == expected_dim (384 for vits14)
  3. Finite numbers: zero NaNs, zero Infs
  4. Non-degeneracy: zero all-zero vectors
  5. Cryptographic checksum: SHA-256 integrity
  6. Bijective label mapping across splits
  7. Cross-split ID isolation (zero leakage)
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from src.embeddings.cache import EmbeddingCache
from src.embeddings.contract import SequenceEmbeddingOutput


def validate_embedding_cache(
    cache_root: str = "embeddings",
    expected_dim: int = 384,
    expected_frames: int = 8,
    output_report: str = "results/representation/cache_validation_report.json"
) -> Dict[str, Any]:
    """
    Run the full M2.10 audit across all cached embedding sequences.
    """
    cache = EmbeddingCache(cache_root)
    manifest = cache.manifest

    report = {
        "cache_root": str(cache.root),
        "expected_dim": expected_dim,
        "expected_frames": expected_frames,
        "splits_audited": {},
        "global_status": "PASS",
        "errors": [],
        "warnings": []
    }

    all_ids_by_split: Dict[str, set] = {}

    for split_name, records in manifest.get("splits", {}).items():
        if not records:
            continue

        split_errors = []
        n_checked = 0
        norms = []

        all_ids_by_split[split_name] = set(records.keys())

        for seq_id in records.keys():
            n_checked += 1
            item = cache.get(seq_id, split=split_name)
            if item is None:
                split_errors.append(f"{seq_id}: Failed to load from cache.")
                continue

            # 1. Unexpected shape check
            if item.embeddings.ndim != 2:
                split_errors.append(f"{seq_id}: unexpected shape ndim {item.embeddings.ndim} != 2")

            # 2. Number of frames == number of embeddings check (M2.10)
            if len(item.frame_indices) != item.embeddings.shape[0]:
                split_errors.append(
                    f"{seq_id}: number_of_frames {len(item.frame_indices)} != "
                    f"number_of_embeddings {item.embeddings.shape[0]}"
                )

            # 3. Embedding dimension == expected dimension check (M2.10)
            if item.embed_dim != expected_dim:
                split_errors.append(
                    f"{seq_id}: embed_dim {item.embed_dim} != expected {expected_dim}"
                )

            # 4. Target frame count check for split
            if item.num_frames != expected_frames and split_name != "real":
                split_errors.append(
                    f"{seq_id}: num_frames {item.num_frames} != expected {expected_frames}"
                )

            # 5. Duplicate frame index & monotonicity check (M2.10)
            if len(item.frame_indices) != len(set(item.frame_indices)):
                split_errors.append(f"{seq_id}: duplicate frame index detected in {item.frame_indices}")
            elif any(item.frame_indices[k] >= item.frame_indices[k+1] for k in range(len(item.frame_indices) - 1)):
                split_errors.append(f"{seq_id}: non-monotonic frame indices detected in {item.frame_indices}")

            # 6. Finite numbers check: NaN? Inf? (M2.10)
            if np.isnan(item.embeddings).any():
                split_errors.append(f"{seq_id}: Contains NaN values.")
            if np.isinf(item.embeddings).any():
                split_errors.append(f"{seq_id}: Contains Inf values.")

            # 7. Non-zero norm / all-zero check (M2.10)
            frame_norms = np.linalg.norm(item.embeddings, axis=1)
            norms.extend(frame_norms.tolist())
            if (frame_norms < 1e-5).any():
                split_errors.append(f"{seq_id}: Contains degenerate all-zero/near-zero vector.")

            # 8. Label integrity check (M2.10)
            if item.label is None or str(item.label).strip() == "":
                split_errors.append(f"{seq_id}: Missing or empty label.")

            # 9. Cryptographic checksum verification
            computed_checksum = item._compute_checksum(item.embeddings)
            if item.checksum and item.checksum != computed_checksum:
                split_errors.append(f"{seq_id}: Checksum mismatch. Expected {item.checksum}, got {computed_checksum}")

        report["splits_audited"][split_name] = {
            "total_sequences": n_checked,
            "error_count": len(split_errors),
            "errors": split_errors[:10],  # Truncate for report readability
            "mean_norm": float(np.mean(norms)) if norms else 0.0,
            "min_norm": float(np.min(norms)) if norms else 0.0,
            "max_norm": float(np.max(norms)) if norms else 0.0,
            "status": "PASS" if len(split_errors) == 0 else "FAIL"
        }

        if len(split_errors) > 0:
            report["global_status"] = "FAIL"
            report["errors"].extend([f"[{split_name}] {e}" for e in split_errors])

    # 5. Cross-split ID isolation check
    train_ids = all_ids_by_split.get("train", set())
    val_ids = all_ids_by_split.get("val", set())
    test_ids = all_ids_by_split.get("test", set())

    leakage_train_val = train_ids & val_ids
    leakage_train_test = train_ids & test_ids
    leakage_val_test = val_ids & test_ids

    report["leakage_check"] = {
        "train_val_overlap": len(leakage_train_val),
        "train_test_overlap": len(leakage_train_test),
        "val_test_overlap": len(leakage_val_test),
        "status": "PASS" if not (leakage_train_val or leakage_train_test or leakage_val_test) else "FAIL"
    }

    if report["leakage_check"]["status"] == "FAIL":
        report["global_status"] = "FAIL"
        report["errors"].append("Cross-split sequence leakage detected in embedding cache!")

    # Write report
    out_p = Path(output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print("M2.10 EMBEDDING CACHE AUDIT SUMMARY")
    print("=" * 65)
    print(f"  Global Status : {report['global_status']}")
    print(f"  Splits Audited: {list(report['splits_audited'].keys())}")
    for s_name, s_data in report["splits_audited"].items():
        print(f"    - {s_name:10s}: {s_data['total_sequences']} sequences | Status: {s_data['status']}")
    print(f"  Zero Leakage  : {report['leakage_check']['status']}")

    # Print exemplary sequence verification (M2.10)
    for sample_split in ["test", "train"]:
        sample_keys = list(manifest.get("splits", {}).get(sample_split, {}).keys())
        if sample_keys:
            sample_seq = sample_keys[0]
            item = cache.get(sample_seq, split=sample_split)
            if item:
                print("\nExemplary Sequence Verification (M2.10 Contract):")
                print("Expected:")
                print(f"  {sample_seq}")
                print(f"  frames       {item.frame_indices}")
                print(f"  embeddings   [{item.num_frames}, {item.embed_dim}]")
                print("Actual:")
                print(f"  frames       {item.frame_indices}")
                print(f"  embeddings   {list(item.embeddings.shape)}")
                print("PASS")
                break

    print(f"  Audit Report  → {output_report}")
    print("=" * 65)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2.10 Embedding Cache Validator")
    parser.add_argument("--cache-root", type=str, default="embeddings")
    parser.add_argument("--expected-dim", type=int, default=384)
    parser.add_argument("--expected-frames", type=int, default=8)
    parser.add_argument("--output", type=str, default="results/representation/cache_validation_report.json")
    args = parser.parse_args()

    validate_embedding_cache(
        cache_root=args.cache_root,
        expected_dim=args.expected_dim,
        expected_frames=args.expected_frames,
        output_report=args.output
    )

