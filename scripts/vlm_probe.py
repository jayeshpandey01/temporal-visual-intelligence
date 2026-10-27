"""
VLM Semantic Change Probe (Day 6 of pipeline).
(research.md §23-25)

Usage:
    export GEMINI_API_KEY=your_key_here
    python scripts/vlm_probe.py --config configs/experiment.yaml --n-samples 20
"""

import sys
import json
import argparse
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.vlm.analyze import SequenceProber, get_vlm_backend


def main(config_path: str, n_samples: int, quadrant_C_dir: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("VLM Semantic Change Probe")
    print(f"Backend: {config.get('vlm', {}).get('backend', 'gemini')}")
    print(f"Samples: {n_samples}")

    vlm = get_vlm_backend(config)
    prober = SequenceProber(vlm, config)

    # Load Quadrant C metadata (temporal rescued examples)
    meta_path = Path(quadrant_C_dir) / 'C_metadata.json'
    if not meta_path.exists():
        print(f"No Quadrant C metadata found at {meta_path}")
        print("Run scripts/evaluate.py --all-experiments first.")
        return

    with open(meta_path) as f:
        samples = json.load(f)[:n_samples]

    print(f"\nProbing {len(samples)} Quadrant C sequences...")
    for i, sample in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] Sequence: {sample.get('sequence_id', sample.get('sample_idx'))}")
        prober.probe_sequence(
            sequence_id=str(sample.get('sample_idx', i)),
            frame_paths=sample.get('frame_paths', []),
            true_label=str(sample.get('true_label', '')),
            baseline_pred=str(sample.get('baseline_pred', '')),
            temporal_pred=str(sample.get('temporal_pred', '')),
        )

    output_path = 'results/vlm_probe.jsonl'
    prober.save_results(output_path)
    summary = prober.summarize()
    print(f"\nVLM Summary:")
    print(f"  VLM accuracy on Quadrant C: {summary.get('vlm_accuracy', 'N/A')}")
    print(f"  % sequences ambiguous from frame 1: {summary.get('pct_ambiguous_from_frame_1', 'N/A')}")
    print(f"  Temporal benefit categories: {summary.get('temporal_benefit_distribution', {})}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment.yaml')
    parser.add_argument('--n-samples', type=int, default=20)
    parser.add_argument('--quadrant-C', default='results/qualitative/quadrant_C')
    args = parser.parse_args()
    main(args.config, args.n_samples, args.quadrant_C)

