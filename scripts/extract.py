"""
Embedding extraction script (Day 2 of pipeline).

Usage:
    python scripts/extract.py --config configs/experiment.yaml
    python scripts/extract.py --config configs/experiment.yaml --force  # re-extract
"""

import sys
import argparse
from pathlib import Path

import json
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import TemporalSequenceDataset
from src.embeddings.extractor import EmbeddingExtractor


def main(config_path: str, force: bool = False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load frozen split manifest to ensure 100% split synchronization
    manifest_path = Path("data/metadata/split_manifest.json")
    manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Loaded locked split manifest from {manifest_path}")

    extractor = EmbeddingExtractor(config)
    all_metadata = []

    for split in ['train', 'val', 'test']:
        dataset = TemporalSequenceDataset(
            root=config['dataset']['root'],
            split=split,
            config=config,
            precomputed_splits=manifest,
        )
        print(f"\n{split}: {len(dataset)} sequences")
        metadata = extractor.extract(dataset, split=split, force_reextract=force)
        all_metadata.extend(metadata)

    extractor.save_metadata(all_metadata)

    # Extract dense continuous real sequences (Tier 2 Real Actions: M1.6 & M2.13)
    dense_real_horse = Path("data/tier2_real/dense_horse_50")
    if dense_real_horse.exists():
        print("\nExtracting dense real sequence: horse_running_gallop_50")
        extractor.extract_directory(
            frame_dir=dense_real_horse,
            sequence_id="horse_running_gallop_50",
            label="horse_galloping",
            split="real"
        )

    dense_real_bmw = Path("data/tier2_real/dense_bmw_50")
    if dense_real_bmw.exists():
        print("Extracting dense real sequence: bmw_m4_drift_50")
        extractor.extract_directory(
            frame_dir=dense_real_bmw,
            sequence_id="bmw_m4_drift_50",
            label="car_drifting",
            split="real"
        )

    print("\nExtraction complete.")
    print(f"Total sequences: {len(all_metadata)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment.yaml')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    main(args.config, args.force)

