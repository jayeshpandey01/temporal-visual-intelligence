# Local Checkpoint Inspector

This is an optional local diagnostic interface. It requires checkpoint weights, feature arrays, and authorized dataset files that are not included in a clean Git clone. Its predictions are checkpoint outputs, not fresh training, and a single sequence is not aggregate evidence. The repository's dataset identity and redistribution rights remain unresolved; see `ASSET_PROVENANCE.md`.

## Commands

```bash
python3 scripts/run_scientific_demo.py
python3 scripts/run_scientific_demo.py --demo-suite
python3 scripts/run_scientific_demo.py --sequence-id vid_07_0001
python3 scripts/demo.py --sequence-id vid_07_0001 --condition canonical --json
```

Supported conditions in `scripts/demo.py` are `canonical`, `shuffled`, `reversed`, and `missing`. The scientific demo supports `canonical`, `shuffled`, `reversed`, and `dropped_2`. Check each script's `--help` output for current options. These perturbations compare model outputs under altered inputs; they do not establish causal mechanisms or robustness beyond the tested examples.

The code records checkpoint paths and, where implemented, hashes. That verifies file identity only; it does not establish data rights, training provenance, or independent replication. Avoid publishing embedded frames or predictions until asset permissions are resolved.
