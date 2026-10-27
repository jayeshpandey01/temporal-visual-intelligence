# Canonical Experimental Ablation Summary

Generated on 2026-10-27 21:48:22 UTC across multi-seed runs.

| Architecture | Observed Frames (K) | Temporal Order | Corruption | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | canonical | none | 5.00 ± 0.00% | 5.00 ± 0.00% | 3 |
| baseline | 8 | canonical | none | 5.00 ± 0.00% | 5.00 ± 0.00% | 3 |
| gru | 1 | canonical | none | 5.00 ± 0.00% | 5.00 ± 0.00% | 3 |
| gru | 2 | canonical | none | 48.67 ± 2.08% | 47.83 ± 1.82% | 3 |
| gru | 4 | canonical | none | 72.33 ± 1.53% | 71.60 ± 1.55% | 3 |
| gru | 8 | canonical | frame_drop | 68.67 ± 1.76% | 67.97 ± 1.80% | 3 |
| gru | 8 | canonical | gaussian_noise | 74.00 ± 1.50% | 73.37 ± 1.55% | 3 |
| gru | 8 | canonical | none | 83.33 ± 1.26% | 82.70 ± 1.25% | 3 |
| gru | 8 | reversed | none | 0.00 ± 0.00% | 0.00 ± 0.00% | 3 |
| gru | 8 | shuffled | none | 5.00 ± 0.00% | 5.00 ± 0.00% | 3 |
| meanpool | 8 | canonical | none | 5.00 ± 0.00% | 5.00 ± 0.00% | 3 |
| meanpool | 8 | shuffled | none | 5.00 ± 0.00% | 5.00 ± 0.00% | 3 |
