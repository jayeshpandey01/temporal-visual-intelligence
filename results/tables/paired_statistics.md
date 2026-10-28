# Recomputed Paired Core Statistics

> Recomputed from sequence-level prediction rows. The five-comparison family is limited to H1-H4 and H14 because the other M6 contrasts lack complete paired prediction artifacts in the audited tree.
> Bootstrap intervals and sign-flip tests resample sequences and condition on the saved checkpoints; they do not capture training-seed variability.

| ID | Paired comparison | N | Mean Δ accuracy (pp) | 95% paired bootstrap CI (pp) | Sign-flip p | BH q (5-test family) |
|---|---|---:|---:|---:|---:|---:|
| H1 | GRU vs single-frame baseline (K=8) | 300 | +26.00 | [+21.67, +30.44] | 9.9999e-06 | 9.9999e-06 |
| H2 | GRU vs mean-pooling control (K=8) | 300 | +26.56 | [+22.11, +31.11] | 9.9999e-06 | 9.9999e-06 |
| H3 | GRU canonical vs shuffled order | 300 | +14.11 | [+10.78, +17.56] | 9.9999e-06 | 9.9999e-06 |
| H4 | GRU canonical vs reversed order | 300 | +18.89 | [+15.11, +22.89] | 9.9999e-06 | 9.9999e-06 |
| H14 | CompactGRU vs mean-pooling control (K=8) | 300 | +15.33 | [+12.33, +18.33] | 9.9999e-06 | 9.9999e-06 |
