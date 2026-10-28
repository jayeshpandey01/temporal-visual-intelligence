# Milestone M4.1 — Experimental Controls Freeze Audit

**Audit Status**: `PASSED`  
**Freeze Timestamp**: `2026-10-28T19:25:00+05:30`  
**Git Commit**: `84b4001d5da9ceb78556990586cd82015ca63f69`  
**Canonical Rule**: *Change exactly one experimental variable at a time.*

---

## Summary of Frozen Controls

| # | Control Axis | Frozen Specification | Status | Key Properties |
|---|---|---|:---:|---|
| 1 | **Dataset** | Something-Something V2 Directional | [PASS] | 20 classes, 2,000 sequences total (100/class) |
| 2 | **Encoder** | DINOv2 ViT-S/14 (Frozen) | [PASS] | 384-dim, eval mode, requires_grad=False (21.8M params) |
| 3 | **Preprocessing** | ImageNet norm, 224x224 | [PASS] | 8 uniform frames, RGB, center crop |
| 4 | **Train Split** | 1,400 sequences (70/class) | [PASS] | ZERO_LEAKAGE (1400 train / 300 val / 300 test) |
| 5 | **Val Split** | 300 sequences (15/class) | [PASS] | Deterministic video ID group split |
| 6 | **Test Split** | 300 sequences (15/class) | [PASS] | Fixed evaluation partition |
| 7 | **Model Checkpoints** | Model A (103.7k), Model B (103.7k), Model C (249.4k) | [PASS] | 3 seeds per model verified on disk |
| 8 | **Evaluation Metrics** | Accuracy, Macro-F1, McNemar, Bootstrap CI | [PASS] | 95% CI (B=1000 resamples) |
| 9 | **Random Seeds** | Model seeds [42, 43, 44], Perturbation seed 42 | [PASS] | Fixed perturbation seed preserves paired comparisons |

---

## Model Checkpoint Inventory & Parameter Accounting

| Model Identifier | Architecture Role | Target Seed | Checkpoint Path | Parameter Count | Matched | SHA-256 (prefix) |
|---|---|:---:|---|:---:|:---:|---|
| `baseline` | SingleFrameMLP | 42 | `results/checkpoints/baseline/best_seed42.pt` | 103,700 | [PASS] YES | `ebd13a05fc17...` |
| `baseline` | SingleFrameMLP | 43 | `results/checkpoints/baseline/best_seed43.pt` | 103,700 | [PASS] YES | `2b2245dc2fc2...` |
| `baseline` | SingleFrameMLP | 44 | `results/checkpoints/baseline/best_seed44.pt` | 103,700 | [PASS] YES | `398ba202e5dc...` |
| `meanpool` | MultiFrameMeanPool | 42 | `results/checkpoints/meanpool/best_seed42.pt` | 103,700 | [PASS] YES | `cecb7a14d3bc...` |
| `meanpool` | MultiFrameMeanPool | 43 | `results/checkpoints/meanpool/best_seed43.pt` | 103,700 | [PASS] YES | `eee28905e558...` |
| `meanpool` | MultiFrameMeanPool | 44 | `results/checkpoints/meanpool/best_seed44.pt` | 103,700 | [PASS] YES | `19f1f6796efd...` |
| `gru` | TemporalGRU | 42 | `results/checkpoints/gru/best_seed42.pt` | 249,364 | [PASS] YES | `a1fe3c5a3781...` |
| `gru` | TemporalGRU | 43 | `results/checkpoints/gru/best_seed43.pt` | 249,364 | [PASS] YES | `36878a33caac...` |
| `gru` | TemporalGRU | 44 | `results/checkpoints/gru/best_seed44.pt` | 249,364 | [PASS] YES | `1de7dd6f395c...` |

---

## Leakage Guard Certification

- **Train $\cap$ Validation Overlap**: 0 video IDs
- **Train $\cap$ Test Overlap**: 0 video IDs
- **Validation $\cap$ Test Overlap**: 0 video IDs
- **Certification Status**: `CERTIFIED ZERO LEAKAGE`

---

## Allowable Single-Variable Intervention Battery

| Milestone | Planned Intervention | Active Variable | Allowed Range / Values | Fixed Control Variables |
|---|---|---|---|---|
| **M4.2** | Progressive Context | `frame_count` | 1, 2, 3, 4, 6, 8 | order=canonical, corruption=none |
| **M4.3** | Order Ablation | `order` | canonical, shuffled, repeated | frame_count=8, corruption=none |
| **M4.4** | Sensor Robustness | `corruption` | zero_mask, gaussian_noise, mean_replace | frame_count=8, order=canonical, frac=0.25 |
| **M4.5** | Directional Semantics | `order` | canonical, reversed | frame_count=8, corruption=none |

> [!IMPORTANT]
> Under the Single-Variable Intervention Engine, any experimental evaluation attempting to change more than one variable at a time will immediately raise `ConfoundViolationError` and abort.