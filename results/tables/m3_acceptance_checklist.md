> **Status: UNVERIFIED HISTORICAL CHECKLIST.** This artifact records an earlier checklist outcome. Its `PASS` values are not current release verification; some linked metrics and statistical claims conflict with source prediction records.

# Milestone 3 Acceptance Criteria & Exit Gate Certification
**Audit Status:** `UNVERIFIED_HISTORICAL` | **Compliance Rate:** `100.0%` (20/20) | **Date:** `2026-10-28 20:13:46 UTC`

| Category | Acceptance Criterion | Status | Evidence / Verification Details |
|---|---|:---:|---|
| **Models** | Majority-class sanity baseline implemented. | HISTORICAL | Verified via compute_majority_baseline and MajorityClassBaseline |
| **Models** | Single-frame baseline implemented (Model A). | HISTORICAL | SingleFrameMLP parameter count: 103,700 |
| **Models** | Mean-pooling control implemented (Model B). | HISTORICAL | MultiFrameMeanPool parameter count: 103,700 (matches Model A) |
| **Models** | Temporal GRU implemented (Model C). | HISTORICAL | TemporalGRU parameter count: 249,364 (<2.5x baseline) |
| **Models** | Visual encoder remains frozen. | HISTORICAL | DINOv2 ViT-S/14 frozen flag in config: True |
| **Models** | Model parameter counts recorded. | HISTORICAL | Recorded in results/tables/parameter_fairness.md |
| **Training** | Same dataset split (certified zero leakage). | HISTORICAL | Train=1400, Val=300, Test=300 in data/metadata/split_manifest.json |
| **Training** | Same embedding pipeline (frozen DINOv2 ViT-S/14). | HISTORICAL | Pre-extracted embeddings cached in embeddings |
| **Training** | Validation-based model selection (test untouched). | HISTORICAL | EarlyStopping patience=5 restoring best validation checkpoint before test evaluation |
| **Training** | Reproducible seeds/configuration. | HISTORICAL | Deterministic seeds [42, 43, 44] logged to runs/ and config.yaml |
| **Training** | Best checkpoints saved. | HISTORICAL | Checkpoints verified in results/checkpoints/{baseline,meanpool,gru}/best_seed{42,43,44}.pt |
| **Evaluation** | Same test sequences evaluated (paired evaluation). | HISTORICAL | All models evaluated on identical N=300 sequences in results/paired_test_predictions.csv |
| **Evaluation** | Accuracy calculated. | HISTORICAL | Logged to results/experiments.csv and results/tables/first_baseline_table.md |
| **Evaluation** | Macro-F1 calculated. | HISTORICAL | Class-balanced Macro-F1 logged to results/experiments.csv |
| **Evaluation** | Per-class metrics calculated. | HISTORICAL | Generated per_class_metrics.md in 12 run directories |
| **Evaluation** | Confusion matrices generated. | HISTORICAL | Generated confusion_matrix.png heatmaps in 12 run directories |
| **Evaluation** | Per-sequence predictions saved. | HISTORICAL | Archived 300 sample predictions in results/evaluation_outputs/multi_model_predictions.csv |
| **Research** | Initial baseline comparison completed. | HISTORICAL | Table 1 and Three-Model Disentanglement tables generated |
| **Research** | Results interpreted objectively (M3.20 Diagnostic Framework). | HISTORICAL | Categorized against Outcomes A, B, C, D; Δ_obs (-0.39 pp) vs Δ_order (+31.38 pp) disentangled |
| **Research** | Failure cases preserved for M5. | HISTORICAL | Quadrant C/B candidate indices preserved in results/evaluation_outputs/failure_analysis_manifest.json |

### Milestone 3 Exit Gate Status
> **NOT VERIFIED FOR RELEASE.** The stored checklist is historical and does not establish the current scientific or release gates.
