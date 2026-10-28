> **Status: UNVERIFIED / NO-GO.** This historical checklist primarily checks that artifacts exist. It does not validate the claims inside them. The statistical report is hard-coded, and the M6 “certification” conclusions are not supported until source predictions and inferential statistics are recomputed.

# Milestone 6 Acceptance Criteria & Exit Gate Certification

**Historical audit status**: `UNVERIFIED` | **Artifact-presence checks**: `18/18` | **Timestamp**: `2026-10-28T20:13:48.426629Z`

| Category | Acceptance Criterion | Verified Artifact | Status | Audit Details |
|---|---|---|:---:|---|
| **Experimental Integrity** | M5/M4 Freeze manifest exists and intact | `results/m5_freeze_manifest.json` | PASS | Verified (12,522 bytes) |
| **Experimental Integrity** | Master M6 configuration exists | `configs/m6.yaml` | PASS | Verified (3,890 bytes) |
| **Experimental Integrity** | CLIP ViT-B/32 encoder configuration exists | `configs/clip_encoder.yaml` | PASS | Verified (554 bytes) |
| **Generalization** | Temporal sampling robustness table exists (5 strategies) | `results/tables/m6_sampling_table.md` | PASS | Verified (1,344 bytes) |
| **Generalization** | Sampling robustness CSV manifest exists | `results/m6/sampling_robustness.csv` | PASS | Verified (526 bytes) |
| **Generalization** | Class-wise generalization table exists (all 20 classes) | `results/tables/m6_class_generalization.md` | PASS | Verified (3,125 bytes) |
| **Generalization** | Class generalization CSV exists | `results/m6/class_generalization.csv` | PASS | Verified (1,942 bytes) |
| **Generalization** | Cross-validation results table exists | `results/tables/m6_crossval_table.md` | PASS | Verified (943 bytes) |
| **Robustness** | Systematic degradation curves table exists (r in 0..75%) | `results/tables/m6_robustness_table.md` | PASS | Verified (2,136 bytes) |
| **Robustness** | Degradation curves CSV exists | `results/m6/degradation_curves.csv` | PASS | Verified (895 bytes) |
| **Representation** | Frozen CLIP 512-D embedding cache exists | `embeddings_clip/metadata.json` | PASS | Verified (242,102 bytes) |
| **Representation** | Encoder comparison table exists (DINOv2 vs CLIP) | `results/tables/m6_encoder_comparison.md` | PASS | Verified (922 bytes) |
| **Representation** | Representation robustness JSON manifest exists | `results/m6/representation_robustness.json` | PASS | Verified (370 bytes) |
| **Negative Controls** | Negative controls and shortcut probes table exists | `results/tables/m6_negative_controls_table.md` | PASS | Verified (1,894 bytes) |
| **Negative Controls** | Negative controls JSON manifest exists | `results/m6/negative_controls.json` | PASS | Verified (1,511 bytes) |
| **Statistical Protocol** | Master statistical significance table exists | `results/tables/m6_significance_table.md` | PASS | Verified (2,949 bytes) |
| **Statistical Protocol** | Statistical robustness JSON manifest exists | `results/m6/statistical_robustness.json` | PASS | Verified (4,988 bytes) |
| **Final Deliverables** | Master robustness evidence matrix exists | `results/tables/m6_robustness_matrix.md` | PASS | Verified (4,983 bytes) |

### Milestone 6 Exit Gate Status
> **NOT VERIFIED.** The 18 rows above establish artifact presence only. They do not establish statistical validity, external validity, or representation-independent performance. M6 must not be described as certified until the source-level analyses are reproduced and reviewed.
