# Milestone 7 Acceptance Checklist & Release Certification

**Release Status**: `NOT RELEASE-READY` | **Audit Status**: Historical checklist, superseded by source discrepancies found during release review

| Category | Release Criterion | Status | Audit Verification Details |
|---|---|:---:|---|
| **Integrity** | Final dataset and checkpoint manifests verified | HISTORICAL PASS | Historical hash checks exist; release data and permission review remains open |
| **Integrity** | M4–M6 results traceable to frozen configurations | PARTIAL | Configs/manifests exist; metric disagreements remain unresolved |
| **Integrity** | No unresolved split or leakage concerns (7/7 Audit) | HISTORICAL PASS | Stored integrity report says 7/7; this release review did not rerun the full audit |
| **Reproducibility** | Clean-environment reproduction succeeds | NOT VERIFIED | Diagnostic ran with local cached data/checkpoints, not a clean checkout |
| **Reproducibility** | Final tests pass across suite | NOT VERIFIED | Test suite not run during this review |
| **Reproducibility** | Headline metrics & statistical tests independently recomputed | BLOCKED | Stored headline, M4 summary, shuffle CSV, and statistical artifacts disagree; see m7_reconciliation_table.md |
| **Reproducibility** | Diagnostic checkpoint recomputation completed | PARTIAL | Current checkpoints reproduce canonical means; shuffle and horizon values differ from historical M4 claims |
| **Demo** | Canonical inference works interactively | NOT VERIFIED | Requires a current run from a clean checkout |
| **Demo** | Shuffle and reversal interventions work | NOT VERIFIED | Requires a current run from a clean checkout |
| **Demo** | Failure case and provenance are visible | PARTIAL | Demo code/docs exist; source predictions and claims require reconciliation |
| **Publication** | Report matches actual empirical results | BLOCKED | Claims require reconciliation; report is a provisional draft |
| **Publication** | VLM and annotation provenance accurately represented | BLOCKED | Saved annotations are offline heuristics; stored “human” labels are rule-generated |
| **Publication** | Setup instructions work from clean checkout & Presentation | NOT VERIFIED | Requires a clean-checkout run and current test result |

### Release review verdict
The historical certification above was not supported by the current cross-artifact evidence and is withdrawn. The repository is **not ready to push as a completed scientific release** until headline results are reconciled, annotation provenance is corrected, permitted data are selected, the PDF is produced, and clean-checkout verification is recorded. This checklist is a release status record, not a claim that all research requirements have passed.
