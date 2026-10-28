"""
Evaluation infrastructure package for Temporal Visual Intelligence.
Exports:
  - compute_metrics
  - aggregate_across_seeds
  - generate_per_class_table
  - analyze_directional_confusion
  - mcnemar_test
  - bootstrap_ci
  - compute_temporal_delta
  - disentangle_temporal_gains
  - plot_confusion_matrix
  - compute_win_loss_quadrants
  - validate_paired_test_alignment
  - compute_paired_metrics
  - build_paired_comparison_matrix
  - generate_paired_comparison_table
"""

from src.evaluation.metrics import (
    compute_metrics,
    aggregate_across_seeds,
    generate_per_class_table,
    analyze_directional_confusion,
    mcnemar_test,
    bootstrap_ci,
    compute_temporal_delta,
    disentangle_temporal_gains,
)
from src.evaluation.analysis import (
    compute_win_loss_quadrants,
    plot_confusion_matrix,
    plot_quadrant_matrix,
    audit_parameter_fairness,
)
from src.evaluation.paired import (
    validate_paired_test_alignment,
    compute_paired_metrics,
    build_paired_comparison_matrix,
    generate_paired_comparison_table,
)
from src.evaluation.output import (
    format_prediction_card,
    compile_evaluation_outputs,
    get_quadrant_candidates,
)
from src.evaluation.tables import (
    generate_first_baseline_table,
    compute_majority_baseline,
)
from src.evaluation.disentangle import (
    compute_three_model_disentanglement,
    classify_m3_20_outcome,
    generate_three_model_comparison_table,
)
from src.evaluation.taxonomy import (
    get_temporal_semantic_taxonomy,
    get_action_class_taxonomy_mapping,
    get_vlm_annotation_schema,
    export_temporal_taxonomy_deliverables,
    audit_taxonomy_integrity,
)
from src.evaluation.rescue_taxonomy_synthesis import (
    RescueTaxonomySynthesizer,
    audit_rescue_taxonomy_synthesis,
)
from src.evaluation.tri_model_comparison import (
    TriModelComparator,
    audit_tri_model_comparison,
)

__all__ = [
    "compute_metrics",
    "aggregate_across_seeds",
    "generate_per_class_table",
    "analyze_directional_confusion",
    "mcnemar_test",
    "bootstrap_ci",
    "compute_temporal_delta",
    "disentangle_temporal_gains",
    "compute_win_loss_quadrants",
    "plot_confusion_matrix",
    "plot_quadrant_matrix",
    "audit_parameter_fairness",
    "validate_paired_test_alignment",
    "compute_paired_metrics",
    "build_paired_comparison_matrix",
    "generate_paired_comparison_table",
    "format_prediction_card",
    "compile_evaluation_outputs",
    "get_quadrant_candidates",
    "generate_first_baseline_table",
    "compute_majority_baseline",
    "compute_three_model_disentanglement",
    "classify_m3_20_outcome",
    "generate_three_model_comparison_table",
    "get_temporal_semantic_taxonomy",
    "get_action_class_taxonomy_mapping",
    "get_vlm_annotation_schema",
    "export_temporal_taxonomy_deliverables",
    "audit_taxonomy_integrity",
    "RescueTaxonomySynthesizer",
    "audit_rescue_taxonomy_synthesis",
    "TriModelComparator",
    "audit_tri_model_comparison",
]
