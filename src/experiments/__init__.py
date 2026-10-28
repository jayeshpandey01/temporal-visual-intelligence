"""
Controlled experiments module for Milestone M4.
Enforces the core research rule: Change exactly one experimental variable at a time.
"""

from src.experiments.controls import (
    ConfoundViolationError,
    ExperimentalCondition,
    FrozenControlsSpec,
    SingleVariableIntervention,
    verify_controls_integrity,
)
from src.experiments.tracking import (
    setup_experiment_tracking_structure,
    build_master_experiment_table,
    build_hypothesis_mapping_table,
    build_falsification_audit_table,
    build_confounds_audit_table,
    audit_experiment_tracking_integrity,
)
from src.experiments.synthesis import (
    export_failure_analysis_partitions,
    generate_comprehensive_experiment_analysis_doc,
    audit_m4_final_acceptance,
)
from src.experiments.freeze import (
    build_m4_frozen_checkpoint,
    verify_m4_frozen_checkpoint,
    load_m4_frozen_checkpoint,
)
from src.experiments.cohorts import (
    construct_five_analysis_cohorts,
    audit_analysis_cohorts,
)
from src.experiments.temporal_ablation import (
    TemporalAblationOperator,
    TemporalAblationEvaluator,
    audit_temporal_ablation,
)
from src.experiments.minimum_context import (
    MinimumTemporalContextEvaluator,
    audit_minimum_temporal_context,
)
from src.experiments.information_formalization import (
    InformationFormalizationEvaluator,
    audit_information_formalization,
)

__all__ = [
    "ConfoundViolationError",
    "ExperimentalCondition",
    "FrozenControlsSpec",
    "SingleVariableIntervention",
    "verify_controls_integrity",
    "setup_experiment_tracking_structure",
    "build_master_experiment_table",
    "build_hypothesis_mapping_table",
    "build_falsification_audit_table",
    "build_confounds_audit_table",
    "audit_experiment_tracking_integrity",
    "export_failure_analysis_partitions",
    "generate_comprehensive_experiment_analysis_doc",
    "audit_m4_final_acceptance",
    "build_m4_frozen_checkpoint",
    "verify_m4_frozen_checkpoint",
    "load_m4_frozen_checkpoint",
    "construct_five_analysis_cohorts",
    "audit_analysis_cohorts",
    "TemporalAblationOperator",
    "TemporalAblationEvaluator",
    "audit_temporal_ablation",
    "MinimumTemporalContextEvaluator",
    "audit_minimum_temporal_context",
    "InformationFormalizationEvaluator",
    "audit_information_formalization",
]
