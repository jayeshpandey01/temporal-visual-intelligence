"""
Milestone M5.3: Temporal Semantic Taxonomy Definition & Formalization.

Defines the 8 core temporal semantic categories (T1-T8) formalized in M5.3
prior to downstream VLM probing (M5.4) and human verification (M5.5).

Taxonomy Categories:
  - T1: Object Displacement (position(t1) -> position(t2))
  - T2: Direction of Motion (A -> B vs. B -> A; reversal sensitive)
  - T3: State Transition (state A -> state B; e.g. open -> closed)
  - T4: Object Interaction (Agent + Object -> interaction -> changed state)
  - T5: Relative Motion (Object A moving relative to Object B)
  - T6: Temporal Ordering of Sub-events (Event A -> Event B vs. B -> A)
  - T7: Persistent / Static Appearance (Negative Control; order invariant)
  - T8: Ambiguous Individual Frames (Frames ambiguous in isolation; sequence disambiguates)
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import pandas as pd


TEMPORAL_TAXONOMY_REGISTRY: List[Dict[str, Any]] = [
    {
        "taxonomy_id": "T1",
        "category_name": "Object Displacement",
        "formal_definition": "position(t1) -> position(t2)",
        "description": "Visible translational or rotational shift of an object across successive frames.",
        "reversal_sensitive": False,
        "order_critical": False,
        "primary_cohort": "Cohort C / Cohort E",
        "diagnostic_relevance": "Differentiates static spatial position from continuous spatial movement.",
        "example_classes": "Pushing something from left to right; Moving something away from the camera",
    },
    {
        "taxonomy_id": "T2",
        "category_name": "Direction of Motion",
        "formal_definition": "Vector(A -> B) != Vector(B -> A)",
        "description": "Trajectory where temporal direction determines action identity; time inversion reverses semantic label.",
        "reversal_sensitive": True,
        "order_critical": True,
        "primary_cohort": "Cohort E / Cohort C",
        "diagnostic_relevance": "Directly tested by temporal reversal intervention (42.7% prediction flip rate).",
        "example_classes": "Pushing something from left to right vs. right to left; Moving something towards vs. away",
    },
    {
        "taxonomy_id": "T3",
        "category_name": "State Transition",
        "formal_definition": "state(t1) = S_A, state(t2) = S_B with S_A != S_B",
        "description": "Discrete or qualitative physical transition between early and late observations.",
        "reversal_sensitive": True,
        "order_critical": True,
        "primary_cohort": "Cohort C (Rescues)",
        "diagnostic_relevance": "Single frame captures only one state; requires temporal transition to identify action.",
        "example_classes": "Opening something; Uncovering something; Tearing something into two pieces",
    },
    {
        "taxonomy_id": "T4",
        "category_name": "Object Interaction",
        "formal_definition": "Agent + Object_1 + Object_2 -> Dynamic Contact -> Boundary Change",
        "description": "Physical contact or dynamic manipulation between an agent and one or more objects.",
        "reversal_sensitive": False,
        "order_critical": True,
        "primary_cohort": "Cohort C",
        "diagnostic_relevance": "Mean pooling averages object co-occurrences without establishing dynamic causality.",
        "example_classes": "Putting something onto something; Taking something out of something",
    },
    {
        "taxonomy_id": "T5",
        "category_name": "Relative Motion",
        "formal_definition": "d/dt (pos(Obj_A) - pos(Obj_B)) != 0",
        "description": "Motion of one entity defined specifically relative to a static or dynamic background object.",
        "reversal_sensitive": True,
        "order_critical": True,
        "primary_cohort": "Cohort E",
        "diagnostic_relevance": "Camera ego-motion vs. physical object displacement disambiguation.",
        "example_classes": "Passing something behind something; Moving something closer to something",
    },
    {
        "taxonomy_id": "T6",
        "category_name": "Temporal Ordering of Sub-events",
        "formal_definition": "Event_A precedes Event_B vs. Event_B precedes Event_A",
        "description": "Multi-phase action requiring chronological sequencing of discrete constituent sub-events.",
        "reversal_sensitive": True,
        "order_critical": True,
        "primary_cohort": "Cohort E / Cohort C",
        "diagnostic_relevance": "Shuffle intervention completely destroys prediction accuracy (13.89 pp collapse).",
        "example_classes": "Pretending to pick something up; Trying but failing to do something",
    },
    {
        "taxonomy_id": "T7",
        "category_name": "Persistent / Static Appearance",
        "formal_definition": "state(t) ~ constant, Delta(E_t, E_{t+1}) ~ 0",
        "description": "Minimal or zero physical transformation across the temporal horizon; static scene.",
        "reversal_sensitive": False,
        "order_critical": False,
        "primary_cohort": "Cohort A (Negative Control)",
        "diagnostic_relevance": "Negative control: single frame is fully sufficient; temporal recurrence yields zero gain.",
        "example_classes": "Holding something; Showing something to the camera",
    },
    {
        "taxonomy_id": "T8",
        "category_name": "Ambiguous Individual Frames",
        "formal_definition": "H(Y | F_t) >> H(Y | F_1...F_K)",
        "description": "Individual frames yield uninformative or misleading representations, disambiguated by temporal sequence.",
        "reversal_sensitive": False,
        "order_critical": True,
        "primary_cohort": "Cohort C (Rescues)",
        "diagnostic_relevance": "Direct mathematical explanation for SingleFrameMLP failure on high-entropy actions.",
        "example_classes": "Something colliding with something and bouncing off; Spilling something onto something",
    },
]


def export_temporal_taxonomy_csv(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Export the standardized 8-class temporal taxonomy to CSV deliverable.

    Args:
        output_path: Target path (defaults to results/m5/temporal_taxonomy.csv).

    Returns:
        DataFrame containing the taxonomy.
    """
    if output_path is None:
        root = Path(__file__).resolve().parent.parent.parent
        output_path = root / "results" / "m5" / "temporal_taxonomy.csv"
    else:
        output_path = Path(output_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(TEMPORAL_TAXONOMY_REGISTRY)
    df.to_csv(output_path, index=False)
    return df


def get_vlm_taxonomy_schema() -> Dict[str, Any]:
    """
    Generate strict JSON schema for downstream VLM sequence annotation (M5.4).
    """
    return {
        "type": "object",
        "properties": {
            "sequence_id": {"type": "string"},
            "observed_categories": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [cat["taxonomy_id"] for cat in TEMPORAL_TAXONOMY_REGISTRY],
                },
                "description": "List of temporal semantic phenomena detected (T1-T8)",
            },
            "primary_temporal_cue": {
                "type": "string",
                "enum": [cat["taxonomy_id"] for cat in TEMPORAL_TAXONOMY_REGISTRY],
                "description": "The single most discriminative temporal phenomenon",
            },
            "critical_frame_window": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": 8},
                "description": "Frames containing the decisive transition (e.g. [3, 4, 5])",
            },
            "directional_arrow_of_time": {
                "type": "boolean",
                "description": "True if reversing the frame sequence inverts or invalidates the event",
            },
            "individual_frame_ambiguity": {
                "type": "boolean",
                "description": "True if Frame 1 or mid-frames in isolation are insufficient to classify",
            },
            "semantic_state_transition": {
                "type": "object",
                "properties": {
                    "initial_state": {"type": "string"},
                    "final_state": {"type": "string"},
                },
                "required": ["initial_state", "final_state"],
            },
            "vlm_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "VLM confidence score on sequence understanding",
            },
            "brief_description": {
                "type": "string",
                "description": "Concise 1-2 sentence description of the physical event progression",
            },
        },
        "required": [
            "sequence_id",
            "observed_categories",
            "primary_temporal_cue",
            "critical_frame_window",
            "directional_arrow_of_time",
            "individual_frame_ambiguity",
            "vlm_confidence",
            "brief_description",
        ],
    }


if __name__ == "__main__":
    out_df = export_temporal_taxonomy_csv()
    print("=" * 80)
    print("Milestone M5.3: Temporal Semantic Taxonomy Registry Exported")
    print("=" * 80)
    for _, row in out_df.iterrows():
        print(f"  [{row['taxonomy_id']}] {row['category_name']:<35} | Reversal: {str(row['reversal_sensitive']):<5} | Cohort: {row['primary_cohort']}")
    print("-" * 80)
    print(f"Exported to: results/m5/temporal_taxonomy.csv ({len(out_df)} categories)")
    print("=" * 80)

