"""
Milestone M5.3: Temporal Semantic Taxonomy Specification & Schema Engine.

Formalizes the 8 candidate temporal semantic phenomena (T1–T8) prior to VLM-assisted
sequence probing and human audit:
  - T1: Object Displacement (spatial translation)
  - T2: Direction of Motion / Arrow of Time (directional asymmetry)
  - T3: State Transition (physical change of state)
  - T4: Object Interaction (contact and manipulation)
  - T5: Relative Motion (coordinate delta relative to reference frame)
  - T6: Sub-Event Temporal Ordering (multi-phase event sequencing)
  - T7: Persistent / Static Appearance (negative control)
  - T8: Ambiguous Individual Frames (single-frame disambiguation)

Rule:
    Define candidate categories and operational criteria BEFORE introducing VLM probing.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import os
import sys
import json
import pandas as pd


def get_temporal_semantic_taxonomy() -> Dict[str, Dict[str, Any]]:
    """
    Return the formal specifications, operational criteria, and kinematic signatures
    for all 8 temporal semantic phenomena (T1 to T8).
    """
    return {
        "T1_displacement": {
            "id": "T1",
            "name": "Object Displacement",
            "vlm_field": "displacement",
            "symbolic_condition": "p(t1) -> p(t2) with ||Delta p|| > tau",
            "operational_definition": (
                "An identifiable entity undergoes visible translation across 2D image coordinates "
                "between early and late observations without necessarily altering its internal physical state."
            ),
            "inclusion_criteria": (
                "Object centroid changes by >= 15% image width/height across sequence; "
                "trajectory is continuous and spatially trackable."
            ),
            "exclusion_criteria": (
                "Motion caused purely by camera pan without independent object movement (see T5)."
            ),
            "reversal_sensitive": "Moderate",
            "permutation_sensitive": "High",
            "kinematic_signature": "Steady monotonic cosine drift (drift_8 > 0.25), moderate inter-frame S_t in [0.80, 0.92].",
            "primary_action_classes": [
                "Pushing something from left to right",
                "Pushing something from right to left",
                "Moving something away from the camera",
                "Moving something towards the camera",
                "Picking something up",
                "Putting something down",
            ],
            "probe_question": "Does any object visibly change its spatial position or translate across the frame?",
            "research_role": "Differentiates spatial movement from stationary objects; tests basic visual motion tracking.",
        },
        "T2_directional_motion": {
            "id": "T2",
            "name": "Direction of Motion (Arrow of Time)",
            "vlm_field": "directional_motion",
            "symbolic_condition": "P(x_{1:T}) != P(x_{T:1})",
            "operational_definition": (
                "The semantic identity of the action depends strictly on the forward chronological "
                "direction of time. Reversing frame order inverts the action class into its reciprocal inverse."
            ),
            "inclusion_criteria": (
                "Inverting the frame sequence produces an opposite action label (e.g., towards -> away, "
                "left-to-right -> right-to-left, clockwise -> counter-clockwise)."
            ),
            "exclusion_criteria": (
                "Symmetric actions or static scenes where reversed playback looks natural or retains the same label."
            ),
            "reversal_sensitive": "Extreme (Direct inversion)",
            "permutation_sensitive": "High",
            "kinematic_signature": "Asymmetric dot product trajectories; reversal drop > 15 pp; flip rate > 30%.",
            "primary_action_classes": [
                "Moving something towards the camera",
                "Moving something away from the camera",
                "Pushing something from left to right",
                "Pushing something from right to left",
                "Rotating something clockwise",
                "Rotating something counter-clockwise",
                "Turning the camera left while filming something",
                "Turning the camera right while filming something",
            ],
            "probe_question": "Does reversing the order of frames alter or invert the meaning or direction of the action?",
            "research_role": "Core causal probe for chronological order sensitivity and arrow of time.",
        },
        "T3_state_transition": {
            "id": "T3",
            "name": "State Transition",
            "vlm_field": "state_transition",
            "symbolic_condition": "S(t) in S_init -> S(t') in S_final",
            "operational_definition": (
                "An entity undergoes a distinct physical, morphological, or configuration change "
                "between early frames and late frames, transitioning between mutually exclusive states."
            ),
            "inclusion_criteria": (
                "Observable transformation of binary or categorical state (e.g. open -> closed, "
                "standing upright -> fallen/tipped, covered -> uncovered, whole -> separated)."
            ),
            "exclusion_criteria": (
                "Continuous motion without any discrete boundary or structural state transition."
            ),
            "reversal_sensitive": "High (Inverts transition boundary)",
            "permutation_sensitive": "High",
            "kinematic_signature": "Sharp drop in consecutive cosine similarity S_t (< 0.75) localized at critical transition frame t_crit.",
            "primary_action_classes": [
                "Opening something",
                "Closing something",
                "Covering something",
                "Uncovering something",
                "Standing something upright",
                "Tipping something over",
            ],
            "probe_question": "Does an object or scene undergo a discrete physical state transition (e.g., open/closed, upright/tipped)?",
            "research_role": "Explains major temporal rescues (86.7% rescue in state transition classes).",
        },
        "T4_object_interaction": {
            "id": "T4",
            "name": "Object Interaction",
            "vlm_field": "object_interaction",
            "symbolic_condition": "Actor + Target -> Contact -> Delta Relationship",
            "operational_definition": (
                "Intentional physical manipulation involving contact, containment, detachment, or placement "
                "between a human agent (hand/effector) and one or more target objects."
            ),
            "inclusion_criteria": (
                "Hand or tool makes contact with target; object enters or leaves containment or support surface."
            ),
            "exclusion_criteria": (
                "Inanimate motion without agent interaction, or passive camera movement."
            ),
            "reversal_sensitive": "Moderate to High",
            "permutation_sensitive": "High",
            "kinematic_signature": "Trajectory inflection point when contact occurs; localized representation variance spike.",
            "primary_action_classes": [
                "Putting something into something",
                "Taking something out of something",
                "Putting something on top of something",
                "Taking something off of something",
                "Picking something up",
                "Putting something down",
            ],
            "probe_question": "Is there a physical interaction between a person (or hand) and an object (e.g., grasping, placing, extracting)?",
            "research_role": "Primary driver of real-world egocentric and manipulation action understanding.",
        },
        "T5_relative_motion": {
            "id": "T5",
            "name": "Relative Motion",
            "vlm_field": "relative_motion",
            "symbolic_condition": "p_A(t) - p_B(t) != const",
            "operational_definition": (
                "Motion where the position of a target object changes relative to a secondary reference "
                "entity (container, secondary object, or moving camera coordinate frame)."
            ),
            "inclusion_criteria": (
                "Relative spatial displacement between two separate visual landmarks; or parallax from camera movement."
            ),
            "exclusion_criteria": (
                "Rigid body translation of entire scene where all entities move in lockstep."
            ),
            "reversal_sensitive": "Moderate",
            "permutation_sensitive": "High",
            "kinematic_signature": "Differential embedding drift between foreground and background patches.",
            "primary_action_classes": [
                "Turning the camera left while filming something",
                "Turning the camera right while filming something",
                "Putting something into something",
                "Taking something out of something",
                "Putting something on top of something",
                "Taking something off of something",
            ],
            "probe_question": "Does one object or entity move relative to another object or relative to the camera frame?",
            "research_role": "Disentangles camera egomotion from object manipulation.",
        },
        "T6_event_order": {
            "id": "T6",
            "name": "Sub-Event Temporal Ordering",
            "vlm_field": "event_order",
            "symbolic_condition": "Event_A prec Event_B != Event_B prec Event_A",
            "operational_definition": (
                "A structured action composed of multiple sequential sub-phases that must occur in a "
                "fixed chronological order to constitute the action category."
            ),
            "inclusion_criteria": (
                "Presence of distinct multi-step sub-phases (e.g. approach -> grasp -> extract; "
                "align -> release -> settle)."
            ),
            "exclusion_criteria": (
                "Monolithic single-phase motions that cannot be decomposed into sequential stages."
            ),
            "reversal_sensitive": "Extreme",
            "permutation_sensitive": "Extreme (Order collapse under shuffle)",
            "kinematic_signature": "Multi-phase trajectory with multiple critical frame boundaries; massive drop under shuffle.",
            "primary_action_classes": [
                "Taking something out of something",
                "Putting something into something",
                "Taking something off of something",
                "Putting something on top of something",
                "Picking something up",
            ],
            "probe_question": "Does the action involve multiple sequential steps where the exact order of steps is essential?",
            "research_role": "Explains why temporal GRU vastly outperforms single frames and mean pooling on complex interactions.",
        },
        "T7_static_appearance": {
            "id": "T7",
            "name": "Persistent / Static Appearance (Negative Control)",
            "vlm_field": "static_appearance",
            "symbolic_condition": "max_t (1 - cos(E_0, E_t)) < epsilon",
            "operational_definition": (
                "Visual sequence where visual content is nearly static across frames; minimal or zero "
                "motion, interaction, or state change occurs. Serves as a vital negative control."
            ),
            "inclusion_criteria": (
                "No significant motion; cumulative cosine drift across 8 frames is < 0.10; frames are near duplicates."
            ),
            "exclusion_criteria": (
                "Any sequence with visible movement, state change, or directional displacement."
            ),
            "reversal_sensitive": "None (Order-invariant)",
            "permutation_sensitive": "None (Order-invariant)",
            "kinematic_signature": "Consecutive cosine similarity S_t > 0.95 across all t; cumulative drift < 0.08.",
            "primary_action_classes": [
                "Negative control across all classes (held static pauses, stationary pre-action frames)",
            ],
            "probe_question": "Is the sequence largely static or unchanging across all frames with minimal visible action?",
            "research_role": "Scientific negative control; verifies that temporal models do not hallucinate temporal benefit on static scenes.",
        },
        "T8_frame_ambiguity": {
            "id": "T8",
            "name": "Ambiguous Individual Frames (Disambiguation)",
            "vlm_field": "frame_ambiguity",
            "symbolic_condition": "H(Y | X_t) >> H(Y | X_{1:T})",
            "operational_definition": (
                "Individual isolated frames do not contain sufficient visual evidence to discriminate the "
                "action label from reciprocal classes, but chronological sequence trajectory resolves the ambiguity."
            ),
            "inclusion_criteria": (
                "A single snapshot could equally be opening or closing, pushing left or pushing right; "
                "temporal direction is required for disambiguation."
            ),
            "exclusion_criteria": (
                "Sequences where an isolated frame contains static artifacts that fully reveal the label."
            ),
            "reversal_sensitive": "High",
            "permutation_sensitive": "High",
            "kinematic_signature": "Single-frame prediction entropy is high (> 2.0); GRU confidence jumps > +0.30.",
            "primary_action_classes": [
                "Opening something",
                "Closing something",
                "Moving something away from the camera",
                "Moving something towards the camera",
                "Pushing something from left to right",
                "Pushing something from right to left",
                "Covering something",
                "Uncovering something",
                "Rotating something clockwise",
                "Rotating something counter-clockwise",
            ],
            "probe_question": "Would a single frozen snapshot from this sequence be ambiguous between two opposite actions?",
            "research_role": "Directly explains why Model A (Single) = 8.78% fails while Model C (GRU) = 34.78% succeeds.",
        },
    }


def get_action_class_taxonomy_mapping() -> pd.DataFrame:
    """
    Return the pre-registered prior hypothesis mapping connecting each of the 20 benchmark
    action classes to its primary, secondary, and reciprocal categories.
    """
    records = [
        {
            "class_idx": 0,
            "class_name": "Closing something",
            "primary_phenomenon": "T3_state_transition",
            "secondary_phenomenon": "T2_directional_motion",
            "reciprocal_class": "Opening something",
            "reciprocal_idx": 4,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Transition from open to closed state; reversed video resembles opening.",
        },
        {
            "class_idx": 1,
            "class_name": "Covering something",
            "primary_phenomenon": "T3_state_transition",
            "secondary_phenomenon": "T4_object_interaction",
            "reciprocal_class": "Uncovering something",
            "reciprocal_idx": 19,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Object occlusion boundary created; reversed video resembles uncovering.",
        },
        {
            "class_idx": 2,
            "class_name": "Moving something away from the camera",
            "primary_phenomenon": "T2_directional_motion",
            "secondary_phenomenon": "T1_displacement",
            "reciprocal_class": "Moving something towards the camera",
            "reciprocal_idx": 3,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Apparent scale shrinkage; inverts to towards camera under reversal.",
        },
        {
            "class_idx": 3,
            "class_name": "Moving something towards the camera",
            "primary_phenomenon": "T2_directional_motion",
            "secondary_phenomenon": "T1_displacement",
            "reciprocal_class": "Moving something away from the camera",
            "reciprocal_idx": 2,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Apparent scale growth; inverts to away from camera under reversal.",
        },
        {
            "class_idx": 4,
            "class_name": "Opening something",
            "primary_phenomenon": "T3_state_transition",
            "secondary_phenomenon": "T2_directional_motion",
            "reciprocal_class": "Closing something",
            "reciprocal_idx": 0,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Transition from closed to open state; reversed video resembles closing.",
        },
        {
            "class_idx": 5,
            "class_name": "Picking something up",
            "primary_phenomenon": "T1_displacement",
            "secondary_phenomenon": "T4_object_interaction",
            "reciprocal_class": "Putting something down",
            "reciprocal_idx": 8,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "Moderate",
            "hypothesis_notes": "Upward vertical displacement following grasp; inverts to putting down.",
        },
        {
            "class_idx": 6,
            "class_name": "Pushing something from left to right",
            "primary_phenomenon": "T1_displacement",
            "secondary_phenomenon": "T2_directional_motion",
            "reciprocal_class": "Pushing something from right to left",
            "reciprocal_idx": 7,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Horizontal displacement vector dx > 0; inverts to right-to-left under reversal.",
        },
        {
            "class_idx": 7,
            "class_name": "Pushing something from right to left",
            "primary_phenomenon": "T1_displacement",
            "secondary_phenomenon": "T2_directional_motion",
            "reciprocal_class": "Pushing something from left to right",
            "reciprocal_idx": 6,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Horizontal displacement vector dx < 0; inverts to left-to-right under reversal.",
        },
        {
            "class_idx": 8,
            "class_name": "Putting something down",
            "primary_phenomenon": "T1_displacement",
            "secondary_phenomenon": "T4_object_interaction",
            "reciprocal_class": "Picking something up",
            "reciprocal_idx": 5,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "Moderate",
            "hypothesis_notes": "Downward vertical displacement terminating in contact with surface.",
        },
        {
            "class_idx": 9,
            "class_name": "Putting something into something",
            "primary_phenomenon": "T4_object_interaction",
            "secondary_phenomenon": "T5_relative_motion",
            "reciprocal_class": "Taking something out of something",
            "reciprocal_idx": 15,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Object containment boundary crossed; inverts to taking out.",
        },
        {
            "class_idx": 10,
            "class_name": "Putting something on top of something",
            "primary_phenomenon": "T4_object_interaction",
            "secondary_phenomenon": "T5_relative_motion",
            "reciprocal_class": "Taking something off of something",
            "reciprocal_idx": 14,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Object support boundary established; inverts to taking off.",
        },
        {
            "class_idx": 11,
            "class_name": "Rotating something clockwise",
            "primary_phenomenon": "T2_directional_motion",
            "secondary_phenomenon": "T1_displacement",
            "reciprocal_class": "Rotating something counter-clockwise",
            "reciprocal_idx": 12,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Angular velocity dtheta > 0; inverts to counter-clockwise under reversal.",
        },
        {
            "class_idx": 12,
            "class_name": "Rotating something counter-clockwise",
            "primary_phenomenon": "T2_directional_motion",
            "secondary_phenomenon": "T1_displacement",
            "reciprocal_class": "Rotating something clockwise",
            "reciprocal_idx": 11,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Angular velocity dtheta < 0; inverts to clockwise under reversal.",
        },
        {
            "class_idx": 13,
            "class_name": "Standing something upright",
            "primary_phenomenon": "T3_state_transition",
            "secondary_phenomenon": "T4_object_interaction",
            "reciprocal_class": "Tipping something over",
            "reciprocal_idx": 16,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "Moderate",
            "hypothesis_notes": "Orientation shift from horizontal to vertical stability; inverts to tipping over.",
        },
        {
            "class_idx": 14,
            "class_name": "Taking something off of something",
            "primary_phenomenon": "T4_object_interaction",
            "secondary_phenomenon": "T6_event_order",
            "reciprocal_class": "Putting something on top of something",
            "reciprocal_idx": 10,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Support boundary severed; inverts to putting on top.",
        },
        {
            "class_idx": 15,
            "class_name": "Taking something out of something",
            "primary_phenomenon": "T4_object_interaction",
            "secondary_phenomenon": "T6_event_order",
            "reciprocal_class": "Putting something into something",
            "reciprocal_idx": 9,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Containment boundary exited; inverts to putting into.",
        },
        {
            "class_idx": 16,
            "class_name": "Tipping something over",
            "primary_phenomenon": "T3_state_transition",
            "secondary_phenomenon": "T1_displacement",
            "reciprocal_class": "Standing something upright",
            "reciprocal_idx": 13,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "Moderate",
            "hypothesis_notes": "Loss of vertical stability terminating in prone state; inverts to standing upright.",
        },
        {
            "class_idx": 17,
            "class_name": "Turning the camera left while filming something",
            "primary_phenomenon": "T5_relative_motion",
            "secondary_phenomenon": "T2_directional_motion",
            "reciprocal_class": "Turning the camera right while filming something",
            "reciprocal_idx": 18,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Camera egomotion induced rightward optical flow; inverts to turning right.",
        },
        {
            "class_idx": 18,
            "class_name": "Turning the camera right while filming something",
            "primary_phenomenon": "T5_relative_motion",
            "secondary_phenomenon": "T2_directional_motion",
            "reciprocal_class": "Turning the camera left while filming something",
            "reciprocal_idx": 17,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Camera egomotion induced leftward optical flow; inverts to turning left.",
        },
        {
            "class_idx": 19,
            "class_name": "Uncovering something",
            "primary_phenomenon": "T3_state_transition",
            "secondary_phenomenon": "T4_object_interaction",
            "reciprocal_class": "Covering something",
            "reciprocal_idx": 1,
            "expected_reversal_inversion": True,
            "expected_frame_ambiguity": "High",
            "hypothesis_notes": "Occlusion boundary removed, revealing hidden entity; inverts to covering.",
        },
    ]
    return pd.DataFrame(records)


def get_vlm_annotation_schema() -> Dict[str, Any]:
    """
    Return the formal JSON Schema governing Milestone M5.4 VLM structured outputs.
    Ensures 100% adherence to Draft-07 specification.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "TemporalSemanticAnnotation",
        "description": "Structured VLM semantic annotation for multi-frame action sequences (Milestone M5.4)",
        "type": "object",
        "properties": {
            "displacement": {
                "type": "boolean",
                "description": "True if an object visibly translates across image coordinates (T1)."
            },
            "directional_motion": {
                "type": "boolean",
                "description": "True if action identity depends on chronological arrow of time (T2)."
            },
            "state_transition": {
                "type": "boolean",
                "description": "True if an object undergoes discrete physical state transition (T3)."
            },
            "object_interaction": {
                "type": "boolean",
                "description": "True if actor makes physical contact or manipulates object (T4)."
            },
            "relative_motion": {
                "type": "boolean",
                "description": "True if object moves relative to another object or camera frame (T5)."
            },
            "event_order": {
                "type": "boolean",
                "description": "True if action consists of multi-step sequential sub-events (T6)."
            },
            "static_appearance": {
                "type": "boolean",
                "description": "True if sequence contains minimal to zero visible motion (Negative Control T7)."
            },
            "frame_ambiguity": {
                "type": "boolean",
                "description": "True if individual isolated frames are visually ambiguous without sequence (T8)."
            },
            "frame_criticality": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 7},
                "description": "0-indexed frame numbers (out of 8) containing the most discriminative transition."
            },
            "temporal_dependency": {
                "type": "string",
                "enum": ["none", "low", "medium", "high"],
                "description": "Degree to which chronological sequence is necessary to classify action."
            },
            "observed_action_description": {
                "type": "string",
                "description": "Concise 1-2 sentence description of the primary visual action observed."
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Model self-reported confidence score in the annotation."
            }
        },
        "required": [
            "displacement",
            "directional_motion",
            "state_transition",
            "object_interaction",
            "relative_motion",
            "event_order",
            "static_appearance",
            "frame_ambiguity",
            "frame_criticality",
            "temporal_dependency"
        ],
        "additionalProperties": False
    }


def export_temporal_taxonomy_deliverables(
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """
    Export all Milestone M5.3 taxonomy deliverables:
      - results/m5/temporal_taxonomy.csv (M5.14 deliverable)
      - results/m5/temporal_taxonomy.md
      - results/m5/class_taxonomy_mapping.csv
      - results/m5/vlm_annotation_schema.json
    """
    if output_dir is None:
        root = Path(__file__).resolve().parent.parent.parent
        out_dir = root / "results" / "m5"
    else:
        out_dir = Path(output_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = get_temporal_semantic_taxonomy()
    df_mapping = get_action_class_taxonomy_mapping()
    schema = get_vlm_annotation_schema()

    # 1. Build temporal_taxonomy.csv
    tax_rows = []
    for cat_id, cat_info in taxonomy.items():
        tax_rows.append({
            "category_id": cat_info["id"],
            "category_key": cat_id,
            "category_name": cat_info["name"],
            "vlm_field": cat_info["vlm_field"],
            "symbolic_condition": cat_info["symbolic_condition"],
            "operational_definition": cat_info["operational_definition"],
            "reversal_sensitive": cat_info["reversal_sensitive"],
            "permutation_sensitive": cat_info["permutation_sensitive"],
            "kinematic_signature": cat_info["kinematic_signature"],
            "research_role": cat_info["research_role"],
            "primary_action_classes": "; ".join(cat_info["primary_action_classes"]),
        })

    df_tax = pd.DataFrame(tax_rows)
    tax_csv_path = out_dir / "temporal_taxonomy.csv"
    df_tax.to_csv(tax_csv_path, index=False)

    # 2. Build class_taxonomy_mapping.csv
    map_csv_path = out_dir / "class_taxonomy_mapping.csv"
    df_mapping.to_csv(map_csv_path, index=False)

    # 3. Build vlm_annotation_schema.json
    schema_path = out_dir / "vlm_annotation_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    # 4. Build comprehensive temporal_taxonomy.md
    tax_md_path = out_dir / "temporal_taxonomy.md"
    md_content = _build_taxonomy_markdown_doc(taxonomy, df_mapping, schema)
    with open(tax_md_path, "w") as f:
        f.write(md_content)

    return {
        "temporal_taxonomy_csv": tax_csv_path,
        "temporal_taxonomy_md": tax_md_path,
        "class_taxonomy_mapping_csv": map_csv_path,
        "vlm_annotation_schema_json": schema_path,
    }


def _build_taxonomy_markdown_doc(
    taxonomy: Dict[str, Any],
    df_mapping: pd.DataFrame,
    schema: Dict[str, Any],
) -> str:
    """Format exhaustive Markdown design document for Milestone M5.3."""
    md = []
    md.append("# Milestone M5.3: Temporal Semantic Taxonomy Specification\n\n")
    md.append("**Pre-Registration Mandate:** This taxonomy is formally defined *before* running VLM-assisted sequence annotation or human audits.\n")
    md.append("**Causal Purpose:** To operationalize the exact visual and kinematic phenomena that cause temporal models to succeed when single-frame or order-invariant representations fail.\n\n")

    md.append("## 1. The Eight Temporal Semantic Phenomena ($T_1$ – $T_8$)\n\n")
    md.append("| ID | Phenomenon Name | Symbolic Condition | Reversal Sensitivity | Permutation Sensitivity | VLM JSON Field |\n")
    md.append("|:---:|---|---|:---:|:---:|---|\n")
    for _, cat in taxonomy.items():
        md.append(f"| **{cat['id']}** | {cat['name']} | `{cat['symbolic_condition']}` | {cat['reversal_sensitive']} | {cat['permutation_sensitive']} | `{cat['vlm_field']}` |\n")

    md.append("\n---\n\n")
    md.append("## 2. Detailed Phenomenon Profiles\n\n")
    for _, cat in taxonomy.items():
        md.append(f"### {cat['id']} — {cat['name']}\n\n")
        md.append(f"- **Formal Symbolic Condition:** `{cat['symbolic_condition']}`\n")
        md.append(f"- **Operational Definition:** {cat['operational_definition']}\n")
        md.append(f"- **Inclusion Criteria:** {cat['inclusion_criteria']}\n")
        md.append(f"- **Exclusion Criteria:** {cat['exclusion_criteria']}\n")
        md.append(f"- **Kinematic / Representation Signature:** {cat['kinematic_signature']}\n")
        md.append(f"- **Expected Reversal Sensitivity:** {cat['reversal_sensitive']}\n")
        md.append(f"- **Expected Permutation Sensitivity:** {cat['permutation_sensitive']}\n")
        md.append(f"- **Research Role:** {cat['research_role']}\n")
        md.append(f"- **VLM Probe Question:** *\"{cat['probe_question']}\"*\n")
        md.append(f"- **Primary Action Classes:**\n")
        for cls in cat['primary_action_classes']:
            md.append(f"  * `{cls}`\n")
        md.append("\n")

    md.append("---\n\n")
    md.append("## 3. Pre-Registered Action Class Mapping (20 Benchmark Classes)\n\n")
    md.append("| Class Idx | Action Class Name | Primary Phenomenon | Secondary Phenomenon | Reciprocal Pair | Reversal Inverts? | Frame Ambiguity |\n")
    md.append("|:---:|---|:---:|:---:|---|:---:|:---:|\n")
    for _, r in df_mapping.iterrows():
        md.append(f"| {r['class_idx']:2d} | `{r['class_name']}` | **{r['primary_phenomenon'][:2]}** | {r['secondary_phenomenon'][:2]} | `{r['reciprocal_class']}` | {r['expected_reversal_inversion']} | {r['expected_frame_ambiguity']} |\n")

    md.append("\n---\n\n")
    md.append("## 4. VLM Structured Output Schema Contract (Milestone M5.4)\n\n")
    md.append("To prevent unstructured narrative drift and hallucination, VLM calls in M5.4 will enforce this strict JSON schema:\n\n")
    md.append("```json\n")
    md.append(json.dumps(schema, indent=2))
    md.append("\n```\n\n")

    md.append("---\n*Generated automatically by `src.evaluation.taxonomy` under Milestone M5.3.*")
    return "".join(md)


def audit_taxonomy_integrity(deliverables: Dict[str, Path]) -> Dict[str, Any]:
    """
    Exhaustively audit the integrity and constraints of the exported taxonomy deliverables.

    Returns:
        Structured audit report dict with status 'CERTIFIED' or 'FAILED'.
    """
    checklist = {}
    failures = []

    # 1. Verify all 4 deliverable files exist and are non-empty
    files_ok = True
    for k, p in deliverables.items():
        if not p.exists() or p.stat().st_size == 0:
            files_ok = False
            failures.append(f"missing_or_empty_{k}")

    checklist["deliverables_exist"] = {
        "description": "All 4 taxonomy deliverable files exported and non-empty.",
        "passed": files_ok,
        "details": f"Verified {len(deliverables)} files in {deliverables['temporal_taxonomy_csv'].parent}",
    }

    # 2. Verify all 8 categories exist in temporal_taxonomy.csv
    df_tax = pd.read_csv(deliverables["temporal_taxonomy_csv"])
    cat_ids = sorted(df_tax["category_id"].unique())
    expected_ids = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    cats_ok = (cat_ids == expected_ids) and (len(df_tax) == 8)
    checklist["eight_phenomena_complete"] = {
        "description": "All 8 temporal semantic categories (T1–T8) populated.",
        "passed": cats_ok,
        "details": f"Found {len(df_tax)} categories: {cat_ids}",
    }
    if not cats_ok:
        failures.append("categories_mismatch")

    # 3. Verify negative control (T7) exists and is specified
    t7_row = df_tax[df_tax["category_id"] == "T7"]
    t7_ok = len(t7_row) == 1 and ("Negative Control" in t7_row.iloc[0]["category_name"])
    checklist["negative_control_defined"] = {
        "description": "Negative control category T7 (Persistent / Static Appearance) formally defined.",
        "passed": t7_ok,
        "details": f"T7 Name: {t7_row.iloc[0]['category_name'] if len(t7_row) else 'None'}",
    }
    if not t7_ok:
        failures.append("t7_negative_control_missing")

    # 4. Verify all 20 benchmark action classes mapped
    df_map = pd.read_csv(deliverables["class_taxonomy_mapping_csv"])
    classes_ok = (len(df_map) == 20) and (df_map["class_idx"].nunique() == 20)
    checklist["twenty_action_classes_mapped"] = {
        "description": "All 20 benchmark action classes mapped to taxonomy categories with zero unmapped classes.",
        "passed": classes_ok,
        "details": f"Mapped {len(df_map)}/20 classes",
    }
    if not classes_ok:
        failures.append("classes_mapping_incomplete")

    # 5. Verify VLM annotation JSON schema Draft-07 compliance
    with open(deliverables["vlm_annotation_schema_json"]) as f:
        schema = json.load(f)
    schema_ok = (
        schema.get("$schema") == "http://json-schema.org/draft-07/schema#"
        and "properties" in schema
        and len(schema["required"]) >= 8
        and "displacement" in schema["properties"]
        and "directional_motion" in schema["properties"]
        and "state_transition" in schema["properties"]
        and "static_appearance" in schema["properties"]
    )
    checklist["vlm_schema_compliant"] = {
        "description": "Strict JSON schema for Milestone M5.4 VLM annotations validated.",
        "passed": schema_ok,
        "details": f"Schema has {len(schema.get('properties', {}))} properties, {len(schema.get('required', []))} required fields.",
    }
    if not schema_ok:
        failures.append("vlm_schema_invalid")

    total_checks = len(checklist)
    passed_checks = sum(1 for c in checklist.values() if c["passed"])
    status = "CERTIFIED" if (passed_checks == total_checks) else "FAILED"

    return {
        "milestone": "M5.3_TEMPORAL_TAXONOMY",
        "status": status,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "compliance_rate": f"{(passed_checks / total_checks) * 100:.1f}%",
        "checklist": checklist,
        "failures": failures,
    }

