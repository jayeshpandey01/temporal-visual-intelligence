# Milestone M5.3: Temporal Semantic Taxonomy Specification

**Pre-Registration Mandate:** This taxonomy is formally defined *before* running VLM-assisted sequence annotation or human audits.
**Causal Purpose:** To operationalize the exact visual and kinematic phenomena that cause temporal models to succeed when single-frame or order-invariant representations fail.

## 1. The Eight Temporal Semantic Phenomena ($T_1$ – $T_8$)

| ID | Phenomenon Name | Symbolic Condition | Reversal Sensitivity | Permutation Sensitivity | VLM JSON Field |
|:---:|---|---|:---:|:---:|---|
| **T1** | Object Displacement | `p(t1) -> p(t2) with ||Delta p|| > tau` | Moderate | High | `displacement` |
| **T2** | Direction of Motion (Arrow of Time) | `P(x_{1:T}) != P(x_{T:1})` | Extreme (Direct inversion) | High | `directional_motion` |
| **T3** | State Transition | `S(t) in S_init -> S(t') in S_final` | High (Inverts transition boundary) | High | `state_transition` |
| **T4** | Object Interaction | `Actor + Target -> Contact -> Delta Relationship` | Moderate to High | High | `object_interaction` |
| **T5** | Relative Motion | `p_A(t) - p_B(t) != const` | Moderate | High | `relative_motion` |
| **T6** | Sub-Event Temporal Ordering | `Event_A prec Event_B != Event_B prec Event_A` | Extreme | Extreme (Order collapse under shuffle) | `event_order` |
| **T7** | Persistent / Static Appearance (Negative Control) | `max_t (1 - cos(E_0, E_t)) < epsilon` | None (Order-invariant) | None (Order-invariant) | `static_appearance` |
| **T8** | Ambiguous Individual Frames (Disambiguation) | `H(Y | X_t) >> H(Y | X_{1:T})` | High | High | `frame_ambiguity` |

---

## 2. Detailed Phenomenon Profiles

### T1 — Object Displacement

- **Formal Symbolic Condition:** `p(t1) -> p(t2) with ||Delta p|| > tau`
- **Operational Definition:** An identifiable entity undergoes visible translation across 2D image coordinates between early and late observations without necessarily altering its internal physical state.
- **Inclusion Criteria:** Object centroid changes by >= 15% image width/height across sequence; trajectory is continuous and spatially trackable.
- **Exclusion Criteria:** Motion caused purely by camera pan without independent object movement (see T5).
- **Kinematic / Representation Signature:** Steady monotonic cosine drift (drift_8 > 0.25), moderate inter-frame S_t in [0.80, 0.92].
- **Expected Reversal Sensitivity:** Moderate
- **Expected Permutation Sensitivity:** High
- **Research Role:** Differentiates spatial movement from stationary objects; tests basic visual motion tracking.
- **VLM Probe Question:** *"Does any object visibly change its spatial position or translate across the frame?"*
- **Primary Action Classes:**
  * `Pushing something from left to right`
  * `Pushing something from right to left`
  * `Moving something away from the camera`
  * `Moving something towards the camera`
  * `Picking something up`
  * `Putting something down`

### T2 — Direction of Motion (Arrow of Time)

- **Formal Symbolic Condition:** `P(x_{1:T}) != P(x_{T:1})`
- **Operational Definition:** The semantic identity of the action depends strictly on the forward chronological direction of time. Reversing frame order inverts the action class into its reciprocal inverse.
- **Inclusion Criteria:** Inverting the frame sequence produces an opposite action label (e.g., towards -> away, left-to-right -> right-to-left, clockwise -> counter-clockwise).
- **Exclusion Criteria:** Symmetric actions or static scenes where reversed playback looks natural or retains the same label.
- **Kinematic / Representation Signature:** Asymmetric dot product trajectories; reversal drop > 15 pp; flip rate > 30%.
- **Expected Reversal Sensitivity:** Extreme (Direct inversion)
- **Expected Permutation Sensitivity:** High
- **Research Role:** Core causal probe for chronological order sensitivity and arrow of time.
- **VLM Probe Question:** *"Does reversing the order of frames alter or invert the meaning or direction of the action?"*
- **Primary Action Classes:**
  * `Moving something towards the camera`
  * `Moving something away from the camera`
  * `Pushing something from left to right`
  * `Pushing something from right to left`
  * `Rotating something clockwise`
  * `Rotating something counter-clockwise`
  * `Turning the camera left while filming something`
  * `Turning the camera right while filming something`

### T3 — State Transition

- **Formal Symbolic Condition:** `S(t) in S_init -> S(t') in S_final`
- **Operational Definition:** An entity undergoes a distinct physical, morphological, or configuration change between early frames and late frames, transitioning between mutually exclusive states.
- **Inclusion Criteria:** Observable transformation of binary or categorical state (e.g. open -> closed, standing upright -> fallen/tipped, covered -> uncovered, whole -> separated).
- **Exclusion Criteria:** Continuous motion without any discrete boundary or structural state transition.
- **Kinematic / Representation Signature:** Sharp drop in consecutive cosine similarity S_t (< 0.75) localized at critical transition frame t_crit.
- **Expected Reversal Sensitivity:** High (Inverts transition boundary)
- **Expected Permutation Sensitivity:** High
- **Research Role:** Explains major temporal rescues (86.7% rescue in state transition classes).
- **VLM Probe Question:** *"Does an object or scene undergo a discrete physical state transition (e.g., open/closed, upright/tipped)?"*
- **Primary Action Classes:**
  * `Opening something`
  * `Closing something`
  * `Covering something`
  * `Uncovering something`
  * `Standing something upright`
  * `Tipping something over`

### T4 — Object Interaction

- **Formal Symbolic Condition:** `Actor + Target -> Contact -> Delta Relationship`
- **Operational Definition:** Intentional physical manipulation involving contact, containment, detachment, or placement between a human agent (hand/effector) and one or more target objects.
- **Inclusion Criteria:** Hand or tool makes contact with target; object enters or leaves containment or support surface.
- **Exclusion Criteria:** Inanimate motion without agent interaction, or passive camera movement.
- **Kinematic / Representation Signature:** Trajectory inflection point when contact occurs; localized representation variance spike.
- **Expected Reversal Sensitivity:** Moderate to High
- **Expected Permutation Sensitivity:** High
- **Research Role:** Primary driver of real-world egocentric and manipulation action understanding.
- **VLM Probe Question:** *"Is there a physical interaction between a person (or hand) and an object (e.g., grasping, placing, extracting)?"*
- **Primary Action Classes:**
  * `Putting something into something`
  * `Taking something out of something`
  * `Putting something on top of something`
  * `Taking something off of something`
  * `Picking something up`
  * `Putting something down`

### T5 — Relative Motion

- **Formal Symbolic Condition:** `p_A(t) - p_B(t) != const`
- **Operational Definition:** Motion where the position of a target object changes relative to a secondary reference entity (container, secondary object, or moving camera coordinate frame).
- **Inclusion Criteria:** Relative spatial displacement between two separate visual landmarks; or parallax from camera movement.
- **Exclusion Criteria:** Rigid body translation of entire scene where all entities move in lockstep.
- **Kinematic / Representation Signature:** Differential embedding drift between foreground and background patches.
- **Expected Reversal Sensitivity:** Moderate
- **Expected Permutation Sensitivity:** High
- **Research Role:** Disentangles camera egomotion from object manipulation.
- **VLM Probe Question:** *"Does one object or entity move relative to another object or relative to the camera frame?"*
- **Primary Action Classes:**
  * `Turning the camera left while filming something`
  * `Turning the camera right while filming something`
  * `Putting something into something`
  * `Taking something out of something`
  * `Putting something on top of something`
  * `Taking something off of something`

### T6 — Sub-Event Temporal Ordering

- **Formal Symbolic Condition:** `Event_A prec Event_B != Event_B prec Event_A`
- **Operational Definition:** A structured action composed of multiple sequential sub-phases that must occur in a fixed chronological order to constitute the action category.
- **Inclusion Criteria:** Presence of distinct multi-step sub-phases (e.g. approach -> grasp -> extract; align -> release -> settle).
- **Exclusion Criteria:** Monolithic single-phase motions that cannot be decomposed into sequential stages.
- **Kinematic / Representation Signature:** Multi-phase trajectory with multiple critical frame boundaries; massive drop under shuffle.
- **Expected Reversal Sensitivity:** Extreme
- **Expected Permutation Sensitivity:** Extreme (Order collapse under shuffle)
- **Research Role:** Explains why temporal GRU vastly outperforms single frames and mean pooling on complex interactions.
- **VLM Probe Question:** *"Does the action involve multiple sequential steps where the exact order of steps is essential?"*
- **Primary Action Classes:**
  * `Taking something out of something`
  * `Putting something into something`
  * `Taking something off of something`
  * `Putting something on top of something`
  * `Picking something up`

### T7 — Persistent / Static Appearance (Negative Control)

- **Formal Symbolic Condition:** `max_t (1 - cos(E_0, E_t)) < epsilon`
- **Operational Definition:** Visual sequence where visual content is nearly static across frames; minimal or zero motion, interaction, or state change occurs. Serves as a vital negative control.
- **Inclusion Criteria:** No significant motion; cumulative cosine drift across 8 frames is < 0.10; frames are near duplicates.
- **Exclusion Criteria:** Any sequence with visible movement, state change, or directional displacement.
- **Kinematic / Representation Signature:** Consecutive cosine similarity S_t > 0.95 across all t; cumulative drift < 0.08.
- **Expected Reversal Sensitivity:** None (Order-invariant)
- **Expected Permutation Sensitivity:** None (Order-invariant)
- **Research Role:** Scientific negative control; verifies that temporal models do not hallucinate temporal benefit on static scenes.
- **VLM Probe Question:** *"Is the sequence largely static or unchanging across all frames with minimal visible action?"*
- **Primary Action Classes:**
  * `Negative control across all classes (held static pauses, stationary pre-action frames)`

### T8 — Ambiguous Individual Frames (Disambiguation)

- **Formal Symbolic Condition:** `H(Y | X_t) >> H(Y | X_{1:T})`
- **Operational Definition:** Individual isolated frames do not contain sufficient visual evidence to discriminate the action label from reciprocal classes, but chronological sequence trajectory resolves the ambiguity.
- **Inclusion Criteria:** A single snapshot could equally be opening or closing, pushing left or pushing right; temporal direction is required for disambiguation.
- **Exclusion Criteria:** Sequences where an isolated frame contains static artifacts that fully reveal the label.
- **Kinematic / Representation Signature:** Single-frame prediction entropy is high (> 2.0); GRU confidence jumps > +0.30.
- **Expected Reversal Sensitivity:** High
- **Expected Permutation Sensitivity:** High
- **Research Role:** Directly explains why Model A (Single) = 8.78% fails while Model C (GRU) = 34.78% succeeds.
- **VLM Probe Question:** *"Would a single frozen snapshot from this sequence be ambiguous between two opposite actions?"*
- **Primary Action Classes:**
  * `Opening something`
  * `Closing something`
  * `Moving something away from the camera`
  * `Moving something towards the camera`
  * `Pushing something from left to right`
  * `Pushing something from right to left`
  * `Covering something`
  * `Uncovering something`
  * `Rotating something clockwise`
  * `Rotating something counter-clockwise`

---

## 3. Pre-Registered Action Class Mapping (20 Benchmark Classes)

| Class Idx | Action Class Name | Primary Phenomenon | Secondary Phenomenon | Reciprocal Pair | Reversal Inverts? | Frame Ambiguity |
|:---:|---|:---:|:---:|---|:---:|:---:|
|  0 | `Closing something` | **T3** | T2 | `Opening something` | True | High |
|  1 | `Covering something` | **T3** | T4 | `Uncovering something` | True | High |
|  2 | `Moving something away from the camera` | **T2** | T1 | `Moving something towards the camera` | True | High |
|  3 | `Moving something towards the camera` | **T2** | T1 | `Moving something away from the camera` | True | High |
|  4 | `Opening something` | **T3** | T2 | `Closing something` | True | High |
|  5 | `Picking something up` | **T1** | T4 | `Putting something down` | True | Moderate |
|  6 | `Pushing something from left to right` | **T1** | T2 | `Pushing something from right to left` | True | High |
|  7 | `Pushing something from right to left` | **T1** | T2 | `Pushing something from left to right` | True | High |
|  8 | `Putting something down` | **T1** | T4 | `Picking something up` | True | Moderate |
|  9 | `Putting something into something` | **T4** | T5 | `Taking something out of something` | True | High |
| 10 | `Putting something on top of something` | **T4** | T5 | `Taking something off of something` | True | High |
| 11 | `Rotating something clockwise` | **T2** | T1 | `Rotating something counter-clockwise` | True | High |
| 12 | `Rotating something counter-clockwise` | **T2** | T1 | `Rotating something clockwise` | True | High |
| 13 | `Standing something upright` | **T3** | T4 | `Tipping something over` | True | Moderate |
| 14 | `Taking something off of something` | **T4** | T6 | `Putting something on top of something` | True | High |
| 15 | `Taking something out of something` | **T4** | T6 | `Putting something into something` | True | High |
| 16 | `Tipping something over` | **T3** | T1 | `Standing something upright` | True | Moderate |
| 17 | `Turning the camera left while filming something` | **T5** | T2 | `Turning the camera right while filming something` | True | High |
| 18 | `Turning the camera right while filming something` | **T5** | T2 | `Turning the camera left while filming something` | True | High |
| 19 | `Uncovering something` | **T3** | T4 | `Covering something` | True | High |

---

## 4. VLM Structured Output Schema Contract (Milestone M5.4)

To prevent unstructured narrative drift and hallucination, VLM calls in M5.4 will enforce this strict JSON schema:

```json
{
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
      "items": {
        "type": "integer",
        "minimum": 0,
        "maximum": 7
      },
      "description": "0-indexed frame numbers (out of 8) containing the most discriminative transition."
    },
    "temporal_dependency": {
      "type": "string",
      "enum": [
        "none",
        "low",
        "medium",
        "high"
      ],
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
  "additionalProperties": false
}
```

---
*Generated automatically by `src.evaluation.taxonomy` under Milestone M5.3.*