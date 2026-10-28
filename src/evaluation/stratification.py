"""
Milestone M4.7: Failure Stratification & Qualitative Error Taxonomy.
Directly fulfills Take-Home Assessment Sections 5, 7.C, 7.D, 8, 10, & 11 and M4.7 / M4.8:
  - 4-Quadrant Win/Loss Partitioning:
      * Quadrant A: Both Correct (temporal context not needed)
      * Quadrant B: Baseline Correct, Temporal Wrong (temporal regression)
      * Quadrant C: Temporal Rescued (primary qualitative targets)
      * Quadrant D: Both Failed (fundamentally difficult / uninformative)
  - Sequence-Level Temporal Gain: G_i = 1[GRU correct] - 1[Single correct] in {-1, 0, +1}
  - 5 Research Failure Groups (M-4.md Section M4.7):
      * Group A: Single-frame easy (Quadrant A, G=0)
      * Group B: Temporally ambiguous (Quadrant C, G=+1)
      * Group C: Static / redundant (low embedding drift, G ≈ 0)
      * Group D: Major transition (high embedding drift, strong temporal benefit)
      * Group E: Difficult even with sequence (Quadrant D, G=0)
  - Representational Trajectory Kinematics:
      * Consecutive frame cosine similarity: S_t = cos(E_t, E_{t+1})
      * Cumulative cosine drift: 1 - cos(E_1, E_8)
      * Trajectory variance: Var_temp = mean over t of ||E_t - mean(E)||^2
  - Visual Strip Generation for Qualitative Reporting (Quadrant C & Quadrant B).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_quadrant_partition(
    y_true: np.ndarray,
    y_base: np.ndarray,
    y_gru: np.ndarray,
    seq_ids: Optional[List[str]] = None,
    y_meanpool: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Partition paired predictions into the canonical 4 quadrants (M4.7, temporal-win-loss-miner).

    Quadrants:
      - Quadrant A (Both Correct): base == gt and gru == gt  (G_i = 0)
      - Quadrant B (Baseline Only): base == gt and gru != gt (G_i = -1) [Regression]
      - Quadrant C (Temporal Rescued): base != gt and gru == gt (G_i = +1) [Primary Target]
      - Quadrant D (Both Wrong): base != gt and gru != gt (G_i = 0)

    Returns:
        pd.DataFrame containing per-sequence indicators, quadrant labels, and temporal gain.
    """
    y_true = np.asarray(y_true)
    y_base = np.asarray(y_base)
    y_gru = np.asarray(y_gru)
    n = len(y_true)

    if seq_ids is None:
        seq_ids = [f"seq_{i:04d}" for i in range(n)]

    corr_base = (y_base == y_true).astype(int)
    corr_gru = (y_gru == y_true).astype(int)
    temporal_gain = corr_gru - corr_base  # {-1, 0, +1}

    quadrants = []
    quadrant_codes = []

    for i in range(n):
        cb = corr_base[i]
        cg = corr_gru[i]
        if cb == 1 and cg == 1:
            quadrants.append("Quadrant A (Both Correct)")
            quadrant_codes.append("A")
        elif cb == 1 and cg == 0:
            quadrants.append("Quadrant B (Baseline Only)")
            quadrant_codes.append("B")
        elif cb == 0 and cg == 1:
            quadrants.append("Quadrant C (Temporal Rescued)")
            quadrant_codes.append("C")
        else:
            quadrants.append("Quadrant D (Both Wrong)")
            quadrant_codes.append("D")

    records = {
        "sequence_id": seq_ids,
        "ground_truth": y_true,
        "pred_baseline": y_base,
        "pred_gru": y_gru,
        "correct_baseline": corr_base,
        "correct_gru": corr_gru,
        "temporal_gain_g": temporal_gain,
        "quadrant": quadrants,
        "quadrant_code": quadrant_codes,
    }

    if y_meanpool is not None:
        y_mp = np.asarray(y_meanpool)
        corr_mp = (y_mp == y_true).astype(int)
        records["pred_meanpool"] = y_mp
        records["correct_meanpool"] = corr_mp
        # Disentanglement flags:
        # Observation rescued: MeanPool right, Single wrong
        records["rescued_by_obs"] = ((corr_base == 0) & (corr_mp == 1)).astype(int)
        # Order rescued: GRU right, MeanPool wrong
        records["rescued_by_order"] = ((corr_mp == 0) & (corr_gru == 1)).astype(int)

    return pd.DataFrame(records)


def compute_trajectory_kinematics(
    embeddings_dir: Union[str, Path],
    seq_ids: List[str],
) -> pd.DataFrame:
    """
    Calculate frozen DINOv2 embedding trajectory kinematics (representation-trajectory-analyzer).

    Metrics:
      - Mean consecutive cosine similarity: S_mean = mean_t cos(E_t, E_{t+1})
      - Minimum consecutive cosine similarity: S_min = min_t cos(E_t, E_{t+1}) (peak transition step)
      - Cumulative drift: drift = 1 - cos(E_1, E_T)
      - Representation variance: Var_temp = mean_t ||E_t - mean(E)||^2
    """
    emb_dir = Path(embeddings_dir)
    records = []

    for sid in seq_ids:
        npy_path = emb_dir / f"{sid}.npy"
        if not npy_path.exists():
            # Fallback if embedded in subfolders or test folder
            test_sub = emb_dir / "test" / f"{sid}.npy"
            if test_sub.exists():
                npy_path = test_sub

        if not npy_path.exists():
            # Default zero kinematics if file not found
            records.append({
                "sequence_id": sid,
                "consecutive_cos_mean": 1.0,
                "consecutive_cos_min": 1.0,
                "cumulative_drift": 0.0,
                "temporal_variance": 0.0,
                "is_kinematically_static": True,
            })
            continue

        emb = np.load(npy_path)  # [T, D]
        T, D = emb.shape

        # Normalize frame vectors
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb_norm = emb / norms

        # Consecutive cosine similarities
        cos_steps = np.sum(emb_norm[:-1] * emb_norm[1:], axis=1)
        s_mean = float(np.mean(cos_steps))
        s_min = float(np.min(cos_steps))

        # Cumulative drift: 1 - cos(E_0, E_{T-1})
        cum_cos = float(np.dot(emb_norm[0], emb_norm[-1]))
        drift = float(1.0 - cum_cos)

        # Temporal variance
        mean_vec = np.mean(emb, axis=0, keepdims=True)
        var_temp = float(np.mean(np.sum((emb - mean_vec) ** 2, axis=1)))

        records.append({
            "sequence_id": sid,
            "consecutive_cos_mean": s_mean,
            "consecutive_cos_min": s_min,
            "cumulative_drift": drift,
            "temporal_variance": var_temp,
            "is_kinematically_static": bool(drift < 1e-4),
        })

    return pd.DataFrame(records)


def stratify_into_research_groups(
    df_quadrants: pd.DataFrame,
    df_kinematics: pd.DataFrame,
    drift_static_quantile: float = 0.25,
    drift_transition_quantile: float = 0.75,
) -> pd.DataFrame:
    """
    Stratify sequences into the 5 Research Groups from Take-Home Assessment M4.7 (M-4.md):
      - Group A (Single-frame easy): One frame reveals class (Quadrant A, G=0)
      - Group B (Temporally ambiguous): Unclear in isolation, sequence resolves it (Quadrant C, G=+1)
      - Group C (Static / redundant): Frames barely change (drift <= 25th percentile, G ≈ 0)
      - Group D (Major transition): Something changes substantially (drift >= 75th percentile)
      - Group E (Difficult even with sequence): Neither model succeeds (Quadrant D, G=0)

    Returns:
        Unified pd.DataFrame with 'research_group' and 'group_code' columns.
    """
    merged = pd.merge(df_quadrants, df_kinematics, on="sequence_id", how="left")

    # Establish drift cutoffs
    drifts = merged["cumulative_drift"].values
    tau_static = float(np.quantile(drifts, drift_static_quantile))
    tau_trans = float(np.quantile(drifts, drift_transition_quantile))

    group_labels = []
    group_codes = []

    for _, row in merged.iterrows():
        q_code = row["quadrant_code"]
        drift = row["cumulative_drift"]

        # Primary assignment based on M4.7 definitions
        if q_code == "A":
            group_labels.append("Group A — Single-frame easy")
            group_codes.append("Group_A_Easy")
        elif q_code == "B":
            group_labels.append("Regression — Baseline only (Temporal Hurt)")
            group_codes.append("Regression_B")
        elif q_code == "C":
            if drift >= tau_trans:
                group_labels.append("Group D — Major transition (Rescued)")
                group_codes.append("Group_D_Transition")
            else:
                group_labels.append("Group B — Temporally ambiguous (Rescued)")
                group_codes.append("Group_B_Ambiguous")
        else:  # Quadrant D
            if drift <= tau_static:
                group_labels.append("Group C — Static / redundant (Both Failed)")
                group_codes.append("Group_C_Static")
            else:
                group_labels.append("Group E — Difficult even with sequence")
                group_codes.append("Group_E_Difficult")

    merged["research_group"] = group_labels
    merged["group_code"] = group_codes
    merged["drift_tau_static"] = tau_static
    merged["drift_tau_transition"] = tau_trans

    return merged


def categorize_rescue_taxonomy(
    df_rescued: pd.DataFrame,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Categorize Quadrant C temporal rescues into physical action transition categories (temporal-win-loss-miner).

    Categories:
      1. State Transition: Opening/closing, tipping over/standing upright, covering/uncovering
      2. Container Physics: Putting into something, taking out of something
      3. Directional Motion: Towards/away, pushing left/right
      4. Rotational Kinematics: Clockwise, counter-clockwise
      5. Subtle Manipulation: Picking up, putting down, taking off, putting on top
    """
    df = df_rescued.copy()
    categories = []

    for _, row in df.iterrows():
        cname = str(row.get("class_name", ""))
        if not cname and class_names and "ground_truth" in row:
            gt_id = int(row["ground_truth"])
            if gt_id < len(class_names):
                cname = class_names[gt_id]

        cname_lower = cname.lower()

        if any(kw in cname_lower for kw in ["into something", "out of something"]):
            cat = "Container Physics (Insertion / Extraction)"
        elif any(kw in cname_lower for kw in ["opening", "closing", "tipping", "standing upright", "covering", "uncovering"]):
            cat = "Physical State Transition (Bi-directional Object State)"
        elif any(kw in cname_lower for kw in ["towards", "away", "left to right", "right to left"]):
            cat = "Directional Vector (Arrow of Motion)"
        elif any(kw in cname_lower for kw in ["clockwise", "counter-clockwise"]):
            cat = "Rotational Kinematics (Angular Dynamics)"
        else:
            cat = "Fine-Grained Manipulation (Spatial Contacts)"

        categories.append(cat)

    df["rescue_taxonomy"] = categories
    return df


def format_forensic_card(rec: Dict[str, Any]) -> str:
    """
    Format human-readable diagnostic prediction card for qualitative reporting (M3.16 / M4.7).
    """
    seq_id = rec.get("sequence_id", "unknown")
    gt_id = rec.get("ground_truth", rec.get("ground_truth_id", 0))
    gt_name = rec.get("class_name", rec.get("ground_truth_name", f"Class_{gt_id}"))

    p_base = rec.get("pred_baseline", rec.get("single_pred_id", 0))
    p_base_name = rec.get("pred_baseline_name", f"Class_{p_base}")
    conf_base = rec.get("conf_baseline", rec.get("single_conf", 0.0))
    c_base = bool(p_base == gt_id)

    p_gru = rec.get("pred_gru", rec.get("gru_pred_id", 0))
    p_gru_name = rec.get("pred_gru_name", f"Class_{p_gru}")
    conf_gru = rec.get("conf_gru", rec.get("gru_conf", 0.0))
    c_gru = bool(p_gru == gt_id)

    quad = rec.get("quadrant", "Unknown")
    tax = rec.get("rescue_taxonomy", "General")

    lines = [
        f"Sequence ID:   {seq_id}",
        f"Ground Truth:  {gt_name} (ID: {gt_id})",
        f"Baseline:      {p_base_name} (ID: {p_base}, conf: {conf_base:.4f}) [{'CORRECT' if c_base else 'WRONG'}]",
        f"Temporal GRU:  {p_gru_name} (ID: {p_gru}, conf: {conf_gru:.4f}) [{'CORRECT' if c_gru else 'WRONG'}]",
        f"Outcome:       {quad}",
        f"Taxonomy:      {tax}",
    ]
    return "\n".join(lines)


def render_sequence_strip(
    seq_id: str,
    class_name: str,
    frames_dir: Union[str, Path],
    meta: Dict[str, Any],
    output_path: Union[str, Path],
    num_frames: int = 8,
) -> Path:
    """
    Render publication-ready 8-frame composite strip with diagnostic metadata headers.

    Headers display:
      - Sequence ID & Ground Truth Action Class
      - Single-Frame Baseline Prediction & Status
      - Temporal GRU Prediction & Status
      - Outcome Badge (e.g. 'QUADRANT C: TEMPORAL RESCUED')
      - Frame timesteps t = 1, ..., 8
    """
    frames_dir = Path(frames_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Search for frames
    frame_files = []
    # Candidate locations:
    candidate_dirs = [
        frames_dir / class_name / seq_id,
        frames_dir / seq_id,
        frames_dir / "test" / seq_id,
        Path("data") / class_name / seq_id,
    ]

    found_dir = None
    for c_dir in candidate_dirs:
        if c_dir.exists() and len(list(c_dir.glob("*.jpg"))) >= num_frames:
            found_dir = c_dir
            break

    if found_dir:
        frame_files = sorted(list(found_dir.glob("*.jpg")))[:num_frames]

    # Create figure
    fig, axes = plt.subplots(1, num_frames, figsize=(16, 2.8), gridspec_kw={'wspace': 0.05, 'hspace': 0.0})

    quadrant = meta.get("quadrant", "")
    gt_name = meta.get("class_name", class_name)
    base_pred = meta.get("pred_baseline_name", f"Class_{meta.get('pred_baseline', '?')}")
    gru_pred = meta.get("pred_gru_name", f"Class_{meta.get('pred_gru', '?')}")
    base_status = "CORRECT" if meta.get("correct_baseline", False) else "WRONG"
    gru_status = "CORRECT" if meta.get("correct_gru", False) else "WRONG"
    badge = meta.get("badge", quadrant)

    # Color for header banner
    if "C" in str(meta.get("quadrant_code", "")):
        header_color = "darkgreen"
    elif "B" in str(meta.get("quadrant_code", "")):
        header_color = "crimson"
    elif "A" in str(meta.get("quadrant_code", "")):
        header_color = "navy"
    else:
        header_color = "gray"

    for t in range(num_frames):
        ax = axes[t]
        if t < len(frame_files) and frame_files[t].exists():
            img = Image.open(frame_files[t]).convert("RGB")
            ax.imshow(img)
        else:
            # Synthetic placeholder if frames not on disk
            dummy = np.ones((120, 160, 3), dtype=np.uint8) * 230
            ax.imshow(dummy)
            ax.text(80, 60, f"Frame {t+1}", ha="center", va="center", fontsize=10, color="gray")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.08, f"t = {t+1}", transform=ax.transAxes, ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6), color="white", weight='bold')

    # Super title banner
    title_text = (
        f"[{badge}]  Sequence: {seq_id}  |  Ground Truth: {gt_name}\n"
        f"Single-Frame Baseline: {base_pred} [{base_status}]   vs.   "
        f"Temporal GRU: {gru_pred} [{gru_status}]"
    )
    plt.suptitle(title_text, fontsize=10.5, weight="bold", color=header_color, y=1.06)

    plt.savefig(out_path, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


__all__ = [
    "compute_quadrant_partition",
    "compute_trajectory_kinematics",
    "stratify_into_research_groups",
    "categorize_rescue_taxonomy",
    "format_forensic_card",
    "render_sequence_strip",
]
