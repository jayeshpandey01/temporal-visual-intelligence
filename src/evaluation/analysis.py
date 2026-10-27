"""
Win/Loss quadrant miner and temporal benefit taxonomy.
(research.md §21-22)

Quadrant matrix:
                  Temporal
               Correct  Wrong
  Baseline
  Correct    |   A   |   B   |
  Wrong      |   C   |   D   |

C = Temporal rescued → PRIMARY qualitative evidence
B = Temporal regressed → investigate failure modes
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def compute_win_loss_quadrants(
    baseline_preds: np.ndarray,
    temporal_preds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Classify each test sample into one of four quadrants.

    Returns:
        dict with quadrant counts, indices, and per-quadrant accuracy
    """
    base_correct = (baseline_preds == labels)
    temp_correct = (temporal_preds == labels)

    # Quadrant masks
    A_mask = base_correct & temp_correct    # Both correct
    B_mask = base_correct & ~temp_correct   # Baseline only (temporal regressed)
    C_mask = ~base_correct & temp_correct   # Temporal rescued ← KEY
    D_mask = ~base_correct & ~temp_correct  # Both wrong

    result = {
        'quadrant_counts': {
            'A_both_correct':       int(A_mask.sum()),
            'B_baseline_only':      int(B_mask.sum()),
            'C_temporal_rescued':   int(C_mask.sum()),
            'D_both_wrong':         int(D_mask.sum()),
        },
        'quadrant_indices': {
            'A': np.where(A_mask)[0].tolist(),
            'B': np.where(B_mask)[0].tolist(),
            'C': np.where(C_mask)[0].tolist(),
            'D': np.where(D_mask)[0].tolist(),
        },
        'total_samples': int(len(labels)),
        'pct_temporal_rescued': float(C_mask.sum() / len(labels) * 100),
        'pct_temporal_regressed': float(B_mask.sum() / len(labels) * 100),
    }

    print("\nWin/Loss Quadrant Analysis:")
    print(f"  A (Both correct):      {result['quadrant_counts']['A_both_correct']:4d} ({A_mask.mean()*100:.1f}%)")
    print(f"  B (Baseline only):     {result['quadrant_counts']['B_baseline_only']:4d} ({B_mask.mean()*100:.1f}%) ← temporal regression")
    print(f"  C (Temporal rescued):  {result['quadrant_counts']['C_temporal_rescued']:4d} ({C_mask.mean()*100:.1f}%) ← KEY EVIDENCE")
    print(f"  D (Both wrong):        {result['quadrant_counts']['D_both_wrong']:4d} ({D_mask.mean()*100:.1f}%)")

    return result


TEMPORAL_BENEFIT_TAXONOMY = [
    'motion_cue',           # Object moving in a direction visible only over time
    'state_transition',     # Opening/closing, filling/emptying
    'occlusion_resolution', # Object hidden in early frames, visible later
    'direction_sensitivity',# Requires knowing trajectory direction
    'context_accumulation', # Background context builds over time
    'disambiguation',       # Single frame ambiguous, sequence resolves it
    'other',
]


def plot_quadrant_matrix(
    quadrant_counts: Dict[str, int],
    output_path: str,
    title: str = "Win/Loss Quadrant Matrix",
):
    """Visualize the 2×2 quadrant matrix."""
    A = quadrant_counts['A_both_correct']
    B = quadrant_counts['B_baseline_only']
    C = quadrant_counts['C_temporal_rescued']
    D = quadrant_counts['D_both_wrong']
    total = A + B + C + D

    fig, ax = plt.subplots(figsize=(6, 5))

    matrix = np.array([[A, B], [C, D]])
    pct_matrix = matrix / max(total, 1) * 100

    im = ax.imshow(pct_matrix, cmap='Blues', vmin=0, vmax=100)

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Temporal Correct', 'Temporal Wrong'], fontsize=11)
    ax.set_yticklabels(['Baseline Correct', 'Baseline Wrong'], fontsize=11)

    labels_text = [
        [f'A\n{A} ({pct_matrix[0,0]:.1f}%)\nBoth correct', f'B\n{B} ({pct_matrix[0,1]:.1f}%)\nBaseline only'],
        [f'C\n{C} ({pct_matrix[1,0]:.1f}%)\nTemporal rescued ⭐', f'D\n{D} ({pct_matrix[1,1]:.1f}%)\nBoth wrong'],
    ]

    for i in range(2):
        for j in range(2):
            color = 'white' if pct_matrix[i, j] > 50 else 'black'
            ax.text(j, i, labels_text[i][j], ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold' if (i, j) == (1, 0) else 'normal')

    plt.colorbar(im, ax=ax, label='% of test samples')
    ax.set_title(title, fontsize=12, pad=15)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved quadrant matrix → {output_path}")


def save_quadrant_strips(
    quadrant_indices: List[int],
    sequence_paths: List[str],
    labels: np.ndarray,
    baseline_preds: np.ndarray,
    temporal_preds: np.ndarray,
    output_dir: str,
    quadrant_name: str,
    max_examples: int = 20,
    n_frames_shown: int = 6,
):
    """
    Save frame strip visualizations for a given quadrant.
    Most useful for Quadrant C (temporal rescued) — show in report.
    """
    os.makedirs(output_dir, exist_ok=True)
    indices = quadrant_indices[:max_examples]

    metadata = []
    for i, idx in enumerate(indices):
        metadata.append({
            'sample_idx': int(idx),
            'true_label': int(labels[idx]),
            'baseline_pred': int(baseline_preds[idx]),
            'temporal_pred': int(temporal_preds[idx]),
            'sequence_path': sequence_paths[idx] if sequence_paths else None,
        })

    # Save metadata
    meta_path = os.path.join(output_dir, f'{quadrant_name}_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(indices)} {quadrant_name} examples → {output_dir}")
    return metadata

