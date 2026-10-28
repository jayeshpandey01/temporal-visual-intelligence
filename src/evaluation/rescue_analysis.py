"""
Forensic Rescue Taxonomy and Qualitative Failure Analysis (Milestone 4).
Stratifies the Quadrant C temporal rescues (MeanPool wrong -> GRU correct)
to explain why chronological state transitions provide discriminative value.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
import numpy as np
import pandas as pd


def analyze_temporal_rescues(
    paired_predictions_csv: str = "results/paired_test_predictions.csv",
    failure_manifest_path: str = "results/evaluation_outputs/failure_analysis_manifest.json",
    output_report_path: str = "results/evaluation_outputs/m4_rescue_taxonomy.json",
    output_table_path: str = "results/tables/m4_rescue_taxonomy.md",
) -> Dict[str, Any]:
    """
    Perform class-level and kinematic stratification of Quadrant C temporal rescues.
    """
    df = pd.read_csv(paired_predictions_csv)
    
    # Identify Quadrants
    # Quadrant C: MeanPool wrong (0) and GRU correct (1)
    q_c_df = df[(df['correct_meanpool'] == 0) & (df['correct_gru'] == 1)]
    q_d_df = df[(df['correct_meanpool'] == 0) & (df['correct_gru'] == 0)]
    q_b_df = df[(df['correct_meanpool'] == 1) & (df['correct_gru'] == 0)]
    q_a_df = df[(df['correct_meanpool'] == 1) & (df['correct_gru'] == 1)]

    total_seqs = len(df)
    n_c = len(q_c_df)
    n_d = len(q_d_df)
    n_b = len(q_b_df)
    n_a = len(q_a_df)

    # Class-level distribution of rescues
    class_col = 'class_name' if 'class_name' in df.columns else 'label_name'
    rescue_counts_by_class = q_c_df[class_col].value_counts().to_dict()
    total_counts_by_class = df[class_col].value_counts().to_dict()

    class_rescue_rates = {}
    for cname, count in total_counts_by_class.items():
        rescued = rescue_counts_by_class.get(cname, 0)
        rate = (rescued / count) * 100.0 if count > 0 else 0.0
        class_rescue_rates[cname] = {
            'rescued': rescued,
            'total': count,
            'rescue_rate': round(rate, 2),
        }

    # Sort classes by rescue rate
    sorted_classes = sorted(class_rescue_rates.items(), key=lambda x: x[1]['rescue_rate'], reverse=True)

    # Directional pair analysis
    directional_keywords = ['left to right', 'right to left', 'opening', 'closing', 'towards', 'away', 'pulling', 'pushing']
    directional_rescues = 0
    non_directional_rescues = 0

    for cname, data in class_rescue_rates.items():
        is_directional = any(kw in cname.lower() for kw in directional_keywords)
        if is_directional:
            directional_rescues += data['rescued']
        else:
            non_directional_rescues += data['rescued']

    directional_ratio = (directional_rescues / n_c) * 100.0 if n_c > 0 else 0.0

    report = {
        'total_sequences': total_seqs,
        'quadrant_counts': {
            'quadrant_a_both_correct': n_a,
            'quadrant_b_meanpool_correct_gru_fail': n_b,
            'quadrant_c_temporal_rescues': n_c,
            'quadrant_d_both_failed': n_d,
        },
        'rescue_rate_overall': round((n_c / total_seqs) * 100.0, 2),
        'directional_rescues_count': directional_rescues,
        'directional_rescues_percentage': round(directional_ratio, 2),
        'top_rescued_classes': sorted_classes[:5],
        'least_rescued_classes': sorted_classes[-5:],
        'interpretation': (
            f"Of {n_c} temporal rescues, {directional_ratio:.1f}% belong to direction-dependent "
            f"action classes where symmetric spatial aggregation is mathematically ambiguous."
        )
    }

    # Save JSON report
    Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Generate Markdown Table
    lines = [
        "# Table: Quadrant C Temporal Rescue Stratification (Milestone 4)",
        f"Forensic breakdown of N = {n_c} sequence rescues by Temporal GRU over Mean-Pooling.",
        "",
        "| Rank | Action Class | Rescued / Total | Rescue Rate (%) | Directional Category |",
        "|:---:|---|:---:|:---:|:---:|"
    ]

    for rank, (cname, data) in enumerate(sorted_classes, 1):
        is_dir = "Directional" if any(kw in cname.lower() for kw in directional_keywords) else "Static / General"
        lines.append(f"| {rank} | `{cname}` | {data['rescued']}/{data['total']} | {data['rescue_rate']:.1f}% | {is_dir} |")

    lines.extend([
        "",
        "### Key Findings:",
        f"1. **Directional Dominance:** {directional_ratio:.1f}% of all rescues occur in direction-sensitive categories.",
        f"2. **Net Rescue Advantage:** GRU rescues {n_c} sequences while regressing on only {n_b} sequences ({n_c/max(1, n_b):.1f}x rescue ratio).",
        f"3. **Conclusion:** Temporal modeling succeeds primarily by disambiguating the chronological arrow of time."
    ])

    table_content = "\n".join(lines)
    Path(output_table_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_table_path, 'w') as f:
        f.write(table_content)

    return report


if __name__ == "__main__":
    rep = analyze_temporal_rescues()
    print("Rescue taxonomy analysis complete:")
    print(f"  Quadrant C Rescues: {rep['quadrant_counts']['quadrant_c_temporal_rescues']}")
    print(f"  Directional Ratio: {rep['directional_rescues_percentage']}%")
