"""
Representation trajectory analysis — run BEFORE training any temporal model.
(research.md §9)

Computes:
  - Consecutive cosine similarity: S_t = cos(E_t, E_{t+1})
  - Cumulative drift from E_0: drift_t = 1 - cos(E_0, E_t)
  - Temporal variance per sequence
  - Inter-class vs intra-class separability

Outputs:
  - results/figures/cos_sim_trajectories.png
  - results/figures/embedding_drift.png
  - results/embedding_analysis.json
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings_for_class(
    embeddings_dir: Path,
    metadata: List[Dict],
    class_label: int,
    split: str = 'train',
    max_sequences: int = 20,
) -> List[np.ndarray]:
    """Load embeddings for sequences of a given class."""
    split_dir = embeddings_dir / split
    sequences = [
        m for m in metadata
        if m['label'] == class_label and m['split'] == split
    ][:max_sequences]

    result = []
    for m in sequences:
        path = split_dir / f"{m['sequence_id']}.npy"
        if path.exists():
            result.append(np.load(str(path)))  # [N, D]
    return result


def consecutive_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute S_t = cos(E_t, E_{t+1}) for all consecutive frame pairs.
    Returns array of shape [N-1].
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    return np.sum(normalized[:-1] * normalized[1:], axis=1)


def cumulative_drift(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute drift_t = 1 - cos(E_0, E_t) for all t.
    Returns array of shape [N].
    """
    e0 = embeddings[0:1]  # [1, D]
    norms_0 = np.linalg.norm(e0, axis=1, keepdims=True)
    norms_t = np.linalg.norm(embeddings, axis=1, keepdims=True)
    e0_norm = e0 / (norms_0 + 1e-8)
    et_norm = embeddings / (norms_t + 1e-8)
    cos_sim = np.sum(e0_norm * et_norm, axis=1)
    return 1 - cos_sim  # drift: 0 at t=0, increases as embedding moves away


def temporal_variance(embeddings: np.ndarray) -> float:
    """Mean squared distance from the sequence mean embedding."""
    mean_emb = embeddings.mean(axis=0)
    return float(np.mean(np.sum((embeddings - mean_emb) ** 2, axis=1)))


def plot_similarity_trajectories(
    sequences_by_class: Dict[str, List[np.ndarray]],
    output_path: str,
    n_frames: int = 8,
):
    """Plot S_t trajectories per class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sequences_by_class)))

    ax1, ax2 = axes

    for (class_name, seqs), color in zip(sequences_by_class.items(), colors):
        all_sims = []
        all_drifts = []

        for emb in seqs:
            if len(emb) < 2:
                continue
            # Subsample to fixed length for plotting
            idx = np.linspace(0, len(emb) - 1, min(n_frames, len(emb)), dtype=int)
            emb_sub = emb[idx]
            sims = consecutive_cosine_similarity(emb_sub)
            drift = cumulative_drift(emb_sub)
            all_sims.append(sims)
            all_drifts.append(drift)

        if not all_sims:
            continue

        # Plot mean ± std
        max_len = max(len(s) for s in all_sims)
        sims_padded = np.array([np.pad(s, (0, max_len - len(s)), constant_values=np.nan) for s in all_sims])
        mean_sim = np.nanmean(sims_padded, axis=0)
        std_sim  = np.nanstd(sims_padded, axis=0)

        x_sim = range(1, len(mean_sim) + 1)
        ax1.plot(x_sim, mean_sim, label=class_name[:12], color=color, linewidth=2)
        ax1.fill_between(x_sim, mean_sim - std_sim, mean_sim + std_sim, alpha=0.15, color=color)

        # Drift
        max_drift_len = max(len(d) for d in all_drifts)
        drifts_padded = np.array([np.pad(d, (0, max_drift_len - len(d)), constant_values=np.nan) for d in all_drifts])
        mean_drift = np.nanmean(drifts_padded, axis=0)

        x_drift = range(len(mean_drift))
        ax2.plot(x_drift, mean_drift, label=class_name[:12], color=color, linewidth=2)

    ax1.set_xlabel('Frame step t')
    ax1.set_ylabel('Cosine similarity S_t = cos(E_t, E_{t+1})')
    ax1.set_title('Consecutive Frame Similarity by Class')
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='High similarity (≥0.9)')
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Frame t')
    ax2.set_ylabel('Drift from E_0: 1 - cos(E_0, E_t)')
    ax2.set_title('Cumulative Embedding Drift from First Frame')
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity trajectories → {output_path}")


def compute_class_separability(
    sequences_by_class: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    """
    Compute mean intra-class vs inter-class cosine similarity.
    High intra / low inter → classes well separated → single-frame baseline may be strong.
    """
    class_names = list(sequences_by_class.keys())
    class_mean_embeddings = {}

    for class_name, seqs in sequences_by_class.items():
        if not seqs:
            continue
        # Use mean of mean embeddings
        means = [emb.mean(axis=0) for emb in seqs]
        class_mean_embeddings[class_name] = np.mean(means, axis=0)

    if len(class_mean_embeddings) < 2:
        return {'error': 'Need at least 2 classes for separability analysis'}

    names = list(class_mean_embeddings.keys())
    means = np.stack([class_mean_embeddings[n] for n in names])
    sim_matrix = cosine_similarity(means)

    # Inter-class: off-diagonal
    mask = ~np.eye(len(names), dtype=bool)
    inter_sim = sim_matrix[mask].mean()

    # Intra-class (comparing individual seqs within class)
    intra_sims = []
    for seqs in sequences_by_class.values():
        if len(seqs) < 2:
            continue
        seq_means = np.stack([emb.mean(axis=0) for emb in seqs])
        intra_sim = cosine_similarity(seq_means)
        mask_i = ~np.eye(len(seq_means), dtype=bool)
        if mask_i.any():
            intra_sims.append(intra_sim[mask_i].mean())

    return {
        'intra_class_similarity': float(np.mean(intra_sims)) if intra_sims else None,
        'inter_class_similarity': float(inter_sim),
        'separability_gap': float(np.mean(intra_sims) - inter_sim) if intra_sims else None,
        'interpretation': (
            'High separability gap → classes well-separated → single-frame baseline may be competitive.'
            if (intra_sims and np.mean(intra_sims) - inter_sim > 0.1) else
            'Low separability gap → temporal context likely needed for disambiguation.'
        ),
    }


def run_analysis(
    embeddings_dir: str,
    metadata_path: str,
    output_dir: str = 'results',
    n_frames: int = 8,
):
    """Run full representation trajectory analysis."""
    print("=" * 60)
    print("REPRESENTATION TRAJECTORY ANALYZER")
    print("=" * 60)

    embeddings_dir = Path(embeddings_dir)
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Group by class
    labels = sorted(set(m['label'] for m in metadata))
    class_names = [str(l) for l in labels]  # Use label idx if no class name

    sequences_by_class = {}
    for label in labels:
        seqs = load_embeddings_for_class(embeddings_dir, metadata, label, split='train')
        if seqs:
            sequences_by_class[str(label)] = seqs
        print(f"  Class {label}: {len(seqs)} sequences loaded")

    # Plots
    plot_similarity_trajectories(
        sequences_by_class,
        output_path=os.path.join(output_dir, 'figures', 'cos_sim_trajectories.png'),
        n_frames=n_frames,
    )

    # Separability analysis
    sep = compute_class_separability(sequences_by_class)
    print(f"\nClass separability:")
    print(f"  Intra-class similarity: {sep.get('intra_class_similarity', 'N/A'):.3f}")
    print(f"  Inter-class similarity: {sep.get('inter_class_similarity', 'N/A'):.3f}")
    print(f"  Gap: {sep.get('separability_gap', 'N/A')}")
    print(f"  → {sep.get('interpretation', '')}")

    # Per-class temporal variance
    variance_by_class = {}
    for class_name, seqs in sequences_by_class.items():
        variances = [temporal_variance(emb) for emb in seqs]
        variance_by_class[class_name] = {
            'mean_temporal_variance': float(np.mean(variances)),
            'std_temporal_variance': float(np.std(variances)),
        }

    analysis_results = {
        'class_separability': sep,
        'temporal_variance_by_class': variance_by_class,
        'n_classes': len(sequences_by_class),
    }

    # Save
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    out_path = os.path.join(output_dir, 'embedding_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\nAnalysis saved → {out_path}")

    return analysis_results

