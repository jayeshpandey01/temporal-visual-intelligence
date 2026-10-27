"""
Milestone 2 Representation Analysis Generator.

Generates the complete suite of M2 deliverables (docs/milestone/M-2.md §M2.11 - §M2.14):
  1. results/representation/cosine_similarity.csv
  2. results/representation/embedding_statistics.csv
  3. results/representation/similarity_plots/
  4. results/representation/embedding_pca/
  5. results/representation/temporal_strips/

Computes adjacent cosine similarity S_t, cumulative drift Delta(t), temporal variance,
and 2D PCA latent trajectories across synthetic (Tier 1) and dense real-world (Tier 2) sequences.
"""

from __future__ import annotations

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

from src.embeddings.cache import EmbeddingCache
from src.embeddings.contract import SequenceEmbeddingOutput


def compute_metrics(embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute consecutive cosine similarity S_t, cumulative drift Delta_t, and temporal variance.
    """
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / (norms + 1e-8)
    
    # S_t = cos(E_t, E_{t+1})
    s_t = np.sum(normed[:-1] * normed[1:], axis=1)
    
    # Delta_t = 1 - cos(E_0, E_t)
    e0_norm = normed[0:1]
    drifts = 1.0 - np.sum(e0_norm * normed, axis=1)
    
    # Temporal variance: mean_t ||E_t - mean(E)||^2
    mean_vec = embs.mean(axis=0, keepdims=True)
    var = float(np.mean(np.sum((embs - mean_vec) ** 2, axis=1)))
    
    return s_t, drifts, var


def main():
    cache_root = Path("embeddings")
    cache = EmbeddingCache(cache_root)
    manifest = cache.manifest

    out_rep = Path("results/representation")
    out_sim_plots = out_rep / "similarity_plots"
    out_pca = out_rep / "embedding_pca"
    out_strips = out_rep / "temporal_strips"
    
    out_sim_plots.mkdir(parents=True, exist_ok=True)
    out_pca.mkdir(parents=True, exist_ok=True)
    out_strips.mkdir(parents=True, exist_ok=True)

    print("Analyzing embedding cache...")

    # 1. Cosine similarity CSV and Embedding Statistics CSV
    cos_sim_rows = []
    stats_rows = []
    
    all_embeddings_by_class: Dict[str, List[np.ndarray]] = {}
    representative_seqs: Dict[str, Tuple[SequenceEmbeddingOutput, str]] = {}

    for split in ["train", "val", "test", "real"]:
        split_dict = manifest["splits"].get(split, {})
        for seq_id in sorted(split_dict.keys()):
            item = cache.get(seq_id, split=split)
            if item is None:
                continue

            embs = item.embeddings
            s_t, drifts, temp_var = compute_metrics(embs)
            
            frame_norms = np.linalg.norm(embs, axis=1)
            
            stats_rows.append({
                "sequence_id": seq_id,
                "split": split,
                "label": item.label,
                "num_frames": item.num_frames,
                "embed_dim": item.embed_dim,
                "mean_norm": float(np.mean(frame_norms)),
                "std_norm": float(np.std(frame_norms)),
                "min_norm": float(np.min(frame_norms)),
                "max_norm": float(np.max(frame_norms)),
                "temporal_variance": temp_var,
                "mean_adjacent_sim": float(np.mean(s_t)) if len(s_t) > 0 else 1.0,
                "min_adjacent_sim": float(np.min(s_t)) if len(s_t) > 0 else 1.0,
                "cumulative_drift_end": float(drifts[-1]) if len(drifts) > 0 else 0.0,
            })

            # Record per-transition similarities
            for t_idx, sim in enumerate(s_t):
                cos_sim_rows.append({
                    "sequence_id": seq_id,
                    "split": split,
                    "label": item.label,
                    "transition": f"t_{t_idx}_to_t_{t_idx+1}",
                    "from_frame": item.frame_indices[t_idx],
                    "to_frame": item.frame_indices[t_idx + 1],
                    "cosine_similarity": float(sim),
                    "cosine_distance": float(1.0 - sim)
                })

            if item.label not in all_embeddings_by_class:
                all_embeddings_by_class[item.label] = []
            all_embeddings_by_class[item.label].append(embs)

            if item.label not in representative_seqs:
                representative_seqs[item.label] = (item, split)

    # Save cosine_similarity.csv
    csv_sim_path = out_rep / "cosine_similarity.csv"
    with open(csv_sim_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sequence_id", "split", "label", "transition", "from_frame", "to_frame", "cosine_similarity", "cosine_distance"
        ])
        writer.writeheader()
        writer.writerows(cos_sim_rows)
    print(f"Saved: {csv_sim_path} ({len(cos_sim_rows)} transitions)")

    # Save embedding_statistics.csv
    csv_stats_path = out_rep / "embedding_statistics.csv"
    with open(csv_stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sequence_id", "split", "label", "num_frames", "embed_dim", "mean_norm", "std_norm",
            "min_norm", "max_norm", "temporal_variance", "mean_adjacent_sim", "min_adjacent_sim", "cumulative_drift_end"
        ])
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f"Saved: {csv_stats_path} ({len(stats_rows)} sequences)")

    # 2. Similarity Plots (M2.11 / M2.13 A)
    # A. Comparative Synthetic Class Dynamics
    fig, ax = plt.subplots(figsize=(10, 6))
    for lbl, seq_list in sorted(all_embeddings_by_class.items()):
        if lbl in ["horse_galloping", "car_drifting"]:
            continue
        # Average similarity profile across sequences of this class
        class_sims = []
        for e in seq_list[:20]:
            s, _, _ = compute_metrics(e)
            if len(s) == 7:  # 8-frame sequences have 7 transitions
                class_sims.append(s)
        if class_sims:
            mean_s = np.mean(class_sims, axis=0)
            ax.plot(range(1, len(mean_s) + 1), mean_s, marker='o', alpha=0.7, label=f"Class {lbl}")
    ax.set_title("Consecutive Frame Cosine Similarity: S_t = cos(E_t, E_{t+1}) across Synthetic Actions", fontsize=12)
    ax.set_xlabel("Transition Step (t -> t+1)", fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_ylim(0.98, 1.002)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_sim_path = out_sim_plots / "synthetic_action_similarity_curves.png"
    plt.savefig(plot_sim_path, dpi=200)
    plt.close()
    print(f"Saved: {plot_sim_path}")

    # B. Real Continuous Action Dynamics (50 Frames)
    horse_item = cache.get("horse_running_gallop_50", split="real")
    bmw_item = cache.get("bmw_m4_drift_50", split="real")
    if horse_item and bmw_item:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        h_sim, h_drift, _ = compute_metrics(horse_item.embeddings)
        b_sim, b_drift, _ = compute_metrics(bmw_item.embeddings)

        axes[0].plot(range(1, len(h_sim) + 1), h_sim, label="Horse Gallop (50 frames)", color="#2ca02c", lw=2)
        axes[0].plot(range(1, len(b_sim) + 1), b_sim, label="BMW Drift (50 frames)", color="#1f77b4", lw=2)
        axes[0].set_title("Adjacent Cosine Similarity Across Dense 50-Frame Manifolds", fontsize=11)
        axes[0].set_xlabel("Frame Transition (t -> t+1)", fontsize=10)
        axes[0].set_ylabel("Cosine Similarity S_t", fontsize=10)
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()

        axes[1].plot(range(len(h_drift)), h_drift, label="Horse Gallop Drift Delta(t)", color="#2ca02c", lw=2)
        axes[1].plot(range(len(b_drift)), b_drift, label="BMW Drift Drift Delta(t)", color="#1f77b4", lw=2)
        axes[1].set_title("Cumulative Cosine Drift: Delta(t) = 1 - cos(E_0, E_t)", fontsize=11)
        axes[1].set_xlabel("Observation Index t", fontsize=10)
        axes[1].set_ylabel("Cumulative Drift", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()

        plt.tight_layout()
        real_sim_path = out_sim_plots / "real_50_frames_similarity_and_drift.png"
        plt.savefig(real_sim_path, dpi=200)
        plt.close()
        print(f"Saved: {real_sim_path}")

    # 3. PCA Trajectory Visualization (M2.13 B)
    # Fit PCA on combined representation of representative sequences
    pca_sources = []
    pca_labels = []
    
    # Sample 4 distinct synthetic classes + 2 real continuous sequences
    selected_classes = list(all_embeddings_by_class.keys())[:4]
    for c in selected_classes:
        for seq_e in all_embeddings_by_class[c][:3]:
            pca_sources.append(seq_e)
            pca_labels.append(f"Synthetic Cls {c}")

    if horse_item:
        pca_sources.append(horse_item.embeddings)
        pca_labels.append("Horse Gallop (50f)")
    if bmw_item:
        pca_sources.append(bmw_item.embeddings)
        pca_labels.append("BMW Drift (50f)")

    stacked_pca = np.concatenate(pca_sources, axis=0)
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(stacked_pca)

    fig, ax = plt.subplots(figsize=(10, 8))
    start_idx = 0
    cmap = plt.cm.tab10
    for i, (seq_e, label_name) in enumerate(zip(pca_sources, pca_labels)):
        end_idx = start_idx + len(seq_e)
        traj_2d = pca_2d[start_idx:end_idx]
        start_idx = end_idx

        color = cmap(i % 10)
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], marker='o', markersize=4, label=label_name, color=color, alpha=0.8)
        
        # Draw progression arrows along the manifold
        for step in range(len(traj_2d) - 1):
            if step % max(1, len(traj_2d) // 5) == 0:
                dx = traj_2d[step + 1, 0] - traj_2d[step, 0]
                dy = traj_2d[step + 1, 1] - traj_2d[step, 1]
                ax.annotate("", xy=(traj_2d[step + 1, 0], traj_2d[step + 1, 1]),
                            xytext=(traj_2d[step, 0], traj_2d[step, 1]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    ax.set_title(f"2D PCA Representation Trajectories (Explained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontsize=12)
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    pca_plot_path = out_pca / "embedding_pca_trajectories.png"
    plt.savefig(pca_plot_path, dpi=200)
    plt.close()
    print(f"Saved: {pca_plot_path}")

    # 4. Temporal Strip with Numerical Alignment (M2.13 C)
    # Generate visual strip for 50-frame horse gallop and BMW drift with aligned metric curve
    for action_item, dir_path, name in [
        (horse_item, "data/tier2_real/dense_horse_50", "horse_gallop_strip_aligned.png"),
        (bmw_item, "data/tier2_real/dense_bmw_50", "bmw_drift_strip_aligned.png")
    ]:
        if not action_item:
            continue
        frame_files = sorted([f for f in Path(dir_path).iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if len(frame_files) >= 8:
            # Pick 8 uniformly sampled frames across the 50 frames
            indices = [int(np.floor(k * (len(frame_files) - 1) / 7)) for k in range(8)]
            sub_frames = [Image.open(frame_files[i]) for i in indices]
            sub_embs = action_item.embeddings[indices]
            s_t, drifts, _ = compute_metrics(sub_embs)

            fig = plt.figure(figsize=(16, 6))
            gs = fig.add_gridspec(2, 8, height_ratios=[1.8, 1.0])

            # Top row: 8 frames
            for k in range(8):
                ax_img = fig.add_subplot(gs[0, k])
                ax_img.imshow(sub_frames[k])
                ax_img.set_title(f"Frame #{indices[k]}", fontsize=9)
                ax_img.axis('off')

            # Bottom row: spanned cosine similarity curve
            ax_curve = fig.add_subplot(gs[1, :])
            ax_curve.plot(range(1, 8), s_t, marker='s', color='#d62728', lw=2, label="Adjacent Cosine Similarity S_t")
            ax_curve.set_xticks(range(1, 8))
            ax_curve.set_xticklabels([f"F{indices[k]} -> F{indices[k+1]}" for k in range(7)], fontsize=9)
            ax_curve.set_ylabel("Similarity S_t", fontsize=10)
            ax_curve.set_title("Aligned Representation Similarity Between Sampled Observation Steps", fontsize=11)
            ax_curve.grid(True, linestyle="--", alpha=0.5)
            ax_curve.legend(loc="lower right")

            plt.tight_layout()
            strip_save_path = out_strips / name
            plt.savefig(strip_save_path, dpi=200)
            plt.close()
            print(f"Saved: {strip_save_path}")

    print("\nMilestone 2 analysis deliverables successfully generated.")


if __name__ == "__main__":
    main()
