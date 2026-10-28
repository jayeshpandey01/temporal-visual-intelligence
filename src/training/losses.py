"""
Loss functions and class imbalance audit for model training (M3.11).
(research.md §12.5)

Formulates standard multiclass Cross-Entropy loss:
    L = -sum_{c=1}^C y_c * log(p_c)
Audits training dataset class distributions to determine if class-weighting is required.
"""

from typing import Dict, Any, Optional, List
from collections import Counter
import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(
    labels: List[int],
    num_classes: int,
) -> torch.Tensor:
    """
    Computes balanced inverse-frequency class weights:
        w_c = N / (C * N_c)
    Normalized so sum(w) == C.

    Args:
        labels: List of integer class labels
        num_classes: Total number of classes C
    Returns:
        weights: torch.Tensor of shape [C]
    """
    counts = Counter(labels)
    total_samples = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        count = counts.get(c, 0)
        if count > 0:
            weights[c] = total_samples / (num_classes * count)
        else:
            weights[c] = 1.0

    return torch.from_numpy(weights)


def audit_class_distribution(
    labels: List[int],
    num_classes: int,
) -> Dict[str, Any]:
    """
    Audits the empirical class distribution of the dataset (M3.11).

    Returns:
        dict containing counts, min/max ratio, and balance determination.
    """
    counts = Counter(labels)
    count_values = [counts.get(c, 0) for c in range(num_classes)]
    min_count = min(count_values) if count_values else 0
    max_count = max(count_values) if count_values else 0
    imbalance_ratio = (max_count / min_count) if min_count > 0 else float('inf')

    # Perfectly balanced if min == max
    is_balanced = (min_count == max_count) and (min_count > 0)

    return {
        'total_samples': len(labels),
        'num_classes': num_classes,
        'min_samples_per_class': min_count,
        'max_samples_per_class': max_count,
        'imbalance_ratio': round(imbalance_ratio, 3),
        'is_perfectly_balanced': is_balanced,
        'counts_per_class': {c: counts.get(c, 0) for c in range(num_classes)},
    }


def build_criterion(
    config: Dict[str, Any],
    train_dataset: Optional[Any] = None,
) -> nn.CrossEntropyLoss:
    """
    Builds the classification loss criterion (M3.11).

    Default: Standard unweighted Cross-Entropy loss.
    If config['training'].get('class_weighting', False) is True, audits the dataset
    and computes inverse-frequency weights w_c = N / (C * N_c).
    """
    num_classes = config.get('dataset', {}).get('subset_classes', 20)
    use_class_weighting = config.get('training', {}).get('class_weighting', False)

    if use_class_weighting and train_dataset is not None:
        if hasattr(train_dataset, 'samples'):
            labels = [s[1] for s in train_dataset.samples]
        elif hasattr(train_dataset, 'labels'):
            labels = list(train_dataset.labels)
        else:
            labels = []

        if labels:
            audit = audit_class_distribution(labels, num_classes)
            if not audit['is_perfectly_balanced']:
                weights = compute_class_weights(labels, num_classes)
                print(f"  [Loss] Applied class-weighted CrossEntropy (imbalance ratio: {audit['imbalance_ratio']}x)")
                return nn.CrossEntropyLoss(weight=weights)
            else:
                print(f"  [Loss] Dataset is perfectly balanced (N_c={audit['min_samples_per_class']} for all {num_classes} classes). Using standard unweighted CrossEntropy.")

    return nn.CrossEntropyLoss()

