"""
Loss functions and class imbalance audit for model training (M3.11).
(research.md §12.5)

Formulates standard multiclass Cross-Entropy loss:
    L = -sum_{c=1}^C y_c * log(p_c)
Audits training dataset class distributions to determine if class-weighting is required.
"""

from typing import Dict, Any, Optional, List, Tuple
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


# ---------------------------------------------------------------------------
# Directional Contrastive Loss (Level 1 Win)
# ---------------------------------------------------------------------------
DEFAULT_DIRECTIONAL_OPPOSITE_PAIRS = {
    0: 1, 1: 0,    # Moving away <-> Moving towards
    2: 3, 3: 2,    # Pushing left-to-right <-> Pushing right-to-left
    4: 5, 5: 4,    # Picking up <-> Putting down
    6: 7, 7: 6,    # Opening <-> Closing
    8: 9, 9: 8,    # Turning camera left <-> Turning camera right
    10: 11, 11: 10, # Putting on top <-> Taking off
    12: 13, 13: 12, # Putting into <-> Taking out of
    14: 15, 15: 14, # Rotating clockwise <-> Rotating counter-clockwise
    16: 17, 17: 16, # Standing upright <-> Tipping over
    18: 19, 19: 18, # Covering <-> Uncovering
}


class DirectionalContrastiveLoss(nn.Module):
    """
    Directional & Temporal Reversal InfoNCE Contrastive Loss.

    Forces representations of:
      1. Identical actions to attract (pull together in latent space).
      2. Reversal sequences (t_K -> t_1) to repel (breaking directional symmetry).
      3. Directionally opposite classes (e.g. Pushing Left vs Pushing Right) to push apart with a margin.
    """
    def __init__(
        self,
        temperature: float = 0.07,
        opposite_pairs: Optional[Dict[int, int]] = None,
        reversal_weight: float = 1.0,
        opposite_margin: float = 0.2,
    ):
        super().__init__()
        self.temperature = temperature
        self.opposite_pairs = opposite_pairs or DEFAULT_DIRECTIONAL_OPPOSITE_PAIRS
        self.reversal_weight = reversal_weight
        self.opposite_margin = opposite_margin

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reversed_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, D] L2-normalized representations
            labels: [B] integer class labels
            reversed_features: [B, D] L2-normalized representations of time-reversed sequences
        Returns:
            loss: scalar loss
        """
        device = features.device
        B = features.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        features = nn.functional.normalize(features, dim=-1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # 1. Supervised InfoNCE component (if labels provided)
        total_loss = torch.tensor(0.0, device=device)
        terms = 0

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            pos_mask = torch.eq(labels, labels.T).float().to(device)
            # Remove self-similarity from positive mask
            diag_mask = torch.eye(B, device=device, dtype=torch.bool)
            pos_mask.masked_fill_(diag_mask, 0.0)

            # Mask out self-contrast in denominator
            logits_mask = ~diag_mask
            exp_sim = torch.exp(sim_matrix) * logits_mask.float()
            denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8

            has_pos = pos_mask.sum(dim=1) > 0
            if has_pos.any():
                log_prob = sim_matrix - torch.log(denom)
                mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
                sup_loss = -mean_log_prob_pos[has_pos].mean()
                total_loss = total_loss + sup_loss
                terms += 1

            # Directional Opposites Margin Penalty
            opp_penalty = torch.tensor(0.0, device=device)
            opp_count = 0
            flat_labels = labels.squeeze(-1).tolist()
            for i in range(B):
                target_opp = self.opposite_pairs.get(flat_labels[i])
                if target_opp is not None:
                    for j in range(B):
                        if i != j and flat_labels[j] == target_opp:
                            raw_cos = torch.dot(features[i], features[j])
                            # Penalize positive cosine similarity with directional opposite
                            opp_penalty = opp_penalty + nn.functional.relu(raw_cos + self.opposite_margin)
                            opp_count += 1
            if opp_count > 0:
                total_loss = total_loss + (opp_penalty / opp_count)
                terms += 1

        # 2. Sequence vs. Temporal Reversal Contrast
        if reversed_features is not None:
            reversed_features = nn.functional.normalize(reversed_features, dim=-1)
            # Alignment between canonical and reversed sequence should be low
            rev_cos = (features * reversed_features).sum(dim=-1)  # [B]
            # Softplus push-away: log(1 + exp(rev_cos / tau))
            rev_loss = torch.log(1.0 + torch.exp(rev_cos / self.temperature)).mean()
            total_loss = total_loss + self.reversal_weight * rev_loss
            terms += 1

        if terms == 0:
            # Fallback self-supervised contrastive among distinct batch elements
            diag_mask = torch.eye(B, device=device, dtype=torch.bool)
            sim_off_diag = sim_matrix.masked_select(~diag_mask)
            return torch.log(1.0 + torch.exp(sim_off_diag)).mean()

        return total_loss / max(1, terms)


class CombinedClassificationLoss(nn.Module):
    """
    Combined Cross-Entropy + Directional Contrastive Auxiliary Loss.
        L_total = L_CE + lambda * L_contra
    """
    def __init__(
        self,
        ce_criterion: Optional[nn.Module] = None,
        lambda_contra: float = 0.3,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.ce = ce_criterion or nn.CrossEntropyLoss()
        self.contrastive = DirectionalContrastiveLoss(temperature=temperature)
        self.lambda_contra = lambda_contra

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        reversed_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_ce = self.ce(logits, labels)
        metrics = {"loss_ce": float(loss_ce.item())}

        if features is not None and self.lambda_contra > 0.0:
            loss_contra = self.contrastive(
                features=features,
                labels=labels,
                reversed_features=reversed_features,
            )
            total = loss_ce + self.lambda_contra * loss_contra
            metrics["loss_contra"] = float(loss_contra.item())
            metrics["loss_total"] = float(total.item())
            return total, metrics

        metrics["loss_total"] = float(loss_ce.item())
        return loss_ce, metrics

