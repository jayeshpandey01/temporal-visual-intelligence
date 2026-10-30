"""
Multimodal Sensor Fusion Architecture for Spatial Computing & XR Systems (Level 3 Upgrade).
Fulfills Take-Home Assessment Section 8 and Vighnesh Inc. XR Headset Specification.

Fuses heterogeneous high-rate and low-rate streams:
  1. RGB Video Patches/CLS (DINOv2, 384-D) -> Photometric appearance & scene context
  2. 6-DoF Headset IMU (1D-ResNet, 64-D) -> Cancels camera ego-motion (Cohort B fix)
  3. Metric Depth / LiDAR (PointNet, 128-D) -> Eliminates 2D optical axis / scale ambiguity
  4. 3D Hand Pose (ST-GCN, 96-D) -> Verifies finger-object contact closure
"""

import math
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseSequenceModel, SequencePrediction


# ---------------------------------------------------------------------------
# 1. Unimodal Sensor Encoders
# ---------------------------------------------------------------------------
class IMUEncoder(nn.Module):
    """
    1D-ResNet / Conv1D encoder for 6-DoF Headset IMU (accel [3] + gyro [3]).
    Native rate: 100-200 Hz. Maps high-rate inertial stream to 64-D latent.
    """
    def __init__(self, in_channels: int = 6, out_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, out_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, imu_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imu_seq: [B, T_imu, 6] or [B, T_frames, T_sub, 6]
        Returns:
            latent: [B, out_dim] or [B, T_frames, out_dim]
        """
        if imu_seq.dim() == 4:
            B, T, Sub, C = imu_seq.shape
            x = imu_seq.view(B * T, Sub, C).transpose(1, 2)
            h = self.relu(self.bn1(self.conv1(x)))
            h = self.relu(self.bn2(self.conv2(h)))
            h = self.relu(self.bn3(self.conv3(h)))
            h = self.pool(h).squeeze(-1)
            return h.view(B, T, -1)

        # Standard 3D: [B, T_imu, 6] -> [B, 6, T_imu]
        x = imu_seq.transpose(1, 2)
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.relu(self.bn3(self.conv3(h)))
        return self.pool(h).squeeze(-1)  # [B, out_dim]


class DepthLiDAREncoder(nn.Module):
    """
    Metric Depth / LiDAR Feature Head.
    Processes metric distance deltas (Z-channel) to eliminate optical axis ambiguity.
    Output: 128-D latent.
    """
    def __init__(self, in_features: int = 128, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, depth_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_data: [B, T, in_features] or [B, in_features]
        Returns:
            latent: [B, T, out_dim] or [B, out_dim]
        """
        return self.mlp(depth_data)


class HandPoseEncoder(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network (ST-GCN) proxy for 3D Hand Pose.
    Tracks 21 3D joint coordinates [x, y, z] across time.
    Output: 96-D kinematic latent.
    """
    def __init__(self, num_joints: int = 21, coords_per_joint: int = 3, out_dim: int = 96):
        super().__init__()
        in_dim = num_joints * coords_per_joint
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, hand_joints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hand_joints: [B, T, 21, 3] or [B, T, 63]
        Returns:
            latent: [B, T, out_dim]
        """
        if hand_joints.dim() == 4:
            B, T, J, C = hand_joints.shape
            hand_joints = hand_joints.view(B, T, J * C)
        return self.proj(hand_joints)


# ---------------------------------------------------------------------------
# 2. Cross-Modal Cross-Attention Fusion Layer
# ---------------------------------------------------------------------------
class CrossModalCrossAttentionFusion(BaseSequenceModel):
    """
    Spatio-Temporal Cross-Modal Cross-Attention Fusion Engine.
    Fuses RGB visual stream, Depth LiDAR, 6-DoF IMU, and 3D Hand Pose.

    Architecture:
      1. Linear projections to shared latent space (D_fused).
      2. Learned Modality-Type Embeddings (E_type).
      3. Modality Dropout Regularization (missing sensor resilience).
      4. Multi-Head Cross-Attention (RGB as query anchor).
      5. Temporal Recurrence / Aggregation.
      6. Fast Action Classification Head (<5ms latency).
    """
    def __init__(
        self,
        rgb_dim: int = 384,
        depth_dim: int = 128,
        imu_dim: int = 64,
        pose_dim: int = 96,
        fused_dim: int = 256,
        num_classes: int = 20,
        num_heads: int = 4,
        dropout: float = 0.1,
        modality_dropout_prob: float = 0.15,
    ):
        super().__init__(embed_dim=rgb_dim, num_classes=num_classes)
        self.fused_dim = fused_dim
        self.modality_dropout_prob = modality_dropout_prob

        # Projections to shared representation space
        self.proj_rgb = nn.Linear(rgb_dim, fused_dim)
        self.proj_depth = nn.Linear(depth_dim, fused_dim)
        self.proj_imu = nn.Linear(imu_dim, fused_dim)
        self.proj_pose = nn.Linear(pose_dim, fused_dim)

        # Modality Type Embeddings (4 modalities: 0=RGB, 1=Depth, 2=IMU, 3=Pose)
        self.modality_embeddings = nn.Parameter(torch.randn(4, fused_dim) * 0.02)

        # Cross-Attention: RGB queries attend to all multimodal tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(fused_dim)
        self.norm2 = nn.LayerNorm(fused_dim)

        # Temporal GRU over fused multimodal timeline
        self.temporal_gru = nn.GRU(
            input_size=fused_dim,
            hidden_size=fused_dim,
            num_layers=1,
            batch_first=True,
        )

        # Fast action loop classifier (<5ms latency budget)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes),
        )

        # Standalone unimodal encoders for raw sensor processing
        self.imu_encoder = IMUEncoder(in_channels=6, out_dim=imu_dim)
        self.depth_encoder = DepthLiDAREncoder(in_features=depth_dim, out_dim=depth_dim)
        self.pose_encoder = HandPoseEncoder(out_dim=pose_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        imu: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multimodal forward pass.
        Args:
            rgb: [B, T, rgb_dim] visual sequence (primary anchor)
            depth: [B, T, depth_dim] metric depth deltas (optional)
            imu: [B, T, imu_dim] inertial ego-motion (optional)
            pose: [B, T, pose_dim] 3D hand skeleton (optional)
        Returns:
            logits: [B, num_classes]
        """
        B, T, _ = rgb.shape
        device = rgb.device

        # Default synthetic placeholders if auxiliary modalities are omitted
        if depth is None:
            depth = torch.zeros(B, T, 128, device=device)
        if imu is None:
            imu = torch.zeros(B, T, 64, device=device)
        if pose is None:
            pose = torch.zeros(B, T, 96, device=device)

        # Apply Modality Dropout during training for fault resilience
        if self.training and self.modality_dropout_prob > 0.0:
            if torch.rand(1).item() < self.modality_dropout_prob:
                depth = torch.zeros_like(depth)
            if torch.rand(1).item() < self.modality_dropout_prob:
                imu = torch.zeros_like(imu)
            if torch.rand(1).item() < self.modality_dropout_prob:
                pose = torch.zeros_like(pose)

        # Project modalities to shared dimension D_fused
        u_rgb = self.proj_rgb(rgb) + self.modality_embeddings[0]      # [B, T, D]
        u_depth = self.proj_depth(depth) + self.modality_embeddings[1]  # [B, T, D]
        u_imu = self.proj_imu(imu) + self.modality_embeddings[2]      # [B, T, D]
        u_pose = self.proj_pose(pose) + self.modality_embeddings[3]    # [B, T, D]

        # Multi-modal context keys/values across time: [B, 4*T, D]
        kv = torch.cat([u_rgb, u_depth, u_imu, u_pose], dim=1)

        # Cross-Attention: RGB queries attend to all sensor modalities
        q = self.norm1(u_rgb)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        fused = self.norm2(u_rgb + attn_out)  # [B, T, D]

        # Temporal modeling across fused sequence
        _, h_n = self.temporal_gru(fused)
        h_last = h_n[-1]  # [B, D]

        return self.classifier(h_last)  # [B, C]

    def predict_sequence(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        imu: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        ground_truth: int = 0,
        sequence_id: str = "multimodal_seq",
        seed: int = 42,
        class_name: Optional[str] = None,
    ) -> SequencePrediction:
        if rgb.dim() == 2:
            rgb = rgb.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            raw_logits = self(rgb, depth, imu, pose)[0].cpu().numpy()
            probs = torch.softmax(torch.from_numpy(raw_logits), dim=-1).numpy()
            probs = probs / np.sum(probs)
            pred_class = int(np.argmax(probs))
            conf = float(probs[pred_class])

        return SequencePrediction(
            sequence_id=sequence_id,
            ground_truth=ground_truth,
            predicted_class=pred_class,
            confidence=conf,
            probabilities=[float(p) for p in probs],
            model_type="multimodal_fusion",
            seed=seed,
            logits=[float(z) for z in raw_logits],
            class_name=class_name,
            metadata={"model": "CrossModalCrossAttentionFusion", "fused_dim": self.fused_dim},
        )

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

