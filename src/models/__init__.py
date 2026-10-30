"""
Sequence model architectures and prediction contracts for temporal visual intelligence.
"""

from src.models.base import BaseSequenceModel, SequencePrediction
from src.models.baseline import (
    SingleFrameMLP,
    MultiFrameMeanPool,
    MajorityClassBaseline,
    build_baseline,
    evaluate_anchor_sweep,
)
from src.models.temporal import (
    TemporalGRU,
    TemporalTransformer,
    TemporalAttentionPooling,
    AttentiveBiGRU,
    build_temporal_model,
)
from src.models.timesformer import TimeSformerModel
from src.models.multimodal_fusion import (
    CrossModalCrossAttentionFusion,
    IMUEncoder,
    DepthLiDAREncoder,
    HandPoseEncoder,
)

__all__ = [
    "BaseSequenceModel",
    "SequencePrediction",
    "SingleFrameMLP",
    "MultiFrameMeanPool",
    "MajorityClassBaseline",
    "TemporalGRU",
    "TemporalTransformer",
    "TemporalAttentionPooling",
    "AttentiveBiGRU",
    "TimeSformerModel",
    "CrossModalCrossAttentionFusion",
    "IMUEncoder",
    "DepthLiDAREncoder",
    "HandPoseEncoder",
    "build_baseline",
    "build_temporal_model",
    "evaluate_anchor_sweep",
    "build_single_frame_model",
    "build_mean_pool_model",
    "build_gru_model",
    "CompactTemporalGRU",
    "build_compact_gru_model",
]

from src.models.single_frame import build_single_frame_model
from src.models.mean_pool import build_mean_pool_model
from src.models.gru import build_gru_model
from src.models.compact_gru import CompactTemporalGRU, build_compact_gru_model
