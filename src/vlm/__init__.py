"""
VLM semantic interpretation and annotation package for Milestone M5.
"""

from src.vlm.annotator import (
    STRUCTURED_VLM_PROMPT_TEMPLATE,
    locate_sequence_frame_paths,
    load_frames_as_pil,
    OfflineKinematicAnnotator,
    LiveGeminiAnnotator,
    VLMAnnotationPipeline,
    audit_vlm_annotations,
)
from src.vlm.analyze import (
    SequenceProber,
    GeminiVLM,
    OpenAIVLM,
    get_vlm_backend,
)

from src.vlm.human_verification import (
    compute_cohens_kappa,
    bootstrap_kappa_and_agreement,
    HumanVerificationAuditor,
    audit_human_verification,
)

__all__ = [
    "STRUCTURED_VLM_PROMPT_TEMPLATE",
    "locate_sequence_frame_paths",
    "load_frames_as_pil",
    "OfflineKinematicAnnotator",
    "LiveGeminiAnnotator",
    "VLMAnnotationPipeline",
    "audit_vlm_annotations",
    "compute_cohens_kappa",
    "bootstrap_kappa_and_agreement",
    "HumanVerificationAuditor",
    "audit_human_verification",
    "SequenceProber",
    "GeminiVLM",
    "OpenAIVLM",
    "get_vlm_backend",
]
