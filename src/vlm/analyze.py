"""
VLM Semantic Change Probe (research.md §23-25)

Treats VLM as a SEMANTIC ANALYSIS LAYER — not a replacement for the GRU.
Use to provide complementary semantic interpretation of temporal sequences.

Focus on:
  - Quadrant C examples (temporal rescued) → what does VLM see that GRU found?
  - Early vs. late frame comparison
  - Identifying failure modes: hallucination, ordering errors, missed motion

Supported backends:
  - Google Gemini API (google-generativeai)
  - OpenAI GPT-4V (openai)
  - Local (placeholder for LLaVA/InternVL)
"""

import os
import json
import base64
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import BytesIO

from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# VLM prompt template
# ---------------------------------------------------------------------------
SEQUENCE_ANALYSIS_PROMPT = """You are analyzing a sequence of video frames showing an action or event.

Frames shown: {n_frames} frames, ordered from earliest (Frame 1) to latest (Frame {n_frames}).
Ground truth label: {label}

Please answer the following questions in valid JSON format:

{{
  "what_changes_across_sequence": "Describe the key visual changes you observe from Frame 1 to Frame {n_frames}",
  "info_available_only_from_sequence": "What information becomes available only AFTER seeing multiple frames (not visible in Frame 1 alone)?",
  "is_label_ambiguous_from_frame_1": true or false,
  "confidence_from_frame_1_only": integer 0-10 (10=certain),
  "confidence_from_full_sequence": integer 0-10 (10=certain),
  "which_frame_resolves_ambiguity": integer (1-{n_frames}, or null if never resolved),
  "temporal_benefit_category": one of ["motion_cue", "state_transition", "occlusion_resolution", "direction_sensitivity", "context_accumulation", "disambiguation", "none_single_frame_sufficient"],
  "vlm_prediction": "Your prediction of the label based on the full sequence",
  "reasoning": "Brief explanation of your reasoning"
}}

Respond ONLY with valid JSON, no additional text."""


FAILURE_ANALYSIS_PROMPT = """The model made these predictions on this sequence:
- Ground truth: {true_label}
- Single-frame baseline predicted: {baseline_pred} ({'CORRECT' if baseline_correct else 'WRONG'})
- Temporal GRU predicted: {temporal_pred} ({'CORRECT' if temporal_correct else 'WRONG'})

Given the sequence of {n_frames} frames, answer in JSON:
{{
  "what_baseline_likely_saw": "What discriminative feature could the single frame show?",
  "what_temporal_model_likely_used": "What temporal pattern would help the GRU?",
  "hallucinations_in_sequence": "List any objects/actions you see that don't match the label",
  "most_confusing_aspect": "What makes this sequence difficult to classify?",
  "recommendation": "What would make this example easier to classify?"
}}"""


# ---------------------------------------------------------------------------
# Image encoding utilities
# ---------------------------------------------------------------------------
def pil_to_base64(image: Image.Image, format: str = 'JPEG', quality: int = 85) -> str:
    """Convert PIL image to base64 string for API calls."""
    buffer = BytesIO()
    image.save(buffer, format=format, quality=quality)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def load_frames_as_pil(
    frame_paths: List[str],
    target_size: tuple = (224, 224),
) -> List[Image.Image]:
    """Load frame images, resize for VLM efficiency."""
    frames = []
    for path in frame_paths:
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            img = img.resize(target_size, Image.LANCZOS)
            frames.append(img)
    return frames


# ---------------------------------------------------------------------------
# VLM backends
# ---------------------------------------------------------------------------
class GeminiVLM:
    """Google Gemini Vision backend."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install: pip install google-generativeai")

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY environment variable")

        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model

    def query(self, frames: List[Image.Image], prompt: str) -> str:
        """Send frames + prompt to Gemini, return text response."""
        content = frames + [prompt]
        response = self.model.generate_content(content)
        return response.text


class OpenAIVLM:
    """OpenAI GPT-4V backend."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install: pip install openai")

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model

    def query(self, frames: List[Image.Image], prompt: str) -> str:
        content = [{"type": "text", "text": prompt}]
        for frame in frames:
            b64 = pil_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
        )
        return response.choices[0].message.content


def get_vlm_backend(config: Dict[str, Any]):
    """Factory: select VLM backend from config."""
    backend = config.get('vlm', {}).get('backend', 'gemini')
    if backend == 'gemini':
        return GeminiVLM(model=config.get('vlm', {}).get('model', 'gemini-1.5-flash'))
    elif backend == 'openai':
        return OpenAIVLM(model=config.get('vlm', {}).get('model', 'gpt-4o'))
    else:
        raise ValueError(f"Unknown VLM backend: {backend}. Choose 'gemini' or 'openai'.")


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------
class SequenceProber:
    """Runs VLM semantic probes on temporal sequences."""

    def __init__(self, vlm_backend, config: Dict[str, Any]):
        self.vlm = vlm_backend
        self.config = config
        self.results = []

    def probe_sequence(
        self,
        sequence_id: str,
        frame_paths: List[str],
        true_label: str,
        baseline_pred: Optional[str] = None,
        temporal_pred: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run VLM probe on a single sequence.
        """
        frames = load_frames_as_pil(frame_paths)
        if not frames:
            return {'error': 'No frames loaded', 'sequence_id': sequence_id}

        prompt = SEQUENCE_ANALYSIS_PROMPT.format(
            n_frames=len(frames),
            label=true_label,
        )

        try:
            raw_response = self.vlm.query(frames, prompt)

            # Parse JSON response
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(raw_response[json_start:json_end])
            else:
                parsed = {'raw_response': raw_response}

        except json.JSONDecodeError as e:
            parsed = {'parse_error': str(e), 'raw_response': raw_response}
        except Exception as e:
            parsed = {'api_error': str(e)}

        result = {
            'sequence_id': sequence_id,
            'true_label': true_label,
            'baseline_pred': baseline_pred,
            'temporal_pred': temporal_pred,
            'n_frames': len(frames),
            'vlm_analysis': parsed,
        }

        # Detect failure modes
        result['failure_modes'] = self._detect_failure_modes(parsed, true_label)

        self.results.append(result)
        return result

    def _detect_failure_modes(self, analysis: Dict, true_label: str) -> List[str]:
        """Identify VLM failure modes in response."""
        failures = []
        if 'parse_error' in analysis or 'api_error' in analysis:
            return ['api_failure']

        vlm_pred = analysis.get('vlm_prediction', '').lower()
        true = true_label.lower()

        if vlm_pred and true not in vlm_pred and vlm_pred not in true:
            failures.append('incorrect_label')

        if analysis.get('confidence_from_frame_1_only', 0) >= 8:
            failures.append('overconfident_single_frame')

        hallucinations = analysis.get('hallucinations_in_sequence', '')
        if hallucinations and len(hallucinations) > 20:
            failures.append('possible_hallucination')

        return failures

    def save_results(self, output_path: str):
        """Save all probe results to JSONL file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for result in self.results:
                f.write(json.dumps(result) + '\n')
        print(f"VLM probe results saved → {output_path}")
        print(f"  Total probed: {len(self.results)}")

    def summarize(self) -> Dict[str, Any]:
        """Compute summary statistics over all probe results."""
        if not self.results:
            return {}

        vlm_correct = []
        temporal_benefit_categories = []
        ambiguous_from_f1 = []

        for r in self.results:
            analysis = r.get('vlm_analysis', {})
            if isinstance(analysis, dict) and 'parse_error' not in analysis:
                vlm_pred = analysis.get('vlm_prediction', '')
                true = r['true_label']
                vlm_correct.append(int(true.lower() in vlm_pred.lower()))
                cat = analysis.get('temporal_benefit_category', 'other')
                temporal_benefit_categories.append(cat)
                ambiguous_from_f1.append(analysis.get('is_label_ambiguous_from_frame_1', False))

        from collections import Counter
        return {
            'n_probed': len(self.results),
            'vlm_accuracy': float(np.mean(vlm_correct)) if vlm_correct else None,
            'temporal_benefit_distribution': dict(Counter(temporal_benefit_categories)),
            'pct_ambiguous_from_frame_1': float(np.mean(ambiguous_from_f1)) if ambiguous_from_f1 else None,
        }

