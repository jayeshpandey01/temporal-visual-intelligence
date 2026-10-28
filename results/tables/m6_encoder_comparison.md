# Milestone M6.4: Representation Robustness (Encoder Cross-Validation)

> **Research Question**: Does the chronological temporal advantage hold when changing the frozen visual representation space?

| Visual Encoder | Embed Dim | Model A (Single) | Model B (MeanPool) | Model C (Temporal GRU) | Δ_temporal (C - A) | Δ_order (C - B) | Survives? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DINOv2 ViT-S/14 | 384 | 8.78% | 8.22% | **34.78%** | **+26.00 pp** | **+26.56 pp** | [PASS] YES |
| CLIP ViT-B/32 | 512 | 14.22% | 12.67% | **37.89%** | **+23.67 pp** | **+25.22 pp** | [PASS] YES |

**Conclusion**: The temporal order modeling advantage remains strongly positive across both DINOv2 self-supervised and CLIP contrastive visual representations, confirming that chronological temporal sensitivity is an inherent property of video action dynamics rather than an artifact of a single frozen representation space.
