Your M0 spec is well aligned with several established lines of research in video understanding. The most useful papers are those that separate **more frames**, **frame order**, and **robustness to missing/corrupted observations**—exactly the distinctions in your H1–H5.

## Most relevant papers

| Paper | Relevance to M0 | What to take from it |
|---|---|---|
| **Shuffle and Learn: Unsupervised Learning Using Temporal Order Verification** (Misra et al., ECCV 2016) | Directly supports Experiment B | Frames are classified as correctly or incorrectly ordered; useful precedent for using shuffled-frame evaluation as an order-sensitivity probe.  [link.springer](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_32) |
| **Uncovering Temporal Context for Video Question and Answering** (Zeng et al., 2015) | Supports Model C and H2/H3 | Uses a GRU-based encoder-decoder over sampled video frames; reports that GRUs help especially for past inference and future prediction, where temporal structure matters.  [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1511.04670) |
| **Temporal Shuffling for Defending Deep Action Recognition Models against Adversarial Attacks** (2021) | Important caution for H2/H4 | Shows some action-recognition models remain fairly strong after frame-order randomization, meaning a small shuffle drop does not automatically prove deep temporal reasoning.  [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2112.07921) |
| **Temporal Reasoning Transfer from Text to Video** (2024) | Strong support for your evaluation design | Explicitly evaluates temporal **order**, attribute change, temporal referring, and grounding; finds Video LLMs can struggle with order even when simple probes on visual embeddings perform well.  [export.arxiv](https://export.arxiv.org/pdf/2410.06166) |
| **Improve Temporal Reasoning in Multimodal LLMs via Video Contrastive Decoding** (NeurIPS 2025) | Relevant to your VLM component | Uses deliberately distorted temporal consistency during decoding to expose and reduce time-insensitive errors in video LLMs.  [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ea6d17af54f827336fc8fed27ca0319d-Abstract-Conference.html) |
| **TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos** (2024) | Relevant to temporal feature modeling | Samples clips with timestamps, fuses local temporal features, then uses a BiLSTM for global sequence aggregation—analogous in spirit to your GRU over frozen frame embeddings.  [arxiv](https://arxiv.org/html/2412.02930v4) |
| **Temporal Chain of Thought: Long-Video Understanding by Thinking in Frames** (NeurIPS 2025) | Relevant to progressive observation / frame sampling | Selects relevant frames adaptively for video QA rather than treating all frames equally; useful background for your online-prefix versus uniform-sampling distinction.  [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2025/file/d285ab51e167a923f07e6137d802305a-Paper-Conference.pdf) |
| **TEMPURA: Temporal Event Masked Prediction and Understanding for Reasoning in Action** (2025) | Optional extension context | Uses masked-event prediction to force models to infer missing events and event dependencies—conceptually related to your missing/corrupted-frame experiment.  [arxiv](https://arxiv.org/html/2505.01583) |
| **Causality Matters: How Temporal Information Emerges in Video Language Models** (2025) | Useful for interpretation rules | Finds that reversing frame order while preserving positional encodings causes substantial performance drops, and analyzes how inter-frame attention builds temporal cues.  [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2508.11576) |

## How they map to M0

### H1 — More observations help
Use **Temporal Chain of Thought** and **TemporalVLM** as background for why the number and relevance of sampled frames can affect performance. Your design is stronger because you distinguish an **online prefix**—first K frames—from a **frame-count control** using K uniformly sampled frames. [arxiv](https://arxiv.org/html/2412.02930v4)

### H2 — Correct order matters
**Shuffle and Learn** is the clearest precedent for treating temporal-order verification as a meaningful task. However, **Temporal Shuffling** warns that shuffled inputs may become out-of-distribution and that some models are surprisingly robust to order disruption; therefore, report a shuffle-related drop as evidence of *order sensitivity*, not proof of general temporal reasoning. [link.springer](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_32)

### H3 — Order versus extra visual evidence
Your mean-pooled multi-frame control is essential. The GRU-based video QA work supports the choice of a recurrent temporal model, while the temporal-reasoning probe study shows that visual embeddings may contain order-relevant information even when a full video model fails to exploit it. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1511.04670)

### H4 — Robustness to degraded frames
**TEMPURA** is conceptually relevant because it trains models to infer missing events from surrounding temporal context. Your experiment is more controlled: hold the corruption type, severity, and affected-frame rule fixed, then compare relative degradation between the single-frame and temporal models. [arxiv](https://arxiv.org/html/2505.01583)

### H5 — Observation saturation
No single paper above directly defines your saturation curve, but **Temporal Chain of Thought** supports the idea that not all frames contribute equally and that frame selection can matter. Your contribution should be descriptive: plot macro-F1 against K and identify where additional frames stop yielding meaningful gains. [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2025/file/d285ab51e167a923f07e6137d802305a-Paper-Conference.pdf)

## Recommended citation set

For the M0 report, cite these five first:

1. Misra et al., *Shuffle and Learn* — temporal-order verification. [link.springer](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_32)
2. Zeng et al., *Uncovering Temporal Context for Video QA* — GRU-based temporal modeling. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1511.04670)
3. *Temporal Shuffling for Defending Action Recognition Models* — caution about interpreting shuffle results. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2112.07921)
4. *Temporal Reasoning Transfer from Text to Video* — order, temporal grounding, and VLM limitations. [export.arxiv](https://export.arxiv.org/pdf/2410.06166)
5. *Temporal Chain of Thought* — frame selection and progressive evidence. [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2025/file/d285ab51e167a923f07e6137d802305a-Paper-Conference.pdf)

Then add **Video Contrastive Decoding** and **TEMPURA** only in the VLM/robustness sections, since they are newer and more model-specific. [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ea6d17af54f827336fc8fed27ca0319d-Abstract-Conference.html)

## Practical framing for your report

A concise related-work paragraph could be:

> Prior work has used temporal-order verification to learn or evaluate video representations, while GRU-based models have been applied to video QA to exploit sequential context. However, temporal shuffling can introduce distribution shift, and some action-recognition models remain robust to randomized frame order. Recent work also shows that video-language models often struggle with explicit temporal reasoning despite informative visual embeddings. Our study therefore separates additional visual evidence from temporal structure using a single-frame baseline, an order-invariant pooled control, and a GRU-based temporal model. [link.springer](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_32)

This gives M0 a credible literature foundation without overstating what a small, controlled experiment can prove.