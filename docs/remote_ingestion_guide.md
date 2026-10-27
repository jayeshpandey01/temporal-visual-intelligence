# Milestone 1.5 — Real Video Pipeline Guide (Hugging Face & Kaggle Ingestion)

This guide documents the automated ingestion, container decoding, and frame extraction pipeline for real action videos from **Hugging Face Hub** and **Kaggle**.

---

## 1. Architecture & Capabilities

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       REMOTE DATASET REPOSITORIES                           │
│     Hugging Face Hub (Datasets API)            Kaggle Datasets API          │
│   (e.g., sayakpaul/ssv2-mini, ucf101)        (e.g., pevogam/ssv2)           │
└───────────────────────┬──────────────────────────────────────┬──────────────┘
                        │                                      │
                        ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 REMOTE INGESTION ENGINE (src/data/remote_ingestion.py)       │
│               - API Authentication (HF_TOKEN, ~/.kaggle/kaggle.json)         │
│               - Targeted archive streaming / snapshot download              │
│               - Checksum verification & data/raw/ immutability               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     UNIVERSAL FFMPEG CONTAINER DECODER                      │
│        - Supports MP4, AVI, WebM, MOV, and animated archives                │
│        - Uniform temporal subsampling: T = 8 or 16 frames                    │
│        - ImageNet spatial normalization: 224 x 224 center-cropped RGB        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STANDARDIZED DATASET ROOT                           │
│             data/{class_name}/{video_id}/frame_{index:04d}.jpg              │
│         - 7-Check Data Integrity Audit (src/data/audit.py)                   │
│         - Leakage-Safe Group Partitioning (src/data/dataset.py)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Ingesting from Hugging Face Hub

### Prerequisites
`huggingface_hub` is pre-installed. For private or gated datasets, export your token:
```bash
export HF_TOKEN="your_hf_token_here"
```

### Ingestion Commands
To download and ingest real video action sequences from a Hugging Face dataset repository:
```bash
# 1. Download and extract from Hugging Face dataset snapshot
python3 scripts/prepare.py \
  --dataset real-video \
  --source hf \
  --repo-id "sayakpaul/something-something-v2-mini" \
  --n-per-class 100 \
  --audit

# 2. Or download a specific archive file from a repository
python3 scripts/prepare.py \
  --dataset real-video \
  --source hf \
  --repo-id "username/video-dataset" \
  --filename "train_videos.tar.gz" \
  --audit
```

---

## 3. Ingesting from Kaggle

### Prerequisites
1. Install Kaggle CLI (if not present):
   ```bash
   pip install kaggle
   ```
2. Configure credentials in `~/.kaggle/kaggle.json`:
   ```json
   {"username": "your_kaggle_username", "key": "your_kaggle_api_key"}
   ```
   Ensure file permissions: `chmod 600 ~/.kaggle/kaggle.json`.

### Ingestion Command
```bash
python3 scripts/prepare.py \
  --dataset real-video \
  --source kaggle \
  --repo-id "pevogam/something-something-v2" \
  --n-per-class 100 \
  --audit
```

---

## 4. Ingesting from Local Raw Videos

If you already have a folder of raw `.mp4` or `.avi` files organized by class:
```bash
python3 scripts/prepare.py \
  --dataset real-video \
  --source local \
  --input-dir "path/to/raw_videos" \
  --audit
```

The pipeline automatically:
1. Walks the directories to identify video containers.
2. Invokes `/opt/homebrew/bin/ffmpeg` to decode uniform frame sequences at $224 \times 224$.
3. Saves to `data/{class_name}/{video_id}/frame_{0001..0008}.jpg`.
4. Runs the 7-check data integrity audit and saves temporal strips to `results/figures/audit_strips.png`.
5. Enforces GroupKFold video-level partitioning via `data/metadata/split_manifest.json`.

