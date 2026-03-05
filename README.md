# speaker_diarization
# Speaker Diarization Pipeline

Offline, production-ready speaker diarization using **Silero VAD** and **TitaNet** — both running as local ONNX models with no cloud dependencies.

---

## Table of Contents

- [How the Pipeline Works](#how-the-pipeline-works)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Running with Docker](#running-with-docker)
- [Output Format](#output-format)
- [Configuration Reference](#configuration-reference)
- [Design Decisions and Trade-offs](#design-decisions-and-trade-offs)
- [Known Issues and Limitations](#known-issues-and-limitations)
- [What I Would Improve Given More Time](#what-i-would-improve-given-more-time)

---

## How the Pipeline Works

```
Audio (.wav)
     │
     ▼
┌─────────────────────────────┐
│  Step 1 — VAD               │  Silero VAD (ONNX)
│  Detect speech segments     │  Chunks audio into speech / non-speech
└────────────┬────────────────┘
             │  speech timestamps (start/end in seconds)
             ▼
┌─────────────────────────────┐
│  Step 2 — Embedding         │  TitaNet Large (ONNX)
│  Extract speaker embeddings │  Log-mel → 192-dim speaker vector per segment
│  Parallel across N threads  │  Segments < 0.5s are skipped
└────────────┬────────────────┘
             │  L2-normalised embedding per segment
             ▼
┌─────────────────────────────┐
│  Step 3 — Clustering        │  Agglomerative (average linkage, cosine)
│  Assign speaker labels      │  Distance threshold = 1 - similarity_threshold
└────────────┬────────────────┘
             │  speaker label per segment
             ▼
┌─────────────────────────────┐
│  Step 4 — Output            │  RTTM + CTM files
│  Write results              │  Rich console report with timeline + summary
└─────────────────────────────┘
```

**Key design choice:** VAD runs first to avoid running the expensive TitaNet model on silence. Only speech segments go through embedding extraction.

---

## Project Structure

```
diarization/
├── diarize.py                   # Entry point — orchestrates all 4 steps
├── Dockerfile                   # Production container definition
│
├── vad/
│   └── silero_vad_offline.py    # Vendored Silero VAD — no silero_vad pip package needed
│
├── embeddings/
│   └── titanet.py               # TitaNet ONNX embedder with parallel extraction
│
├── clustering/
│   └── agglomerative.py         # Cosine agglomerative clustering
│
└── output/
    └── reporter.py              # RTTM + CTM writer + console report
```

---

## Quick Start

### Prerequisites

```bash
python 3.13.1
# Python dependencies
pip install torch==2.9.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu121

pip install onnxruntime-gpu==1.24.2 onnxruntime librosa==0.11.0 soundfile==0.13.1 \
    numpy==2.2.6 scipy==1.16.3 packaging==24.2
```

### Models required

| Model | Purpose | Download |
|---|---|---|
| `silero_vad.onnx` | Voice activity detection | `wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx` |
| `titanet_large.onnx` | Speaker embeddings | https://github.com/NutanChoudhary/speaker_diarization/releases/tag/itanet_large.onnx | export code is provided in model_export.py

### Run

```bash
python3 diarize.py \
    --audio   /path/to/audio.wav \
    --silero  /path/to/silero_vad.onnx \
    --titanet /path/to/titanet_large.onnx \
    --outdir  /path/to/output/
```

Or edit the `CONFIG` block at the top of `diarize.py` and run without arguments:

```bash
python3 diarize.py \
    --batch-dir   /data/test_audio\
    --silero  /path/to/silero_vad.onnx \
    --titanet /path/to/titanet_large.onnx \
    --outdir  /path/to/output/
```

---

## Running with Docker

## Download Models

Download the required ONNX models and place them in a `models/` folder: silero_vad.onnx

wget -O docker_speaker_diarization/models/titanet_large.onnx https://github.com/NutanChoudhary/speaker_diarization/releases/tag/itanet_large.onnx

### Step 1 — Pre-download wheels (one-time, on a machine with internet)

Because the build server may not have internet access, download all wheels first on the host:

```bash
mkdir -p wheels

# PyTorch with CUDA
pip download torch==2.9.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    -d wheels

# Everything else
pip download onnxruntime-gpu==1.24.1 librosa==0.11.0 soundfile==0.13.1 \
    numpy==2.2.6 scipy==1.16.3 packaging==24.2 \
    -d wheels
```

### Step 2 — Build

```bash
docker build -t diarization:latest .


### Step 3 — Run

```bash
docker run --gpus 1 --rm \
  -v docker_speaker_diarization/models:/models:ro \
  -v /mnt/disk5/nutan:/data \
  diarization:latest \
    --batch-dir  /test_audios \
    --silero     /models/silero_vad.onnx \
    --titanet    /models/titanet_large.onnx \
    --outdir     /data/diarization_output
```

### Volume mount layout

| Host path | Container path | Purpose |
|---|---|---|
| `/your/models/` | `/models` (read-only) | ONNX model files |
| `/your/audio/` | `/data/input` (read-only) | Input audio files |
| `/your/output/` | `/data/output` | RTTM + CTM output files |


## Output Format

### RTTM (Rich Transcription Time Marks) — primary output

Standard NIST format, compatible with `pyannote`, `dscore`, and `md-eval`:

```
SPEAKER audio_name 1 0.066 3.996 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER audio_name 1 5.154 3.068 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER audio_name 1 9.314 1.180 <NA> <NA> SPEAKER_02 <NA> <NA>
```

Fields: `SPEAKER <file_id> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>`

### CTM (Conversation Time Marks) — secondary output

```
audio_name A 0.066 3.996 SPEAKER_00 1.000
audio_name A 5.154 3.068 SPEAKER_00 1.000
```

### Console report

Printed to stdout on every run — includes per-speaker summary with percentage share, and an overlap analysis section.

---

## Configuration Reference

All parameters can be set in the `CONFIG` block in `diarize.py` or passed as CLI flags:

| Parameter | Default | Description |
|---|---|---|
| `vad_threshold` | `0.5` | Speech probability cutoff. Raise for noisy audio. |
| `min_speech_duration_ms` | `250` | Discard speech segments shorter than this. |
| `min_silence_duration_ms` | `100` | Minimum silence gap to split two speech segments. |
| `speech_pad_ms` | `30` | Padding added to each side of a speech segment. |
| `similarity_threshold` | `0.75` | Cosine similarity above which two segments are the same speaker. Raise → fewer speakers. Lower → more speakers. |
| `min_segment_duration` | `0.5` | Skip segments shorter than this for embedding (too short to be reliable). |
| `num_embedding_workers` | `4` | Parallel threads for TitaNet inference. |

---

## Design Decisions and Trade-offs

**TitaNet over simpler speaker models**
TitaNet Large produces high-quality 192-dimensional embeddings trained specifically for speaker verification. The trade-off is model size (~97 MB ONNX) and inference time versus lighter alternatives like ECAPA-TDNN. For production accuracy, TitaNet is the right call.

**Agglomerative clustering over spectral clustering**
Agglomerative clustering with average linkage requires no prior knowledge of speaker count and is deterministic. Spectral clustering can produce better boundaries but requires estimating the number of speakers first (e.g. via eigenvalue analysis), adding complexity. For a first-pass production system, agglomerative is the safer choice.

---

## Known Issues and Limitations

- **Speaker count is not capped.** If `similarity_threshold` is set too low, every segment gets its own speaker label. A reasonable range is 0.65–0.85 depending on audio quality.
- **Overlapping speech is detected but not separated.** The overlap analysis in the report flags regions where two speaker turns overlap in time, but the embeddings are not decomposed — the segment is assigned to one speaker only.
- **Short segments are unreliable.** Segments under 0.5s are skipped for embedding. If a speaker only appears in very short bursts they will be missed.
- **No speaker re-identification across files.** Each run is independent. There is no persistent speaker database, so `SPEAKER_00` in one file is not guaranteed to be the same person as `SPEAKER_00` in another.


## What I Would Improve Given More Time

**1. Automatic speaker count estimation**
Currently the number of speakers is inferred purely from the clustering threshold. Adding eigenvalue gap analysis on the affinity matrix would give a data-driven estimate of the true speaker count before clustering, reducing sensitivity to the `similarity_threshold` parameter.

**2. Overlap-aware diarization**
The current pipeline assigns each segment to exactly one speaker. A secondary overlap detection pass (e.g. using a separate classifier on the VAD probabilities) would allow flagging overlapping speech regions and assigning multiple labels, which matters for meeting transcription accuracy.

**3. Streaming / incremental processing**
For long recordings (>30 min), loading the full audio into memory is inefficient. The VAD module already supports streaming via `VADIterator` — wiring this into the main pipeline with chunk-based embedding extraction would reduce peak memory from O(audio_length) to O(chunk_size).

**4. Speaker re-identification across files**
A persistent embedding database (e.g. FAISS index) would allow the pipeline to assign consistent speaker identities across multiple files or sessions — useful for meeting series or multi-session interviews.

**5. Confidence scores in output**
The RTTM and CTM files currently have no confidence information. Attaching the within-cluster cosine similarity as a confidence score to each turn would make downstream filtering and human review much easier.

**7. Replace agglomerative clustering with spectral clustering**
For recordings with a known or estimable number of speakers, spectral clustering on the cosine affinity matrix consistently outperforms agglomerative in DER benchmarks. The main blocker is robustly estimating speaker count first (see point 1).
