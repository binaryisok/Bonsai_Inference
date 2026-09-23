# Nemotron-3-Diarization Inference & Development Iteration Log

## Iteration 1: Initial Architecture Compatibility Test
- **Target**: Run inference with default pre-installed libraries in Colab runtime (`transformers 5.17.0`).
- **Execution**: Initialized `AutoProcessor` and attempted `AutoModelForAudioFrameClassification.from_pretrained("nvidia/Nemotron-3-Diarization")`.
- **Result**: **FAILED** with `ValueError: The checkpoint you are trying to load has model type nemotron3_diarization but Transformers does not recognize this architecture.`
- **Root Cause**: `nvidia/Nemotron-3-Diarization` is an open-weight model released on September 23, 2026. The `nemotron3_diarization` model architecture was added in `transformers 5.18.0.dev0` (main branch). The older 5.17.0 version could not recognize the model type.
- **Commit**: `fix(diarization-iter-1): Transformers 5.17.0 lacks nemotron3_diarization architecture; upgrade to git+https://github.com/huggingface/transformers.git`

---

## Iteration 2: Source Installation & Model GPU Loading
- **Target**: Install latest transformers from upstream git source, restart Jupyter kernel, and verify model weight loading on Tesla T4 GPU.
- **Execution**:
  - Ran `pip install -U git+https://github.com/huggingface/transformers.git accelerate soundfile librosa` on the Colab instance.
  - Restarted the kernel via Colab Studio API (`POST /api/sessions/diarization-test/restart`).
  - Tested `AutoModelForAudioFrameClassification.from_pretrained("nvidia/Nemotron-3-Diarization", device_map="cuda", torch_dtype=torch.float32)`.
- **Result**: **SUCCESSFUL**
  - Transformers upgraded to `5.18.0.dev0`.
  - Model loaded successfully to `cuda:0` with 99.2M parameters.
- **Observations**:
  - Warning surfaced: `[transformers] torch_dtype is deprecated! Use dtype instead!`.
  - Feature extractor expects 16,000 Hz audio sampling rate.

---

## Iteration 3: Full End-to-End Audio Diarization on `ben10.mp3`
- **Target**: Run end-to-end diarization pipeline on `/content/ben10.mp3` (4.5 min Cartoon Network audio), evaluate RTFx speed, and generate RTTM & JSON outputs.
- **Execution**:
  - Uploaded `ben10.mp3` (2.7 MB) and `inference_diarization.py` to Colab VM.
  - Applied FFmpeg audio normalization: extracted 16kHz mono audio (`-vn -ac 1 -ar 16000`), discarding the embedded cover art video stream.
  - Fed normalized audio tensor into `Nemotron3DiarizationForAudioFrameClassification`.
  - Generated frame logits and converted to speaker segments via `processor.extract_speaker_dict()`.
- **Result**: **SUCCESSFUL**
  - Total Audio Duration: **268.45 seconds (~4.5 minutes)**
  - GPU Inference Time: **2.29 seconds**
  - Real-Time Factor (RTFx): **117.15x faster than real-time**
  - Speakers Detected: **4 unique speakers** (Speaker 0, Speaker 1, Speaker 2, Speaker 3)
  - Total Segments: **36 speech turns**
  - Speaking Time Breakdown:
    - Speaker 0: 1.82s (0.7%)
    - Speaker 1: 26.04s (9.7%)
    - Speaker 2: 15.78s (5.9%)
    - Speaker 3: 13.83s (5.2%)
  - Clean RTTM (`ben10_diarization.rttm`) and JSON (`ben10_diarization.json`) downloaded and verified locally.
