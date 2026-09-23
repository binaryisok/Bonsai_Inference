#!/usr/bin/env python3
"""
Nemotron-3-Diarization Inference Pipeline
Model: nvidia/Nemotron-3-Diarization
Architecture: Streaming / Offline Sortformer Frame Classification (up to 8 speakers)
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
import torch

def preprocess_audio(input_path: str, output_wav: str = "/content/ben10_16k.wav") -> str:
    """
    Ensure audio is 16kHz mono WAV without video/cover streams.
    Nemotron-3-Diarization requires 16000Hz mono audio.
    """
    input_path = str(input_path)
    output_wav = str(output_wav)
    print(f"[*] Preprocessing audio: {input_path} -> {output_wav}")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn",             # disable video / attached pictures
        "-ac", "1",        # mono
        "-ar", "16000",    # 16kHz sample rate
        "-f", "wav",
        output_wav
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg audio conversion failed:\n{res.stderr}")
    
    # Verify duration
    probe_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", output_wav
    ]
    duration = float(subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip())
    print(f"[*] Audio converted successfully. Duration: {duration:.2f} seconds.")
    return output_wav

def run_diarization_transformers(audio_path: str, model_id: str = "nvidia/Nemotron-3-Diarization", device: str = "cuda"):
    """
    Run diarization using native Hugging Face Transformers pipeline.
    """
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor
    from transformers.audio_utils import load_audio

    print(f"[*] Loading model & processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    model = AutoModelForAudioFrameClassification.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device
    )
    model.eval()
    print(f"[*] Model loaded on {model.device} with dtype {model.dtype}")

    sr = processor.feature_extractor.sampling_rate
    print(f"[*] Loading audio with target sampling rate: {sr} Hz...")
    audio = load_audio(audio_path, sampling_rate=sr)
    print(f"[*] Audio waveform loaded: shape {audio.shape}, length {audio.shape[0] / sr:.2f}s")

    print("[*] Feature extraction...")
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(model.device, dtype=model.dtype)

    print("[*] Running inference...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (batch_size, num_frames, 8)
    infer_time = time.perf_counter() - t0
    print(f"[*] Inference finished in {infer_time:.2f}s (RTFx: {(audio.shape[0] / sr) / infer_time:.2f}x)")

    print("[*] Extracting speaker segments...")
    segments = processor.extract_speaker_dict(logits, inputs.get("attention_mask"))[0]
    return segments, audio.shape[0] / sr, infer_time

def run_diarization_nemo(audio_path: str, model_id: str = "nvidia/Nemotron-3-Diarization"):
    """
    Run diarization using NVIDIA NeMo Speech SortformerEncLabelModel.
    """
    from nemo.collections.asr.models import SortformerEncLabelModel
    print(f"[*] Loading NeMo model: {model_id}...")
    diar_model = SortformerEncLabelModel.from_pretrained(model_id)
    diar_model.eval()

    # Offline / high accuracy configuration
    diar_model.sortformer_modules.chunk_len = 340
    diar_model.sortformer_modules.chunk_right_context = 40
    diar_model.sortformer_modules.fifo_len = 40
    diar_model.sortformer_modules.spkcache_update_period = 300
    diar_model._check_streaming_parameters()

    print("[*] Running NeMo diarize...")
    t0 = time.perf_counter()
    predicted_segments = diar_model.diarize(audio=[audio_path], batch_size=1)
    infer_time = time.perf_counter() - t0

    # Parse NeMo output segments
    segments = []
    for item in predicted_segments[0]:
        if isinstance(item, str):
            # Format: 'start end speaker_X'
            parts = item.strip().split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                spk = parts[2].replace("speaker_", "")
                segments.append({"Speaker": int(spk) if spk.isdigit() else spk, "Start": start, "End": end})
        elif isinstance(item, dict):
            segments.append(item)
    return segments, 0.0, infer_time

def format_rttm(segments, file_id: str = "audio") -> str:
    """Format segments into standard RTTM file."""
    lines = []
    for seg in sorted(segments, key=lambda x: x.get("Start", 0.0)):
        start = float(seg.get("Start", 0.0))
        end = float(seg.get("End", 0.0))
        dur = max(0.0, end - start)
        spk = seg.get("Speaker", 0)
        lines.append(f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> speaker_{spk} <NA> <NA>")
    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="Nemotron-3-Diarization Runner")
    parser.add_argument("--audio", required=True, help="Input audio file (mp3/wav)")
    parser.add_argument("--output-json", default="diarization_results.json", help="Output JSON path")
    parser.add_argument("--output-rttm", default="diarization_results.rttm", help="Output RTTM path")
    parser.add_argument("--engine", choices=["transformers", "nemo"], default="transformers")
    args = parser.parse_args()

    audio_wav = preprocess_audio(args.audio)
    
    if args.engine == "transformers":
        segments, duration, infer_time = run_diarization_transformers(audio_wav)
    else:
        segments, duration, infer_time = run_diarization_nemo(audio_wav)

    print(f"\n{'='*70}")
    print(f"DIARIZATION COMPLETE: {len(segments)} SEGMENTS DETECTED")
    print(f"{'='*70}")

    # Calculate speaker statistics
    spk_stats = {}
    for s in segments:
        spk = s.get("Speaker")
        dur = s.get("End", 0.0) - s.get("Start", 0.0)
        spk_stats[spk] = spk_stats.get(spk, 0.0) + dur

    print("\n--- Speaker Speaking Time Breakdown ---")
    for spk, total_dur in sorted(spk_stats.items(), key=lambda x: str(x[0])):
        print(f"  Speaker {spk}: {total_dur:.2f} seconds ({total_dur/max(1, duration)*100:.1f}%)")

    print("\n--- Sample Timeline (First 15 segments) ---")
    for i, s in enumerate(segments[:15]):
        print(f"  [{s.get('Start', 0.0):6.2f}s -> {s.get('End', 0.0):6.2f}s] Speaker {s.get('Speaker')}")

    if len(segments) > 15:
        print(f"  ... and {len(segments) - 15} more segments.")

    result_data = {
        "model": "nvidia/Nemotron-3-Diarization",
        "audio_file": str(args.audio),
        "duration_seconds": duration,
        "inference_time_seconds": infer_time,
        "total_segments": len(segments),
        "unique_speakers": len(spk_stats),
        "speaker_durations": {f"speaker_{k}": v for k, v in spk_stats.items()},
        "segments": segments
    }

    with open(args.output_json, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n[+] Saved detailed JSON to: {args.output_json}")

    rttm_text = format_rttm(segments, file_id=Path(args.audio).stem)
    with open(args.output_rttm, "w") as f:
        f.write(rttm_text)
    print(f"[+] Saved standard RTTM to: {args.output_rttm}")

if __name__ == "__main__":
    main()
