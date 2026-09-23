#!/usr/bin/env python3
"""
Automated Colab Runner for Nemotron-3-Diarization
Connects to Colab Studio, spins up a Tesla T4 GPU session,
uploads audio, runs the inference pipeline, downloads RTTM/JSON,
and outputs speaker segmentation metrics.
"""

import os
import sys
import json
import time
import requests
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run Nemotron-3-Diarization on Google Colab")
    parser.add_argument("--studio-url", default="http://localhost:8791", help="Colab Studio base URL")
    parser.add_argument("--session-name", default="diarization-run", help="Colab session name")
    parser.add_argument("--audio", default="/home/001230629076_0/Documents/Bonsai_Inference/ben10.mp3", help="Audio file path")
    parser.add_argument("--clean-up", action="store_true", help="Release session when finished")
    args = parser.parse_args()

    base = args.studio_url.rstrip("/")
    sess_name = args.session_name

    print(f"[*] Checking Colab Studio at {base}...")
    try:
        health = requests.get(f"{base}/api/health", timeout=10).json()
        print(f"[+] Health: {health.get('status')}, Authenticated: {health.get('authenticated')}")
    except Exception as e:
        print(f"[-] Could not connect to Colab Studio: {e}")
        sys.exit(1)

    # Check existing sessions
    sessions = requests.get(f"{base}/api/sessions", timeout=10).json().get("sessions", [])
    active_session = next((s for s in sessions if s.get("name") == sess_name), None)

    if not active_session:
        print(f"[*] Allocating Tesla T4 GPU session '{sess_name}'...")
        sess_r = requests.post(f"{base}/api/sessions", json={
            "name": sess_name,
            "gpu": "t4",
            "connect": True
        }, timeout=120)
        if sess_r.status_code != 200:
            print(f"[-] Session allocation failed: {sess_r.text}")
            sys.exit(1)
        active_session = sess_r.json().get("session", {})
        print(f"[+] Allocated session on endpoint: {active_session.get('endpoint')}")
    else:
        print(f"[+] Reusing existing session '{sess_name}' on {active_session.get('endpoint')}")

    # Upload inference script
    script_path = Path(__file__).parent / "inference_diarization.py"
    with open(script_path, "rb") as f:
        print(f"[*] Uploading {script_path.name} to Colab...")
        up_r = requests.post(
            f"{base}/api/sessions/{sess_name}/files/upload?path=/content/inference_diarization.py",
            files={"file": (script_path.name, f, "text/x-python")},
            timeout=60
        )
        print(f"[+] Script upload: {up_r.status_code}")

    # Upload audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[-] Audio file not found: {audio_path}")
        sys.exit(1)

    with open(audio_path, "rb") as f:
        print(f"[*] Uploading {audio_path.name} ({audio_path.stat().st_size / 1e6:.2f} MB) to Colab...")
        up_r = requests.post(
            f"{base}/api/sessions/{sess_name}/files/upload?path=/content/{audio_path.name}",
            files={"file": (audio_path.name, f, "application/octet-stream")},
            timeout=120
        )
        print(f"[+] Audio upload: {up_r.status_code}")

    # Remote Execution Code
    remote_code = f"""
import subprocess, sys

# 1. Ensure latest transformers from source
try:
    import transformers
    if not hasattr(transformers, "Nemotron3DiarizationForAudioFrameClassification"):
        print("[*] Installing transformers from git source...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "git+https://github.com/huggingface/transformers.git", "accelerate", "soundfile", "librosa"], check=True)
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "git+https://github.com/huggingface/transformers.git", "accelerate", "soundfile", "librosa"], check=True)

# 2. Run inference
cmd = [
    sys.executable, "/content/inference_diarization.py",
    "--audio", "/content/{audio_path.name}",
    "--output-json", "/content/diarization_results.json",
    "--output-rttm", "/content/diarization_results.rttm"
]
print("[*] Executing:", " ".join(cmd))
res = subprocess.run(cmd, text=True, capture_output=False)
"""

    print("\n" + "="*70)
    print("STARTING REMOTE INFERENCE ON TESLA T4 GPU")
    print("="*70)
    exec_resp = requests.post(
        f"{base}/api/sessions/{sess_name}/execute",
        json={"code": remote_code, "timeout": 600},
        stream=True,
        timeout=620
    )

    for line in exec_resp.iter_lines():
        if line:
            ev = json.loads(line.decode("utf-8"))
            if ev.get("type") == "stream":
                print(ev.get("text"), end="", flush=True)
            elif ev.get("type") == "error":
                print(f"\n[-] Execution Error: {ev.get('ename')} - {ev.get('evalue')}")
                for t in ev.get("traceback", []):
                    print(t)

    # Download results
    print("\n[*] Fetching results from Colab...")
    out_dir = Path(__file__).parent
    for remote_fn, local_fn in [
        ("content/diarization_results.json", out_dir / f"{audio_path.stem}_diarization.json"),
        ("content/diarization_results.rttm", out_dir / f"{audio_path.stem}_diarization.rttm")
    ]:
        dl_resp = requests.get(f"{base}/api/sessions/{sess_name}/files/download?path={remote_fn}", timeout=30)
        if dl_resp.status_code == 200:
            wrapper = dl_resp.json() if "application/json" in dl_resp.headers.get("content-type", "") else None
            if wrapper and "content" in wrapper:
                content = wrapper["content"]
                if local_fn.suffix == ".json":
                    content = json.dumps(json.loads(content), indent=2)
            else:
                content = dl_resp.text

            with open(local_fn, "w") as f:
                f.write(content)
            print(f"[+] Downloaded: {local_fn}")

    if args.clean_up:
        print(f"\n[*] Cleaning up session {sess_name}...")
        requests.delete(f"{base}/api/sessions/{sess_name}")
        print("[+] Session released.")

if __name__ == "__main__":
    main()
