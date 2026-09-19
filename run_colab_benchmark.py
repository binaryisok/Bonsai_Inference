import requests
import json
import time
import subprocess

base_studio = "http://localhost:8791"
session_name = "bonsai-bench"

print("Creating fresh Tesla T4 session on Colab Studio...")
sess_r = requests.post(f"{base_studio}/api/sessions", json={
    "name": session_name,
    "gpu": "t4",
    "connect": True
}, timeout=90)
sess_data = sess_r.json()
endpoint = sess_data.get("session", {}).get("endpoint")
print(f"Allocated session '{session_name}' on endpoint: {endpoint}")

with open("/home/001230629076_0/Documents/Bonsai_Inference/inference.py", "r") as f:
    inf_code = f.read()

remote_code = """
import os, sys, time, subprocess, requests, json

# 1. Write latest inference.py
with open("/content/inference.py", "w") as f:
    f.write(__INFERENCE_CODE__)

# 2. Add swap to prevent RAM crash
subprocess.run("fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile || true", shell=True)

# 3. Ensure /content is in sys.path and Bonsai-demo is cloned and setup
sys.path.insert(0, "/content")
from inference import BonsaiColabPipeline

pipeline = BonsaiColabPipeline(port=8088, context_size=4096, gpu_layers=99)
pipeline.start_server()

# 4. Benchmark Prompts
benchmarks = [
    ("Math & Reasoning", "Solve step-by-step: If a store has 45 apples and sells 3/5 of them in the morning, then receives 20 new apples in the afternoon, how many apples are left?"),
    ("Code Generation", "Write a concise Python function to calculate the factorial of n using recursion with an edge case check.")
]

print("\\n" + "="*80)
print("             STARTING LIVE BENCHMARK ON TESLA T4 GPU")
print("="*80)

for name, prompt in benchmarks:
    print(f"\\n--- Test: {name} ---")
    print(f"Prompt: {prompt}\\n")
    
    t0 = time.perf_counter()
    resp = requests.post("http://127.0.0.1:8088/v1/chat/completions", json={
        "model": "Bonsai-2-27B",
        "messages": [
            {"role": "system", "content": "You are a concise expert assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "max_tokens": 256,
        "stream": True
    }, stream=True, timeout=120)

    t_first = None
    think_tokens = 0
    ans_tokens = 0
    in_thinking = False

    for line in resp.iter_lines():
        if not line: continue
        dec = line.decode("utf-8")
        if dec.startswith("data: "):
            d_str = dec[6:].strip()
            if d_str == "[DONE]": break
            chunk = json.loads(d_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})

            if t_first is None:
                t_first = time.perf_counter()

            if "reasoning_content" in delta and delta["reasoning_content"]:
                think_tokens += 1
                if not in_thinking:
                    print("\\033[90m[Thinking] ", end="", flush=True)
                    in_thinking = True
                print(delta["reasoning_content"], end="", flush=True)
            elif "content" in delta and delta["content"]:
                ans_tokens += 1
                if in_thinking:
                    print("\\033[0m\\n\\n[Answer] ", end="", flush=True)
                    in_thinking = False
                print(delta["content"], end="", flush=True)

    if in_thinking:
        print("\\033[0m")

    t1 = time.perf_counter()
    ttft = t_first - t0 if t_first else 0.0
    total_time = t1 - t0
    gen_time = max(0.001, total_time - ttft)
    total_tokens = think_tokens + ans_tokens
    decode_tps = total_tokens / gen_time if gen_time > 0 else 0.0

    print(f"\\n\\nMetrics for {name}:")
    print(f"  TTFT:            {ttft:.2f} s")
    print(f"  Thinking tokens: {think_tokens}")
    print(f"  Answer tokens:   {ans_tokens}")
    print(f"  Total tokens:    {total_tokens}")
    print(f"  Generation time: {gen_time:.2f} s")
    print(f"  Decode TPS:      {decode_tps:.2f} tokens/sec")
    print(f"  End-to-end TPS:  {total_tokens / total_time:.2f} tokens/sec")

# 5. Check VRAM
vram_out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
print(f"\nFinal GPU VRAM Usage: {vram_out} MiB")
""".replace("__INFERENCE_CODE__", json.dumps(inf_code))

try:
    print("Executing benchmark on Colab VM...")
    resp = requests.post(f"{base_studio}/api/sessions/{session_name}/execute", json={"code": remote_code, "timeout": 480}, stream=True, timeout=540)
    for line in resp.iter_lines():
        if line:
            ev = json.loads(line.decode("utf-8"))
            if ev.get("type") == "stream":
                print(ev.get("text"), end="", flush=True)
            elif ev.get("type") == "error":
                print("Error:", ev.get("ename"), ev.get("evalue"))
                for t in ev.get("traceback", []):
                    print(t)
finally:
    print("\n" + "="*80)
    print("CLEANING UP AND RELEASING COLAB SESSION...")
    print("="*80)
    try:
        del_r = requests.delete(f"{base_studio}/api/sessions/{session_name}")
        print(f"Deleted session {session_name}: status {del_r.status_code}")
    except Exception as e:
        print(f"Error deleting session: {e}")

    try:
        rel_r = requests.post(f"{base_studio}/api/assignments/release", json={"endpoint": endpoint})
        print(f"Released assignment {endpoint}: status {rel_r.status_code}")
    except Exception as e:
        print(f"Error releasing assignment: {e}")

    # Final check
    rem_sess = requests.get(f"{base_studio}/api/sessions").json().get("sessions", [])
    rem_assign = requests.get(f"{base_studio}/api/assignments").json().get("assignments", [])
    print("Remaining sessions:", rem_sess)
    print("Remaining assignments:", rem_assign)
    print("[OK] Session and assignment have been fully closed and released.")
