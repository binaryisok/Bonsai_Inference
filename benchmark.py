"""
benchmark.py - Simple TPS (Tokens Per Second) and Latency Benchmark
for Ternary Bonsai 2 27B on Google Colab (Tesla T4 GPU).
"""

import os
import sys
import time
import json
import requests
import subprocess
from typing import Dict, Any, List
from inference import BonsaiColabPipeline


def get_gpu_vram() -> str:
    """Returns current VRAM usage string from nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        used, total = out.split(",")
        return f"{used.strip()} MiB / {total.strip()} MiB"
    except Exception:
        return "N/A"


def run_single_benchmark(
    base_url: str,
    name: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.6
) -> Dict[str, Any]:
    """Runs a single prompt through the OpenAI-compatible endpoint and computes metrics."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "Bonsai-2-27B",
        "messages": [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }

    t_start = time.perf_counter()
    resp = requests.post(url, json=payload, stream=True, timeout=120)
    resp.raise_for_status()

    t_first_token = None
    thinking_tokens = 0
    answer_tokens = 0
    thinking_text = []
    answer_text = []

    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            data_str = decoded[6:].strip()
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})

            if t_first_token is None:
                t_first_token = time.perf_counter()

            if "reasoning_content" in delta and delta["reasoning_content"]:
                thinking_tokens += 1
                thinking_text.append(delta["reasoning_content"])
            elif "content" in delta and delta["content"]:
                answer_tokens += 1
                answer_text.append(delta["content"])

    t_end = time.perf_counter()

    ttft = (t_first_token - t_start) if t_first_token else 0.0
    total_time = t_end - t_start
    gen_time = max(0.001, total_time - ttft)
    total_tokens = thinking_tokens + answer_tokens

    decode_tps = total_tokens / gen_time if gen_time > 0 else 0.0
    e2e_tps = total_tokens / total_time if total_time > 0 else 0.0

    return {
        "name": name,
        "prompt": prompt,
        "ttft_sec": round(ttft, 2),
        "total_time_sec": round(total_time, 2),
        "thinking_tokens": thinking_tokens,
        "answer_tokens": answer_tokens,
        "total_tokens": total_tokens,
        "decode_tps": round(decode_tps, 2),
        "e2e_tps": round(e2e_tps, 2),
        "vram": get_gpu_vram(),
        "sample_answer": "".join(answer_text)[:120] + ("..." if len("".join(answer_text)) > 120 else "")
    }


def main():
    print("==================================================================")
    print("  Bonsai 2 27B Benchmark — Tesla T4 GPU on Google Colab")
    print("==================================================================")

    # 1. Initialize and ensure server is up
    pipeline = BonsaiColabPipeline(port=8088, context_size=8192, gpu_layers=99)
    pipeline.start_server()

    print(f"\n[INFO] Initial GPU VRAM: {get_gpu_vram()}\n")

    # 2. Benchmark test suite
    test_cases = [
        {
            "name": "Short Fact Retrieval",
            "prompt": "What is the speed of light in vacuum? Answer in 1 short sentence.",
            "max_tokens": 128
        },
        {
            "name": "Reasoning & Math",
            "prompt": "If a train travels at 60 mph for 2.5 hours, then increases its speed by 20% for another 1.5 hours, what is the total distance traveled? Show reasoning briefly.",
            "max_tokens": 256
        },
        {
            "name": "Code Generation",
            "prompt": "Write a Python function to compute the Fibonacci sequence up to n using dynamic programming with comments.",
            "max_tokens": 256
        },
        {
            "name": "Summarization & Concept",
            "prompt": "Summarize the importance of low-bit ternary representation in modern artificial intelligence in 3 bullet points.",
            "max_tokens": 256
        }
    ]

    results = []
    for i, tc in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Running: {tc['name']}...")
        res = run_single_benchmark(
            base_url=pipeline.base_url,
            name=tc["name"],
            prompt=tc["prompt"],
            max_tokens=tc["max_tokens"]
        )
        results.append(res)
        print(f"       TTFT: {res['ttft_sec']}s | Tokens: {res['total_tokens']} (Think: {res['thinking_tokens']}, Ans: {res['answer_tokens']})")
        print(f"       Decode TPS: {res['decode_tps']} tok/s | Total Time: {res['total_time_sec']}s")
        print(f"       Sample: {res['sample_answer']}\n")

    # 3. Print Summary Table
    print("\n===============================================================================================")
    print("                               BENCHMARK SUMMARY RESULTS                                      ")
    print("===============================================================================================")
    header = f"{'Test Name':<26} | {'TTFT (s)':<8} | {'Tokens':<8} | {'Decode TPS':<11} | {'E2E TPS':<9} | {'VRAM Usage':<15}"
    print(header)
    print("-" * len(header))

    total_decode_tps = 0.0
    total_tokens = 0
    total_time = 0.0

    for r in results:
        total_decode_tps += r["decode_tps"]
        total_tokens += r["total_tokens"]
        total_time += r["total_time_sec"]
        row = f"{r['name']:<26} | {r['ttft_sec']:<8.2f} | {r['total_tokens']:<8} | {r['decode_tps']:<11.2f} | {r['e2e_tps']:<9.2f} | {r['vram']:<15}"
        print(row)

    avg_decode_tps = total_decode_tps / len(results) if results else 0
    avg_e2e_tps = total_tokens / total_time if total_time > 0 else 0
    print("-" * len(header))
    print(f"{'Average / Aggregate':<26} | {'-':<8} | {total_tokens:<8} | {avg_decode_tps:<11.2f} | {avg_e2e_tps:<9.2f} | {results[-1]['vram']:<15}")
    print("===============================================================================================\n")

    # Output JSON for programmatic consumption
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[OK] Results saved to benchmark_results.json")


if __name__ == "__main__":
    main()
