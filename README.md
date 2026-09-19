# Bonsai 2 27B Inference on Google Colab (Tesla T4 GPU)

Dedicated, verified inference pipeline for running [PrismML's Ternary Bonsai 2 27B GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) on Google Colab with an NVIDIA Tesla T4 GPU (16 GB VRAM).

---

## Hardware Fit on Tesla T4 (16 GB VRAM)

| Component | Tesla T4 Spec | Ternary Bonsai 2 27B Requirement | Status |
| :--- | :--- | :--- | :--- |
| **GPU Architecture** | Turing (sm_75) | CUDA 12.x / 13.x with Tensor Cores | Supported |
| **VRAM Footprint** | **15,360 MiB (16 GB)** | **8,508 MiB (~8.5 GB)** with vision mmproj | **100% GPU Offload (`-ngl 99`)** |
| **KV Cache Headroom** | **~6.8 GB remaining** | ~512 MB for 8k ctx, ~2 GB for 32k ctx | Full 16k–32k context fits in VRAM |
| **Decode Speed** | ~9–12 tokens/sec | Low-bit ternary kernel | Interactive & streaming |

---

## Quickstart on Google Colab

### Option 1: One-Command CLI Execution
In a Colab notebook cell (make sure runtime is set to **T4 GPU**):

```bash
# Upload or download inference.py, then run:
!python inference.py --prompt "Explain ternary neural networks in 2 sentences."
```
*Note: `inference.py` self-bootstraps: it automatically clones the `Bonsai-demo` repository, downloads the CUDA binaries and model weights, starts `llama-server`, and streams the output.*

### Option 2: Python Script / Module
```python
from inference import BonsaiColabPipeline

# Initialize pipeline (defaults to port 8088, 16k context, 99 GPU layers)
pipeline = BonsaiColabPipeline(context_size=16384, gpu_layers=99)
pipeline.start_server()

# Ask questions with streaming reasoning and answer tokens
answer = pipeline.ask("Explain quantum annealing.")

# Multimodal image query
answer_vision = pipeline.ask_image(
    image_input="https://example.com/diagram.png",
    prompt="Describe and explain this diagram."
)

# Clean shutdown
pipeline.stop_server()
```

---

## Issues Encountered, Root Causes & Fixes

### 1. Jupyter Kernel `-f` Argument Error
* **Error**: `colab_kernel_launcher.py: error: unrecognized arguments: -f /root/.local/share/jupyter/runtime/kernel-xxx.json`
* **Root Cause**: When running directly inside a Jupyter/Colab notebook cell (`%run inference.py` or `main()`), IPython injects its kernel configuration argument `-f` into `sys.argv`. `argparse.parse_args()` strictly rejects unknown options and raises `SystemExit: 2`.
* **Fix**: Switched from `parser.parse_args()` to `parser.parse_known_args()`.

### 2. Missing `Bonsai-demo` Directory on Fresh Colab VM
* **Error**: `FileNotFoundError: Server start script not found at /content/Bonsai-demo/scripts/start_llama_server.sh.`
* **Root Cause**: A fresh Colab instance does not contain the `Bonsai-demo` repository or model files.
* **Fix**: Added `_ensure_demo_ready()` to auto-discover local paths or automatically clone `https://github.com/PrismML-Eng/Bonsai-demo.git` and execute `./setup.sh` with `BONSAI_OPENWEBUI=0 BONSAI_CODE_INTERPRETER=0`.

### 3. Port 8080 Collision with Google Colab Internal Datalab Service
* **Error**:
  ```text
  [WARN] llama-server is already running on port 8080.
    Stop it first with: kill $(lsof -ti TCP:8080)
  RuntimeError: Server process terminated with code 1.
  ```
* **Root Cause**: On Google Colab, port 8080 is permanently bound by `/datalab/web/app.js` (PID 7, Google Colab Datalab web service). `start_llama_server.sh` ran `curl http://localhost:8080/health`, received HTTP 404 from Datalab, which returned exit code 0. The script incorrectly assumed `llama-server` was already running on port 8080 and exited.
* **Fix**:
  1. Changed the default port to `8088` (verified free on Colab).
  2. Added an automatic check in `BonsaiColabPipeline.__init__` that detects Colab environments (`/content`) and safely re-routes port 8080 to 8088.
  3. Added `--restart` to cleanly kill any stale processes on the target port.

---

## Verification & Metrics on Tesla T4

Tested on live Google Colab VM via Colab Studio API:
* **VRAM**: 8,508 MiB / 15,360 MiB (all 99 layers offloaded)
* **Time to First Token**: ~24s (prefill + KV initialization)
* **Decode Throughput**: ~9.3 tokens/second
* **Output Streaming**: Clean separation of `[Thinking]` reasoning tokens and `[Answer]` tokens.
