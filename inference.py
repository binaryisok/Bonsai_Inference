"""
inference.py - Dedicated Inference Pipeline for Ternary Bonsai 2 27B on Google Colab (Tesla T4 GPU).

Features:
  - Automates llama-server lifecycle management on Linux / Google Colab with CUDA.
  - Full GPU offloading (-ngl 99) for Tesla T4 (16 GB VRAM).
  - Streaming generation with real-time separation of thinking (reasoning_content) and answer tokens.
  - Multimodal Vision support (local image path, URL, or PIL Image).
  - Native OpenAI-compatible tool/function calling support.
  - Context size and 4-bit KV cache knobs (BONSAI_KV4).
"""

import os
import sys
import time
import signal
import base64
import argparse
import requests
import json
import subprocess
from io import BytesIO
from typing import Generator, List, Dict, Any, Optional, Union
from PIL import Image


class BonsaiColabPipeline:
    """
    Dedicated inference pipeline for Ternary Bonsai 2 27B on Google Colab / Tesla T4 GPU.
    Manages the background llama-server process and provides high-level text, vision, and tool APIs.
    """

    def __init__(
        self,
        demo_dir: str = "/content/Bonsai-demo",
        port: int = 8088,
        context_size: int = 16384,
        gpu_layers: int = 99,
        kv_4bit: bool = False,
        host: str = "127.0.0.1",
        auto_setup: bool = True,
    ):
        # On Google Colab, port 8080 is permanently bound by /datalab/web/app.js (Datalab service)
        if port == 8080 and os.path.exists("/content"):
            print("[WARN] Port 8080 is reserved by Google Colab's internal Datalab service. Switching to port 8088.")
            port = 8088

        self.demo_dir = os.path.abspath(demo_dir)
        self.port = port
        self.host = host
        self.base_url = f"http://{host}:{port}"
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.kv_4bit = kv_4bit
        self.auto_setup = auto_setup
        self.process: Optional[subprocess.Popen] = None

    def _ensure_demo_ready(self) -> None:
        """Finds or automatically clones and sets up Bonsai-demo."""
        candidate_paths = [
            self.demo_dir,
            "/content/Bonsai-demo",
            os.path.abspath("Bonsai-demo"),
            os.path.abspath("../Bonsai-demo"),
        ]
        for path in candidate_paths:
            if os.path.isfile(os.path.join(path, "scripts", "start_llama_server.sh")):
                self.demo_dir = path
                return

        if not self.auto_setup:
            raise FileNotFoundError(
                f"Bonsai-demo not found at {self.demo_dir}. "
                f"Clone it first with: git clone https://github.com/PrismML-Eng/Bonsai-demo.git {self.demo_dir}"
            )

        print(f"==> Bonsai-demo not found. Automatically cloning to '{self.demo_dir}'...")
        os.makedirs(os.path.dirname(self.demo_dir), exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/PrismML-Eng/Bonsai-demo.git", self.demo_dir],
            check=True
        )

        print("==> Running setup.sh to download model weights and CUDA binaries...")
        env = os.environ.copy()
        env["BONSAI_OPENWEBUI"] = "0"
        env["BONSAI_CODE_INTERPRETER"] = "0"
        env["BONSAI_MODEL"] = "27B"
        env["BONSAI_FAMILY"] = "bonsai2"

        # Make scripts executable
        setup_script = os.path.join(self.demo_dir, "setup.sh")
        os.chmod(setup_script, 0o755)
        for s in os.listdir(os.path.join(self.demo_dir, "scripts")):
            if s.endswith(".sh"):
                os.chmod(os.path.join(self.demo_dir, "scripts", s), 0o755)

        subprocess.run([setup_script], cwd=self.demo_dir, env=env, check=True)
        print("[OK] Bonsai-demo setup completed successfully.")

    def kill_existing_server(self) -> None:
        """Kills any running llama-server process or process occupying the port."""
        print(f"==> Clearing port {self.port} / killing any existing llama-server...")
        try:
            subprocess.run(["pkill", "-9", "-f", "llama-server"], stderr=subprocess.DEVNULL)
        except Exception:
            pass
        try:
            subprocess.run(["fuser", "-k", f"{self.port}/tcp"], stderr=subprocess.DEVNULL)
        except Exception:
            pass
        time.sleep(1)

    def start_server(self, wait_timeout: int = 180, restart: bool = False) -> None:
        """
        Starts the llama.cpp server in the background with full GPU offload.
        If a server is already running, connects to it (or restarts if restart=True).
        Waits until the server responds to /health with status 200.
        """
        # Ensure demo repository and scripts exist
        self._ensure_demo_ready()

        if restart:
            self.kill_existing_server()

        # Check if already running on the given port
        for host_candidate in [self.host, "127.0.0.1", "localhost"]:
            try:
                r = requests.get(f"http://{host_candidate}:{self.port}/health", timeout=2)
                if r.status_code == 200:
                    print(f"[OK] Bonsai server is already active and ready at http://{host_candidate}:{self.port}")
                    self.base_url = f"http://{host_candidate}:{self.port}"
                    return
                elif r.status_code == 503:
                    print(f"==> Existing server found at http://{host_candidate}:{self.port} (loading model)...")
                    self.base_url = f"http://{host_candidate}:{self.port}"
                    break
            except Exception:
                pass

        start_script = os.path.join(self.demo_dir, "scripts", "start_llama_server.sh")
        if not os.path.exists(start_script):
            raise FileNotFoundError(
                f"Server start script not found at {start_script}. "
                f"Ensure Bonsai-demo is cloned at '{self.demo_dir}'."
            )

        print(
            f"==> Launching Bonsai 2 27B server on Tesla T4 "
            f"(context={self.context_size}, ngl={self.gpu_layers}, kv_4bit={self.kv_4bit})..."
        )

        env = os.environ.copy()
        env["PORT"] = str(self.port)
        env["BONSAI_HOST"] = self.host
        env["BONSAI_CTX"] = str(self.context_size)
        env["BONSAI_NGL"] = str(self.gpu_layers)
        if self.kv_4bit:
            env["BONSAI_KV4"] = "1"

        log_path = os.path.join(self.demo_dir, "server.log")
        log_file = open(log_path, "w")

        self.process = subprocess.Popen(
            ["./scripts/start_llama_server.sh"],
            cwd=self.demo_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # Detach process group for clean teardown
        )

        start_time = time.time()
        print("==> Waiting for model weights to load into VRAM...")
        while time.time() - start_time < wait_timeout:
            # Check if process exited
            if self.process.poll() is not None:
                log_file.close()
                with open(log_path, "r", errors="ignore") as lf:
                    logs = lf.read()

                # If server is already running, gracefully adopt it instead of failing!
                if "llama-server is already running on port" in logs:
                    print(f"[INFO] Server is already running on port {self.port}. Connecting to it...")
                    # Wait for it to be fully ready (HTTP 200)
                    while time.time() - start_time < wait_timeout:
                        try:
                            r = requests.get(f"{self.base_url}/health", timeout=2)
                            if r.status_code == 200:
                                print(f"[OK] Connected to existing server at {self.base_url}!")
                                return
                        except Exception:
                            pass
                        time.sleep(2)

                raise RuntimeError(
                    f"Server process terminated with code {self.process.returncode}.\nLogs:\n{logs[-1500:]}"
                )

            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    print(
                        f"[OK] Server is live and ready at {self.base_url} "
                        f"(took {int(time.time() - start_time)}s)"
                    )
                    return
            except Exception:
                time.sleep(3)

        raise TimeoutError(
            f"Server failed to start within {wait_timeout}s. Check {log_path} for details."
        )

    def stop_server(self) -> None:
        """Stops the running server process and child processes cleanly."""
        if self.process:
            print("==> Stopping Bonsai server...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.process = None
            print("[OK] Server stopped.")

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.6,
        max_tokens: int = 1024,
        reasoning_budget: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streams completions, separating reasoning/thinking tokens from content.
        Yields dicts with format:
          {'type': 'thinking' | 'answer' | 'tool_calls', 'data': ...}
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": "Bonsai-2-27B",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if reasoning_budget is not None:
            payload["reasoning_budget"] = reasoning_budget
        if tools:
            payload["tools"] = tools

        with requests.post(url, json=payload, stream=True) as resp:
            resp.raise_for_status()
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

                    # Reasoning/thinking tokens
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        yield {"type": "thinking", "data": delta["reasoning_content"]}
                    # Answer content tokens
                    elif "content" in delta and delta["content"]:
                        yield {"type": "answer", "data": delta["content"]}
                    # Tool call tokens / descriptors
                    elif "tool_calls" in delta and delta["tool_calls"]:
                        yield {"type": "tool_calls", "data": delta["tool_calls"]}

    def ask(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.6,
        reasoning_budget: Optional[int] = None,
    ) -> str:
        """
        Convenience method for interactive one-shot queries.
        Prints thoughts in dimmed color, then prints the answer.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        print(f"\nUser: {prompt}\n")
        in_thinking = False
        full_answer = []

        for chunk in self.chat_stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_budget=reasoning_budget,
        ):
            if chunk["type"] == "thinking":
                if not in_thinking:
                    print("\033[90m[Thinking] ", end="", flush=True)
                    in_thinking = True
                print(chunk["data"], end="", flush=True)
            elif chunk["type"] == "answer":
                if in_thinking:
                    print("\033[0m\n\nAssistant: ", end="", flush=True)
                    in_thinking = False
                print(chunk["data"], end="", flush=True)
                full_answer.append(chunk["data"])

        if in_thinking:
            print("\033[0m")
        print("\n")
        return "".join(full_answer)

    def ask_image(
        self,
        image_input: Union[str, Image.Image],
        prompt: str,
        system_prompt: str = "You are a helpful assistant with computer vision capabilities.",
        max_tokens: int = 1024,
    ) -> str:
        """
        Multimodal query with an image (file path, URL, or PIL Image) and a text prompt.
        """
        if isinstance(image_input, str) and (
            image_input.startswith("http://") or image_input.startswith("https://")
        ):
            img_url = image_input
        else:
            if isinstance(image_input, str):
                img = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                raise ValueError("image_input must be a file path, URL, or PIL Image")

            buf = BytesIO()
            img.convert("RGB").save(buf, format="JPEG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            img_url = f"data:image/jpeg;base64,{b64_str}"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            },
        ]

        print(f"\nUser [Image + Text]: {prompt}\n")
        in_thinking = False
        full_answer = []

        for chunk in self.chat_stream(messages, max_tokens=max_tokens):
            if chunk["type"] == "thinking":
                if not in_thinking:
                    print("\033[90m[Thinking] ", end="", flush=True)
                    in_thinking = True
                print(chunk["data"], end="", flush=True)
            elif chunk["type"] == "answer":
                if in_thinking:
                    print("\033[0m\n\nAssistant: ", end="", flush=True)
                    in_thinking = False
                print(chunk["data"], end="", flush=True)
                full_answer.append(chunk["data"])

        if in_thinking:
            print("\033[0m")
        print("\n")
        return "".join(full_answer)


def main():
    parser = argparse.ArgumentParser(
        description="Bonsai 2 27B Inference Pipeline (Colab / Tesla T4)"
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default="/content/Bonsai-demo",
        help="Path to Bonsai-demo directory",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8088,
        help="Port for llama-server (default: 8088, avoids Colab 8080 conflict)",
    )
    parser.add_argument(
        "--ctx", type=int, default=16384, help="Context size (default: 16384)"
    )
    parser.add_argument(
        "--ngl", type=int, default=99, help="Number of GPU layers to offload (default: 99)"
    )
    parser.add_argument(
        "--kv4", action="store_true", help="Enable 4-bit KV cache (BONSAI_KV4=1)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain why ternary representation enables high efficiency in LLMs.",
        help="Prompt to run",
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Optional image path or URL for vision input"
    )
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Start server and keep running without exiting",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Kill any existing server process on the port and start fresh",
    )
    args, _ = parser.parse_known_args()

    pipeline = BonsaiColabPipeline(
        demo_dir=args.demo_dir,
        port=args.port,
        context_size=args.ctx,
        gpu_layers=args.ngl,
        kv_4bit=args.kv4,
    )

    try:
        pipeline.start_server(restart=args.restart)

        if args.server_only:
            print("Server is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)

        if args.image:
            pipeline.ask_image(image_input=args.image, prompt=args.prompt)
        else:
            pipeline.ask(prompt=args.prompt)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if not args.server_only:
            pipeline.stop_server()


if __name__ == "__main__":
    main()
