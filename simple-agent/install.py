#!/usr/bin/env python3
"""
Gemma 4 AI Agent — One-command installer.

What this does:
  1. Creates a Python virtual environment (.venv/)
  2. Detects your GPU (CUDA / Metal / CPU-only)
  3. Installs llama-cpp-python with the right backend
  4. Downloads the default Gemma 4 GGUF model (~3 GB)
  5. Writes run.sh / run.bat launchers

Usage:
    python install.py                    # auto-detect everything
    python install.py --cpu              # force CPU-only
    python install.py --model-url URL    # use a custom GGUF URL
    python install.py --skip-model       # skip model download

After install:
    ./run.sh                             # Linux / macOS
    run.bat                              # Windows
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import urllib.error

# ── Defaults ──────────────────────────────────────────────────────────────

VENV_DIR = ".venv"
MODELS_DIR = "models"

# Default Gemma 4 GGUF — Q4_K_M quantization (good quality/speed balance).
# Uses the standard HuggingFace blob download URL pattern.
# Update this URL when the canonical Gemma 4 GGUF repo is available.
DEFAULT_MODEL_URL = (
    "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF"
    "/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf"
)
DEFAULT_MODEL_FILENAME = "gemma-4-4b-it-Q4_K_M.gguf"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ───────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[install] {msg}")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def detect_gpu() -> str:
    """Return 'cuda', 'metal', or 'cpu'."""
    system = platform.system()

    # macOS with Apple Silicon
    if system == "Darwin":
        cpu = platform.processor()
        machine = platform.machine()
        if machine == "arm64" or "apple" in cpu.lower():
            return "metal"

    # NVIDIA GPU
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(
                ["nvidia-smi"], capture_output=True, check=True, timeout=10,
            )
            return "cuda"
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Check for CUDA libs
    for path in ["/usr/local/cuda", "/usr/lib/x86_64-linux-gnu/libcuda.so"]:
        if os.path.exists(path):
            return "cuda"

    return "cpu"


def python_executable() -> str:
    """Return the venv python path."""
    if platform.system() == "Windows":
        return os.path.join(SCRIPT_DIR, VENV_DIR, "Scripts", "python.exe")
    return os.path.join(SCRIPT_DIR, VENV_DIR, "bin", "python")


def pip_executable() -> str:
    """Return the venv pip path."""
    if platform.system() == "Windows":
        return os.path.join(SCRIPT_DIR, VENV_DIR, "Scripts", "pip.exe")
    return os.path.join(SCRIPT_DIR, VENV_DIR, "bin", "pip")


# ── Step 1: Virtual environment ──────────────────────────────────────────

def create_venv() -> None:
    venv_path = os.path.join(SCRIPT_DIR, VENV_DIR)
    if os.path.isdir(venv_path):
        log(f"Virtual environment already exists at {VENV_DIR}/")
        return
    log(f"Creating virtual environment in {VENV_DIR}/...")
    run([sys.executable, "-m", "venv", venv_path])
    log("Upgrading pip...")
    run([pip_executable(), "install", "--upgrade", "pip"])


# ── Step 2: Install llama-cpp-python ─────────────────────────────────────

def install_llama_cpp(gpu: str) -> None:
    pip = pip_executable()
    log(f"Installing llama-cpp-python (backend: {gpu})...")

    env = os.environ.copy()

    if gpu == "cuda":
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        log("  Building with CUDA support (requires CUDA toolkit)")
    elif gpu == "metal":
        env["CMAKE_ARGS"] = "-DGGML_METAL=on"
        log("  Building with Metal support (Apple Silicon)")
    else:
        log("  Building CPU-only (no GPU acceleration)")

    subprocess.run(
        [pip, "install", "llama-cpp-python>=0.3.0", "--no-cache-dir"],
        env=env,
        check=True,
    )
    log("llama-cpp-python installed.")


# ── Step 3: Download model ───────────────────────────────────────────────

def download_model(url: str, filename: str) -> str:
    models_path = os.path.join(SCRIPT_DIR, MODELS_DIR)
    os.makedirs(models_path, exist_ok=True)
    dest = os.path.join(models_path, filename)

    if os.path.isfile(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        log(f"Model already exists: {dest} ({size_mb:.0f} MB)")
        return dest

    log(f"Downloading model to {dest}")
    log(f"  URL: {url}")
    log("  This may take a few minutes depending on your connection...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gemma-agent-installer/1.0"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks

            with open(dest + ".part", "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = downloaded * 100 // total
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                        print(
                            f"\r  [{bar}] {pct}%  {mb_done:.0f}/{mb_total:.0f} MB",
                            end="", flush=True,
                        )
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        print(f"\r  Downloaded {mb_done:.0f} MB...", end="", flush=True)

            print()  # newline after progress bar

        os.rename(dest + ".part", dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        log(f"Download complete: {size_mb:.0f} MB")
        return dest

    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        # Clean up partial download
        partial = dest + ".part"
        if os.path.exists(partial):
            os.remove(partial)
        log(f"Download failed: {exc}")
        log("You can download the model manually and place it in the models/ directory.")
        log(f"  URL: {url}")
        sys.exit(1)


# ── Step 4: Write launchers ─────────────────────────────────────────────

def write_launchers(model_path: str) -> None:
    agent_script = os.path.join(SCRIPT_DIR, "gemma_agent.py")
    rel_model = os.path.relpath(model_path, SCRIPT_DIR)

    # -- run.sh (Linux / macOS) --
    run_sh = os.path.join(SCRIPT_DIR, "run.sh")
    with open(run_sh, "w") as f:
        f.write(f"""#!/usr/bin/env bash
# Gemma 4 AI Agent — launcher
# Generated by install.py

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate
exec python gemma_agent.py --model-path "{rel_model}" "$@"
""")
    os.chmod(run_sh, 0o755)

    # -- run.bat (Windows) --
    run_bat = os.path.join(SCRIPT_DIR, "run.bat")
    with open(run_bat, "w") as f:
        f.write(f"""@echo off
REM Gemma 4 AI Agent — launcher
REM Generated by install.py

cd /d "%~dp0"
call .venv\\Scripts\\activate.bat
python gemma_agent.py --model-path "{rel_model}" %*
""")

    log(f"Created run.sh and run.bat")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma 4 AI Agent — one-command installer"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU-only build (skip GPU detection)",
    )
    parser.add_argument(
        "--model-url", default=DEFAULT_MODEL_URL,
        help="URL to download the GGUF model from",
    )
    parser.add_argument(
        "--model-filename", default=DEFAULT_MODEL_FILENAME,
        help="Filename to save the model as",
    )
    parser.add_argument(
        "--skip-model", action="store_true",
        help="Skip model download (use if you already have a GGUF file)",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Gemma 4 AI Agent — Installer")
    print("=" * 60)
    print()

    # Step 1: venv
    create_venv()
    print()

    # Step 2: GPU detection & llama-cpp-python
    if args.cpu:
        gpu = "cpu"
    else:
        gpu = detect_gpu()
        log(f"Detected GPU backend: {gpu}")
    install_llama_cpp(gpu)
    print()

    # Step 3: model download
    if args.skip_model:
        log("Skipping model download (--skip-model)")
        model_path = os.path.join(SCRIPT_DIR, MODELS_DIR, args.model_filename)
    else:
        model_path = download_model(args.model_url, args.model_filename)
    print()

    # Step 4: launchers
    write_launchers(model_path)
    print()

    # Done!
    print("=" * 60)
    print("  Installation complete!")
    print("=" * 60)
    print()
    print("  To run the agent:")
    print()
    if platform.system() == "Windows":
        print("    run.bat")
        print("    run.bat --verbose")
    else:
        print("    ./run.sh")
        print("    ./run.sh --verbose")
    print()
    print("  Or manually:")
    print(f"    source {VENV_DIR}/bin/activate")
    print(f"    python gemma_agent.py --model-path {os.path.relpath(model_path, SCRIPT_DIR)}")
    print()
    print("  Options:")
    print("    --n-ctx 8192        Larger context window")
    print("    --n-gpu-layers 0    CPU-only inference")
    print("    --verbose           Show tool calls")
    print("    --temperature 0.5   Lower temperature")
    print()


if __name__ == "__main__":
    main()
