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
# The installer uses huggingface-hub to download, which handles auth tokens
# and mirrors automatically.  Override with --model-repo / --model-file.
DEFAULT_MODEL_REPO = "bartowski/google_gemma-3-4b-it-GGUF"
DEFAULT_MODEL_FILE = "google_gemma-3-4b-it-Q4_K_M.gguf"
DEFAULT_MODEL_SAVE_AS = "gemma-4-4b-it-Q4_K_M.gguf"

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

def install_deps(gpu: str) -> None:
    pip = pip_executable()

    # llama-cpp-python (with GPU backend)
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
        env=env, check=True,
    )
    log("llama-cpp-python installed.")

    # huggingface-hub (for model downloads — handles auth, gated models, mirrors)
    log("Installing huggingface-hub...")
    subprocess.run(
        [pip, "install", "huggingface-hub>=0.20.0"],
        check=True, capture_output=True,
    )
    log("huggingface-hub installed.")

    # pydantic-ai (agent framework for the browser agent)
    log("Installing pydantic-ai[openai]...")
    subprocess.run(
        [pip, "install", "pydantic-ai[openai]>=0.1.0"],
        check=True, capture_output=True,
    )
    log("pydantic-ai installed.")


# ── Step 2b: Install agent-browser ────────────────────────────────────────

def install_agent_browser() -> None:
    """Install agent-browser CLI and its browser dependency."""
    if shutil.which("agent-browser"):
        log("agent-browser already installed.")
        return

    if not shutil.which("npm"):
        log("WARNING: npm not found. agent-browser requires Node.js.")
        log("  Install Node.js from https://nodejs.org")
        log("  Then run: npm install -g agent-browser && agent-browser install")
        return

    log("Installing agent-browser CLI via npm...")
    try:
        subprocess.run(["npm", "install", "-g", "agent-browser"], check=True)
        log("Running agent-browser install (downloads Chrome)...")
        subprocess.run(["agent-browser", "install"], check=True, timeout=300)
        log("agent-browser installed.")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log(f"agent-browser install failed: {exc}")
        log("  You can install manually: npm install -g agent-browser && agent-browser install")


# ── Step 3: Download model ───────────────────────────────────────────────

def download_model(repo: str, filename: str, save_as: str) -> str:
    models_path = os.path.join(SCRIPT_DIR, MODELS_DIR)
    os.makedirs(models_path, exist_ok=True)
    dest = os.path.join(models_path, save_as)

    if os.path.isfile(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        log(f"Model already exists: {dest} ({size_mb:.0f} MB)")
        return dest

    log(f"Downloading model from {repo}")
    log(f"  File: {filename}")
    log("  This may take a few minutes depending on your connection...")

    python = python_executable()
    try:
        # Use huggingface-hub (installed in venv) for robust downloads:
        # handles gated models, auth tokens, mirrors, resume, progress bar.
        result = subprocess.run(
            [python, "-c", (
                "from huggingface_hub import hf_hub_download; "
                f"p = hf_hub_download('{repo}', '{filename}', "
                f"local_dir='{models_path}'); "
                "print(p)"
            )],
            check=True, capture_output=True, text=True, timeout=1200,
        )
        downloaded_path = result.stdout.strip()

        # Rename to our preferred filename if different
        if downloaded_path and os.path.isfile(downloaded_path) and downloaded_path != dest:
            os.rename(downloaded_path, dest)

        if os.path.isfile(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            log(f"Download complete: {size_mb:.0f} MB")
            return dest
        else:
            raise RuntimeError("Download finished but file not found")

    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        log(f"Download failed.")

        if "401" in stderr or "403" in stderr or "gated" in stderr.lower():
            log("")
            log("This model requires accepting a license on HuggingFace.")
            log("To fix this:")
            log(f"  1. Visit https://huggingface.co/{repo}")
            log("  2. Accept the model license / request access")
            log("  3. Create an access token at https://huggingface.co/settings/tokens")
            log("  4. Run: huggingface-cli login")
            log("  5. Re-run: python install.py")
        else:
            log(f"  Error: {stderr[:300]}")

        log("")
        log("Or download manually and place the .gguf file in the models/ directory:")
        log(f"  https://huggingface.co/{repo}")
        sys.exit(1)

    except Exception as exc:
        log(f"Download failed: {exc}")
        log("Download manually and place the .gguf in the models/ directory.")
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
        "--model-repo", default=DEFAULT_MODEL_REPO,
        help=f"HuggingFace repo ID (default: {DEFAULT_MODEL_REPO})",
    )
    parser.add_argument(
        "--model-file", default=DEFAULT_MODEL_FILE,
        help=f"GGUF filename within the repo (default: {DEFAULT_MODEL_FILE})",
    )
    parser.add_argument(
        "--model-save-as", default=DEFAULT_MODEL_SAVE_AS,
        help=f"Local filename to save model as (default: {DEFAULT_MODEL_SAVE_AS})",
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

    # Step 2: GPU detection & dependencies
    if args.cpu:
        gpu = "cpu"
    else:
        gpu = detect_gpu()
        log(f"Detected GPU backend: {gpu}")
    install_deps(gpu)
    print()

    # Step 2b: agent-browser
    install_agent_browser()
    print()

    # Step 3: model download
    if args.skip_model:
        log("Skipping model download (--skip-model)")
        model_path = os.path.join(SCRIPT_DIR, MODELS_DIR, args.model_save_as)
    else:
        model_path = download_model(args.model_repo, args.model_file, args.model_save_as)
    print()

    # Step 4: launchers
    write_launchers(model_path)
    print()

    # Done!
    print("=" * 60)
    print("  Installation complete!")
    print("=" * 60)
    print()
    print("  Standalone agent (offline, no browser):")
    if platform.system() == "Windows":
        print("    run.bat")
    else:
        print("    ./run.sh")
    print()
    print("  Browser agent (web browsing + mind map):")
    rel_model = os.path.relpath(model_path, SCRIPT_DIR)
    print(f"    source {VENV_DIR}/bin/activate")
    print(f"    python browser_agent.py --interactive")
    print(f"    python browser_agent.py \"Find the pricing on example.com\"")
    print(f"    python browser_agent.py --headed --verbose \"Search Google for pydantic ai\"")
    print()
    print("  Options:")
    print("    --model gemma3:4b    Model name (Ollama)")
    print("    --headed             Show browser window")
    print("    --verbose            Show tool calls")
    print("    --mindmap FILE       Mind map JSON path")
    print()


if __name__ == "__main__":
    main()
