#!/usr/bin/env python3
"""
Simple AI Agent with Gemma 4 — interactive CLI.

Usage (offline — no server needed):
    pip install llama-cpp-python
    python main.py --model-path ./gemma-4-4b-it-Q4_K_M.gguf

Usage (remote — Ollama or any OpenAI-compatible server):
    python main.py --base-url http://localhost:11434/v1 --model gemma4

Prerequisites for offline mode:
    1. pip install llama-cpp-python
       (GPU: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python)
    2. Download a Gemma 4 GGUF file from HuggingFace
    3. python main.py --model-path /path/to/gemma4.gguf
"""

from __future__ import annotations

import argparse
import sys

from agent import Agent
from inference import LocalInferenceClient, RemoteInferenceClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple AI Agent powered by Gemma 4"
    )

    # Model source — pick one
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model-path",
        help="Path to a local GGUF file for fully offline inference (no server needed).",
    )
    group.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL (default: Ollama at localhost:11434/v1)",
    )

    parser.add_argument(
        "--model", default="gemma4", help="Model name for remote mode (default: gemma4)"
    )
    parser.add_argument(
        "--api-key", default="", help="API key (remote mode only)"
    )

    # Local model settings (mirrors Jan's llamacpp-extension/settings.json)
    parser.add_argument(
        "--n-ctx", type=int, default=4096,
        help="Context window size in tokens (default: 4096)",
    )
    parser.add_argument(
        "--n-gpu-layers", type=int, default=-1,
        help="Layers to offload to GPU (-1 = all, 0 = CPU only, default: -1)",
    )
    parser.add_argument(
        "--flash-attn", action="store_true", default=True,
        help="Enable flash attention (default: on)",
    )
    parser.add_argument(
        "--no-flash-attn", action="store_true",
        help="Disable flash attention",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Number of CPU threads (default: auto)",
    )

    # Sampling
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show tool calls and intermediate steps"
    )
    args = parser.parse_args()

    flash_attn = args.flash_attn and not args.no_flash_attn

    # Build the right client
    if args.model_path:
        client = LocalInferenceClient(
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            flash_attn=flash_attn,
            n_threads=args.threads,
            verbose=args.verbose,
        )
        mode_label = f"Offline | {args.model_path}"
    else:
        base_url = args.base_url or "http://localhost:11434/v1"
        client = RemoteInferenceClient(
            base_url=base_url,
            api_key=args.api_key,
        )
        mode_label = f"Remote | {args.model} @ {base_url}"

    agent = Agent(
        client=client,
        model=args.model,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    print("Simple AI Agent (Gemma 4)")
    print(f"Mode: {mode_label}")
    print("Tools: calculate, get_current_time, read_file, write_file")
    print('Type "quit" to exit, "reset" to clear history.\n')

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("(conversation cleared)\n")
            continue

        try:
            response = agent.run(user_input)
            print(f"\nAgent: {response}\n")
        except Exception as exc:
            print(f"\nError: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
