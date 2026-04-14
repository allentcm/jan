#!/usr/bin/env python3
"""
Simple AI Agent with Gemma 4 — interactive CLI.

Usage:
    # Start with defaults (Ollama at localhost:11434, model "gemma4")
    python main.py

    # Custom endpoint / model
    python main.py --base-url http://localhost:11434/v1 --model gemma4 --verbose

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull Gemma 4:   ollama pull gemma4
    3. Run this script: python main.py
"""

from __future__ import annotations

import argparse
import sys

from agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple AI Agent powered by Gemma 4"
    )
    parser.add_argument(
        "--model", default="gemma4", help="Model name (default: gemma4)"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible API base URL (default: Ollama)",
    )
    parser.add_argument(
        "--api-key", default="", help="API key (if required by the server)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show tool calls and intermediate steps"
    )
    args = parser.parse_args()

    agent = Agent(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    print("Simple AI Agent (Gemma 4)")
    print(f"Model: {args.model} | Server: {args.base_url}")
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
