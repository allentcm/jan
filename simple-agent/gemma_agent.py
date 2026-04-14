#!/usr/bin/env python3
"""
Gemma 4 AI Agent — single-file, fully offline.

All-in-one: inference engine, tool system, agent loop, and CLI.
Loads a local GGUF model via llama-cpp-python (the same llama.cpp engine
that powers Jan's llamacpp-extension). No server, no internet at runtime.

Run via the install.py installer, or directly:
    python gemma_agent.py [--model-path PATH] [--verbose]
"""

from __future__ import annotations

import json
import math
import uuid
import datetime
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Inference types & client
# Extracted from Jan's core/src/browser/extensions/engines/AIEngine.ts
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class ChatCompletionRequest:
    model: str
    messages: list[Message]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = "auto"
    temperature: float = 0.7
    top_p: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.tools:
            d["tools"] = self.tools
            d["tool_choice"] = self.tool_choice or "auto"
        return d


@dataclass
class ChatCompletion:
    id: str
    model: str
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    finish_reason: str | None
    usage: dict[str, int] | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


class LocalInferenceClient:
    """
    Offline inference — loads a GGUF model via llama-cpp-python.
    Mirrors Jan's llamacpp-extension (extensions/llamacpp-extension/).
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        flash_attn: bool = True,
        n_threads: int | None = None,
        verbose: bool = False,
    ):
        from llama_cpp import Llama  # installed by install.py

        print(f"Loading model: {model_path}")
        print(f"  context={n_ctx}  gpu_layers={n_gpu_layers}  flash_attn={flash_attn}")

        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "flash_attn": flash_attn,
            "verbose": verbose,
        }
        if n_threads is not None:
            kwargs["n_threads"] = n_threads

        self._llm = Llama(**kwargs, chat_format="chatml-function-calling")
        self._model_path = model_path
        print("Model loaded.\n")

    def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        messages = [m.to_dict() for m in request.messages]
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = request.tool_choice or "auto"

        result = self._llm.create_chat_completion(**kwargs)
        choice = result["choices"][0]
        message = choice["message"]

        return ChatCompletion(
            id=result.get("id", f"local-{uuid.uuid4().hex[:8]}"),
            model=result.get("model", self._model_path),
            content=message.get("content"),
            tool_calls=message.get("tool_calls"),
            finish_reason=choice.get("finish_reason"),
            usage=result.get("usage"),
            raw=result,
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Tool system
# Extracted from Jan's Tool / ToolFunction interfaces (AIEngine.ts)
# ═══════════════════════════════════════════════════════════════════════════

_TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def _register(name: str, description: str, parameters: dict, fn):
    _TOOL_REGISTRY[name] = {
        "spec": {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        },
        "fn": fn,
    }


def get_tool_specs() -> list[dict[str, Any]]:
    return [t["spec"] for t in _TOOL_REGISTRY.values()]


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    entry = _TOOL_REGISTRY.get(name)
    if entry is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return entry["fn"](**arguments)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# -- built-in tools --------------------------------------------------------

def _calculate(expression: str) -> str:
    allowed = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sqrt": math.sqrt, "log": math.log,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return json.dumps({"result": result})
    except Exception as exc:
        return json.dumps({"error": f"Cannot evaluate '{expression}': {exc}"})

_register("calculate",
    "Evaluate a math expression. Supports +, -, *, /, sqrt, log, sin, cos, tan, pi, e.",
    {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression, e.g. 'sqrt(2) * pi'"}}, "required": ["expression"]},
    _calculate)


def _get_current_time(timezone: str = "UTC") -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return json.dumps({"datetime": now.isoformat(), "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"), "timezone": timezone})

_register("get_current_time",
    "Get the current date and time.",
    {"type": "object", "properties": {"timezone": {"type": "string", "description": "Timezone (currently UTC only)."}}, "required": []},
    _get_current_time)


def _read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return json.dumps({"content": f.read(10_000)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

_register("read_file",
    "Read a local text file (up to 10 KB).",
    {"type": "object", "properties": {"path": {"type": "string", "description": "Path to file."}}, "required": ["path"]},
    _read_file)


def _write_file(path: str, content: str) -> str:
    try:
        with open(path, "w") as f:
            f.write(content)
        return json.dumps({"status": "ok", "bytes_written": len(content)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

_register("write_file",
    "Write text to a local file (creates or overwrites).",
    {"type": "object", "properties": {"path": {"type": "string", "description": "Path to file."}, "content": {"type": "string", "description": "Text content."}}, "required": ["path", "content"]},
    _write_file)


def _list_directory(path: str = ".") -> str:
    try:
        entries = os.listdir(path)
        return json.dumps({"entries": sorted(entries)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

_register("list_directory",
    "List files and folders in a directory.",
    {"type": "object", "properties": {"path": {"type": "string", "description": "Directory path (default: current dir)."}}, "required": []},
    _list_directory)


def _run_python(code: str) -> str:
    """Execute a Python snippet and capture stdout."""
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__})
        output = buf.getvalue()
        return json.dumps({"output": output[:5000]})
    except Exception as exc:
        return json.dumps({"error": str(exc), "partial_output": buf.getvalue()[:2000]})

_register("run_python",
    "Execute a Python code snippet and return stdout. Use for computation, data processing, etc.",
    {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute."}}, "required": ["code"]},
    _run_python)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Agent loop
# Extracted from Jan's assistant-extension + AIEngine.chat() flow
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful AI agent. You think step by step and use tools when needed.

When handling queries:
1. Think about what information you need.
2. Use available tools to gather data or perform actions.
3. Synthesize the results into a clear answer.

If you don't need a tool, just answer directly.\
"""

MAX_TOOL_ROUNDS = 10


class Agent:
    def __init__(
        self,
        client: LocalInferenceClient,
        model: str = "gemma4",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        verbose: bool = False,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.verbose = verbose
        self.client = client
        self.messages: list[Message] = []

    def run(self, user_input: str) -> str:
        if not self.messages:
            self.messages.append(Message(role="system", content=self.system_prompt))

        self.messages.append(Message(role="user", content=user_input))
        tool_specs = get_tool_specs()

        for round_idx in range(MAX_TOOL_ROUNDS):
            if self.verbose:
                print(f"\n--- Agent round {round_idx + 1} ---")

            request = ChatCompletionRequest(
                model=self.model,
                messages=list(self.messages),
                tools=tool_specs if tool_specs else None,
                temperature=self.temperature,
            )
            completion = self.client.chat(request)

            if completion.tool_calls:
                self.messages.append(Message(
                    role="assistant", content=completion.content,
                    tool_calls=completion.tool_calls,
                ))
                for tc in completion.tool_calls:
                    fn = tc["function"]
                    tool_name = fn["name"]
                    try:
                        tool_args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                    except json.JSONDecodeError:
                        tool_args = {}

                    if self.verbose:
                        print(f"  Tool: {tool_name}({json.dumps(tool_args)})")

                    result = call_tool(tool_name, tool_args)
                    if self.verbose:
                        print(f"  Result: {result[:200]}")

                    self.messages.append(Message(
                        role="tool", content=result,
                        tool_call_id=tc.get("id", ""),
                    ))
                continue

            answer = completion.content or ""
            self.messages.append(Message(role="assistant", content=answer))
            if self.verbose:
                print(f"  Answer: {answer[:200]}")
            return answer

        return "(Agent reached maximum tool-call rounds without a final answer.)"

    def reset(self) -> None:
        self.messages.clear()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLI
# ═══════════════════════════════════════════════════════════════════════════

def find_default_model() -> str | None:
    """Look for a .gguf file in the models/ subdirectory next to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    if os.path.isdir(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".gguf"):
                return os.path.join(models_dir, f)
    # Also check script dir itself
    for f in sorted(os.listdir(script_dir)):
        if f.endswith(".gguf"):
            return os.path.join(script_dir, f)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 AI Agent — fully offline")
    parser.add_argument("--model-path", default=None, help="Path to GGUF model file (auto-detected from models/ if omitted)")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context window (default: 4096)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1=all, 0=CPU only)")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (default: auto)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--verbose", action="store_true", help="Show tool calls and intermediate steps")
    args = parser.parse_args()

    model_path = args.model_path or find_default_model()
    if not model_path:
        print("Error: No model found. Run install.py first, or pass --model-path.", file=sys.stderr)
        print("  python install.py            # downloads model + deps", file=sys.stderr)
        print("  python gemma_agent.py --model-path /path/to/model.gguf", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    client = LocalInferenceClient(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        flash_attn=not args.no_flash_attn,
        n_threads=args.threads,
        verbose=args.verbose,
    )

    agent = Agent(client=client, temperature=args.temperature, verbose=args.verbose)

    print("Gemma 4 AI Agent (offline)")
    print(f"Model: {os.path.basename(model_path)}")
    print("Tools: calculate, get_current_time, read_file, write_file, list_directory, run_python")
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
