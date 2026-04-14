"""
Core inference module — extracted from Jan's AI engine patterns.

Jan uses an OpenAI-compatible chat completion interface
(see core/src/browser/extensions/engines/AIEngine.ts) and loads local
GGUF models via llama.cpp (see extensions/llamacpp-extension/).

This module provides two backends:
  1. LocalInferenceClient  — loads a GGUF file directly via llama-cpp-python
                             (fully offline, no server needed).
  2. RemoteInferenceClient — talks to an OpenAI-compatible HTTP endpoint
                             (Ollama, vLLM, etc.).
"""

from __future__ import annotations

import json
import uuid
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Generator, Protocol


# ---------------------------------------------------------------------------
# Types — mirrors Jan's chatCompletionRequest / chatCompletion
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """Single chat message (maps to Jan's chatCompletionRequestMessage)."""
    role: str  # "system" | "user" | "assistant" | "tool"
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
    """
    Minimal chat completion request.
    Mirrors Jan's chatCompletionRequest (AIEngine.ts).
    """
    model: str
    messages: list[Message]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = "auto"
    temperature: float = 0.7
    top_p: float = 0.8
    stream: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
        }
        if self.tools:
            d["tools"] = self.tools
            d["tool_choice"] = self.tool_choice or "auto"
        return d


@dataclass
class ChatCompletion:
    """Parsed non-streaming response (maps to Jan's chatCompletion)."""
    id: str
    model: str
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    finish_reason: str | None
    usage: dict[str, int] | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol — both clients implement this
# ---------------------------------------------------------------------------

class InferenceClient(Protocol):
    def chat(self, request: ChatCompletionRequest) -> ChatCompletion: ...


# ---------------------------------------------------------------------------
# 1. Local offline inference via llama-cpp-python
#    Mirrors Jan's llamacpp-extension: loads GGUF, runs chat completions
#    in-process with the same settings (ctx_size, threads, flash_attn, etc.)
# ---------------------------------------------------------------------------

class LocalInferenceClient:
    """
    Offline inference — loads a GGUF model directly using llama-cpp-python.
    No server process, no network. This is the Python equivalent of what
    Jan's llamacpp-extension does via its Tauri plugin.

    Install:  pip install llama-cpp-python
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
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "Offline inference requires llama-cpp-python.\n"
                "Install it with:  pip install llama-cpp-python\n"
                "For GPU support:   CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
            )

        print(f"Loading model: {model_path}")
        print(f"  n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}, flash_attn={flash_attn}")

        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "flash_attn": flash_attn,
            "verbose": verbose,
        }
        if n_threads is not None:
            kwargs["n_threads"] = n_threads

        self._llm = Llama(
            **kwargs,
            chat_format="chatml-function-calling",
        )
        self._model_path = model_path
        print("Model loaded.\n")

    def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        """Run chat completion locally — no network needed."""
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


# ---------------------------------------------------------------------------
# 2. Remote inference via HTTP (Ollama / vLLM / any OpenAI-compatible server)
#    Mirrors Jan's OAIEngine HTTP path.
# ---------------------------------------------------------------------------

class RemoteInferenceClient:
    """
    Talks to an OpenAI-compatible HTTP endpoint.
    Works with Ollama (localhost:11434/v1), vLLM, LM Studio, etc.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        request.stream = False
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(request.to_dict()).encode()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode() if exc.fp else ""
            raise RuntimeError(
                f"Inference request failed ({exc.code}): {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach inference server at {self.base_url}: {exc.reason}"
            ) from exc

        choice = data["choices"][0]
        message = choice["message"]

        return ChatCompletion(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            content=message.get("content"),
            tool_calls=message.get("tool_calls"),
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
            raw=data,
        )

    def chat_stream(
        self, request: ChatCompletionRequest
    ) -> Generator[str, None, None]:
        """Yield content deltas from a streaming chat completion."""
        request.stream = True
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(request.to_dict()).encode()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text
