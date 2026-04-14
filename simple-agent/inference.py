"""
Core inference module — extracted from Jan's AI engine patterns.

Jan uses an OpenAI-compatible chat completion interface (see core/src/browser/extensions/engines/AIEngine.ts).
This module distills that into a minimal Python client that speaks the same protocol,
targeting a local Gemma 4 model served via Ollama or any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Generator


# ---------------------------------------------------------------------------
# Types — mirrors Jan's chatCompletionRequest / chatCompletion / chatCompletionChunk
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
    Mirrors Jan's chatCompletionRequest (AIEngine.ts) — only the fields
    that matter for a simple agent loop.
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
# Client — the actual HTTP call, analogous to Jan's OAIEngine.chat()
# ---------------------------------------------------------------------------

class InferenceClient:
    """
    Lightweight OpenAI-compatible inference client.
    Works with Ollama (default http://localhost:11434/v1) or any
    OpenAI-compatible server.
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

    # -- non-streaming chat -------------------------------------------------

    def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        """Send a chat completion request and return the parsed response."""
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

    # -- streaming chat -----------------------------------------------------

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
