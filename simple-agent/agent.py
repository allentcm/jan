"""
Simple AI Agent — a minimal agentic loop powered by Gemma 4.

Extracted from Jan's inference architecture:
  - Jan's AIEngine.chat() handles a single request/response turn.
  - Jan's assistant-extension provides system instructions and tool definitions.
  - Jan's OAIEngine ties them together with event-based message routing.

This module collapses that into a single Agent class that runs a
think-act-observe loop with tool calling, using the OpenAI-compatible
chat completion API that Gemma 4 supports (via Ollama or similar).
"""

from __future__ import annotations

import json
from typing import Any

from inference import InferenceClient, ChatCompletionRequest, Message
from tools import get_tool_specs, call_tool


# Default system prompt — inspired by Jan's assistant-extension default instructions
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful AI agent. You think step by step and use tools when needed.

When handling queries:
1. Think about what information you need.
2. Use available tools to gather data or perform actions.
3. Synthesize the results into a clear answer.

If you don't need a tool, just answer directly.\
"""

MAX_TOOL_ROUNDS = 10  # safety cap on agent loop iterations


class Agent:
    """
    A simple agentic loop: send messages to Gemma 4, let it call tools,
    feed results back, repeat until the model produces a final answer.
    """

    def __init__(
        self,
        model: str = "gemma4",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        verbose: bool = False,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.verbose = verbose
        self.client = InferenceClient(base_url=base_url, api_key=api_key)
        self.messages: list[Message] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """
        Run the agent loop for a single user turn.
        Returns the final assistant text response.
        """
        # Seed system prompt on first turn
        if not self.messages:
            self.messages.append(Message(role="system", content=self.system_prompt))

        # Append the user message
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

            # If the model wants to call tools, execute them and loop
            if completion.tool_calls:
                # Record the assistant message (with tool_calls, possibly empty content)
                self.messages.append(
                    Message(
                        role="assistant",
                        content=completion.content,
                        tool_calls=completion.tool_calls,
                    )
                )

                for tc in completion.tool_calls:
                    fn = tc["function"]
                    tool_name = fn["name"]
                    try:
                        tool_args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                    except json.JSONDecodeError:
                        tool_args = {}

                    if self.verbose:
                        print(f"  Tool call: {tool_name}({json.dumps(tool_args)})")

                    result = call_tool(tool_name, tool_args)

                    if self.verbose:
                        print(f"  Result: {result[:200]}")

                    self.messages.append(
                        Message(
                            role="tool",
                            content=result,
                            tool_call_id=tc.get("id", ""),
                        )
                    )

                continue  # let the model see tool results

            # No tool calls — this is the final answer
            answer = completion.content or ""
            self.messages.append(Message(role="assistant", content=answer))

            if self.verbose:
                print(f"  Final answer: {answer[:200]}")

            return answer

        # Exhausted rounds — return whatever we have
        return "(Agent reached maximum tool-call rounds without a final answer.)"

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
