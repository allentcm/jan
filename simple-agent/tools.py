"""
Tool definitions for the simple Gemma 4 agent.

Jan's assistant extension supports tools via the OpenAI function-calling format
(see core/src/browser/extensions/engines/AIEngine.ts — Tool / ToolFunction types).
This module provides concrete tool implementations the agent can call.
"""

from __future__ import annotations

import json
import math
import datetime
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

# Each tool is: { "spec": <OpenAI tool schema>, "fn": <callable(args) -> str> }
_TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    fn: Callable[..., str],
) -> None:
    """Register a tool the agent can call."""
    _TOOL_REGISTRY[name] = {
        "spec": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "fn": fn,
    }


def get_tool_specs() -> list[dict[str, Any]]:
    """Return OpenAI-format tool specs for all registered tools."""
    return [t["spec"] for t in _TOOL_REGISTRY.values()]


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a registered tool by name and return the result string."""
    entry = _TOOL_REGISTRY.get(name)
    if entry is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return entry["fn"](**arguments)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

def _calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    allowed = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sqrt": math.sqrt, "log": math.log,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return json.dumps({"result": result})
    except Exception as exc:
        return json.dumps({"error": f"Cannot evaluate '{expression}': {exc}"})


register_tool(
    name="calculate",
    description="Evaluate a mathematical expression. Supports basic arithmetic, sqrt, log, sin, cos, tan, pi, e.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g. 'sqrt(2) * pi'",
            }
        },
        "required": ["expression"],
    },
    fn=_calculate,
)


def _get_current_time(timezone: str = "UTC") -> str:
    """Return the current date and time."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return json.dumps({
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": timezone,
    })


register_tool(
    name="get_current_time",
    description="Get the current date and time.",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone name (currently only UTC is supported).",
            }
        },
        "required": [],
    },
    fn=_get_current_time,
)


def _read_file(path: str) -> str:
    """Read and return the contents of a local file."""
    try:
        with open(path, "r") as f:
            content = f.read(10_000)  # cap at 10 KB
        return json.dumps({"content": content})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


register_tool(
    name="read_file",
    description="Read the contents of a local text file (up to 10 KB).",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            }
        },
        "required": ["path"],
    },
    fn=_read_file,
)


def _write_file(path: str, content: str) -> str:
    """Write content to a local file."""
    try:
        with open(path, "w") as f:
            f.write(content)
        return json.dumps({"status": "ok", "bytes_written": len(content)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


register_tool(
    name="write_file",
    description="Write text content to a local file (creates or overwrites).",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write.",
            },
        },
        "required": ["path", "content"],
    },
    fn=_write_file,
)
