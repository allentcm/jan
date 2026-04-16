#!/usr/bin/env python3
"""
End-to-end test suite for gemma_agent.py.

Run with:
    .venv/bin/python3 test_agent.py

Tests the full pipeline: imports, tool system, model loading,
chat completion, agent loop, multi-turn, and reset.
Requires a GGUF model in models/ (test-tiny.gguf is fine).
"""

import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemma_agent import (
    Message, ChatCompletionRequest, ChatCompletion,
    LocalInferenceClient, Agent,
    get_tool_specs, call_tool,
    find_default_model,
)

passed = 0
failed = 0


def test(name):
    global passed, failed
    class _ctx:
        def __enter__(self):
            print(f"Test: {name}...", end=" ", flush=True)
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            global passed, failed
            if exc_type:
                failed += 1
                print(f"FAILED: {exc_val}")
                return True  # suppress
            passed += 1
            print("OK")
    return _ctx()


# ── Tool system ──────────────────────────────────────────────────────────

with test("Tool registry has 6 tools"):
    specs = get_tool_specs()
    assert len(specs) == 6

with test("calculate(2+2) == 4"):
    r = json.loads(call_tool("calculate", {"expression": "2 + 2"}))
    assert r["result"] == 4

with test("calculate(sqrt(9)) == 3"):
    r = json.loads(call_tool("calculate", {"expression": "sqrt(9)"}))
    assert r["result"] == 3.0

with test("get_current_time returns datetime"):
    r = json.loads(call_tool("get_current_time", {}))
    assert "datetime" in r and "date" in r

with test("list_directory finds gemma_agent.py"):
    r = json.loads(call_tool("list_directory", {"path": "."}))
    assert "gemma_agent.py" in r["entries"]

with test("run_python executes code"):
    r = json.loads(call_tool("run_python", {"code": "print(sum(range(10)))"}))
    assert r["output"].strip() == "45"

with test("read_file reads this test"):
    r = json.loads(call_tool("read_file", {"path": __file__}))
    assert "End-to-end test suite" in r["content"]

with test("write_file + read_file roundtrip"):
    tmp = "/tmp/_agent_test_tmp.txt"
    call_tool("write_file", {"path": tmp, "content": "hello"})
    r = json.loads(call_tool("read_file", {"path": tmp}))
    assert r["content"] == "hello"
    os.remove(tmp)

with test("Unknown tool returns error"):
    r = json.loads(call_tool("no_such_tool", {}))
    assert "error" in r

# ── Model auto-detection ─────────────────────────────────────────────────

with test("find_default_model locates GGUF"):
    found = find_default_model()
    assert found is not None and found.endswith(".gguf")

# ── Inference client ─────────────────────────────────────────────────────

model_path = find_default_model()
client = None

with test("LocalInferenceClient loads model"):
    client = LocalInferenceClient(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=0,
        flash_attn=False,
        verbose=False,
    )

with test("Chat completion returns valid structure"):
    req = ChatCompletionRequest(
        model="test",
        messages=[Message(role="user", content="Hello")],
        temperature=0.8,
    )
    c = client.chat(req)
    assert isinstance(c, ChatCompletion)
    assert c.finish_reason is not None
    assert c.usage is not None
    assert c.usage["prompt_tokens"] > 0

# ── Agent loop ───────────────────────────────────────────────────────────

with test("Agent.run returns a string"):
    agent = Agent(client=client, temperature=0.8)
    r = agent.run("Hello")
    assert isinstance(r, str)

with test("Agent history has system + user + assistant"):
    assert len(agent.messages) >= 3
    assert agent.messages[0].role == "system"
    assert agent.messages[1].role == "user"
    assert agent.messages[-1].role == "assistant"

with test("Multi-turn preserves history"):
    r2 = agent.run("Tell me more")
    assert len(agent.messages) >= 5

with test("Agent.reset clears history"):
    agent.reset()
    assert len(agent.messages) == 0

# ── Summary ──────────────────────────────────────────────────────────────

print()
total = passed + failed
print(f"{'=' * 50}")
if failed == 0:
    print(f"  ALL {passed} TESTS PASSED")
else:
    print(f"  {passed}/{total} passed, {failed} FAILED")
print(f"{'=' * 50}")

sys.exit(1 if failed else 0)
