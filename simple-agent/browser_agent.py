#!/usr/bin/env python3
"""
Browser Agent — a Pydantic AI agent that browses the web with a mind map.

Uses:
  - pydantic-ai: agent framework with typed tool calling
  - agent-browser: Rust CLI for fast browser automation
  - mindmap.py: persistent graph of everything the agent has learned

The mind map records page structure, links, and navigation paths.
On revisit, the agent skips exploration and navigates directly.

Usage:
    python browser_agent.py "Find the pricing on example.com"
    python browser_agent.py --interactive
    python browser_agent.py --headed "Search for pydantic ai on Google"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from browser import Browser
from mindmap import MindMap


# ═══════════════════════════════════════════════════════════════════════════
# Dependencies (shared state passed to all tools via RunContext)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Deps:
    browser: Browser
    mindmap: MindMap
    current_url: str = ""
    verbose: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# System prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a browser agent that navigates the web to accomplish tasks.

You have a MIND MAP — a memory of every website you've visited before.
Before opening any URL, ALWAYS check the mind map first with `mindmap_lookup`.
If you've been there before, use the stored navigation hints and page structure
to act faster instead of re-exploring the whole page.

Workflow:
1. Check the mind map for prior knowledge about the target URL/domain.
2. Open the URL (or navigate using known shortcuts).
3. Take a snapshot to see the page structure (interactive elements with @ref IDs).
4. Interact: click links (@e1), fill forms, scroll, etc.
5. The mind map updates automatically after each page visit.

Tips:
- Use @ref IDs from snapshots to click/fill (e.g. click "@e5").
- Take a new snapshot after navigation to see updated elements.
- Use `mindmap_search` to find pages you've visited about a topic.
- Scroll down if you don't see what you need in the snapshot.
- When done, provide a clear answer based on what you found.\
"""


# ═══════════════════════════════════════════════════════════════════════════
# Agent definition
# ═══════════════════════════════════════════════════════════════════════════

def create_agent(model: OpenAIChatModel) -> Agent[Deps, str]:
    """Build the Pydantic AI agent with all browser + mind map tools."""

    agent = Agent(
        model,
        deps_type=Deps,
        result_type=str,
        system_prompt=SYSTEM_PROMPT,
    )

    # ── Mind map tools ────────────────────────────────────────────────

    @agent.tool
    async def mindmap_lookup(ctx: RunContext[Deps], url: str) -> str:
        """Check what the mind map knows about a URL (prior visits, page structure, links).
        ALWAYS call this before opening a URL to see if we have prior knowledge."""
        context = ctx.deps.mindmap.get_context_for_url(url)
        return context

    @agent.tool
    async def mindmap_search(ctx: RunContext[Deps], query: str) -> str:
        """Search the mind map for pages matching a keyword. Returns previously visited
        pages related to the query."""
        results = ctx.deps.mindmap.search(query)
        if not results:
            return f"No pages in mind map matching '{query}'."
        lines = [f"Found {len(results)} pages matching '{query}':"]
        for node in results[:10]:
            lines.append(f"  - {node.url}: {node.title} (visited {node.visit_count}x)")
            if node.summary:
                lines.append(f"    Summary: {node.summary[:100]}")
        return "\n".join(lines)

    @agent.tool
    async def mindmap_stats(ctx: RunContext[Deps]) -> str:
        """Get mind map statistics (total pages, domains, links)."""
        return ctx.deps.mindmap.get_stats()

    # ── Browser navigation ────────────────────────────────────────────

    @agent.tool
    async def browse_open(ctx: RunContext[Deps], url: str) -> str:
        """Open a URL in the browser. Returns the page title.
        The mind map is automatically updated with what we find."""
        parent_url = ctx.deps.current_url

        result = ctx.deps.browser.open(url)
        if not result.success:
            return f"Failed to open {url}: {result.error}"

        # Get page info
        title_result = ctx.deps.browser.get_title()
        title = title_result.data.get("title", "") if title_result.success else ""
        url_result = ctx.deps.browser.get_url()
        actual_url = url_result.data.get("url", url) if url_result.success else url

        ctx.deps.current_url = actual_url

        # Take a snapshot and update mind map
        snap = ctx.deps.browser.snapshot(interactive=True)
        snapshot_text = snap.data.get("text", snap.raw_stdout) if snap.success else ""

        ctx.deps.mindmap.record_visit(
            url=actual_url,
            title=title,
            snapshot_text=snapshot_text,
            parent_url=parent_url if parent_url else None,
        )

        if ctx.deps.verbose:
            print(f"  [browse] Opened: {actual_url} — {title}")
            print(f"  [mindmap] {ctx.deps.mindmap.get_stats()}")

        return f"Opened: {title} ({actual_url}). Take a snapshot to see page elements."

    @agent.tool
    async def browse_snapshot(ctx: RunContext[Deps]) -> str:
        """Get the current page's accessibility tree. Shows interactive elements
        with @ref IDs (like @e1, @e2) that you can use for clicking and filling."""
        result = ctx.deps.browser.snapshot(interactive=True)
        if not result.success:
            return f"Snapshot failed: {result.error}"

        text = result.data.get("text", result.raw_stdout)

        # Update mind map with fresh snapshot
        if ctx.deps.current_url:
            ctx.deps.mindmap.record_visit(
                url=ctx.deps.current_url,
                snapshot_text=text,
            )

        # Truncate if very long (LLM context limit)
        if len(text) > 8000:
            text = text[:8000] + "\n... (truncated, scroll or use a more specific selector)"

        return text

    @agent.tool
    async def browse_click(ctx: RunContext[Deps], selector: str) -> str:
        """Click an element. Use @ref IDs from the snapshot (e.g. "@e5").
        After clicking a link, take a new snapshot to see the new page."""
        old_url = ctx.deps.current_url

        result = ctx.deps.browser.click(selector)
        if not result.success:
            return f"Click failed: {result.error}"

        # Check if navigation happened
        ctx.deps.browser.wait(500)
        url_result = ctx.deps.browser.get_url()
        new_url = url_result.data.get("url", "") if url_result.success else ""

        if new_url and new_url != old_url:
            ctx.deps.current_url = new_url
            # Record the navigation path in mind map
            ctx.deps.mindmap.record_navigation(old_url, new_url, f"click {selector}")

            title_result = ctx.deps.browser.get_title()
            title = title_result.data.get("title", "") if title_result.success else ""

            snap = ctx.deps.browser.snapshot(interactive=True)
            snapshot_text = snap.data.get("text", snap.raw_stdout) if snap.success else ""
            ctx.deps.mindmap.record_visit(
                url=new_url, title=title, snapshot_text=snapshot_text, parent_url=old_url,
            )

            return f"Clicked {selector} → navigated to {new_url} ({title}). Take a snapshot to see elements."

        return f"Clicked {selector}. Page may have updated — take a snapshot to see changes."

    @agent.tool
    async def browse_fill(ctx: RunContext[Deps], selector: str, text: str) -> str:
        """Fill a text input field. Use @ref IDs from the snapshot."""
        result = ctx.deps.browser.fill(selector, text)
        if not result.success:
            return f"Fill failed: {result.error}"
        return f"Filled {selector} with '{text}'."

    @agent.tool
    async def browse_press_key(ctx: RunContext[Deps], key: str) -> str:
        """Press a keyboard key (e.g. 'Enter', 'Tab', 'Escape', 'Control+a')."""
        result = ctx.deps.browser.press(key)
        if not result.success:
            return f"Key press failed: {result.error}"
        return f"Pressed {key}."

    @agent.tool
    async def browse_scroll(ctx: RunContext[Deps], direction: str = "down") -> str:
        """Scroll the page. Direction: 'up' or 'down'."""
        result = ctx.deps.browser.scroll(direction, pixels=500)
        if not result.success:
            return f"Scroll failed: {result.error}"
        return f"Scrolled {direction}. Take a snapshot to see new content."

    @agent.tool
    async def browse_get_text(ctx: RunContext[Deps], selector: str) -> str:
        """Get the text content of a specific element."""
        result = ctx.deps.browser.get_text(selector)
        if not result.success:
            return f"Get text failed: {result.error}"
        return result.data.get("text", result.raw_stdout)

    @agent.tool
    async def browse_back(ctx: RunContext[Deps]) -> str:
        """Go back to the previous page."""
        result = ctx.deps.browser.back()
        if not result.success:
            return f"Back failed: {result.error}"

        ctx.deps.browser.wait(500)
        url_result = ctx.deps.browser.get_url()
        ctx.deps.current_url = url_result.data.get("url", "") if url_result.success else ""
        return f"Went back to {ctx.deps.current_url}."

    @agent.tool
    async def browse_screenshot(ctx: RunContext[Deps], filename: str = "screenshot.png") -> str:
        """Take a screenshot of the current page."""
        result = ctx.deps.browser.screenshot(filename)
        if not result.success:
            return f"Screenshot failed: {result.error}"
        return f"Screenshot saved to {filename}."

    @agent.tool
    async def browse_eval_js(ctx: RunContext[Deps], script: str) -> str:
        """Execute JavaScript in the browser and return the result."""
        result = ctx.deps.browser.eval_js(script)
        if not result.success:
            return f"JS eval failed: {result.error}"
        return json.dumps(result.data)

    return agent


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

async def run_task(agent: Agent[Deps, str], deps: Deps, task: str) -> str:
    """Run a single task and return the result."""
    result = await agent.run(task, deps=deps)
    return result.data


async def interactive_loop(agent: Agent[Deps, str], deps: Deps) -> None:
    """Run an interactive session."""
    print("Browser Agent (Pydantic AI + agent-browser + Mind Map)")
    print(f"Model: {os.environ.get('MODEL_NAME', 'gemma3:4b')}")
    print(f"Mind map: {deps.mindmap.get_stats()}")
    print('Type "quit" to exit, "stats" for mind map stats, "reset" to clear history.\n')

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "stats":
            print(f"\n{deps.mindmap.get_stats()}\n")
            continue

        try:
            result = await agent.run(user_input, deps=deps)
            print(f"\nAgent: {result.data}\n")
        except Exception as exc:
            print(f"\nError: {exc}\n", file=sys.stderr)

    # Cleanup
    deps.browser.close()
    print("Browser closed. Bye!")


def build_model(model_name: str, base_url: str, api_key: str) -> OpenAIChatModel:
    """Build a Pydantic AI OpenAI-compatible model."""
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIChatModel(model_name, provider=provider)


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser Agent with Mind Map")
    parser.add_argument("task", nargs="?", default=None, help="Task to perform (or omit for interactive mode)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gemma3:4b"), help="Model name")
    parser.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL", "http://localhost:11434/v1"), help="OpenAI-compatible API URL")
    parser.add_argument("--api-key", default=os.environ.get("MODEL_API_KEY", "ollama"), help="API key")
    parser.add_argument("--mindmap", default="mindmap.json", help="Path to mind map file")
    parser.add_argument("--headed", action="store_true", help="Show the browser window")
    parser.add_argument("--verbose", action="store_true", help="Show tool calls")
    parser.add_argument("--session", default="agent", help="Browser session name")
    args = parser.parse_args()

    model = build_model(args.model, args.base_url, args.api_key)
    agent = create_agent(model)

    deps = Deps(
        browser=Browser(session=args.session, headed=args.headed),
        mindmap=MindMap(path=args.mindmap),
        verbose=args.verbose,
    )

    if not deps.browser.is_installed:
        print("Error: agent-browser is not installed.", file=sys.stderr)
        print("Install with: npm install -g agent-browser && agent-browser install", file=sys.stderr)
        sys.exit(1)

    if args.task and not args.interactive:
        # Single task mode
        result = asyncio.run(run_task(agent, deps, args.task))
        print(result)
        deps.browser.close()
    else:
        # Interactive mode
        asyncio.run(interactive_loop(agent, deps))


if __name__ == "__main__":
    main()
