"""
Python wrapper for the agent-browser CLI.

agent-browser is a Rust-native CLI for browser automation designed for AI agents.
This module wraps it via subprocess + JSON, giving Python code clean access to
navigation, snapshots, clicking, form-filling, and more.

See: https://github.com/vercel-labs/agent-browser
"""

from __future__ import annotations

import json
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BrowserResult:
    """Parsed response from an agent-browser command."""
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    raw_stdout: str = ""


class Browser:
    """
    Thin wrapper around the agent-browser CLI.
    Every method runs a subprocess with --json and returns structured data.
    """

    def __init__(self, session: str = "agent", headed: bool = False, timeout: int = 30):
        self.session = session
        self.headed = headed
        self.timeout = timeout
        self._binary = shutil.which("agent-browser")

    @property
    def is_installed(self) -> bool:
        return self._binary is not None

    # ── Core command runner ───────────────────────────────────────────────

    def _run(self, *args: str, timeout: int | None = None) -> BrowserResult:
        """Execute an agent-browser command and return parsed result."""
        cmd = [
            self._binary or "agent-browser",
            "--session", self.session,
            "--json",
        ]
        if self.headed:
            cmd.append("--headed")
        cmd.extend(args)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
            )
        except FileNotFoundError:
            return BrowserResult(
                success=False,
                error="agent-browser not found. Install with: npm install -g agent-browser && agent-browser install",
            )
        except subprocess.TimeoutExpired:
            return BrowserResult(success=False, error=f"Command timed out after {timeout or self.timeout}s")

        stdout = proc.stdout.strip()
        if not stdout:
            if proc.returncode != 0:
                return BrowserResult(success=False, error=proc.stderr.strip() or f"Exit code {proc.returncode}", raw_stdout="")
            return BrowserResult(success=True, raw_stdout="")

        # Parse JSON response
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                return BrowserResult(
                    success=parsed.get("success", proc.returncode == 0),
                    data=parsed.get("data", parsed),
                    error=parsed.get("error"),
                    raw_stdout=stdout,
                )
            return BrowserResult(success=True, data={"result": parsed}, raw_stdout=stdout)
        except json.JSONDecodeError:
            # Non-JSON output (e.g. snapshot text)
            return BrowserResult(success=proc.returncode == 0, data={"text": stdout}, raw_stdout=stdout)

    # ── Navigation ────────────────────────────────────────────────────────

    def open(self, url: str) -> BrowserResult:
        """Navigate to a URL."""
        return self._run("open", url, timeout=60)

    def back(self) -> BrowserResult:
        return self._run("back")

    def forward(self) -> BrowserResult:
        return self._run("forward")

    def reload(self) -> BrowserResult:
        return self._run("reload")

    # ── Page inspection ───────────────────────────────────────────────────

    def snapshot(self, interactive: bool = True, compact: bool = False) -> BrowserResult:
        """Get the accessibility tree snapshot with @ref element IDs."""
        args = ["snapshot"]
        if interactive:
            args.append("-i")
        if compact:
            args.append("-c")
        return self._run(*args, timeout=30)

    def get_title(self) -> BrowserResult:
        return self._run("get", "title")

    def get_url(self) -> BrowserResult:
        return self._run("get", "url")

    def get_text(self, selector: str) -> BrowserResult:
        """Get the text content of an element."""
        return self._run("get", "text", selector)

    def get_html(self, selector: str) -> BrowserResult:
        return self._run("get", "html", selector)

    # ── Interaction ───────────────────────────────────────────────────────

    def click(self, selector: str) -> BrowserResult:
        """Click an element (use @ref IDs from snapshot)."""
        return self._run("click", selector)

    def fill(self, selector: str, text: str) -> BrowserResult:
        """Clear and fill a text input."""
        return self._run("fill", selector, text)

    def type_text(self, selector: str, text: str) -> BrowserResult:
        """Type text into an element (appends, doesn't clear)."""
        return self._run("type", selector, text)

    def select(self, selector: str, value: str) -> BrowserResult:
        """Select a dropdown option."""
        return self._run("select", selector, value)

    def hover(self, selector: str) -> BrowserResult:
        return self._run("hover", selector)

    def check(self, selector: str) -> BrowserResult:
        return self._run("check", selector)

    def press(self, key: str) -> BrowserResult:
        """Press a key (e.g. 'Enter', 'Tab', 'Control+a')."""
        return self._run("press", key)

    def scroll(self, direction: str = "down", pixels: int = 300, selector: str | None = None) -> BrowserResult:
        args = ["scroll", direction, str(pixels)]
        if selector:
            args.extend(["--selector", selector])
        return self._run(*args)

    # ── Screenshots ───────────────────────────────────────────────────────

    def screenshot(self, path: str | None = None, full: bool = False) -> BrowserResult:
        args = ["screenshot"]
        if path:
            args.append(path)
        if full:
            args.append("--full")
        return self._run(*args)

    # ── JavaScript ────────────────────────────────────────────────────────

    def eval_js(self, script: str) -> BrowserResult:
        """Execute JavaScript in the page context."""
        return self._run("eval", script)

    # ── Waiting ───────────────────────────────────────────────────────────

    def wait(self, ms: int = 1000) -> BrowserResult:
        return self._run("wait", str(ms))

    def wait_for_text(self, text: str, timeout: int = 10000) -> BrowserResult:
        return self._run("wait", "--text", text, "--timeout", str(timeout))

    def wait_for_url(self, pattern: str, timeout: int = 10000) -> BrowserResult:
        return self._run("wait", "--url", pattern, "--timeout", str(timeout))

    # ── Tabs ──────────────────────────────────────────────────────────────

    def new_tab(self, url: str | None = None) -> BrowserResult:
        args = ["tab", "new"]
        if url:
            args.append(url)
        return self._run(*args)

    def list_tabs(self) -> BrowserResult:
        return self._run("tab")

    # ── Session ───────────────────────────────────────────────────────────

    def close(self) -> BrowserResult:
        return self._run("close")

    # ── Batch (multiple commands in one call) ─────────────────────────────

    def batch(self, commands: list[str]) -> BrowserResult:
        """Run multiple commands in a single invocation (much faster)."""
        return self._run("batch", *commands, timeout=120)
