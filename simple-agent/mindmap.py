"""
Mind Map — a persistent graph of browsed websites.

Stores what the agent learned about each page: title, summary, links,
interactive elements, and navigation paths.  On revisit, the agent can
skip exploration and go straight to the right element.

Storage: a single JSON file (mindmap.json) using Pydantic models.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse, urljoin

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════

class PageLink(BaseModel):
    """A link found on a page."""
    text: str = ""
    href: str = ""
    context: str = ""  # e.g. "navigation bar", "main content", "footer"


class InteractiveElement(BaseModel):
    """A clickable/fillable element from a snapshot."""
    ref: str = ""        # e.g. "@e1"
    role: str = ""       # e.g. "button", "link", "input"
    text: str = ""       # visible text
    description: str = ""  # extra context


class PageNode(BaseModel):
    """Everything the agent knows about a single page."""
    url: str
    domain: str = ""
    title: str = ""
    summary: str = ""                    # AI-generated summary of what this page is about
    content_snippet: str = ""            # first ~500 chars of visible text
    links: list[PageLink] = Field(default_factory=list)
    interactive_elements: list[InteractiveElement] = Field(default_factory=list)
    navigation_hints: dict[str, str] = Field(default_factory=dict)
    # ^ maps target_url -> how to get there, e.g. "click @e3 'About' link"
    last_visited: str = ""
    visit_count: int = 0
    parent_urls: list[str] = Field(default_factory=list)  # pages that link here


class Edge(BaseModel):
    """A directed edge in the mind map graph."""
    source: str   # source URL
    target: str   # target URL
    label: str = ""  # link text or action description


class MindMapData(BaseModel):
    """The full mind map — serialized to/from JSON."""
    nodes: dict[str, PageNode] = Field(default_factory=dict)  # url -> PageNode
    edges: list[Edge] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Mind map manager
# ═══════════════════════════════════════════════════════════════════════════

class MindMap:
    """
    Persistent mind map of browsed websites.

    Automatically saves to a local JSON file after every update.
    Provides lookup, search, and context generation for the AI agent.
    """

    def __init__(self, path: str = "mindmap.json"):
        self.path = path
        self._data = self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> MindMapData:
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r") as f:
                    return MindMapData.model_validate_json(f.read())
            except Exception:
                return MindMapData()
        return MindMapData()

    def save(self) -> None:
        with open(self.path, "w") as f:
            f.write(self._data.model_dump_json(indent=2))

    # ── Query ─────────────────────────────────────────────────────────────

    def get_node(self, url: str) -> PageNode | None:
        """Get what we know about a URL."""
        return self._data.nodes.get(self._normalize_url(url))

    def has_visited(self, url: str) -> bool:
        return self._normalize_url(url) in self._data.nodes

    def get_domain_pages(self, domain: str) -> list[PageNode]:
        """Get all known pages for a domain."""
        domain = domain.lower().removeprefix("www.")
        return [n for n in self._data.nodes.values() if n.domain == domain]

    def search(self, query: str) -> list[PageNode]:
        """Search mind map by keyword (title, summary, content)."""
        query_lower = query.lower()
        results = []
        for node in self._data.nodes.values():
            searchable = f"{node.title} {node.summary} {node.content_snippet}".lower()
            if query_lower in searchable:
                results.append(node)
        return sorted(results, key=lambda n: n.visit_count, reverse=True)

    def get_context_for_url(self, url: str) -> str:
        """
        Generate a context string for the AI agent about a URL.
        Includes what we know about the page and its neighbors.
        """
        url = self._normalize_url(url)
        node = self._data.nodes.get(url)

        if not node:
            # Check if we know anything about this domain
            domain = urlparse(url).netloc.lower().removeprefix("www.")
            domain_pages = self.get_domain_pages(domain)
            if domain_pages:
                lines = [f"Never visited {url}, but know {len(domain_pages)} pages on {domain}:"]
                for p in domain_pages[:5]:
                    lines.append(f"  - {p.url}: {p.title} ({p.summary[:80]})")
                return "\n".join(lines)
            return f"No prior knowledge of {url} or its domain."

        lines = [
            f"Previously visited {url} ({node.visit_count}x, last: {node.last_visited})",
            f"Title: {node.title}",
        ]
        if node.summary:
            lines.append(f"Summary: {node.summary}")
        if node.content_snippet:
            lines.append(f"Content preview: {node.content_snippet[:200]}...")
        if node.links:
            lines.append(f"Known links ({len(node.links)}):")
            for link in node.links[:10]:
                lines.append(f"  - [{link.text}] -> {link.href}")
        if node.interactive_elements:
            lines.append(f"Known interactive elements ({len(node.interactive_elements)}):")
            for el in node.interactive_elements[:10]:
                lines.append(f"  - {el.ref} [{el.role}] \"{el.text}\"")
        if node.navigation_hints:
            lines.append("Navigation shortcuts:")
            for target, hint in list(node.navigation_hints.items())[:5]:
                lines.append(f"  - To {target}: {hint}")

        return "\n".join(lines)

    def get_stats(self) -> str:
        """Return a summary of the mind map."""
        n = len(self._data.nodes)
        e = len(self._data.edges)
        domains = set(node.domain for node in self._data.nodes.values())
        return f"Mind map: {n} pages, {e} links, {len(domains)} domains"

    # ── Update ────────────────────────────────────────────────────────────

    def record_visit(
        self,
        url: str,
        title: str = "",
        snapshot_text: str = "",
        summary: str = "",
        parent_url: str | None = None,
    ) -> PageNode:
        """
        Record or update a page visit.
        Parses the snapshot to extract links and interactive elements.
        """
        url = self._normalize_url(url)
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        now = datetime.now(timezone.utc).isoformat()

        existing = self._data.nodes.get(url)
        if existing:
            node = existing
            node.visit_count += 1
            node.last_visited = now
            if title:
                node.title = title
            if summary:
                node.summary = summary
        else:
            node = PageNode(
                url=url,
                domain=domain,
                title=title,
                summary=summary,
                last_visited=now,
                visit_count=1,
            )

        # Extract content snippet from snapshot
        if snapshot_text:
            # Strip ref markers for content snippet
            clean = re.sub(r"@e\d+\s*", "", snapshot_text)
            node.content_snippet = clean[:500]

            # Parse interactive elements from snapshot
            node.interactive_elements = self._parse_elements(snapshot_text)

            # Parse links from snapshot
            node.links = self._parse_links(snapshot_text, url)

        # Track parent
        if parent_url:
            parent_url = self._normalize_url(parent_url)
            if parent_url not in node.parent_urls:
                node.parent_urls.append(parent_url)
            # Add edge
            edge = Edge(source=parent_url, target=url, label=title or url)
            if not any(e.source == edge.source and e.target == edge.target for e in self._data.edges):
                self._data.edges.append(edge)

        self._data.nodes[url] = node
        self.save()
        return node

    def record_navigation(self, from_url: str, to_url: str, action: str) -> None:
        """Record how to navigate from one page to another."""
        from_url = self._normalize_url(from_url)
        to_url = self._normalize_url(to_url)

        node = self._data.nodes.get(from_url)
        if node:
            node.navigation_hints[to_url] = action
            self.save()

    # ── Parsing helpers ───────────────────────────────────────────────────

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL for consistent keys (strip trailing slash, fragment)."""
        url = url.split("#")[0]  # remove fragment
        url = url.rstrip("/")
        return url

    @staticmethod
    def _parse_elements(snapshot: str) -> list[InteractiveElement]:
        """Extract interactive elements from a snapshot accessibility tree."""
        elements = []
        # Pattern: @e1 [role] "text" or @e1 role "text"
        pattern = re.compile(
            r'(@e\d+)\s+\[?(\w+)\]?\s*"([^"]*)"',
        )
        for match in pattern.finditer(snapshot):
            ref, role, text = match.groups()
            if role in ("button", "link", "input", "textbox", "checkbox",
                        "radio", "select", "combobox", "menuitem", "tab",
                        "searchbox", "slider", "switch"):
                elements.append(InteractiveElement(
                    ref=ref, role=role, text=text,
                ))
        return elements

    @staticmethod
    def _parse_links(snapshot: str, base_url: str) -> list[PageLink]:
        """Extract links from snapshot text."""
        links = []
        seen_hrefs = set()
        # Look for href patterns in snapshot
        pattern = re.compile(r'(@e\d+)\s+\[?link\]?\s*"([^"]*)".*?href="([^"]*)"', re.IGNORECASE)
        for match in pattern.finditer(snapshot):
            _, text, href = match.groups()
            if href.startswith("/"):
                href = urljoin(base_url, href)
            if href not in seen_hrefs and href.startswith("http"):
                seen_hrefs.add(href)
                links.append(PageLink(text=text, href=href))

        # Also catch simpler link patterns
        simple = re.compile(r'\[link\]\s*"([^"]*)"')
        for match in simple.finditer(snapshot):
            text = match.group(1)
            if text and len(text) < 100:
                # We don't have href here, but record the link text
                if not any(l.text == text for l in links):
                    links.append(PageLink(text=text))

        return links
