"""
METIS Pipeline — Real Web Search Backend (Tool-Augmented Dreaming)

Multi-backend retrieval engine for breaking knowledge gaps during
night-time dreaming. Provides real internet search capabilities so
the model can autonomously acquire new factual knowledge.

Backend priority:
    1. Tavily Search API  (highest quality, requires TAVILY_API_KEY)
    2. DuckDuckGo (ddgs)  (free, no API key, good fallback)
    3. Wikipedia           (free, structured, good for factual queries)

All backends return a unified format: a single concatenated context
string suitable for direct injection into EGTS augmented prompts
and CPT training corpora.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("metis.night.search")


@dataclass
class SearchResult:
    """Unified search result from any backend."""
    query: str
    context: str                    # Concatenated text for LLM consumption
    sources: List[str] = field(default_factory=list)  # Source URLs
    backend: str = "none"           # Which backend produced this result
    latency_s: float = 0.0
    success: bool = False


# ─────────────────────────────────────────────────────────────────────
# Backend 1: Tavily Search API (best quality)
# ─────────────────────────────────────────────────────────────────────

def _search_tavily(query: str, max_results: int = 5) -> Optional[SearchResult]:
    """Search via Tavily API. Requires TAVILY_API_KEY env var."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None

    try:
        import json
        import urllib.request

        t0 = time.time()
        payload = json.dumps({
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_raw_content": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = data.get("results", [])
        if not results:
            return None

        chunks = []
        sources = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            if content:
                chunks.append(f"[{title}] {content}")
                sources.append(url)

        context = "\n\n".join(chunks)
        return SearchResult(
            query=query,
            context=context,
            sources=sources,
            backend="tavily",
            latency_s=time.time() - t0,
            success=bool(context.strip()),
        )
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# Backend 2: DuckDuckGo (ddgs) — free, no API key
# ─────────────────────────────────────────────────────────────────────

def _search_ddgs(query: str, max_results: int = 5) -> Optional[SearchResult]:
    """Search via DuckDuckGo using the ddgs package."""
    try:
        from ddgs import DDGS

        t0 = time.time()
        results = DDGS().text(query, max_results=max_results)

        if not results:
            return None

        chunks = []
        sources = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            if body:
                chunks.append(f"[{title}] {body}")
                sources.append(href)

        context = "\n\n".join(chunks)
        return SearchResult(
            query=query,
            context=context,
            sources=sources,
            backend="ddgs",
            latency_s=time.time() - t0,
            success=bool(context.strip()),
        )
    except Exception as e:
        logger.warning(f"DDGS search failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# Backend 3: Wikipedia — structured factual knowledge
# ─────────────────────────────────────────────────────────────────────

def _search_wikipedia(query: str, max_results: int = 3) -> Optional[SearchResult]:
    """Search Wikipedia for factual content."""
    try:
        import wikipedia

        t0 = time.time()
        search_results = wikipedia.search(query, results=max_results)
        if not search_results:
            return None

        chunks = []
        sources = []
        for title in search_results[:max_results]:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                # Take first 500 chars of summary for concise context
                summary = page.summary[:500]
                if summary:
                    chunks.append(f"[Wikipedia: {page.title}] {summary}")
                    sources.append(page.url)
            except (wikipedia.exceptions.DisambiguationError,
                    wikipedia.exceptions.PageError):
                continue

        if not chunks:
            return None

        context = "\n\n".join(chunks)
        return SearchResult(
            query=query,
            context=context,
            sources=sources,
            backend="wikipedia",
            latency_s=time.time() - t0,
            success=bool(context.strip()),
        )
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# Unified Multi-Backend Search Interface
# ─────────────────────────────────────────────────────────────────────

_BACKENDS = [
    ("tavily", _search_tavily),
    ("ddgs", _search_ddgs),
    ("wikipedia", _search_wikipedia),
]


def web_search(
    query: str,
    max_results: int = 5,
    preferred_backend: Optional[str] = None,
) -> SearchResult:
    """
    Execute a real web search with automatic fallback across backends.

    Tries backends in priority order (Tavily > DDGS > Wikipedia).
    If ``preferred_backend`` is set, that backend is tried first.

    Args:
        query: The search query string.
        max_results: Maximum number of results per backend.
        preferred_backend: Force a specific backend first ("tavily", "ddgs", "wikipedia").

    Returns:
        SearchResult with concatenated context, or an empty-context
        SearchResult if all backends fail.
    """
    backends = list(_BACKENDS)

    # Re-order if a preferred backend is specified
    if preferred_backend:
        backends.sort(key=lambda b: 0 if b[0] == preferred_backend else 1)

    for name, fn in backends:
        logger.info(f"    [Tool] Trying {name} for: {query[:80]}...")
        try:
            result = fn(query, max_results=max_results)
            if result and result.success:
                logger.info(
                    f"    [Tool] {name} succeeded ({len(result.sources)} sources, "
                    f"{result.latency_s:.1f}s)"
                )
                return result
        except Exception as e:
            logger.warning(f"    [Tool] {name} raised unexpected error: {e}")
            continue

    logger.warning(f"    [Tool] All search backends failed for: {query[:80]}")
    return SearchResult(query=query, context="", backend="none", success=False)
