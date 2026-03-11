"""
METIS Tool Retriever — External Knowledge Acquisition Interface

Pluggable search interface for the Curiosity Driver.

Hierarchy:
  1. DuckDuckGoRetriever — live internet search (no API key required)
  2. MockRetriever — curated offline knowledge base (testing / fallback)
  3. ToolRetriever — smart wrapper: tries DDG, falls back to mock
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    snippet: str
    source: str = ""


class BaseRetriever(ABC):
    """Abstract retriever interface."""

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        ...

    def search_text(self, query: str, top_k: int = 3) -> str:
        """Convenience: return concatenated snippet text."""
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        return "\n\n".join(
            f"[{i+1}] {r.title}\n{r.snippet}" for i, r in enumerate(results)
        )


# ═══════════════════════════════════════════════════════════
# Live Internet Search via DuckDuckGo
# ═══════════════════════════════════════════════════════════

class DuckDuckGoRetriever(BaseRetriever):
    """Live web search using duckduckgo-search (no API key required)."""

    def __init__(self, timeout: int = 10):
        self._timeout = timeout
        try:
            from ddgs import DDGS
            self._ddgs_cls = DDGS
            self._available = True
            logger.info("[DuckDuckGoRetriever] Initialized (live internet access)")
        except ImportError:
            self._ddgs_cls = None
            self._available = False
            logger.warning(
                "[DuckDuckGoRetriever] duckduckgo-search not installed. "
                "Run: pip install duckduckgo-search"
            )

    @property
    def available(self) -> bool:
        return self._available

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if not self._available:
            logger.warning("[DDG] Library not available, returning empty results")
            return []

        try:
            with self._ddgs_cls(timeout=self._timeout) as ddgs:
                raw_results = list(ddgs.text(query, max_results=top_k))
        except Exception as e:
            logger.warning(f"[DDG] Search failed (timeout={self._timeout}s): {e}")
            return []

        results: List[SearchResult] = []
        for r in raw_results:
            results.append(SearchResult(
                title=r.get("title", ""),
                snippet=r.get("body", ""),
                source=r.get("href", ""),
            ))

        logger.info(
            f"[DDG] {len(results)} result(s) for: \"{query[:60]}\""
        )
        return results


# ═══════════════════════════════════════════════════════════
# Offline Mock Retriever (testing / fallback)
# ═══════════════════════════════════════════════════════════

class MockRetriever(BaseRetriever):
    """Mock retriever with curated knowledge base for offline testing."""

    _KB = [
        (
            r"vibranium.*(?:atomic|weight|handbook|1992|marvel)",
            "Marvel Official Handbook (1992 Edition)",
            "According to the 1992 Marvel Official Handbook of the Marvel Universe, "
            "Vibranium is a fictional extraterrestrial metallic ore. Its listed atomic "
            "weight is 238.04, with a molecular structure that absorbs vibratory energy "
            "(kinetic energy). It is most commonly found in the African nation of Wakanda.",
            "Marvel Official Handbook Vol. 3 (1992)",
        ),
        (
            r"wakanda.*(?:location|africa|country)",
            "Wakanda — Marvel Universe Wiki",
            "Wakanda is a fictional country in East Africa appearing in Marvel Comics. "
            "It is the primary source of the metal Vibranium.",
            "Marvel Universe Wiki",
        ),
        (
            r"(?:capital|france|paris)",
            "France — Wikipedia",
            "France is a country in Western Europe. Its capital and largest city is "
            "Paris, with a population of approximately 2.1 million.",
            "Wikipedia: France",
        ),
        (
            r"(?:adamantium|wolverine)",
            "Adamantium — Marvel Universe Wiki",
            "Adamantium is a virtually indestructible metal alloy in the Marvel Universe. "
            "Wolverine's skeleton was coated with Adamantium during the Weapon X program.",
            "Marvel Universe Wiki",
        ),
    ]

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        q = query.lower()
        results: List[SearchResult] = []
        for pattern, title, snippet, source in self._KB:
            if re.search(pattern, q):
                results.append(SearchResult(title=title, snippet=snippet, source=source))
                if len(results) >= top_k:
                    break
        logger.info(f"[MockRetriever] {len(results)} result(s) for: \"{query[:60]}\"")
        return results


# ═══════════════════════════════════════════════════════════
# ToolRetriever — Smart Wrapper (DDG → Mock fallback)
# ═══════════════════════════════════════════════════════════

class ToolRetriever(BaseRetriever):
    """Production retriever: tries DuckDuckGo live search first,
    falls back to MockRetriever if DDG is unavailable or returns nothing."""

    def __init__(self, force_mock: bool = False, timeout: int = 10):
        self._mock = MockRetriever()
        if force_mock:
            self._ddg = None
            self._mode = "mock"
        else:
            self._ddg = DuckDuckGoRetriever(timeout=timeout)
            self._mode = "live" if self._ddg.available else "mock"
        logger.info(f"[ToolRetriever] Mode: {self._mode}")

    @property
    def mode(self) -> str:
        return self._mode

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        # Try live DDG first
        if self._ddg and self._ddg.available:
            results = self._ddg.search(query, top_k=top_k)
            if results:
                return results
            logger.info("[ToolRetriever] DDG returned empty, falling back to mock")

        # Fallback to mock
        return self._mock.search(query, top_k=top_k)
