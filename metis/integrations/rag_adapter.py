"""
METIS RAG Adapter — Inline Context Injection for BoundaryAction.SEEK

Phase 18: When the Epistemic Boundary Guard fires SEEK during token-by-token
generation, this adapter:
  1. Extracts the uncertain topic from prompt + generated text
  2. Searches via ToolRetriever (DuckDuckGo → Mock fallback)
  3. Formats the result as <metis_pause_and_search> + <grounding_context>
  4. Returns the injection string for the inference engine to append

The inference engine handles KV-cache rebuild and continuation.

Injection topology (Inline Append with explicit break semantics):
    [model generated so far]...
    <metis_pause_and_search query="extracted_topic" />
    <grounding_context>
    [retrieved facts]
    </grounding_context>
    Based on the above verified information,
    [model continues generating]
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class RAGAdapter:
    """Inline RAG for BoundaryAction.SEEK — search + context injection.

    Usage:
        from metis.integrations.rag_adapter import RAGAdapter
        from metis.search.retriever import ToolRetriever

        adapter = RAGAdapter(retriever=ToolRetriever())
        # Called by MetisInference.generate() on SEEK:
        topic = adapter.extract_topic(prompt, generated_text)
        injection = adapter.search_and_format(topic)
    """

    # Max chars from search results to inject (prevent context explosion)
    _MAX_CONTEXT_CHARS: int = 800
    # Resumption cue after grounding context
    _RESUMPTION_CUE: str = "Based on the above verified information, "

    def __init__(
        self,
        retriever: Optional[object] = None,
        force_mock: bool = False,
        max_context_chars: int = 800,
    ):
        """
        Args:
            retriever: BaseRetriever instance (ToolRetriever recommended).
                       If None, a ToolRetriever is lazily created on first use.
            force_mock: Force MockRetriever (for testing).
            max_context_chars: Max characters from search to inject.
        """
        self._retriever = retriever
        self._force_mock = force_mock
        self._MAX_CONTEXT_CHARS = max_context_chars
        self._initialized = False

    def _ensure_retriever(self) -> None:
        """Lazy-init retriever on first use."""
        if self._initialized:
            return
        self._initialized = True
        if self._retriever is not None:
            return
        try:
            from metis.search.retriever import ToolRetriever
            self._retriever = ToolRetriever(force_mock=self._force_mock)
            logger.info("[RAGAdapter] ToolRetriever initialized")
        except Exception as e:
            logger.warning(f"[RAGAdapter] Failed to init ToolRetriever: {e}")
            # Fallback: create a no-op retriever
            self._retriever = None

    def extract_topic(self, prompt: str, generated_text: str) -> str:
        """Extract the uncertain topic from prompt + generated context.

        Heuristic strategy:
          1. Take the last 1-2 sentences of generated_text (where uncertainty peaked)
          2. Combine with key nouns from the original prompt
          3. Clean up to form a concise search query

        Args:
            prompt: Original user prompt.
            generated_text: Text generated so far by the model.

        Returns:
            Search query string (max ~120 chars).
        """
        # Extract last 1-2 sentences from generated text
        tail = generated_text[-500:] if len(generated_text) > 500 else generated_text
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?。！？])\s+', tail.strip())
        recent = " ".join(sentences[-2:]) if len(sentences) >= 2 else tail[-200:]

        # Extract key terms from prompt (nouns/proper nouns heuristic)
        # Keep capitalized words and longer words as likely important
        prompt_clean = prompt[:300]
        prompt_words = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', prompt_clean)
        if not prompt_words:
            # Fallback: take first meaningful words
            prompt_words = [w for w in prompt_clean.split()[:5] if len(w) > 3]

        # Combine: prompt keywords + recent context
        prompt_part = " ".join(prompt_words[:4])
        recent_part = recent[:80].strip()

        if prompt_part and recent_part:
            query = f"{prompt_part} {recent_part}"
        elif prompt_part:
            query = prompt_part
        else:
            query = recent_part or prompt[:100]

        # Clean: remove thinking tags, excessive whitespace
        query = re.sub(r'</?thinking>', '', query)
        query = re.sub(r'\s+', ' ', query).strip()

        # Cap length for search API
        if len(query) > 120:
            query = query[:120].rsplit(' ', 1)[0]

        logger.info(f"[RAGAdapter] Extracted topic: \"{query[:80]}\"")
        return query

    def search_and_format(self, query: str) -> str:
        """Search for the query and format as injection text.

        Returns the full injection string including:
          <metis_pause_and_search query="..." />
          <grounding_context>...</grounding_context>
          Resumption cue

        Returns empty string if no results found or retriever unavailable.
        """
        self._ensure_retriever()
        if self._retriever is None:
            logger.warning("[RAGAdapter] No retriever available, skipping search")
            return ""

        try:
            results_text: str = self._retriever.search_text(query, top_k=3)
        except Exception as e:
            logger.warning(f"[RAGAdapter] Search failed: {e}")
            return ""

        if not results_text:
            logger.info("[RAGAdapter] No search results found")
            return ""

        # Truncate to max context chars
        if len(results_text) > self._MAX_CONTEXT_CHARS:
            results_text = results_text[:self._MAX_CONTEXT_CHARS].rsplit('\n', 1)[0]

        # Build injection with explicit break semantics
        injection = self._build_injection(query, results_text)
        logger.info(
            f"[RAGAdapter] Injection prepared: {len(injection)} chars, "
            f"context={len(results_text)} chars"
        )
        return injection

    def _build_injection(self, query: str, results_text: str) -> str:
        """Build the full injection string with pause/search/grounding tags."""
        # Escape query for XML attribute
        safe_query = query.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        return (
            f' <metis_pause_and_search query="{safe_query}" />\n'
            f'<grounding_context>\n'
            f'{results_text}\n'
            f'</grounding_context>\n'
            f'{self._RESUMPTION_CUE}'
        )
