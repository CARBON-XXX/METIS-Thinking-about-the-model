"""
METIS × LlamaIndex Integration — Cognitive-Aware Query Engine.

Provides LlamaIndex-compatible components that use METIS metacognitive
signals for intelligent RAG decisions:
  - MetisQueryTransform: Rewrites queries when METIS detects uncertainty
  - MetisResponseEvaluator: Post-generation hallucination risk scoring
  - MetisRetrieverGuard: Gates retrieval based on epistemic state

Usage:
    from metis.integrations.llamaindex import MetisResponseEvaluator

    evaluator = MetisResponseEvaluator(metis_instance)
    result = query_engine.query("What is dark matter?")
    eval_result = evaluator.evaluate(result)

    if eval_result.should_retrieve_more:
        # METIS detected high uncertainty — fetch more context
        ...

Requires: pip install llama-index-core
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from llama_index.core.base.response.schema import Response
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
    _HAS_LLAMAINDEX = True
except ImportError:
    _HAS_LLAMAINDEX = False


def _require_llamaindex() -> None:
    if not _HAS_LLAMAINDEX:
        raise ImportError(
            "llama-index-core is required for METIS LlamaIndex integration. "
            "Install with: pip install llama-index-core"
        )


@dataclass
class MetisEvalResult:
    """Result of METIS metacognitive evaluation on a LlamaIndex response."""
    epistemic_confidence: float
    hallucination_risk: float
    cognitive_load: float
    suggested_action: str
    boundary_status: str
    reasoning: str

    @property
    def should_retrieve_more(self) -> bool:
        """METIS recommends fetching additional context."""
        return self.suggested_action in ("verify", "seek")

    @property
    def should_hedge(self) -> bool:
        """METIS recommends hedging the response with uncertainty markers."""
        return self.suggested_action == "hedge"

    @property
    def is_reliable(self) -> bool:
        """METIS considers the response reliable."""
        return (
            self.epistemic_confidence > 0.6
            and self.hallucination_risk < 0.3
            and self.boundary_status == "stable"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epistemic_confidence": round(self.epistemic_confidence, 4),
            "hallucination_risk": round(self.hallucination_risk, 4),
            "cognitive_load": round(self.cognitive_load, 4),
            "suggested_action": self.suggested_action,
            "boundary_status": self.boundary_status,
            "is_reliable": self.is_reliable,
            "should_retrieve_more": self.should_retrieve_more,
            "reasoning": self.reasoning,
        }


class MetisResponseEvaluator:
    """
    Evaluate LlamaIndex query results with METIS metacognition.

    Uses the cognitive trace from the most recent generation to assess
    whether the response is reliable, needs verification, or should be
    hedged with uncertainty markers.

    Works with any Metis instance (doesn't require LlamaIndex at eval time).
    """

    def __init__(self, metis: Any) -> None:
        self._metis = metis

    def evaluate_from_trace(self) -> MetisEvalResult:
        """
        Evaluate based on the current METIS trace (call after generation).

        Returns:
            MetisEvalResult with epistemic confidence, risk, and recommended action.
        """
        judgment = self._metis.introspect()
        return MetisEvalResult(
            epistemic_confidence=judgment.epistemic_confidence,
            hallucination_risk=judgment.hallucination_risk,
            cognitive_load=judgment.cognitive_load,
            suggested_action=judgment.suggested_action,
            boundary_status=judgment.boundary_status,
            reasoning=judgment.reasoning,
        )

    def should_rag(self, threshold: float = 0.4) -> bool:
        """
        Quick check: should we trigger RAG/retrieval?

        Args:
            threshold: Hallucination risk threshold above which RAG is recommended.
        """
        try:
            judgment = self._metis.introspect()
            return (
                judgment.hallucination_risk > threshold
                or judgment.suggested_action in ("verify", "seek")
            )
        except Exception:
            return False


class MetisRetrieverGuard:
    """
    Guards retrieval decisions based on METIS epistemic state.

    Wraps a retriever and only triggers actual retrieval when METIS
    signals indicate the model is uncertain. For confident generations,
    retrieval is skipped to save latency/cost.

    Usage:
        guard = MetisRetrieverGuard(metis, retriever)
        nodes = guard.retrieve_if_needed(query)
        # Returns [] if METIS says the model is confident
        # Returns retriever results if METIS detects uncertainty
    """

    def __init__(
        self,
        metis: Any,
        retriever: Any = None,
        uncertainty_threshold: float = 0.4,
    ) -> None:
        self._metis = metis
        self._retriever = retriever
        self._threshold = uncertainty_threshold
        self._retrieval_count = 0
        self._skip_count = 0

    def retrieve_if_needed(self, query: str) -> List[Any]:
        """
        Conditionally retrieve based on METIS epistemic state.

        Returns empty list if model is confident (saves retrieval cost).
        """
        try:
            uncertainty = self._metis.get_uncertainty_score()
            last_signal = self._metis.last_signal

            needs_retrieval = False

            if uncertainty > self._threshold:
                needs_retrieval = True
            elif last_signal is not None:
                from ..core.types import BoundaryAction
                if last_signal.boundary_action == BoundaryAction.SEEK:
                    needs_retrieval = True
        except Exception:
            needs_retrieval = True  # Default to retrieval on error

        if needs_retrieval and self._retriever is not None:
            self._retrieval_count += 1
            logger.debug(
                f"[METIS×LlamaIndex] Triggering retrieval "
                f"(uncertainty={uncertainty:.2f}, count={self._retrieval_count})"
            )
            return self._retriever.retrieve(query)
        else:
            self._skip_count += 1
            logger.debug(
                f"[METIS×LlamaIndex] Skipping retrieval — model confident "
                f"(skipped={self._skip_count})"
            )
            return []

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "retrieval_count": self._retrieval_count,
            "skip_count": self._skip_count,
            "total": self._retrieval_count + self._skip_count,
        }


if _HAS_LLAMAINDEX:

    class MetisCallbackHandler(BaseCallbackHandler):
        """
        LlamaIndex callback handler for METIS cognitive telemetry.

        Logs metacognitive signals at query engine boundaries.
        """

        def __init__(self, metis: Any) -> None:
            super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
            self._metis = metis
            self.evaluations: List[MetisEvalResult] = []

        def on_event_end(
            self,
            event_type: "CBEventType",
            payload: Optional[Dict[str, Any]] = None,
            event_id: str = "",
            **kwargs: Any,
        ) -> None:
            """Capture LLM completion events for METIS evaluation."""
            if event_type == CBEventType.LLM and payload is not None:
                try:
                    evaluator = MetisResponseEvaluator(self._metis)
                    result = evaluator.evaluate_from_trace()
                    self.evaluations.append(result)

                    logger.debug(
                        f"[METIS×LlamaIndex] LLM complete: "
                        f"confidence={result.epistemic_confidence:.2f} "
                        f"risk={result.hallucination_risk:.2f} "
                        f"action={result.suggested_action}"
                    )
                except Exception as e:
                    logger.warning(f"[METIS×LlamaIndex] Eval failed: {e}")

        def on_event_start(
            self,
            event_type: "CBEventType",
            payload: Optional[Dict[str, Any]] = None,
            event_id: str = "",
            parent_id: str = "",
            **kwargs: Any,
        ) -> str:
            """Start events — no-op for METIS."""
            return event_id

        def start_trace(self, trace_id: Optional[str] = None) -> None:
            pass

        def end_trace(
            self,
            trace_id: Optional[str] = None,
            trace_map: Optional[Dict[str, List[str]]] = None,
        ) -> None:
            pass

else:

    class MetisCallbackHandler:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_llamaindex()
