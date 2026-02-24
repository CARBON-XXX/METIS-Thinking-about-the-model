"""
METIS × LangChain Integration — Cognitive-Aware LLM Wrapper.

Provides a LangChain-compatible LLM wrapper that instruments generation
with METIS metacognitive signals. Enables:
  - Automatic RAG trigger when BoundaryAction.SEEK fires
  - Hallucination risk scoring per generation
  - Cognitive load monitoring for agent step decomposition
  - Epistemic confidence metadata on every LLM call

Usage:
    from metis.integrations.langchain import MetisLLM, MetisCallbackHandler

    llm = MetisLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    result = llm.invoke("What is quantum entanglement?")
    print(result.metadata)  # cognitive signals included

    # Or as a callback handler for existing chains:
    handler = MetisCallbackHandler(metis_instance)
    chain.invoke(input, config={"callbacks": [handler]})

Requires: pip install langchain-core
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.language_models.llms import BaseLLM
    from langchain_core.outputs import Generation, GenerationChunk, LLMResult
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


def _require_langchain() -> None:
    if not _HAS_LANGCHAIN:
        raise ImportError(
            "langchain-core is required for METIS LangChain integration. "
            "Install with: pip install langchain-core"
        )


if _HAS_LANGCHAIN:

    class MetisCallbackHandler(BaseCallbackHandler):
        """
        LangChain callback that logs METIS cognitive signals.

        Attach to any chain/agent to get per-step metacognitive telemetry.
        """

        def __init__(self, metis: Any = None) -> None:
            """
            Args:
                metis: Optional Metis instance. If provided, reads signals
                       from it after each LLM call.
            """
            super().__init__()
            self._metis = metis
            self.last_judgment: Optional[Dict[str, Any]] = None
            self.signals: List[Dict[str, Any]] = []

        def on_llm_end(self, response: "LLMResult", **kwargs: Any) -> None:
            """Called after LLM generation completes."""
            if self._metis is None:
                return
            try:
                judgment = self._metis.introspect()
                judgment_dict = {
                    "epistemic_confidence": judgment.epistemic_confidence,
                    "cognitive_load": judgment.cognitive_load,
                    "hallucination_risk": judgment.hallucination_risk,
                    "suggested_action": judgment.suggested_action,
                    "boundary_status": judgment.boundary_status,
                    "reasoning": judgment.reasoning,
                }
                self.last_judgment = judgment_dict
                self.signals.append(judgment_dict)

                # Inject metadata into response generations
                for gen_list in response.generations:
                    for gen in gen_list:
                        if gen.generation_info is None:
                            gen.generation_info = {}
                        gen.generation_info["metis"] = judgment_dict

                logger.debug(
                    f"[METIS×LangChain] confidence={judgment.epistemic_confidence:.2f} "
                    f"risk={judgment.hallucination_risk:.2f} "
                    f"action={judgment.suggested_action}"
                )
            except Exception as e:
                logger.warning(f"[METIS×LangChain] Introspection failed: {e}")

        @property
        def should_seek(self) -> bool:
            """Check if METIS recommends external retrieval (RAG trigger)."""
            if self.last_judgment is None:
                return False
            return self.last_judgment.get("suggested_action") in ("verify", "seek")

        @property
        def should_hedge(self) -> bool:
            """Check if METIS recommends hedging the response."""
            if self.last_judgment is None:
                return False
            return self.last_judgment.get("suggested_action") == "hedge"

        @property
        def should_decompose(self) -> bool:
            """Check if cognitive load is too high (agent should decompose)."""
            if self.last_judgment is None:
                return False
            return self.last_judgment.get("cognitive_load", 0) > 0.7

    class MetisLLM(BaseLLM):
        """
        LangChain LLM wrapper with built-in METIS metacognition.

        Wraps any HuggingFace model with METIS instrumentation.
        Generation metadata includes full cognitive signals.
        """

        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
        device: str = "auto"
        max_tokens: int = 512
        temperature: float = 0.7

        # Internal state (not serialized)
        _metis: Any = None
        _model: Any = None
        _tokenizer: Any = None
        _inference: Any = None

        class Config:
            arbitrary_types_allowed = True

        @property
        def _llm_type(self) -> str:
            return "metis"

        def _setup(self) -> None:
            """Lazy initialization of model + METIS."""
            if self._model is not None:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from ..metis import Metis
            from ..inference import MetisInferenceEngine

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
            if device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            ).to(device)
            self._model.eval()

            self._metis = Metis.attach(self._model, self._tokenizer)
            self._inference = MetisInferenceEngine(self._metis)

        def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional["CallbackManagerForLLMRun"] = None,
            **kwargs: Any,
        ) -> "LLMResult":
            """Generate text with METIS instrumentation."""
            self._setup()

            generations: List[List[Generation]] = []
            for prompt in prompts:
                result = self._inference.generate(
                    prompt,
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                )

                judgment = self._metis.introspect()
                meta = {
                    "epistemic_confidence": judgment.epistemic_confidence,
                    "cognitive_load": judgment.cognitive_load,
                    "hallucination_risk": judgment.hallucination_risk,
                    "suggested_action": judgment.suggested_action,
                    "boundary_status": judgment.boundary_status,
                    "dominant_state": judgment.dominant_state.value
                    if hasattr(judgment.dominant_state, "value")
                    else str(judgment.dominant_state),
                }

                gen = Generation(
                    text=result.text,
                    generation_info={"metis": meta},
                )
                generations.append([gen])

            return LLMResult(generations=generations)

        @property
        def metis(self) -> Any:
            """Access the underlying Metis instance."""
            self._setup()
            return self._metis

else:
    # Stub classes when langchain not installed
    class MetisCallbackHandler:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_langchain()

    class MetisLLM:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_langchain()
