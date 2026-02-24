"""
METIS Integrations
Integration layer - non-invasive LLM inference pipeline integration

Submodules:
  - hook: PyTorch forward hook (zero-code METIS instrumentation)
  - langchain: LangChain LLM wrapper + callback handler
  - llamaindex: LlamaIndex evaluator, retriever guard, callback handler
"""
from .hook import MetisHook

__all__ = [
    "MetisHook",
]

# Lazy imports — only load when explicitly accessed (avoids hard deps)
def __getattr__(name: str):
    if name in ("MetisLLM", "MetisCallbackHandler"):
        from .langchain import MetisLLM, MetisCallbackHandler as _LC
        if name == "MetisLLM":
            return MetisLLM
        return _LC
    if name in ("MetisResponseEvaluator", "MetisRetrieverGuard"):
        from .llamaindex import MetisResponseEvaluator, MetisRetrieverGuard
        if name == "MetisResponseEvaluator":
            return MetisResponseEvaluator
        return MetisRetrieverGuard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
