"""
METIS Integrations
Integration layer - non-invasive LLM inference pipeline integration
"""
from .hook import MetisHook

# Backward compatibility
SEDACHook = MetisHook

__all__ = ["MetisHook", "SEDACHook"]
