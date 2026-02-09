"""
METIS PyTorch Hook
Non-invasive LLM integration

Does not modify model internals, only observes logits output.
One line of code to integrate with any HuggingFace model.

Usage:
    from metis import Metis
    
    model = AutoModelForCausalLM.from_pretrained(...)
    metis = Metis.attach(model)
    
    # Automatic monitoring during inference
    outputs = model(input_ids)
    signal = metis.last_signal  # Get cognitive signal
"""
import torch
import torch.nn as nn
from typing import Optional, Any

from ..metis import Metis


class MetisHook:
    """
    METIS integration via PyTorch Hook.
    
    Listens to lm_head output via register_forward_hook,
    performs semantic entropy analysis on logits without modifying the model.
    """
    
    def __init__(self, metis: Metis):
        self._metis = metis
        self._handle = None
    
    def attach(self, model: nn.Module) -> None:
        """
        Attach to model.
        
        Automatically finds lm_head and registers hook.
        """
        # Extract embedding for semantic entropy
        try:
            embed = model.get_input_embeddings()
            if embed is not None and hasattr(embed, 'weight'):
                self._metis.set_embedding_matrix(embed.weight.data)
        except Exception:
            pass
        
        # Find lm_head
        lm_head = self._find_lm_head(model)
        if lm_head is not None:
            self._handle = lm_head.register_forward_hook(self._hook_fn)
    
    def detach(self) -> None:
        """Remove hook"""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
    
    def _hook_fn(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        """Hook callback: automatically invoked on each lm_head forward pass"""
        if isinstance(output, torch.Tensor) and output.dim() >= 2:
            self._metis.step(output)
    
    @staticmethod
    def _find_lm_head(model: nn.Module) -> Optional[nn.Module]:
        """Automatically find lm_head"""
        if hasattr(model, 'lm_head'):
            return model.lm_head
        if hasattr(model, 'output'):
            return model.output
        # Traverse to find last Linear layer
        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        return last_linear
