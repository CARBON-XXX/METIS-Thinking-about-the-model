"""
SEDAC Zero-Configuration Auto Mode

用户拿到手不用做任何设置，直接用。

Usage:
    from sedac import auto
    
    # 一行代码启动
    monitor = auto()
    
    # 或者指定模式
    monitor = auto("aggressive")  # 速度优先
    monitor = auto("conservative")  # 质量优先

Author: SEDAC Research
Date: 2026-01-23
"""

import torch
from typing import Optional, Literal, Union, Dict, Any
from dataclasses import dataclass
import warnings


# =============================================================================
# Auto Configuration
# =============================================================================

@dataclass
class AutoConfig:
    """Auto-detected configuration."""
    mode: str = "balanced"
    tau: float = 0.95
    k: int = 3
    min_layer_ratio: float = 0.20
    
    # Model info (auto-detected)
    num_layers: int = 36
    hidden_dim: int = 2048
    model_size: str = "unknown"  # 7B, 13B, 72B, etc.
    
    # Device
    device: str = "cuda"
    
    # Spectral guard
    enable_spectral_guard: bool = True
    min_layers_before_exit: int = 12


def _detect_model_size(num_layers: int, hidden_dim: int) -> str:
    """Detect model size from architecture."""
    # Common configurations
    if num_layers <= 32 and hidden_dim <= 4096:
        return "7B"
    elif num_layers <= 40 and hidden_dim <= 5120:
        return "13B"
    elif num_layers <= 64 and hidden_dim <= 8192:
        return "70B"
    elif num_layers > 64:
        return "72B+"
    return "unknown"


def _get_optimal_params(model_size: str, mode: str) -> Dict[str, Any]:
    """Get optimal parameters based on model size and mode."""
    
    # Base params per model size
    base_params = {
        "7B": {"tau": 0.94, "k": 3, "min_layer_ratio": 0.20, "min_spectral": 10},
        "13B": {"tau": 0.95, "k": 3, "min_layer_ratio": 0.22, "min_spectral": 12},
        "70B": {"tau": 0.96, "k": 3, "min_layer_ratio": 0.25, "min_spectral": 14},
        "72B+": {"tau": 0.96, "k": 4, "min_layer_ratio": 0.30, "min_spectral": 16},
        "unknown": {"tau": 0.95, "k": 3, "min_layer_ratio": 0.20, "min_spectral": 12},
    }
    
    params = base_params.get(model_size, base_params["unknown"]).copy()
    
    # Mode adjustments
    if mode == "aggressive":
        params["tau"] -= 0.05
        params["k"] = max(2, params["k"] - 1)
        params["min_layer_ratio"] -= 0.05
    elif mode == "conservative":
        params["tau"] += 0.02
        params["k"] = min(5, params["k"] + 1)
        params["min_layer_ratio"] += 0.05
    elif mode == "maximum_speed":
        params["tau"] -= 0.08
        params["k"] = 2
        params["min_layer_ratio"] = 0.15
    
    # Clamp values
    params["tau"] = max(0.85, min(0.995, params["tau"]))
    params["min_layer_ratio"] = max(0.10, min(0.40, params["min_layer_ratio"]))
    
    return params


# =============================================================================
# Auto Monitor
# =============================================================================

class SEDACAutoMonitor:
    """
    Zero-configuration SEDAC monitor.
    
    Auto-detects model architecture and applies optimal settings.
    """
    
    def __init__(
        self,
        mode: Literal["conservative", "balanced", "aggressive", "maximum_speed"] = "balanced",
        device: Optional[str] = None,
    ):
        self.mode = mode
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Will be set on first forward
        self._initialized = False
        self._config: Optional[AutoConfig] = None
        
        # State
        self._prev_hidden: Optional[torch.Tensor] = None
        self._consecutive: Optional[torch.Tensor] = None
        self._stability_history: Optional[torch.Tensor] = None
        self._layer_idx: int = 0
        self._exited_mask: Optional[torch.Tensor] = None
    
    def _auto_init(self, hidden: torch.Tensor, num_layers: int) -> None:
        """Auto-initialize from first hidden state."""
        batch_size = hidden.shape[0]
        hidden_dim = hidden.shape[-1]
        
        model_size = _detect_model_size(num_layers, hidden_dim)
        params = _get_optimal_params(model_size, self.mode)
        
        self._config = AutoConfig(
            mode=self.mode,
            tau=params["tau"],
            k=params["k"],
            min_layer_ratio=params["min_layer_ratio"],
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            model_size=model_size,
            device=str(self.device),
            enable_spectral_guard=True,
            min_layers_before_exit=params["min_spectral"],
        )
        
        # Initialize state tensors
        self._consecutive = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._exited_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self._stability_history = torch.zeros(
            batch_size, num_layers - 1, dtype=torch.float32, device=self.device
        )
        
        self._initialized = True
    
    def reset(self, batch_size: Optional[int] = None) -> None:
        """Reset state for new sequence."""
        self._prev_hidden = None
        self._layer_idx = 0
        
        if batch_size is not None and self._config is not None:
            self._consecutive = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            self._exited_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            self._stability_history = torch.zeros(
                batch_size, self._config.num_layers - 1, 
                dtype=torch.float32, device=self.device
            )
        elif self._consecutive is not None:
            self._consecutive.zero_()
            self._exited_mask.zero_()
            self._stability_history.zero_()
    
    def step(
        self, 
        hidden: torch.Tensor, 
        layer_idx: int,
        num_layers: int
    ) -> torch.Tensor:
        """
        Process one layer and return exit mask.
        
        Args:
            hidden: [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
            layer_idx: Current layer index (0-based)
            num_layers: Total number of layers
            
        Returns:
            exit_mask: [batch_size] bool tensor, True = should exit
        """
        # Flatten to [batch_size, hidden_dim] if needed
        if hidden.dim() == 3:
            hidden = hidden[:, -1, :]  # Take last token
        
        hidden = hidden.to(self.device)
        batch_size = hidden.shape[0]
        
        # Auto-initialize on first call
        if not self._initialized:
            self._auto_init(hidden, num_layers)
        
        # Handle batch size change
        if self._consecutive.shape[0] != batch_size:
            self.reset(batch_size)
        
        cfg = self._config
        min_layer = max(
            int(cfg.num_layers * cfg.min_layer_ratio),
            cfg.min_layers_before_exit
        )
        
        # Layer 0: just store hidden
        if layer_idx == 0:
            self._prev_hidden = hidden.clone()
            self._layer_idx = 0
            return torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Compute stability
        with torch.no_grad():
            cos_sim = torch.nn.functional.cosine_similarity(
                self._prev_hidden.float(), hidden.float(), dim=1
            )
            stability = (cos_sim + 1.0) / 2.0
        
        # Store in history
        if layer_idx - 1 < self._stability_history.shape[1]:
            self._stability_history[:, layer_idx - 1] = stability
        
        # Update state
        self._prev_hidden = hidden.clone()
        self._layer_idx = layer_idx
        
        # Before min_layer: no exit
        if layer_idx < min_layer:
            return torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Check stability threshold
        stable_mask = stability >= cfg.tau
        
        # Update consecutive count
        self._consecutive = torch.where(
            stable_mask & ~self._exited_mask,
            self._consecutive + 1,
            torch.zeros_like(self._consecutive)
        )
        
        # Check for exit (consecutive >= k)
        want_exit = (self._consecutive >= cfg.k) & ~self._exited_mask
        
        # Spectral guard (simplified)
        if cfg.enable_spectral_guard and layer_idx >= cfg.min_layers_before_exit:
            # Compute spectral features
            history = self._stability_history[:, :layer_idx]
            stability_mean = history.mean(dim=1)
            
            # Simple pseudo-convergence detection
            # Too stable + low variance = suspicious
            stability_std = history.std(dim=1)
            pseudo_mask = (stability_mean > 0.97) & (stability_std < 0.01)
            
            # Block pseudo-converged
            want_exit = want_exit & ~pseudo_mask
        
        # Mark as exited
        self._exited_mask = self._exited_mask | want_exit
        
        return want_exit
    
    def get_config(self) -> Optional[AutoConfig]:
        """Get current configuration."""
        return self._config
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        if not self._initialized:
            return {}
        
        exited = self._exited_mask.sum().item()
        total = self._exited_mask.shape[0]
        
        return {
            "mode": self.mode,
            "model_size": self._config.model_size,
            "tau": self._config.tau,
            "k": self._config.k,
            "exited": exited,
            "total": total,
            "exit_rate": exited / total if total > 0 else 0,
            "current_layer": self._layer_idx,
        }


# =============================================================================
# Public API
# =============================================================================

def auto(
    mode: Literal["conservative", "balanced", "aggressive", "maximum_speed"] = "balanced",
    device: Optional[str] = None,
) -> SEDACAutoMonitor:
    """
    Create a zero-configuration SEDAC monitor.
    
    Usage:
        monitor = auto()  # Balanced mode (default)
        monitor = auto("aggressive")  # Speed priority
        monitor = auto("conservative")  # Quality priority
        
    In your forward loop:
        for layer_idx, layer in enumerate(model.layers):
            hidden = layer(hidden)
            exit_mask = monitor.step(hidden, layer_idx, len(model.layers))
            
            if exit_mask.all():
                break
    
    Args:
        mode: One of "conservative", "balanced", "aggressive", "maximum_speed"
        device: Device to use (auto-detects if None)
        
    Returns:
        SEDACAutoMonitor instance
    """
    return SEDACAutoMonitor(mode=mode, device=device)


# Alias for convenience
SEDAC = auto


def create_monitor(
    mode: str = "balanced",
    **kwargs
) -> SEDACAutoMonitor:
    """Alias for auto()."""
    return auto(mode=mode, **kwargs)
