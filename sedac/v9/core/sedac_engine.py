"""
SEDAC V9.0 - Universal Early Exit Engine
Framework-agnostic implementation for any LLM

Supports:
- HuggingFace Transformers (LLaMA, Qwen, Mistral, etc.)
- vLLM
- TensorRT-LLM
- Custom PyTorch models
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math

# Try to load CUDA kernels
try:
    import sys
    sys.path.insert(0, str(__file__).replace("core/sedac_engine.py", "cuda_ext"))
    import sedac_cuda_v2 as sedac_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


@dataclass
class SEDACConfig:
    """SEDAC configuration - universal across all models"""
    
    # Exit thresholds
    exit_threshold: float = 0.7
    min_exit_layer: int = 4
    max_exit_layer: int = -1  # -1 means num_layers - 1
    
    # Anchor layers (never skip)
    anchor_interval: int = 4
    protect_first_n: int = 2
    protect_last_n: int = 1
    
    # Attention sinks protection
    attention_sink_tokens: int = 4
    protect_recent_tokens: int = 2
    
    # Statistics
    entropy_ema_alpha: float = 0.1
    initial_entropy_mean: float = 3.0
    initial_entropy_std: float = 1.0
    
    # Performance
    use_cuda_kernels: bool = True
    use_fp16: bool = True
    
    # Ghost KV
    enable_ghost_kv: bool = False
    ghost_kv_checkpoint: Optional[str] = None
    
    # Debug
    verbose: bool = False
    collect_stats: bool = True


@dataclass
class SEDACState:
    """Runtime state for SEDAC inference"""
    
    # Running statistics
    entropy_mean: float = 3.0
    entropy_std: float = 1.0
    entropy_count: int = 0
    
    # Per-token tracking
    token_exit_layers: Optional[torch.Tensor] = None
    
    # Batch statistics
    total_tokens: int = 0
    total_layers_computed: int = 0
    total_layers_skipped: int = 0
    exit_layer_histogram: Dict[int, int] = field(default_factory=dict)


class SEDACDecisionMaker:
    """
    Core decision logic - framework agnostic
    
    Input: logits, hidden_states, prev_hidden_states
    Output: exit_mask (which tokens should exit)
    """
    
    def __init__(self, config: SEDACConfig):
        self.config = config
        self.state = SEDACState(
            entropy_mean=config.initial_entropy_mean,
            entropy_std=config.initial_entropy_std,
        )
        self._use_cuda = CUDA_AVAILABLE and config.use_cuda_kernels
    
    def decide(
        self,
        logits: torch.Tensor,          # [batch, seq, vocab] or [N, vocab]
        hidden: torch.Tensor,          # [batch, seq, hidden] or [N, hidden]
        prev_hidden: torch.Tensor,     # [batch, seq, hidden] or [N, hidden]
        layer_idx: int,
        num_layers: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decide which tokens should exit at this layer.
        
        Returns:
            exit_mask: bool tensor, True = should exit
            stats: dict with entropy, confidence, etc.
        """
        # Check layer constraints
        if layer_idx < self.config.min_exit_layer:
            return self._no_exit(hidden), {}
        
        max_exit = self.config.max_exit_layer if self.config.max_exit_layer > 0 else num_layers - 1
        if layer_idx >= max_exit:
            return self._no_exit(hidden), {}
        
        # Check anchor layers
        if self._is_anchor_layer(layer_idx, num_layers):
            return self._no_exit(hidden), {}
        
        layer_progress = layer_idx / num_layers
        
        # Flatten if needed
        original_shape = hidden.shape[:-1]
        if hidden.dim() == 3:
            batch, seq, hdim = hidden.shape
            logits = logits.view(batch * seq, -1)
            hidden = hidden.view(batch * seq, -1)
            prev_hidden = prev_hidden.view(batch * seq, -1)
        
        # Compute decision
        if self._use_cuda and hidden.is_cuda:
            entropy, confidence, decision, load = sedac_cuda.fused_entropy_decision_v2(
                logits.float() if logits.dtype != torch.float32 else logits,
                hidden.float() if hidden.dtype != torch.float32 else hidden,
                prev_hidden.float() if prev_hidden.dtype != torch.float32 else prev_hidden,
                self.state.entropy_mean,
                self.state.entropy_std,
                layer_progress,
                self.config.exit_threshold,
            )
        else:
            entropy, confidence, decision, load = self._pytorch_decision(
                logits, hidden, prev_hidden, layer_progress
            )
        
        # Update running stats
        self._update_stats(entropy)
        
        # Apply attention sink protection
        if attention_mask is not None:
            decision = self._protect_attention_sinks(decision, original_shape)
        
        # Reshape
        exit_mask = decision.view(original_shape)
        
        stats = {
            "entropy": entropy.view(original_shape),
            "confidence": confidence.view(original_shape),
            "cognitive_load": load.view(original_shape),
        }
        
        return exit_mask, stats
    
    def _pytorch_decision(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        prev_hidden: torch.Tensor,
        layer_progress: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure PyTorch fallback implementation"""
        import torch.nn.functional as F
        
        # Entropy
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        entropy = -torch.sum(probs * log_probs, dim=-1) / math.log(2)
        
        # Stability
        diff = hidden.float() - prev_hidden.float()
        diff_norm = torch.norm(diff, p=2, dim=-1)
        hidden_norm = torch.norm(hidden.float(), p=2, dim=-1)
        stability = 1.0 / (1.0 + diff_norm / (hidden_norm + 1e-6))
        
        # Confidence
        z_score = (self.state.entropy_mean - entropy) / (self.state.entropy_std + 1e-6)
        confidence = torch.sigmoid(z_score * 2.0)
        
        # Cognitive load
        load = (1.0 - confidence) * 0.5 + (1.0 - stability) * 0.3 + (1.0 - layer_progress) * 0.2
        
        # Decision
        current_thresh = self.config.exit_threshold - layer_progress * 0.2
        decision = (confidence * stability * layer_progress) > current_thresh
        
        return entropy, confidence, decision, load
    
    def _is_anchor_layer(self, layer_idx: int, num_layers: int) -> bool:
        """Check if layer is an anchor (must be computed)"""
        if layer_idx < self.config.protect_first_n:
            return True
        if layer_idx >= num_layers - self.config.protect_last_n:
            return True
        if layer_idx % self.config.anchor_interval == 0:
            return True
        return False
    
    def _no_exit(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return all-False exit mask"""
        shape = hidden.shape[:-1]
        return torch.zeros(shape, dtype=torch.bool, device=hidden.device)
    
    def _protect_attention_sinks(
        self,
        decision: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Protect attention sink tokens from early exit"""
        if len(original_shape) == 2:
            batch, seq = original_shape
            decision = decision.view(batch, seq)
            
            # Protect first N tokens (attention sinks)
            if self.config.attention_sink_tokens > 0:
                decision[:, :self.config.attention_sink_tokens] = False
            
            # Protect last N tokens (recent context)
            if self.config.protect_recent_tokens > 0:
                decision[:, -self.config.protect_recent_tokens:] = False
            
            decision = decision.view(-1)
        
        return decision
    
    def _update_stats(self, entropy: torch.Tensor):
        """Update running entropy statistics with EMA"""
        batch_mean = entropy.mean().item()
        batch_std = entropy.std().item() if entropy.numel() > 1 else self.state.entropy_std
        
        alpha = self.config.entropy_ema_alpha
        self.state.entropy_mean = (1 - alpha) * self.state.entropy_mean + alpha * batch_mean
        self.state.entropy_std = max(0.1, (1 - alpha) * self.state.entropy_std + alpha * batch_std)
        self.state.entropy_count += entropy.numel()
    
    def reset_stats(self):
        """Reset running statistics"""
        self.state = SEDACState(
            entropy_mean=self.config.initial_entropy_mean,
            entropy_std=self.config.initial_entropy_std,
        )


class SEDACTokenRouter:
    """
    Token routing for dynamic batch processing
    
    Splits tokens into:
    - Active tokens (continue processing)
    - Exit tokens (use current hidden states)
    """
    
    def __init__(self, use_cuda: bool = True):
        self._use_cuda = CUDA_AVAILABLE and use_cuda
    
    def split(
        self,
        hidden: torch.Tensor,      # [N, hidden]
        exit_mask: torch.Tensor,   # [N] bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split tokens into active and exit groups.
        
        Returns:
            active_hidden: [n_active, hidden]
            active_indices: [n_active] original indices
            exit_hidden: [n_exit, hidden]
            exit_indices: [n_exit] original indices
        """
        if self._use_cuda and hidden.is_cuda:
            return sedac_cuda.token_router_split_v2(hidden, exit_mask)
        else:
            return self._pytorch_split(hidden, exit_mask)
    
    def merge(
        self,
        active_hidden: torch.Tensor,   # [n_active, hidden]
        active_indices: torch.Tensor,  # [n_active]
        exit_hidden: torch.Tensor,     # [n_exit, hidden]
        exit_indices: torch.Tensor,    # [n_exit]
        total_size: int,
    ) -> torch.Tensor:
        """Merge active and exit tokens back to original order"""
        if self._use_cuda and active_hidden.is_cuda:
            return sedac_cuda.token_router_merge_v2(
                active_hidden, active_indices,
                exit_hidden, exit_indices,
                total_size
            )
        else:
            return self._pytorch_merge(
                active_hidden, active_indices,
                exit_hidden, exit_indices,
                total_size
            )
    
    def _pytorch_split(
        self,
        hidden: torch.Tensor,
        exit_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """PyTorch fallback for split"""
        active_mask = ~exit_mask
        
        active_indices = torch.where(active_mask)[0]
        exit_indices = torch.where(exit_mask)[0]
        
        active_hidden = hidden[active_indices]
        exit_hidden = hidden[exit_indices]
        
        return active_hidden, active_indices, exit_hidden, exit_indices
    
    def _pytorch_merge(
        self,
        active_hidden: torch.Tensor,
        active_indices: torch.Tensor,
        exit_hidden: torch.Tensor,
        exit_indices: torch.Tensor,
        total_size: int,
    ) -> torch.Tensor:
        """PyTorch fallback for merge"""
        output = torch.empty(
            total_size, active_hidden.size(1),
            dtype=active_hidden.dtype,
            device=active_hidden.device
        )
        
        if active_hidden.numel() > 0:
            output[active_indices] = active_hidden
        if exit_hidden.numel() > 0:
            output[exit_indices] = exit_hidden
        
        return output


class ModelAdapter(ABC):
    """
    Abstract adapter for different model architectures
    Implement this for each model family
    """
    
    @abstractmethod
    def get_num_layers(self) -> int:
        """Return number of transformer layers"""
        pass
    
    @abstractmethod
    def get_hidden_size(self) -> int:
        """Return hidden dimension"""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        pass
    
    @abstractmethod
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings"""
        pass
    
    @abstractmethod
    def get_layer(self, layer_idx: int) -> nn.Module:
        """Get transformer layer by index"""
        pass
    
    @abstractmethod
    def get_lm_head(self) -> nn.Module:
        """Get language model head"""
        pass
    
    @abstractmethod
    def get_final_norm(self) -> nn.Module:
        """Get final layer norm"""
        pass
    
    @abstractmethod
    def forward_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """Forward through a single layer"""
        pass


class HuggingFaceAdapter(ModelAdapter):
    """Adapter for HuggingFace Transformers models"""
    
    def __init__(self, model):
        self.model = model
        self.config = model.config
        
        # Detect model type
        self._detect_architecture()
    
    def _detect_architecture(self):
        """Detect model architecture and set attribute names"""
        model_type = getattr(self.config, "model_type", "").lower()
        
        if hasattr(self.model, "model"):
            self._base = self.model.model
        elif hasattr(self.model, "transformer"):
            self._base = self.model.transformer
        else:
            self._base = self.model
        
        # Layers
        if hasattr(self._base, "layers"):
            self._layers = self._base.layers
        elif hasattr(self._base, "h"):
            self._layers = self._base.h
        elif hasattr(self._base, "decoder") and hasattr(self._base.decoder, "layers"):
            self._layers = self._base.decoder.layers
        else:
            raise ValueError(f"Cannot find layers in model type: {model_type}")
        
        # Embeddings
        if hasattr(self._base, "embed_tokens"):
            self._embed = self._base.embed_tokens
        elif hasattr(self._base, "wte"):
            self._embed = self._base.wte
        elif hasattr(self._base, "embed_in"):
            self._embed = self._base.embed_in
        else:
            raise ValueError(f"Cannot find embeddings in model type: {model_type}")
        
        # Final norm
        if hasattr(self._base, "norm"):
            self._norm = self._base.norm
        elif hasattr(self._base, "ln_f"):
            self._norm = self._base.ln_f
        elif hasattr(self._base, "final_layernorm"):
            self._norm = self._base.final_layernorm
        else:
            self._norm = nn.Identity()
        
        # LM head
        if hasattr(self.model, "lm_head"):
            self._lm_head = self.model.lm_head
        elif hasattr(self.model, "embed_out"):
            self._lm_head = self.model.embed_out
        else:
            raise ValueError(f"Cannot find lm_head in model type: {model_type}")
        
        # RoPE (if present)
        if hasattr(self._base, "rotary_emb"):
            self._rotary = self._base.rotary_emb
        else:
            self._rotary = None
    
    def get_num_layers(self) -> int:
        return len(self._layers)
    
    def get_hidden_size(self) -> int:
        return self.config.hidden_size
    
    def get_vocab_size(self) -> int:
        return self.config.vocab_size
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embed(input_ids)
    
    def get_layer(self, layer_idx: int) -> nn.Module:
        return self._layers[layer_idx]
    
    def get_lm_head(self) -> nn.Module:
        return self._lm_head
    
    def get_final_norm(self) -> nn.Module:
        return self._norm
    
    def forward_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward through a single layer"""
        # Get position embeddings if needed
        position_embeddings = None
        if self._rotary is not None and position_ids is not None:
            position_embeddings = self._rotary(hidden_states, position_ids)
        
        # Forward
        outputs = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs


class SEDACEngine:
    """
    Main SEDAC engine for early exit inference
    
    Usage:
        engine = SEDACEngine(model, config)
        output = engine.generate(input_ids, max_new_tokens=100)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[SEDACConfig] = None,
        adapter: Optional[ModelAdapter] = None,
    ):
        self.config = config or SEDACConfig()
        
        # Create adapter
        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = HuggingFaceAdapter(model)
        
        self.model = model
        self.decision_maker = SEDACDecisionMaker(self.config)
        self.router = SEDACTokenRouter(use_cuda=self.config.use_cuda_kernels)
        
        # Model info
        self.num_layers = self.adapter.get_num_layers()
        self.hidden_size = self.adapter.get_hidden_size()
        self.vocab_size = self.adapter.get_vocab_size()
        
        # Determine anchor layers
        self.anchor_layers = self._compute_anchor_layers()
        
        if self.config.verbose:
            print(f"SEDAC Engine initialized:")
            print(f"  Layers: {self.num_layers}")
            print(f"  Hidden: {self.hidden_size}")
            print(f"  Vocab: {self.vocab_size}")
            print(f"  Anchors: {sorted(self.anchor_layers)}")
            print(f"  CUDA kernels: {CUDA_AVAILABLE and self.config.use_cuda_kernels}")
    
    def _compute_anchor_layers(self) -> set:
        """Compute anchor layer indices"""
        anchors = set()
        
        # First N layers
        for i in range(self.config.protect_first_n):
            anchors.add(i)
        
        # Last N layers
        for i in range(self.num_layers - self.config.protect_last_n, self.num_layers):
            anchors.add(i)
        
        # Interval anchors
        for i in range(0, self.num_layers, self.config.anchor_interval):
            anchors.add(i)
        
        return anchors
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass with early exit
        
        Returns:
            logits: [batch, seq, vocab]
            exit_info: dict with layer statistics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        hidden_states = self.adapter.get_embeddings(input_ids)
        
        # Position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Track exits
        exit_layers = torch.full((batch_size, seq_len), self.num_layers, device=device)
        active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        final_hidden = torch.zeros_like(hidden_states)
        
        prev_hidden = hidden_states.clone()
        
        # Layer-by-layer processing
        for layer_idx in range(self.num_layers):
            layer = self.adapter.get_layer(layer_idx)
            
            # Check for early exit (not on anchor layers)
            if layer_idx not in self.anchor_layers and layer_idx >= self.config.min_exit_layer:
                # Get logits for decision
                with torch.no_grad():
                    temp_norm = self.adapter.get_final_norm()(hidden_states)
                    logits = self.adapter.get_lm_head()(temp_norm)
                
                exit_mask, stats = self.decision_maker.decide(
                    logits, hidden_states, prev_hidden,
                    layer_idx, self.num_layers,
                    attention_mask,
                )
                
                # Mark exits
                newly_exited = exit_mask & active_mask
                if newly_exited.any():
                    final_hidden[newly_exited] = hidden_states[newly_exited]
                    exit_layers[newly_exited] = layer_idx
                    active_mask[newly_exited] = False
                    
                    if self.config.verbose:
                        n_exit = newly_exited.sum().item()
                        print(f"Layer {layer_idx}: {n_exit} tokens exited")
            
            # If all tokens exited, stop
            if not active_mask.any():
                break
            
            # Forward through layer
            prev_hidden = hidden_states.clone()
            hidden_states = self.adapter.forward_layer(
                layer, hidden_states, layer_idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
        
        # Copy remaining active tokens
        final_hidden[active_mask] = hidden_states[active_mask]
        
        # Final norm and LM head
        final_hidden = self.adapter.get_final_norm()(final_hidden)
        logits = self.adapter.get_lm_head()(final_hidden)
        
        # Compute stats
        avg_exit_layer = exit_layers.float().mean().item()
        skip_ratio = 1.0 - avg_exit_layer / self.num_layers
        
        return {
            "logits": logits,
            "hidden_states": final_hidden,
            "exit_layers": exit_layers,
            "avg_exit_layer": avg_exit_layer,
            "skip_ratio": skip_ratio,
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get decision maker statistics"""
        return {
            "entropy_mean": self.decision_maker.state.entropy_mean,
            "entropy_std": self.decision_maker.state.entropy_std,
            "entropy_count": self.decision_maker.state.entropy_count,
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.decision_maker.reset_stats()


def create_sedac_engine(
    model: nn.Module,
    exit_threshold: float = 0.7,
    min_exit_layer: int = 4,
    anchor_interval: int = 4,
    verbose: bool = False,
) -> SEDACEngine:
    """
    Factory function to create SEDAC engine
    
    Args:
        model: HuggingFace model or compatible
        exit_threshold: Threshold for early exit (0-1)
        min_exit_layer: Minimum layer before exit allowed
        anchor_interval: Interval between anchor layers
        verbose: Print debug info
    
    Returns:
        SEDACEngine instance
    """
    config = SEDACConfig(
        exit_threshold=exit_threshold,
        min_exit_layer=min_exit_layer,
        anchor_interval=anchor_interval,
        verbose=verbose,
    )
    
    return SEDACEngine(model, config)
