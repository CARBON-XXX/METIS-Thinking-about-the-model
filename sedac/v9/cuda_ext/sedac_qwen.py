"""
SEDAC V9.0 - Qwen Model Integration with CUDA Kernels

Inject SEDAC early-exit logic into Qwen2.5 inference.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import time

# Load CUDA kernels
try:
    import sedac_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: sedac_cuda not available, using PyTorch fallback")


@dataclass
class SEDACConfig:
    """SEDAC configuration"""
    exit_threshold: float = 0.7
    min_layers: int = 4          # Minimum layers before exit allowed
    anchor_interval: int = 4     # Anchor layers (no skip)
    entropy_window: int = 100    # Window for running stats
    enable_ghost_kv: bool = False
    verbose: bool = False


class SEDACQwenWrapper(nn.Module):
    """
    SEDAC wrapper for Qwen models with CUDA kernel acceleration.
    
    Features:
    - Early exit based on entropy/confidence
    - CUDA-accelerated decision making
    - KV cache management for skipped layers
    - Statistics tracking
    """
    
    def __init__(self, model, config: SEDACConfig = None):
        super().__init__()
        self.model = model
        self.config = config or SEDACConfig()
        
        # Model info
        self.num_layers = model.config.num_hidden_layers
        self.hidden_size = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        
        # Running statistics
        self.entropy_sum = 0.0
        self.entropy_sq_sum = 0.0
        self.entropy_count = 0
        self.entropy_mean = 3.0  # Initial estimate
        self.entropy_std = 1.0
        
        # Stats tracking
        self.stats = {
            "total_tokens": 0,
            "total_layers_computed": 0,
            "total_layers_skipped": 0,
            "exit_decisions": [],
        }
        
        # Anchor layers (never skip)
        self.anchor_layers = set(
            range(0, self.num_layers, self.config.anchor_interval)
        )
        self.anchor_layers.add(self.num_layers - 1)  # Always compute last layer
        
        print(f"SEDAC initialized: {self.num_layers} layers, anchors={sorted(self.anchor_layers)}")
    
    def _update_entropy_stats(self, entropy: torch.Tensor):
        """Update running entropy statistics"""
        batch_mean = entropy.mean().item()
        batch_var = entropy.var().item() if entropy.numel() > 1 else 0.0
        
        n = self.entropy_count
        m = entropy.numel()
        
        if n == 0:
            self.entropy_mean = batch_mean
            self.entropy_std = batch_var ** 0.5 if batch_var > 0 else 1.0
        else:
            # Welford's online algorithm
            delta = batch_mean - self.entropy_mean
            self.entropy_mean += delta * m / (n + m)
            self.entropy_std = max(0.1, (self.entropy_std ** 2 * 0.9 + batch_var * 0.1) ** 0.5)
        
        self.entropy_count = min(self.entropy_count + m, self.config.entropy_window)
    
    def _should_exit(
        self,
        hidden_states: torch.Tensor,
        prev_hidden: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decide which tokens should exit at this layer.
        
        Returns:
            exit_mask: [batch, seq] bool tensor
            stats: dict with entropy, confidence, etc.
        """
        # Never exit from anchor layers or before min_layers
        if layer_idx in self.anchor_layers or layer_idx < self.config.min_layers:
            return torch.zeros(hidden_states.shape[:-1], dtype=torch.bool, device=hidden_states.device), {}
        
        # Compute logits for decision (use LM head)
        with torch.no_grad():
            logits = self.model.lm_head(hidden_states)  # [batch, seq, vocab]
        
        # Flatten for kernel
        batch, seq, hidden = hidden_states.shape
        logits_flat = logits.view(batch * seq, -1).float()
        hidden_flat = hidden_states.view(batch * seq, hidden).float()
        prev_flat = prev_hidden.view(batch * seq, hidden).float()
        
        layer_progress = layer_idx / self.num_layers
        
        # Use CUDA kernel
        if CUDA_AVAILABLE:
            entropy, confidence, decision, load = sedac_cuda.fused_entropy_decision(
                logits_flat,
                hidden_flat,
                prev_flat,
                self.entropy_mean,
                self.entropy_std,
                layer_progress,
                self.config.exit_threshold,
            )
        else:
            # PyTorch fallback
            import torch.nn.functional as F
            import math
            
            log_probs = F.log_softmax(logits_flat, dim=-1)
            probs = log_probs.exp()
            entropy = -torch.sum(probs * log_probs, dim=-1) / math.log(2)
            
            diff = hidden_flat - prev_flat
            stability = 1.0 / (1.0 + torch.norm(diff, dim=-1) / (torch.norm(hidden_flat, dim=-1) + 1e-6))
            
            z_score = (self.entropy_mean - entropy) / (self.entropy_std + 1e-6)
            confidence = torch.sigmoid(z_score * 2.0)
            
            current_thresh = self.config.exit_threshold - layer_progress * 0.2
            decision = (confidence * stability * layer_progress) > current_thresh
            load = (1.0 - confidence) * 0.5 + (1.0 - stability) * 0.3
        
        # Update stats
        self._update_entropy_stats(entropy)
        
        exit_mask = decision.view(batch, seq)
        
        return exit_mask, {
            "entropy": entropy.view(batch, seq),
            "confidence": confidence.view(batch, seq),
        }
    
    def forward_with_sedac(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass with SEDAC early exit.
        
        Note: This is a simplified version that demonstrates the concept.
        Full integration requires modifying the model's forward pass.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        hidden_states = self.model.model.embed_tokens(input_ids)
        
        # Track which tokens are still active
        active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        final_hidden = torch.zeros_like(hidden_states)
        
        # Layer-by-layer processing
        prev_hidden = hidden_states.clone()
        layers_computed = 0
        layers_skipped = 0
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Check for early exit
            exit_mask, decision_stats = self._should_exit(hidden_states, prev_hidden, layer_idx)
            
            # Tokens that exit here
            newly_exited = exit_mask & active_mask
            if newly_exited.any():
                # Store final hidden states for exited tokens
                final_hidden[newly_exited] = hidden_states[newly_exited]
                active_mask[newly_exited] = False
                
                if self.config.verbose:
                    print(f"Layer {layer_idx}: {newly_exited.sum().item()} tokens exited")
            
            # If all tokens exited, stop
            if not active_mask.any():
                layers_skipped += self.num_layers - layer_idx - 1
                break
            
            # Compute layer for active tokens
            prev_hidden = hidden_states.clone()
            
            # Generate position_ids
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Get position embeddings from model
            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            
            # Full layer computation
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[layer_idx] if past_key_values else None,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_output[0]
            
            layers_computed += 1
        
        # Remaining active tokens use final hidden states
        final_hidden[active_mask] = hidden_states[active_mask]
        
        # Apply final norm and LM head
        final_hidden = self.model.model.norm(final_hidden)
        logits = self.model.lm_head(final_hidden)
        
        # Update stats
        self.stats["total_tokens"] += batch_size * seq_len
        self.stats["total_layers_computed"] += layers_computed
        self.stats["total_layers_skipped"] += layers_skipped
        
        return {
            "logits": logits,
            "hidden_states": final_hidden,
            "layers_computed": layers_computed,
            "layers_skipped": layers_skipped,
            "exit_ratio": 1.0 - active_mask.float().mean().item(),
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get SEDAC statistics"""
        total_possible = self.stats["total_layers_computed"] + self.stats["total_layers_skipped"]
        skip_ratio = self.stats["total_layers_skipped"] / max(1, total_possible)
        
        return {
            "total_tokens": self.stats["total_tokens"],
            "layers_computed": self.stats["total_layers_computed"],
            "layers_skipped": self.stats["total_layers_skipped"],
            "skip_ratio": skip_ratio,
            "entropy_mean": self.entropy_mean,
            "entropy_std": self.entropy_std,
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_tokens": 0,
            "total_layers_computed": 0,
            "total_layers_skipped": 0,
            "exit_decisions": [],
        }


def benchmark_sedac_qwen(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Benchmark SEDAC on Qwen model"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 70)
    print("SEDAC V9.0 - Qwen Benchmark")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    
    # Wrap with SEDAC
    config = SEDACConfig(
        exit_threshold=0.6,
        min_layers=4,
        anchor_interval=6,
        verbose=True,
    )
    sedac_model = SEDACQwenWrapper(model, config)
    
    # Test prompts
    prompts = [
        "What is 2+2?",
        "Hello, how are you?",
        "Explain machine learning.",
    ]
    
    print("\n" + "-" * 70)
    print("SEDAC Inference Test")
    print("-" * 70)
    
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            result = sedac_model.forward_with_sedac(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                use_cache=False,
            )
        
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"\nPrompt: {prompt}")
        print(f"  Time: {elapsed:.1f}ms")
        print(f"  Layers computed: {result['layers_computed']}")
        print(f"  Exit ratio: {result['exit_ratio']*100:.1f}%")
    
    # Print stats
    stats = sedac_model.get_stats()
    print("\n" + "-" * 70)
    print("Overall Statistics")
    print("-" * 70)
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Skip ratio: {stats['skip_ratio']*100:.1f}%")
    print(f"  Entropy: mean={stats['entropy_mean']:.3f}, std={stats['entropy_std']:.3f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    benchmark_sedac_qwen()
