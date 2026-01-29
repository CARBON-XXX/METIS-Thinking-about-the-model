"""
SEDAC V9.0 - Ghost KV Generator
Lightweight MLP to predict KV cache for skipped layers

Architecture:
- Input: hidden_states from layer L
- Output: predicted K, V for layers L+1 to L+skip
- Training: distillation from full model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class GhostKVConfig:
    """Configuration for Ghost KV Generator"""
    
    hidden_size: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    num_kv_heads: int = 8  # GQA support
    max_skip_layers: int = 4
    
    # MLP config
    mlp_hidden_mult: float = 0.5  # Reduction factor
    mlp_layers: int = 2
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Loss weights
    kv_loss_weight: float = 1.0
    output_loss_weight: float = 0.5


class GhostKVGenerator(nn.Module):
    """
    Lightweight MLP to predict KV cache for skipped layers
    
    Input: hidden_states [batch, seq, hidden]
    Output: List of (K, V) tuples for skipped layers
    """
    
    def __init__(self, config: GhostKVConfig):
        super().__init__()
        self.config = config
        
        hidden = config.hidden_size
        mlp_hidden = int(hidden * config.mlp_hidden_mult)
        kv_dim = config.num_kv_heads * config.head_dim
        
        # Shared encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(hidden, mlp_hidden))
        encoder_layers.append(nn.GELU())
        encoder_layers.append(nn.Dropout(config.dropout))
        
        for _ in range(config.mlp_layers - 1):
            encoder_layers.append(nn.Linear(mlp_hidden, mlp_hidden))
            encoder_layers.append(nn.GELU())
            encoder_layers.append(nn.Dropout(config.dropout))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Per-layer K/V predictors
        self.k_predictors = nn.ModuleList([
            nn.Linear(mlp_hidden, kv_dim)
            for _ in range(config.max_skip_layers)
        ])
        
        self.v_predictors = nn.ModuleList([
            nn.Linear(mlp_hidden, kv_dim)
            for _ in range(config.max_skip_layers)
        ])
        
        # Learnable layer embeddings
        self.layer_embed = nn.Embedding(config.max_skip_layers, mlp_hidden)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden]
        num_skip_layers: int = 1,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate Ghost KV for skipped layers
        
        Returns:
            List of (K, V) tuples, each [batch, num_kv_heads, seq, head_dim]
        """
        batch, seq, _ = hidden_states.shape
        
        # Encode
        encoded = self.encoder(hidden_states)  # [batch, seq, mlp_hidden]
        
        kv_pairs = []
        for i in range(min(num_skip_layers, self.config.max_skip_layers)):
            # Add layer embedding
            layer_emb = self.layer_embed.weight[i]  # [mlp_hidden]
            layer_encoded = encoded + layer_emb.unsqueeze(0).unsqueeze(0)
            
            # Predict K, V
            k = self.k_predictors[i](layer_encoded)  # [batch, seq, kv_dim]
            v = self.v_predictors[i](layer_encoded)
            
            # Reshape to [batch, num_kv_heads, seq, head_dim]
            k = k.view(batch, seq, self.config.num_kv_heads, self.config.head_dim)
            k = k.transpose(1, 2)
            
            v = v.view(batch, seq, self.config.num_kv_heads, self.config.head_dim)
            v = v.transpose(1, 2)
            
            kv_pairs.append((k, v))
        
        return kv_pairs
    
    def compute_loss(
        self,
        predicted_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        target_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss against teacher KV cache
        
        Args:
            predicted_kv: List of (K, V) from Ghost KV
            target_kv: List of (K, V) from teacher model
        
        Returns:
            Dict with loss components
        """
        total_k_loss = 0.0
        total_v_loss = 0.0
        
        for (pred_k, pred_v), (tgt_k, tgt_v) in zip(predicted_kv, target_kv):
            # Normalize and compute cosine similarity loss
            k_loss = 1 - F.cosine_similarity(
                pred_k.flatten(2), tgt_k.flatten(2), dim=-1
            ).mean()
            v_loss = 1 - F.cosine_similarity(
                pred_v.flatten(2), tgt_v.flatten(2), dim=-1
            ).mean()
            
            total_k_loss += k_loss
            total_v_loss += v_loss
        
        num_layers = len(predicted_kv)
        k_loss = total_k_loss / num_layers
        v_loss = total_v_loss / num_layers
        
        total_loss = self.config.kv_loss_weight * (k_loss + v_loss)
        
        return {
            "loss": total_loss,
            "k_loss": k_loss,
            "v_loss": v_loss,
        }


class GhostKVTrainer:
    """
    Trainer for Ghost KV Generator using knowledge distillation
    """
    
    def __init__(
        self,
        ghost_kv: GhostKVGenerator,
        teacher_model: nn.Module,
        config: GhostKVConfig,
        device: torch.device = None,
    ):
        self.ghost_kv = ghost_kv
        self.teacher = teacher_model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move to device
        self.ghost_kv.to(self.device)
        self.teacher.to(self.device)
        self.teacher.eval()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            ghost_kv.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Stats
        self.step = 0
        self.best_loss = float("inf")
    
    def extract_kv_cache(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        start_layer: int,
        num_layers: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract KV cache from teacher model layers"""
        kv_list = []
        
        # This is model-specific - simplified example
        # In practice, hook into the model's attention layers
        
        current_hidden = hidden_states
        layers = model.model.layers if hasattr(model, "model") else model.layers
        
        for i in range(start_layer, min(start_layer + num_layers, len(layers))):
            layer = layers[i]
            
            # Get attention module
            attn = layer.self_attn
            
            # Compute K, V projections
            batch, seq, hidden = current_hidden.shape
            
            # Project to K, V
            if hasattr(attn, "k_proj"):
                k = attn.k_proj(current_hidden)
                v = attn.v_proj(current_hidden)
            else:
                # Fallback for different architectures
                qkv = attn.qkv_proj(current_hidden) if hasattr(attn, "qkv_proj") else None
                if qkv is not None:
                    # Split QKV
                    q_size = attn.num_heads * attn.head_dim
                    kv_size = attn.num_kv_heads * attn.head_dim
                    k = qkv[:, :, q_size:q_size + kv_size]
                    v = qkv[:, :, q_size + kv_size:q_size + 2*kv_size]
                else:
                    # Skip if can't extract
                    continue
            
            # Reshape
            num_kv_heads = self.config.num_kv_heads
            head_dim = self.config.head_dim
            
            k = k.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)
            
            kv_list.append((k.detach(), v.detach()))
            
            # Forward through layer for next iteration
            with torch.no_grad():
                current_hidden = layer(current_hidden)[0]
        
        return kv_list
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        skip_start_layer: int = 4,
        num_skip_layers: int = 2,
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            input_ids: [batch, seq] input tokens
            skip_start_layer: Layer to start skipping from
            num_skip_layers: Number of layers to skip
        
        Returns:
            Dict with loss values
        """
        self.ghost_kv.train()
        
        # Get hidden states at skip_start_layer from teacher
        with torch.no_grad():
            # Forward through teacher up to skip_start_layer
            hidden = self.teacher.model.embed_tokens(input_ids)
            
            for i in range(skip_start_layer):
                layer = self.teacher.model.layers[i]
                hidden = layer(hidden)[0]
            
            input_hidden = hidden.clone()
            
            # Get target KV from teacher
            target_kv = self.extract_kv_cache(
                self.teacher, hidden,
                skip_start_layer, num_skip_layers
            )
        
        # Generate Ghost KV
        predicted_kv = self.ghost_kv(input_hidden, num_skip_layers)
        
        # Compute loss
        if len(target_kv) > 0 and len(predicted_kv) > 0:
            loss_dict = self.ghost_kv.compute_loss(predicted_kv, target_kv)
            
            # Backward
            self.optimizer.zero_grad()
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.ghost_kv.parameters(), 1.0)
            self.optimizer.step()
            
            self.step += 1
            
            return {k: v.item() for k, v in loss_dict.items()}
        else:
            return {"loss": 0.0, "k_loss": 0.0, "v_loss": 0.0}
    
    def train_epoch(
        self,
        dataloader,
        skip_start_layer: int = 4,
        num_skip_layers: int = 2,
    ) -> Dict[str, float]:
        """Train for one epoch"""
        total_loss = 0.0
        total_k_loss = 0.0
        total_v_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            
            losses = self.train_step(input_ids, skip_start_layer, num_skip_layers)
            
            total_loss += losses["loss"]
            total_k_loss += losses["k_loss"]
            total_v_loss += losses["v_loss"]
            num_batches += 1
        
        return {
            "loss": total_loss / max(num_batches, 1),
            "k_loss": total_k_loss / max(num_batches, 1),
            "v_loss": total_v_loss / max(num_batches, 1),
        }
    
    def save(self, path: str):
        """Save Ghost KV model"""
        torch.save({
            "model_state_dict": self.ghost_kv.state_dict(),
            "config": self.config,
            "step": self.step,
            "best_loss": self.best_loss,
        }, path)
    
    def load(self, path: str):
        """Load Ghost KV model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.ghost_kv.load_state_dict(checkpoint["model_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))


def create_ghost_kv_for_model(model: nn.Module) -> GhostKVGenerator:
    """
    Create Ghost KV Generator matching model configuration
    
    Args:
        model: HuggingFace model
    
    Returns:
        GhostKVGenerator instance
    """
    config = model.config
    
    # Determine KV heads (GQA support)
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    
    ghost_config = GhostKVConfig(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
    )
    
    return GhostKVGenerator(ghost_config)
