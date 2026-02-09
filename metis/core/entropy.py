"""
METIS Token-level Entropy Heuristic
Fast cognitive signal

NOTE: This is NOT the generation-level semantic entropy defined by Kuhn et al. (ICLR 2023).
For the academic version, see: metis.core.semantic_entropy.SemanticEntropyEstimator

This module provides a fast per-token heuristic signal:
    H_heuristic = H_shannon × (1 + λ·D_embedding)

where D_embedding measures top-k candidate dispersion in embedding space.
This is the low-latency approximation used by System 1 (fast thinking).

Differences from academic definition:
    - This module: single token position softmax distribution x embedding diversity -> O(1) per token
    - Kuhn et al.: sample N complete generations -> bidirectional entailment clustering -> cluster entropy -> O(N^2) per prompt
    The former for per-step fast monitoring, the latter for System 2 authoritative verification.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math

from .types import ControllerConfig

LN2 = 0.6931471805599453  # ln(2) for bits conversion


class SemanticEntropyComputer:
    """
    Semantic Entropy Computer.
    
    Beyond Shannon's symbolic uncertainty:
    Shannon entropy only measures the "shape" of the probability distribution,
    not the semantic relationships between candidate tokens.
    Two distributions with identical probabilities but completely different semantics
    have the same Shannon entropy, but different cognitive uncertainty.
    
    Example:
    - P("happy")=0.5, P("glad")=0.5 -> low semantic diversity, model actually "knows"
    - P("cat")=0.5, P("table")=0.5 -> high semantic diversity, model genuinely confused
    """
    
    def __init__(self, config: ControllerConfig = None):
        self._config = config or ControllerConfig()
        self._embedding_matrix: Optional[torch.Tensor] = None
        self._lambda = self._config.semantic_weight
        self._top_k = self._config.top_k
    
    def set_embedding_matrix(self, embedding_matrix: torch.Tensor) -> None:
        """
        Set embedding matrix for semantic distance computation.
        
        Args:
            embedding_matrix: [vocab_size, hidden_dim] model's token embedding weights
        """
        self._embedding_matrix = embedding_matrix
    
    @torch.no_grad()
    def compute(self, logits: torch.Tensor) -> Tuple[float, float, float, float]:
        """
        Compute semantic entropy.
        
        Args:
            logits: [batch, seq, vocab] or [batch, vocab] or [vocab]
            
        Returns:
            (semantic_entropy, token_entropy, semantic_diversity, confidence)
        """
        # Normalize dimensions
        if logits.dim() == 3:
            logits = logits[:, -1, :]   # Take last token
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        # Shannon Entropy (bits)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        token_entropy = (-torch.sum(probs * log_probs, dim=-1) / LN2).item()
        
        # Handle nan/inf
        if math.isnan(token_entropy) or math.isinf(token_entropy):
            token_entropy = 0.0
        
        # Confidence
        confidence = probs.max(dim=-1).values.item()
        
        # Semantic Diversity
        semantic_diversity = self._compute_semantic_diversity(logits)
        
        # Combined Semantic Entropy
        # Amplify entropy when semantic diversity is high; suppress when candidates are semantically close
        semantic_entropy = token_entropy * (1.0 + self._lambda * semantic_diversity)
        
        return semantic_entropy, token_entropy, semantic_diversity, confidence
    
    @torch.no_grad()
    def _compute_semantic_diversity(self, logits: torch.Tensor) -> float:
        """
        Compute semantic diversity of top-k tokens.
        
        Uses average cosine distance of top-k tokens in embedding space.
        Diversity in [0, 1]
        - 0: top-k tokens semantically identical (e.g., synonyms)
        - 1: top-k tokens semantically disjoint (genuine confusion)
        """
        if self._embedding_matrix is None:
            return 0.0
        
        k = min(self._top_k, logits.shape[-1])
        _, top_indices = torch.topk(logits, k, dim=-1)  # [batch, k]
        
        # Get embeddings and normalize
        top_embeddings = self._embedding_matrix[top_indices]         # [batch, k, hidden]
        top_embeddings = F.normalize(top_embeddings, p=2, dim=-1)
        
        # Compute cosine similarity matrix
        sim_matrix = torch.bmm(top_embeddings, top_embeddings.transpose(1, 2))  # [batch, k, k]
        
        # Extract upper triangle (excluding self-similarity diagonal)
        mask = torch.triu(torch.ones(k, k, device=logits.device), diagonal=1)
        n_pairs = k * (k - 1) / 2
        
        avg_similarity = (sim_matrix * mask).sum(dim=(1, 2)) / (n_pairs + 1e-8)
        diversity = (1.0 - avg_similarity).clamp(0, 1).item()
        
        return diversity
    
    def compute_detailed(self, logits: torch.Tensor) -> Dict[str, float]:
        """Compute detailed entropy metrics"""
        se, te, sd, conf = self.compute(logits)
        return {
            "semantic_entropy": se,
            "token_entropy": te,
            "semantic_diversity": sd,
            "confidence": conf,
        }
