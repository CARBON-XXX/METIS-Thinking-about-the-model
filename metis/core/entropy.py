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
        Compute probability-weighted semantic diversity of top tokens.
        
        Uses softmax-weighted cosine distance in embedding space.
        Diversity in [0, 1]
        - 0: high-probability tokens are semantically similar (synonyms)
        - 1: high-probability tokens are semantically disjoint (genuine confusion)
        
        Key insight: unweighted cosine distance fails in high-dimensional space
        (concentration of measure → all pairs ≈ orthogonal → diversity ≈ 1.0).
        Probability weighting focuses on the tokens that actually matter:
        if top-2 are synonyms at 40%+35%, their high-weight pair drives diversity DOWN.
        """
        if self._embedding_matrix is None:
            return 0.0
        
        # Use fewer tokens to focus on high-probability candidates
        k = min(5, self._top_k, logits.shape[-1])
        top_values, top_indices = torch.topk(logits, k, dim=-1)  # [batch, k]
        
        # Softmax over top-k only (re-normalize among candidates)
        top_probs = F.softmax(top_values.float(), dim=-1)  # [batch, k]
        
        # Get embeddings and normalize
        top_embeddings = self._embedding_matrix[top_indices]         # [batch, k, hidden]
        top_embeddings = F.normalize(top_embeddings.float(), p=2, dim=-1)
        
        # Compute cosine similarity matrix
        sim_matrix = torch.bmm(top_embeddings, top_embeddings.transpose(1, 2))  # [batch, k, k]
        
        # Probability-weighted pairwise distance
        # Weight of pair (i, j) = p_i * p_j
        # This ensures high-probability synonym pairs dominate the score
        weight_matrix = torch.bmm(
            top_probs.unsqueeze(2),   # [batch, k, 1]
            top_probs.unsqueeze(1),   # [batch, 1, k]
        )  # [batch, k, k]
        
        # Upper triangle mask (exclude self-similarity on diagonal)
        mask = torch.triu(torch.ones(k, k, device=logits.device), diagonal=1)
        
        weighted_sim = (sim_matrix * weight_matrix * mask).sum(dim=(1, 2))
        weight_sum = (weight_matrix * mask).sum(dim=(1, 2))
        
        avg_weighted_sim = weighted_sim / (weight_sum + 1e-8)
        diversity = (1.0 - avg_weighted_sim).clamp(0, 1).item()
        
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
