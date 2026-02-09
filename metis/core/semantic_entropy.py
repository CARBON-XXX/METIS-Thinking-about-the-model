"""
METIS Generation-level Semantic Entropy
Kuhn et al. (ICLR 2023)

Academic definition (Kuhn et al., "Semantic Uncertainty: Linguistic Invariances
for Uncertainty Estimation in Natural Language Generation"):

    1. Sample N complete generations for the same prompt
    2. For each pair (i, j), perform bidirectional entailment check
       - If a entails b AND b entails a, then a == b (semantic equivalence)
    3. Cluster by semantic equivalence into K equivalence classes C_1, ..., C_K
    4. Compute semantic entropy:
       SE = -sum_k p(C_k) log2 p(C_k)

       where p(C_k) can be:
       - Frequency estimate: |C_k| / N
       - Probability-weighted: sum_{gen in C_k} p(gen) / sum_all p(gen)

    SE = 0  -> All generations semantically consistent, model is certain
    SE > 0  -> Semantic disagreement exists, model is genuinely uncertain

Differences from token-level heuristic (entropy.py):
    - token-level: single token position softmax entropy x embedding diversity
    - generation-level: semantic equivalence between complete answers
    The latter is the academically recognized uncertainty measure, the former is an engineering approximation.

Supports two semantic equivalence methods:
    1. NLI (Natural Language Inference) - academic standard, using DeBERTa or similar NLI models
    2. Embedding similarity - lightweight fallback, using sentence embedding cosine similarity
"""
from __future__ import annotations

import math
import logging
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .types import (
    GenerationSample,
    SemanticCluster,
    SemanticEntropyResult,
)

logger = logging.getLogger(__name__)

LN2 = 0.6931471805599453


# =============================================================
# Semantic Equivalence Checker
# =============================================================

class NLIEquivalenceChecker:
    """
    NLI-based semantic equivalence checker (academic standard).

    Uses cross-encoder NLI model for bidirectional entailment:
    - entailment(a->b) AND entailment(b->a) -> semantic equivalence
    - Corresponds to bidirectional entailment in Kuhn et al.

    Ref: Kuhn et al. (2023), Section 3.2
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: Optional[str] = None,
        entailment_threshold: float = 0.5,
    ):
        self._model_name = model_name
        self._entailment_threshold = entailment_threshold
        self._model = None
        self._tokenizer = None

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    def _ensure_loaded(self) -> None:
        """Lazy-load NLI model"""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for NLI equivalence checking. "
                "Install with: pip install transformers"
            )

        logger.info(f"Loading NLI model: {self._model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        ).to(self._device)
        self._model.eval()

        # Determine label mapping (different NLI models have different label orders)
        id2label = self._model.config.id2label
        self._entailment_id = None
        for idx, label in id2label.items():
            if label.lower() in ("entailment", "entail"):
                self._entailment_id = int(idx)
                break
        if self._entailment_id is None:
            # Fallback: assume label 2 = entailment (MNLI standard)
            self._entailment_id = 2
            logger.warning(
                f"Could not find 'entailment' label in model config. "
                f"Falling back to id={self._entailment_id}."
            )

    @torch.no_grad()
    def check_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Check entailment probability: premise -> hypothesis.

        Returns:
            float: entailment probability in [0, 1]
        """
        self._ensure_loaded()

        inputs = self._tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._device)

        logits = self._model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        return probs[0, self._entailment_id].item()

    def check_bidirectional(self, text_a: str, text_b: str) -> float:
        """
        Bidirectional entailment check (core operation of Kuhn et al.).

        Returns:
            float: min(entailment(a->b), entailment(b->a))
                   i.e., minimum probability of bidirectional entailment
        """
        p_ab = self.check_entailment(text_a, text_b)
        p_ba = self.check_entailment(text_b, text_a)
        return min(p_ab, p_ba)

    def compute_entailment_matrix(
        self, texts: List[str]
    ) -> List[List[float]]:
        """
        Compute NxN bidirectional entailment matrix.

        Returns:
            matrix[i][j] = min(entailment(i->j), entailment(j->i))
        """
        self._ensure_loaded()
        n = len(texts)
        matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                score = self.check_bidirectional(texts[i], texts[j])
                matrix[i][j] = score
                matrix[j][i] = score

        return matrix

    def are_equivalent(self, text_a: str, text_b: str) -> bool:
        """Determine if two texts are semantically equivalent"""
        return self.check_bidirectional(text_a, text_b) >= self._entailment_threshold


class EmbeddingEquivalenceChecker:
    """
    Embedding-based semantic equivalence checker (lightweight fallback).

    Uses sentence embedding cosine similarity for semantic equivalence.
    Lower precision than NLI, but requires no additional model.

    Embedding source priority:
    1. sentence-transformers model (if specified)
    2. Generative model's own hidden states mean-pooling (zero extra overhead)
       - Uses model's own representation space to judge semantic equivalence
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        # Generative model reference (for hidden-state embedding)
        generative_model: Optional[torch.nn.Module] = None,
        generative_tokenizer=None,
    ):
        self._threshold = similarity_threshold
        self._sentence_model_name = model_name
        self._sentence_model = None
        self._generative_model = generative_model
        self._generative_tokenizer = generative_tokenizer

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    def set_generative_model(
        self, model: torch.nn.Module, tokenizer
    ) -> None:
        """Set generative model reference (for hidden-state embedding)"""
        self._generative_model = model
        self._generative_tokenizer = tokenizer

    def _ensure_sentence_model(self) -> None:
        """Lazy-load sentence-transformers model"""
        if self._sentence_model is not None:
            return
        if self._sentence_model_name is None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence model: {self._sentence_model_name}")
            self._sentence_model = SentenceTransformer(
                self._sentence_model_name, device=self._device
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Will use generative model hidden states."
            )

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into vectors.

        Priority: sentence-transformers > generative model hidden states

        Returns:
            [N, dim] normalized embeddings
        """
        self._ensure_sentence_model()

        if self._sentence_model is not None:
            embeddings = self._sentence_model.encode(
                texts, convert_to_tensor=True, normalize_embeddings=True,
            )
            return embeddings

        # Use generative model's own hidden states
        if self._generative_model is not None and self._generative_tokenizer is not None:
            return self._encode_with_generative_model(texts)

        raise RuntimeError(
            "EmbeddingEquivalenceChecker requires at least one embedding source: "
            "pass model_name (sentence-transformers) or "
            "generative_model + generative_tokenizer (zero-overhead hidden-state)"
        )

    @torch.no_grad()
    def _encode_with_generative_model(self, texts: List[str]) -> torch.Tensor:
        """
        Use generative model's own last hidden state with mean-pooling.

        Rationale: CausalLM hidden states already encode rich semantic information.
        Mean-pooling followed by normalization yields sentence embeddings.
        No extra model needed, reuses the inference model's parameters.

        Returns:
            [N, dim] normalized embeddings
        """
        model = self._generative_model
        tokenizer = self._generative_tokenizer
        embeddings = []

        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=512, padding=False,
            ).to(model.device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # Take last layer hidden states, apply mean-pooling
            # shape: [1, seq_len, hidden_dim]
            last_hidden = outputs.hidden_states[-1]
            # attention_mask excludes padding (single sequence so generally no padding)
            mask = inputs.attention_mask.unsqueeze(-1).float()  # [1, seq_len, 1]
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [1, hidden_dim]
            embeddings.append(pooled.squeeze(0))

        result = torch.stack(embeddings)  # [N, hidden_dim]
        return F.normalize(result, p=2, dim=-1)

    def compute_entailment_matrix(
        self, texts: List[str]
    ) -> List[List[float]]:
        """
        Compute NxN cosine similarity matrix.

        Returns:
            matrix[i][j] = cosine_similarity(embed_i, embed_j)
        """
        embeddings = self.encode_texts(texts)  # [N, dim]
        sim = torch.mm(embeddings, embeddings.T)  # [N, N]
        return sim.cpu().tolist()

    def are_equivalent(self, text_a: str, text_b: str) -> bool:
        """Determine if two texts are semantically equivalent"""
        embeddings = self.encode_texts([text_a, text_b])
        sim = F.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
        return sim >= self._threshold


# =============================================================
# Multi-Generation Sampler
# =============================================================

class GenerationSampler:
    """
    Sample N complete generations for the same prompt.

    Kuhn et al. use temperature sampling to obtain diverse answers.
    Each generation carries sequence log-probability for probability weighting.
    """

    @staticmethod
    @torch.no_grad()
    def sample(
        model: torch.nn.Module,
        tokenizer,
        prompt: str,
        n_samples: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
        chat_template: bool = True,
    ) -> List[GenerationSample]:
        """
        Sample N complete generations.

        Args:
            model: HuggingFace CausalLM
            tokenizer: Corresponding tokenizer
            prompt: User input
            n_samples: Number of samples (Kuhn et al. recommend 5-10)
            temperature: Sampling temperature (>0, higher = more diverse)
            max_new_tokens: Maximum generation length
            top_p: Nucleus sampling threshold
            chat_template: Whether to use chat template

        Returns:
            List[GenerationSample]: N sampled results
        """
        if chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        samples: List[GenerationSample] = []

        # Try batch sampling (num_return_sequences)
        # If model supports it, one forward generates N sequences, much faster than N independent calls
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=n_samples,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # outputs.sequences: [n_samples, seq_len]
            for i in range(n_samples):
                generated_ids = outputs.sequences[i, input_len:]
                generated_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                # In batch mode, scores share the step dimension,
                # need to extract using batch dimension i
                log_prob = GenerationSampler._compute_sequence_log_prob_batch(
                    outputs.scores, generated_ids, batch_idx=i
                )
                samples.append(GenerationSample(
                    text=generated_text,
                    log_prob=log_prob,
                    tokens=generated_ids.cpu().tolist(),
                ))
        except Exception:
            # Fallback: sequential sampling (compatible with models that don't support num_return_sequences)
            logger.debug("Batch sampling failed, falling back to sequential")
            for _ in range(n_samples):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                generated_ids = outputs.sequences[0, input_len:]
                generated_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                log_prob = GenerationSampler._compute_sequence_log_prob_batch(
                    outputs.scores, generated_ids, batch_idx=0
                )
                samples.append(GenerationSample(
                    text=generated_text,
                    log_prob=log_prob,
                    tokens=generated_ids.cpu().tolist(),
                ))

        return samples

    @staticmethod
    def _compute_sequence_log_prob_batch(
        scores: Tuple[torch.Tensor, ...],
        generated_ids: torch.Tensor,
        batch_idx: int = 0,
    ) -> float:
        """
        Compute sequence log-probability (supports batch sampling).

        Args:
            scores: output_scores from model.generate(), per-step logits
                    shape per step: [batch_size, vocab_size]
            generated_ids: Actually generated token ids (single sequence)
            batch_idx: Batch dimension index

        Returns:
            float: Sequence log-probability (length-normalized)
        """
        total_log_prob = 0.0
        n_tokens = min(len(scores), len(generated_ids))

        for step in range(n_tokens):
            step_logits = scores[step]
            # step_logits shape: [batch_size, vocab_size]
            if step_logits.dim() == 1:
                log_probs = F.log_softmax(step_logits.float(), dim=-1)
            else:
                log_probs = F.log_softmax(step_logits[batch_idx].float(), dim=-1)
            token_id = generated_ids[step].item()
            total_log_prob += log_probs[token_id].item()

        # Length-normalized to avoid bias toward shorter sequences
        if n_tokens > 0:
            total_log_prob /= n_tokens

        return total_log_prob


# =============================================================
# Semantic Clustering
# =============================================================

def cluster_by_equivalence(
    entailment_matrix: List[List[float]],
    threshold: float = 0.5,
) -> List[List[int]]:
    """
    Cluster into semantic equivalence classes based on entailment matrix.

    Uses Union-Find algorithm:
    - If matrix[i][j] >= threshold, then i and j belong to the same equivalence class
    - Transitivity: if a==b and b==c, then a==c

    Corresponds to equivalence class construction in Kuhn et al.

    Args:
        entailment_matrix: NxN bidirectional entailment scores
        threshold: Equivalence judgment threshold

    Returns:
        List[List[int]]: Each sublist contains member indices of one equivalence class
    """
    n = len(entailment_matrix)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if entailment_matrix[i][j] >= threshold:
                union(i, j)

    # Collect equivalence classes
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    return list(clusters.values())


# =============================================================
# Semantic Entropy Computation
# =============================================================

def compute_semantic_entropy(
    clusters: List[List[int]],
    generations: List[GenerationSample],
    use_probability_weighting: bool = True,
) -> Tuple[float, List[SemanticCluster]]:
    """
    Compute semantic entropy (Kuhn et al., ICLR 2023, Equation 3).

    SE = -sum_k p(C_k) log2 p(C_k)

    Two p(C_k) estimates:
    1. Frequency: p(C_k) = |C_k| / N
    2. Probability-weighted: p(C_k) = sum_{gen in C_k} exp(log_prob(gen)) / Z
       where Z = sum_all exp(log_prob(gen))

    The paper recommends probability weighting as it leverages model confidence per generation.

    Args:
        clusters: Clustering result (list of member index lists)
        generations: All generation samples
        use_probability_weighting: Whether to use probability weighting

    Returns:
        (semantic_entropy, list_of_SemanticCluster)
    """
    n = len(generations)
    if n == 0:
        return 0.0, []

    semantic_clusters: List[SemanticCluster] = []

    if use_probability_weighting:
        # Probability-weighted: p(C_k) = sum exp(log_prob) / Z
        # Use log-sum-exp for numerical stability
        all_log_probs = [g.log_prob for g in generations]
        log_Z = _log_sum_exp(all_log_probs)

        for member_indices in clusters:
            cluster_log_probs = [generations[i].log_prob for i in member_indices]
            log_cluster_prob = _log_sum_exp(cluster_log_probs) - log_Z
            cluster_prob = math.exp(log_cluster_prob)

            semantic_clusters.append(SemanticCluster(
                members=member_indices,
                probability=cluster_prob,
            ))
    else:
        # Frequency estimate: p(C_k) = |C_k| / N
        for member_indices in clusters:
            cluster_prob = len(member_indices) / n
            semantic_clusters.append(SemanticCluster(
                members=member_indices,
                probability=cluster_prob,
            ))

    # SE = -Σ p(C_k) log₂ p(C_k)
    se = 0.0
    for cluster in semantic_clusters:
        p = cluster.probability
        if p > 1e-10:
            se -= p * math.log2(p)

    return se, semantic_clusters


def _log_sum_exp(values: List[float]) -> float:
    """Numerically stable log-sum-exp"""
    if not values:
        return float('-inf')
    max_val = max(values)
    if math.isinf(max_val):
        return float('-inf')
    return max_val + math.log(sum(math.exp(v - max_val) for v in values))


# =============================================================
# Main Class: SemanticEntropyEstimator
# =============================================================

class SemanticEntropyEstimator:
    """
    Kuhn et al. (ICLR 2023) Semantic Entropy Estimator.

    Full pipeline:
        prompt -> sample N generations -> bidirectional entailment
        -> cluster by equivalence -> compute SE

    This is the academically recognized LLM uncertainty measurement method.
    Unlike token-level heuristics, it measures semantic consistency between complete answers.

    Usage:
        estimator = SemanticEntropyEstimator(method="nli")
        result = estimator.estimate(model, tokenizer, "What is the capital of France?")

        if result.is_uncertain:
            print("Model is genuinely uncertain")
            print(f"SE = {result.semantic_entropy:.2f} bits")
            print(f"{result.n_clusters} semantic clusters from {result.n_samples} samples")
        else:
            print(f"Confident answer: {result.majority_answer}")
    """

    def __init__(
        self,
        method: str = "nli",
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
        embedding_model_name: Optional[str] = None,
        entailment_threshold: float = 0.5,
        embedding_similarity_threshold: float = 0.85,
        n_samples: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
        uncertainty_threshold: float = 0.5,
        use_probability_weighting: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            method: "nli" (academic standard) or "embedding" (lightweight fallback)
            nli_model_name: NLI cross-encoder model name
            embedding_model_name: sentence-transformers model name (embedding mode)
            entailment_threshold: NLI entailment judgment threshold
            embedding_similarity_threshold: Embedding similarity judgment threshold
            n_samples: Number of samples (recommended 5-10)
            temperature: Sampling temperature
            max_new_tokens: Maximum generation length
            top_p: Nucleus sampling threshold
            uncertainty_threshold: SE > this value is judged as uncertain
            use_probability_weighting: Whether to use probability weighting (paper recommends True)
            device: Compute device
        """
        self._method = method
        self._n_samples = n_samples
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._top_p = top_p
        self._uncertainty_threshold = uncertainty_threshold
        self._use_prob_weighting = use_probability_weighting

        if method == "nli":
            self._checker = NLIEquivalenceChecker(
                model_name=nli_model_name,
                device=device,
                entailment_threshold=entailment_threshold,
            )
            self._equiv_threshold = entailment_threshold
        elif method == "embedding":
            self._checker = EmbeddingEquivalenceChecker(
                similarity_threshold=embedding_similarity_threshold,
                model_name=embedding_model_name,
                device=device,
            )
            self._equiv_threshold = embedding_similarity_threshold
        else:
            raise ValueError(f"method must be 'nli' or 'embedding', got '{method}'")

    def estimate(
        self,
        model: torch.nn.Module,
        tokenizer,
        prompt: str,
        n_samples: Optional[int] = None,
        chat_template: bool = True,
    ) -> SemanticEntropyResult:
        """
        Full pipeline: sample -> entailment check -> cluster -> compute SE.

        Args:
            model: HuggingFace CausalLM
            tokenizer: Corresponding tokenizer
            prompt: User input
            n_samples: Override default sample count
            chat_template: Whether to use chat template

        Returns:
            SemanticEntropyResult: Complete semantic entropy analysis result
        """
        n = n_samples or self._n_samples

        # If using embedding method, auto-pass generative model reference
        if isinstance(self._checker, EmbeddingEquivalenceChecker):
            self._checker.set_generative_model(model, tokenizer)

        # Step 1: Sample N complete generations
        generations = GenerationSampler.sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n_samples=n,
            temperature=self._temperature,
            max_new_tokens=self._max_new_tokens,
            top_p=self._top_p,
            chat_template=chat_template,
        )

        return self.estimate_from_generations(generations)

    def estimate_from_generations(
        self,
        generations: List[GenerationSample],
    ) -> SemanticEntropyResult:
        """
        Compute semantic entropy from existing generations (skip sampling step).

        Suitable for:
        - Already having multiple generation results
        - Custom sampling strategies
        - Unit testing

        Args:
            generations: Pre-sampled generation list

        Returns:
            SemanticEntropyResult
        """
        if not generations:
            return SemanticEntropyResult()

        texts = [g.text for g in generations]
        n = len(texts)

        # Step 2: Compute NxN entailment matrix
        entailment_matrix = self._checker.compute_entailment_matrix(texts)

        # Step 3: Cluster
        raw_clusters = cluster_by_equivalence(
            entailment_matrix, threshold=self._equiv_threshold
        )

        # Step 4: Compute semantic entropy
        se, semantic_clusters = compute_semantic_entropy(
            raw_clusters, generations,
            use_probability_weighting=self._use_prob_weighting,
        )

        # Find largest cluster and its representative answer
        majority_cluster = max(semantic_clusters, key=lambda c: c.probability)
        majority_idx = majority_cluster.members[0]
        majority_answer = generations[majority_idx].text
        majority_prob = majority_cluster.probability

        return SemanticEntropyResult(
            semantic_entropy=se,
            n_clusters=len(semantic_clusters),
            n_samples=n,
            clusters=semantic_clusters,
            generations=generations,
            entailment_matrix=entailment_matrix,
            majority_answer=majority_answer,
            majority_cluster_prob=majority_prob,
            is_uncertain=se > self._uncertainty_threshold,
        )
