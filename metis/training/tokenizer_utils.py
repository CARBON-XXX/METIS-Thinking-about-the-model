"""
METIS Tokenizer Utilities — Special Token Registration

Ensures all METIS cognitive tags are registered as Special Tokens in the
tokenizer vocabulary, preventing them from being split into sub-tokens.

Without this, <thinking> might tokenize as ['<', 'think', 'ing', '>'] (4 tokens)
instead of ['<thinking>'] (1 token), causing:
  1. SFT learns wrong token distribution
  2. DPO π_ref assigns near-zero probability → KL explosion
  3. Generation never produces the tag as a single unit

Usage:
    from metis.training.tokenizer_utils import register_metis_special_tokens
    tokenizer, model = register_metis_special_tokens(tokenizer, model)
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# METIS Special Tokens — canonical list
# ─────────────────────────────────────────────────────

METIS_SPECIAL_TOKENS: List[str] = [
    # Thinking block delimiters
    "<thinking>",
    "</thinking>",
    # Cognitive state markers
    "[COGNITIVE_STATE: FAST]",
    "[COGNITIVE_STATE: NORMAL]",
    "[COGNITIVE_STATE: DEEP]",
    # Entropy markers
    "[ENTROPY: LOW]",
    "[ENTROPY: MEDIUM]",
    "[ENTROPY: HIGH]",
    # Self-critique markers
    "[SELF-CRITIQUE:",
    # Final answer marker (used by generative eval parser)
    "FINAL ANSWER:",
]


def register_metis_special_tokens(
    tokenizer: Any,
    model: Optional[Any] = None,
) -> Tuple[Any, Optional[Any]]:
    """Register METIS cognitive tags as special tokens and resize model embeddings.

    Args:
        tokenizer: HuggingFace tokenizer instance
        model: Optional model to resize embeddings for

    Returns:
        (tokenizer, model) — both potentially modified in-place
    """
    # Filter tokens that are NOT already in the vocabulary
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in METIS_SPECIAL_TOKENS if t not in existing_vocab]

    if not new_tokens:
        logger.info(f"[Tokenizer] All {len(METIS_SPECIAL_TOKENS)} METIS tokens already in vocab")
        # Still resize model if needed (fresh model reload with already-augmented tokenizer)
        if model is not None:
            vocab_size = len(tokenizer)
            embed_size = model.get_input_embeddings().weight.shape[0]
            if embed_size != vocab_size:
                model.resize_token_embeddings(vocab_size)
                logger.info(f"[Tokenizer] Resized model embeddings {embed_size} → {vocab_size}")
        return tokenizer, model

    # Add as additional_special_tokens (preserved during encoding, never split)
    n_added = tokenizer.add_special_tokens({
        "additional_special_tokens": tokenizer.additional_special_tokens + new_tokens
    })

    logger.info(
        f"[Tokenizer] Added {n_added} METIS special tokens to vocab "
        f"(total vocab: {len(tokenizer)})"
    )

    # Resize model embeddings to match new vocab size
    if model is not None and n_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(
            f"[Tokenizer] Resized model embeddings → {len(tokenizer)} tokens"
        )

    return tokenizer, model


def verify_metis_tokens(tokenizer: Any) -> bool:
    """Verify all METIS special tokens are properly registered.

    Returns True if all tokens encode as single token IDs.
    Logs warnings for any that don't.
    """
    all_ok = True
    for token in METIS_SPECIAL_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            logger.warning(
                f"[Tokenizer] METIS token '{token}' encodes as {len(ids)} tokens "
                f"(expected 1): {ids}"
            )
            all_ok = False

    if all_ok:
        logger.info(f"[Tokenizer] ✓ All {len(METIS_SPECIAL_TOKENS)} METIS tokens verified (single-token)")
    return all_ok
