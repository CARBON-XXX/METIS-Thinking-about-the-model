"""
METIS Inference Pipeline
Cognitive-aware inference pipeline — signal consumer

Usage:
    from metis import Metis, MetisInference

    metis = Metis.attach(model, tokenizer)
    engine = MetisInference(metis)
    result = engine.generate("What is the capital of France?")
"""
from __future__ import annotations

import re
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metis import Metis

# ── Try Rust native accelerator for repetition detection ──
try:
    from metis_native import detect_repetition_hybrid as _native_rep_detect
    _HAS_NATIVE_REP = True
except ImportError:
    _HAS_NATIVE_REP = False

from .core.types import (
    Decision,
    EpistemicState,
    BoundaryAction,
    CoTStrategy,
    CognitiveSignal,
    InferenceResult,
    MetaJudgment,
    SemanticEntropyResult,
)
from .cognitive.cot import CoTManager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Reasoning Scaffold Templates
# ═══════════════════════════════════════════════════════════
# Problem: 1.5B models treat <thinking> as "answer area 2" — they dump
# knowledge instead of reasoning. Fix: inject structured scaffolds that
# force meta-cognitive operations BEFORE content generation.
#
# Design principles:
#   1. Echo user input → forces model to re-examine the question
#   2. Strategy-specific prompt → steers toward reasoning, not recitation
#   3. Short (< 30 tokens) → doesn't waste thinking budget
#   4. Ends mid-sentence → model must CONTINUE the reasoning, not start fresh

def _has_cjk(text: str) -> bool:
    """Detect if text contains CJK characters (Chinese/Japanese/Korean)."""
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF      # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF    # CJK Extension A
            or 0x3000 <= cp <= 0x303F    # CJK Punctuation
            or 0x3040 <= cp <= 0x30FF):  # Hiragana + Katakana
            return True
    return False


def _build_reasoning_scaffold(
    strategy: CoTStrategy,
    user_input: str,
    context: str = "",
) -> str:
    """
    Build a language-adaptive reasoning scaffold for CoT injection.

    Key design (v2 — "按头小分队"):
      1. Same language as user input (Chinese scaffold for CJK, English otherwise)
      2. Echo FULL input in quotes (critical for catching trailing typos)
      3. End with a strong auto-regressive hook ("我注意到" / "I notice")
         that FORCES the model to identify something before knowledge dumping
      4. Ultra-short (< 15 tokens) — don't waste thinking budget

    Args:
        strategy: CoT strategy (determines scaffold type)
        user_input: Original user prompt (echoed back for re-examination)
        context: Optional additional context (e.g., what was generated so far)

    Returns:
        Scaffold string to inject after <thinking> tag
    """
    # Echo full input up to 120 chars (longer than before — must catch trailing chars)
    preview = user_input[:120].strip()
    if len(user_input) > 120:
        preview += "..."

    use_chinese = _has_cjk(user_input)

    if use_chinese:
        # ── Chinese scaffolds ──
        # "我注意到" = "I notice that" — model MUST say what it noticed
        if strategy == CoTStrategy.CLARIFICATION:
            return f'仔细审题："{preview}"\n这句话有个问题：'
        elif strategy == CoTStrategy.DECOMPOSITION:
            return f'审题："{preview}"\n这个问题需要分步骤，首先'
        elif strategy == CoTStrategy.REFLECTION:
            return f'等等，重新看："{preview}"\n之前的回答可能有误，因为'
        else:
            # STANDARD — strongest general-purpose hook
            return f'仔细审题："{preview}"\n我注意到'
    else:
        # ── English scaffolds ──
        if strategy == CoTStrategy.CLARIFICATION:
            return f'Input: "{preview}"\nI notice a potential issue: '
        elif strategy == CoTStrategy.DECOMPOSITION:
            return f'Input: "{preview}"\nBreaking this down, first '
        elif strategy == CoTStrategy.REFLECTION:
            return f'Wait, re-reading: "{preview}"\nMy previous response may be wrong because '
        else:
            return f'Input: "{preview}"\nI notice '


class MetisInference:
    """
    METIS Cognitive-aware Inference Pipeline

    Dual-System Architecture:
        System 1 (fast): Standard token-by-token generation
        System 2 (deep): Post-generation Kuhn et al. semantic entropy verification

    Boundary Guard Actions:
        GENERATE → Normal output
        HEDGE    → Annotate uncertainty
        SEEK     → External retrieval callback
        REFUSE   → Knowledge boundary rejection
    """

    def __init__(
        self,
        metis: Metis,
        # System 2 trigger conditions
        system2_deep_ratio: float = 0.15,
        system2_uncertainty_score: float = 2.0,  # 50% of CUSUM_HEDGE_H=4.0
        # Boundary behavior
        hedge_prefix: str = "",
        hedge_suffix: str = "",  # Hedge is metadata-only; text injection removed to preserve natural output
        refuse_message: str = "I apologize, but this question is beyond my reliable knowledge base. I cannot provide a certain answer.",
        # REFUSE strategy
        refuse_grace_period: int = 8,
        refuse_consecutive_threshold: int = 3,
        # Self-correction budget control
        max_correction_tokens: int = 96,
        # Maximum thinking tokens before forced closure
        max_thinking_tokens: int = 128,
        # Callbacks
        on_seek: Optional[Callable[[str, str], Optional[str]]] = None,
        on_token: Optional[Callable[[str, 'CognitiveSignal'], None]] = None,
    ):
        """
        Args:
            metis: Metis instance (must be attached to model+tokenizer)
            system2_deep_ratio: DEEP ratio above this triggers System 2 verification
            system2_uncertainty_score: Accumulated uncertainty above this triggers System 2
            hedge_prefix: Uncertainty prefix text
            hedge_suffix: Uncertainty suffix text
            refuse_message: Text when refusing to answer
            refuse_grace_period: Within first N tokens, REFUSE triggers immediately;
                after that, requires consecutive occurrences, else downgraded to HEDGE.
                Reason: entropy spikes when model corrects false premises, not ignorance.
            refuse_consecutive_threshold: Consecutive REFUSE count needed after grace period
            on_seek: External retrieval callback fn(query, context) -> Optional[extra_info]
            on_token: Streaming callback fn(token_text, signal) -- called per token
        """
        self._metis = metis
        self._system2_deep_ratio = system2_deep_ratio
        self._system2_uncertainty_score = system2_uncertainty_score
        self._hedge_prefix = hedge_prefix
        self._hedge_suffix = hedge_suffix
        self._refuse_message = refuse_message
        self._refuse_grace_period = refuse_grace_period
        self._refuse_consecutive = refuse_consecutive_threshold
        self._max_correction_tokens = max_correction_tokens
        self._max_thinking_tokens = max_thinking_tokens
        self._cot_manager = CoTManager()
        self._on_seek = on_seek
        self._on_token = on_token

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.9,
        enable_system2: bool = True,
        se_n_samples: Optional[int] = None,
        chat_template: bool = True,
        use_thinking_protocol: bool = False,
    ) -> InferenceResult:
        """
        METIS-aware generation.
        
        Args:
            use_thinking_protocol: If True, forces "think before output" mode.
                                   Appends <thinking> after prompt and guides
                                   the model to generate a reasoning process.
        """
        model = self._metis.model
        tokenizer = self._metis.tokenizer

        if model is None or tokenizer is None:
            raise ValueError(
                "model and tokenizer required. "
                "Use Metis.attach(model, tokenizer) first."
            )

        start_time = time.perf_counter()

        # Build input
        if chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            
            # ─── Magic Mod: Thinking Protocol System Prompt ───
            # Anti-performative: explicitly forbid knowledge dumping in thinking.
            # Force the model to do meta-cognitive work (intent analysis, error
            # detection, decomposition) instead of "performing" reasoning.
            if use_thinking_protocol:
                system_content = (
                    "You are a careful reasoning AI. You think inside <thinking>...</thinking> tags before answering.\n"
                    "CRITICAL RULES for your thinking:\n"
                    "1. FIRST analyze what the user is really asking. Check for typos, ambiguity, or missing context.\n"
                    "2. NEVER list facts or bullet points inside thinking. That belongs in the answer.\n"
                    "3. Use thinking ONLY for: questioning assumptions, detecting errors, planning your response structure.\n"
                    "4. Write in natural flowing narrative, not numbered lists.\n"
                    "5. Keep thinking SHORT (2-4 sentences). Then give a thorough answer outside the tags."
                )
                messages.insert(0, {"role": "system", "content": system_content})

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
            
        # Thinking Protocol injection (Force start) + reasoning scaffold
        if use_thinking_protocol:
            scaffold = _build_reasoning_scaffold(CoTStrategy.STANDARD, prompt)
            text += f"<thinking>\n{scaffold}"

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        # Save prompt IDs for KV cache regeneration if needed
        prompt_ids = input_ids
        past_key_values = None

        # Start METIS session
        self._metis.start_session(prompt)
        self._cot_manager.reset()

        generated_tokens: List[int] = []
        vis_buffer: List[Tuple[str, CognitiveSignal]] = []
        signals: List[CognitiveSignal] = []
        boundary_interventions = 0
        was_refused = False
        consecutive_refuse = 0
        cot_injected = False
        cot_inject_token_idx = -1  # Token index where CoT was injected
        cot_strategies_used: List[CoTStrategy] = []
        seek_results: List[str] = []
        
        # Thinking Protocol state tracking
        is_thinking = use_thinking_protocol
        thinking_start_step = 0 if is_thinking else -1
        min_thinking_tokens = 64  # Minimum thinking length (Anti-Lazy)
        dynamic_thinking_budget = self._max_thinking_tokens  # L2: overridden at injection time

        # L1: Recitation Detector — track entropy/decisions inside thinking block
        # Real reasoning = exploratory (mid-high entropy, NORMAL/DEEP mode)
        # Recitation  = highly predictable (low entropy, FAST mode)
        thinking_entropies: List[float] = []
        thinking_decisions: List[Decision] = []
        _RECITATION_WINDOW = 8       # Sliding window size
        _RECITATION_ENTROPY_CEIL = 0.3  # bits — below this = reciting, not reasoning
        _RECITATION_FAST_RATIO = 0.75   # FAST ratio above this = not thinking
        _RECITATION_MIN_TOKENS = 16     # Don't check until scaffold has had effect

        # Repetition detection state
        repetition_events = 0
        thinking_failed = False    # Set True when thinking was closed due to repetition
        _REP_CHECK_INTERVAL = 5    # Check every N steps (saves compute + reduces false positives)
        _REP_CHECK_START = 40      # Start after enough tokens for meaningful detection
        _REP_FORCE_STOP = 3        # Force stop after N repetition events

        # Deferred CoT injection state: wait for sentence boundary
        cot_pending = False
        cot_pending_strategy: Optional[CoTStrategy] = None
        cot_pending_since = -1
        _COT_DEFER_MAX = 30  # Max tokens to wait for sentence boundary
        _SENTENCE_BREAKS = frozenset('\u3002\uff01\uff1f\n.!?\uff1a:')

        # If thinking protocol enabled, write <thinking> tag to generated_tokens and notify callback
        if use_thinking_protocol:
            think_open = "<thinking>\n"
            think_open_ids = tokenizer.encode(think_open, add_special_tokens=False)
            for tid in think_open_ids:
                generated_tokens.append(tid)
            if self._on_token is not None:
                open_signal = CognitiveSignal(decision=Decision.DEEP, introspection="[Thinking Start]")
                self._on_token(think_open, open_signal)

        # --- Token-by-token generation + METIS real-time monitoring ---
        for step in range(max_tokens):
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # METIS cognitive signal
            signal = self._metis.step(logits)
            signals.append(signal)

            # --- G3: Dynamic Thinking Trigger (deferred to sentence boundary) ---
            # CUSUM detects sustained difficulty, but injection waits for a
            # natural sentence break so the model enters thinking from a
            # coherent context — not mid-sentence.
            self._cot_manager.observe(signal)

            if not cot_pending and self._cot_manager.should_inject() and not is_thinking:
                cot_pending = True
                cot_pending_strategy = self._cot_manager.select_strategy(signal)
                cot_pending_since = step
                logger.info(
                    f"[METIS] CoT CUSUM triggered at step {step}, "
                    f"deferring to sentence boundary"
                )

            # Check if deferred CoT should fire now (sentence boundary or timeout)
            if cot_pending and not is_thinking:
                # Check last generated token for sentence-ending punctuation
                last_char = ''
                if generated_tokens:
                    last_text = tokenizer.decode(generated_tokens[-1:], skip_special_tokens=True)
                    last_char = last_text[-1] if last_text else ''
                deferred_too_long = (step - cot_pending_since) >= _COT_DEFER_MAX

                if last_char in _SENTENCE_BREAKS or deferred_too_long:
                    strategy = cot_pending_strategy
                    cot_pending = False
                    cot_pending_strategy = None

                    if deferred_too_long:
                        logger.info(f"[METIS] CoT defer timeout at step {step}")

                    # Check for existing repetition before starting thinking
                    max_window = min(len(generated_tokens) // 2, 256)
                    rep_len, _ = self._detect_repetition_hybrid(
                        generated_tokens, max_window
                    )
                    if rep_len > 0:
                        logger.info(
                            f"[METIS] Pre-CoT Repetition Trim: removed {rep_len} tokens"
                        )
                        generated_tokens = generated_tokens[:-rep_len]
                        vis_buffer.clear()

                        if generated_tokens:
                            gen_ids = torch.tensor([generated_tokens], device=model.device)
                            full_input = torch.cat([prompt_ids, gen_ids], dim=1)
                        else:
                            full_input = prompt_ids

                        with torch.no_grad():
                            clean_out = model(
                                input_ids=full_input,
                                use_cache=True,
                                return_dict=True,
                            )
                        past_key_values = clean_out.past_key_values
                        logits = clean_out.logits[:, -1, :]

                    # Open <thinking> tag + reasoning scaffold at sentence boundary
                    # Scaffold forces meta-cognitive analysis before knowledge dumping
                    scaffold = _build_reasoning_scaffold(strategy, prompt)
                    think_open = f"\n<thinking>\n{scaffold}"
                    think_open_ids = tokenizer.encode(think_open, add_special_tokens=False)
                    for tid in think_open_ids:
                        generated_tokens.append(tid)
                    if self._on_token is not None:
                        open_signal = CognitiveSignal(
                            decision=Decision.DEEP,
                            introspection=f"[Thinking: {strategy.value}]",
                        )
                        self._on_token(think_open, open_signal)

                    open_input = torch.tensor([think_open_ids], device=model.device)
                    open_out = model(
                        input_ids=open_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = open_out.past_key_values
                    logits = open_out.logits[:, -1, :]

                    is_thinking = True
                    thinking_start_step = step
                    self._cot_manager.record_injection(strategy)
                    cot_injected = True
                    # Record token position BEFORE the <thinking> tag
                    # so we can cleanly split answer from thinking later
                    cot_inject_token_idx = len(generated_tokens) - len(think_open_ids)
                    cot_strategies_used.append(strategy)

                    # L2: Dynamic thinking budget based on CUSUM magnitude
                    # Higher CUSUM at trigger = deeper confusion = more thinking time
                    # Note: CUSUM was just reset by record_injection(), use pre-reset stats
                    _THINK_BUDGET_MIN = 24   # 1-2 sentences of analysis
                    _THINK_BUDGET_MAX = 128  # No encyclopedias allowed
                    _cusum_at_trigger = signal.z_score  # Use z-score as proxy (CUSUM already reset)
                    _cusum_ratio = min(max(_cusum_at_trigger / 2.0, 0.0), 2.0)
                    dynamic_thinking_budget = int(
                        _THINK_BUDGET_MIN + (_THINK_BUDGET_MAX - _THINK_BUDGET_MIN) * _cusum_ratio
                    )
                    # Reset thinking trackers for this new block
                    thinking_entropies.clear()
                    thinking_decisions.clear()

                    logger.info(
                        f"[METIS] Thinking injected at step {step}: "
                        f"reason={strategy.value}, z={signal.z_score:.2f}, "
                        f"budget={dynamic_thinking_budget} tokens"
                    )

            # --- Boundary guard action execution ---
            if signal.boundary_action == BoundaryAction.REFUSE:
                if step < self._refuse_grace_period:
                    # First N tokens: model hasn't started answering yet, refuse immediately
                    was_refused = True
                    logger.info(
                        f"[METIS] Boundary REFUSE at step {step} "
                        f"(early, within grace period): {signal.introspection}"
                    )
                    break
                else:
                    # After content has been generated: may be model correcting false premise,
                    # entropy spike doesn't mean "doesn't know". Accumulate consecutive REFUSE count.
                    consecutive_refuse += 1
                    if consecutive_refuse >= self._refuse_consecutive:
                        was_refused = True
                        logger.info(
                            f"[METIS] Boundary REFUSE at step {step} "
                            f"({consecutive_refuse} consecutive): "
                            f"{signal.introspection}"
                        )
                        break
                    else:
                        # Downgrade to HEDGE, continue generating
                        boundary_interventions += 1
                        logger.info(
                            f"[METIS] REFUSE downgraded to HEDGE at step {step} "
                            f"(consecutive={consecutive_refuse}/{self._refuse_consecutive})"
                        )
            else:
                consecutive_refuse = 0  # Non-REFUSE signal resets counter

            if signal.boundary_action == BoundaryAction.SEEK:
                boundary_interventions += 1
                if self._on_seek is not None:
                    context = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    extra = self._on_seek(prompt, context)
                    if extra:
                        seek_results.append(extra)
                logger.info(
                    f"[METIS] Boundary SEEK at step {step}: "
                    f"{signal.introspection}"
                )

            if signal.boundary_action == BoundaryAction.HEDGE:
                boundary_interventions += 1

            # --- G5: Repetition Loop Detection & Intervention ---
            # Detects when model is stuck repeating the same token sequence.
            # LaTeX/math tokens have low entropy even during loops, so
            # token-level signals alone cannot catch this.
            if step >= _REP_CHECK_START and step % _REP_CHECK_INTERVAL == 0:
                # Dense scan: check all possible window sizes up to N/2
                # Limit max_window to 256 to allow catching paragraph-level loops
                max_window = min(len(generated_tokens) // 2, 256)
                
                # Hybrid detection: 
                # - Short patterns: Strict positional matching (avoids false positives on math steps)
                # - Long patterns: Loose Jaccard matching (catches semantic loops/rephrasing)
                rep_len, match_score = self._detect_repetition_hybrid(
                    generated_tokens, max_window
                )
                
                if rep_len > 0:
                    repetition_events += 1
                    # Ensure log is visible
                    print(
                        f"\n[METIS] DETECTED REPETITION len={rep_len} "
                        f"score={match_score:.2f} event={repetition_events}\n"
                    )
                    logger.warning(
                        f"[METIS] Repetition detected at step {step}: "
                        f"len={rep_len}, score={match_score:.2f}, "
                        f"event #{repetition_events}"
                    )

                    if repetition_events >= _REP_FORCE_STOP:
                        # Escalation 3: Force stop — model is hopelessly stuck
                        logger.warning(
                            f"[METIS] Force stopping: {repetition_events} "
                            f"repetition events"
                        )
                        # Trim the repeated tail
                        generated_tokens = generated_tokens[:-rep_len]
                        # Close thinking block before break so _split_thinking
                        # can separate thinking content from the answer.
                        if is_thinking:
                            close_tag = "\n</thinking>\n"
                            close_ids = tokenizer.encode(
                                close_tag, add_special_tokens=False
                            )
                            for tid in close_ids:
                                generated_tokens.append(tid)
                            if self._on_token is not None:
                                self._on_token(
                                    close_tag,
                                    CognitiveSignal(
                                        decision=Decision.DEEP,
                                        introspection="[Thinking End: force-stop]",
                                    ),
                                )
                            is_thinking = False
                        break

                    if thinking_failed and not is_thinking:
                        # Case 3: Thinking already failed, model STILL looping
                        # outside thinking. No point continuing — force stop now.
                        logger.warning(
                            "[METIS] Post-thinking repetition -> "
                            "force stopping (thinking already failed)"
                        )
                        generated_tokens = generated_tokens[:-rep_len]
                        break

                    elif not is_thinking and use_thinking_protocol:
                        # Escalation 1: Trigger thinking (think=ON only)
                        # If model repeats itself, it's stuck. Stop and think.
                        logger.info(
                            "[METIS] Repetition -> triggering thinking "
                            "to break loop"
                        )
                        # Trim repeated tail first
                        generated_tokens = generated_tokens[:-rep_len]
                        vis_buffer.clear()

                        # Regenerate KV cache to remove dirty context
                        if generated_tokens:
                            gen_ids = torch.tensor([generated_tokens], device=model.device)
                            full_input = torch.cat([prompt_ids, gen_ids], dim=1)
                        else:
                            full_input = prompt_ids

                        with torch.no_grad():
                            clean_out = model(
                                input_ids=full_input,
                                use_cache=True,
                                return_dict=True,
                            )
                        past_key_values = clean_out.past_key_values
                        logits = clean_out.logits[:, -1, :]

                        # Inject <thinking> tag + REFLECTION scaffold (model is stuck)
                        scaffold = _build_reasoning_scaffold(CoTStrategy.REFLECTION, prompt)
                        think_open = f"\n<thinking>\n{scaffold}"
                        think_open_ids = tokenizer.encode(
                            think_open, add_special_tokens=False
                        )
                        for tid in think_open_ids:
                            generated_tokens.append(tid)
                        if self._on_token is not None:
                            self._on_token(
                                think_open,
                                CognitiveSignal(
                                    decision=Decision.DEEP,
                                    introspection="[Thinking: repetition-break]",
                                ),
                            )
                        open_input = torch.tensor(
                            [think_open_ids], device=model.device
                        )
                        open_out = model(
                            input_ids=open_input,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                        past_key_values = open_out.past_key_values
                        logits = open_out.logits[:, -1, :]
                        is_thinking = True
                        thinking_start_step = step
                        cot_injected = True
                        cot_inject_token_idx = len(generated_tokens) - len(think_open_ids)
                        # L2: Repetition-triggered thinking gets modest budget
                        dynamic_thinking_budget = 64
                        thinking_entropies.clear()
                        thinking_decisions.clear()

                    elif not is_thinking:
                        # think=OFF: trim repeated tail + rebuild KV cache
                        # Don't inject thinking — just give model fresh context
                        logger.info(
                            f"[METIS] Repetition (think=OFF) -> "
                            f"trim {rep_len} tokens + rebuild KV"
                        )
                        generated_tokens = generated_tokens[:-rep_len]
                        vis_buffer.clear()

                        if generated_tokens:
                            gen_ids = torch.tensor(
                                [generated_tokens], device=model.device
                            )
                            full_input = torch.cat(
                                [prompt_ids, gen_ids], dim=1
                            )
                        else:
                            full_input = prompt_ids
                        with torch.no_grad():
                            clean_out = model(
                                input_ids=full_input,
                                use_cache=True,
                                return_dict=True,
                            )
                        past_key_values = clean_out.past_key_values
                        logits = clean_out.logits[:, -1, :]

                    else:
                        # Escalation 2: Repetition INSIDE thinking block.
                        # Force-close thinking and let model answer
                        # with whatever reasoning it has so far.
                        thinking_failed = True
                        logger.info(
                            "[METIS] Repetition in thinking -> "
                            "force-closing to produce answer"
                        )
                        generated_tokens = generated_tokens[:-rep_len]
                        vis_buffer.clear()

                        # Rebuild KV cache without repeated tail
                        if generated_tokens:
                            gen_ids = torch.tensor(
                                [generated_tokens], device=model.device
                            )
                            full_input = torch.cat(
                                [prompt_ids, gen_ids], dim=1
                            )
                        else:
                            full_input = prompt_ids
                        clean_out = model(
                            input_ids=full_input,
                            use_cache=True,
                            return_dict=True,
                        )
                        past_key_values = clean_out.past_key_values
                        logits = clean_out.logits[:, -1, :]

                        # Inject </thinking> to close the block
                        close_tag = "\n</thinking>\n"
                        close_tag_ids = tokenizer.encode(
                            close_tag, add_special_tokens=False
                        )
                        for tid in close_tag_ids:
                            generated_tokens.append(tid)
                        if self._on_token is not None:
                            self._on_token(
                                close_tag,
                                CognitiveSignal(
                                    decision=Decision.DEEP,
                                    introspection="[Thinking End: repetition]",
                                ),
                            )
                        # Feed closing tag into model
                        close_input = torch.tensor(
                            [close_tag_ids], device=model.device
                        )
                        close_out = model(
                            input_ids=close_input,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                        past_key_values = close_out.past_key_values
                        logits = close_out.logits[:, -1, :]
                        is_thinking = False

            # --- Token surprise: compute from RAW logits before sampling modifies them ---
            # -log2(p(token)) = prediction error in bits (neuroscience: surprise signal)
            with torch.no_grad():
                _log_probs = F.log_softmax(logits.float(), dim=-1)

            # --- Token sampling (cognitive-aware) ---
            next_token_id = self._cognitive_sample(
                logits, signal, temperature, top_p, generated_tokens
            )

            # Attach surprise to signal (after sampling, we know which token was chosen)
            _token_log_prob = _log_probs[0, next_token_id].item()
            signal.token_surprise = max(0.0, -_token_log_prob / 0.6931471805599453)  # nats→bits

            # Feed surprise back to boundary guard (1-step lag: affects NEXT step's CUSUM)
            self._metis.feed_surprise(signal.token_surprise)
            
            # --- L1: Recitation Detector + L2: Dynamic Budget ---
            # Two-layer defense against performative reasoning:
            #   L1: If model is reciting (low entropy + FAST mode) → force-close
            #   L2: Dynamic budget based on CUSUM magnitude at injection time
            if is_thinking:
                tokens_in_block = step - thinking_start_step

                # Track entropy/decisions inside thinking block
                thinking_entropies.append(signal.semantic_entropy)
                thinking_decisions.append(signal.decision)

                # L1: Recitation Detector — sliding window check
                # Skip first _RECITATION_MIN_TOKENS to let scaffold take effect
                should_truncate_recitation = False
                if (
                    tokens_in_block >= _RECITATION_MIN_TOKENS
                    and len(thinking_entropies) >= _RECITATION_WINDOW
                ):
                    window_h = thinking_entropies[-_RECITATION_WINDOW:]
                    window_d = thinking_decisions[-_RECITATION_WINDOW:]
                    avg_h = sum(window_h) / len(window_h)
                    fast_ratio = sum(
                        1 for d in window_d if d == Decision.FAST
                    ) / len(window_d)

                    if avg_h < _RECITATION_ENTROPY_CEIL and fast_ratio >= _RECITATION_FAST_RATIO:
                        should_truncate_recitation = True
                        logger.info(
                            f"[METIS] Recitation detected at step {step}: "
                            f"avg_H={avg_h:.3f} < {_RECITATION_ENTROPY_CEIL}, "
                            f"FAST={fast_ratio:.0%} >= {_RECITATION_FAST_RATIO:.0%} "
                            f"→ force-closing thinking (you're reciting, not reasoning)"
                        )

                # L2: Dynamic budget exhaustion
                should_truncate_budget = tokens_in_block >= dynamic_thinking_budget

                if should_truncate_budget and not should_truncate_recitation:
                    logger.info(
                        f"[METIS] Thinking budget exhausted "
                        f"({tokens_in_block}/{dynamic_thinking_budget}), "
                        f"force-closing thinking block"
                    )

                if should_truncate_recitation or should_truncate_budget:
                    # Inject </thinking>\n to close the block
                    close_tag = "\n</thinking>\n"
                    close_tag_ids = tokenizer.encode(
                        close_tag, add_special_tokens=False
                    )
                    for tid in close_tag_ids:
                        generated_tokens.append(tid)

                    # Notify streaming callback
                    if self._on_token is not None:
                        _reason = "recitation" if should_truncate_recitation else "budget"
                        close_signal = CognitiveSignal(
                            decision=Decision.DEEP,
                            introspection=f"[Thinking End: {_reason}]",
                        )
                        self._on_token(close_tag, close_signal)

                    # Feed closing tag into model so it continues with answer
                    close_input = torch.tensor(
                        [close_tag_ids], device=model.device
                    )
                    close_out = model(
                        input_ids=close_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = close_out.past_key_values
                    logits = close_out.logits[:, -1, :]

                    is_thinking = False
                    # Flush buffered thinking tokens
                    if vis_buffer:
                        for t, s in vis_buffer:
                            if self._on_token:
                                self._on_token(t, s)
                        vis_buffer.clear()

                    # Re-sample from post-closing logits for the answer
                    next_token_id = self._cognitive_sample(
                        logits, signal, temperature, top_p, generated_tokens
                    )
                    # Skip Anti-Lazy check since we just closed
                    generated_tokens.append(next_token_id)
                    if self._on_token is not None and next_token_id != tokenizer.eos_token_id:
                        token_text = tokenizer.decode(
                            [next_token_id], skip_special_tokens=True
                        )
                        self._on_token(token_text, signal)
                    if next_token_id == tokenizer.eos_token_id:
                        break
                    input_ids = torch.tensor(
                        [[next_token_id]], device=model.device
                    )
                    continue

            # --- Anti-Lazy Thinking: enforce minimum reasoning depth ---
            # If model tries to close <thinking> too early, suppress the closing
            # tag and let the model continue reasoning naturally.
            # NO injected text — just token suppression + re-sample.
            if is_thinking:
                check_window = 12
                recent_ids = generated_tokens[-check_window:] + [next_token_id]
                recent_text = tokenizer.decode(recent_ids, skip_special_tokens=True)

                if "</thinking>" in recent_text:
                    tokens_in_block = step - thinking_start_step
                    if tokens_in_block < min_thinking_tokens:
                        logger.info(
                            f"[METIS] Anti-Lazy: Suppressed premature </thinking> "
                            f"at {tokens_in_block}/{min_thinking_tokens} tokens"
                        )

                        # 1. Rollback: Remove closing tag partial tokens
                        _closing_partials = [
                            "</thinking>", "</thinking", "</thinkin", "</thinki",
                            "</think", "</thin", "</thi", "</th", "</t", "</",
                        ]
                        while generated_tokens:
                            tail = tokenizer.decode(generated_tokens[-6:], skip_special_tokens=True)
                            if any(tail.rstrip().endswith(p) for p in _closing_partials):
                                generated_tokens.pop()
                            else:
                                break

                        # 2. Clear visualizer buffer (hide rejected tag)
                        vis_buffer.clear()

                        # 3. Rebuild KV cache after rollback
                        # Without this, past_key_values still contains attention
                        # memory for the deleted tokens (ghost context), causing
                        # incoherent continuation.
                        if generated_tokens:
                            gen_ids = torch.tensor(
                                [generated_tokens], device=model.device
                            )
                            full_input = torch.cat(
                                [prompt_ids, gen_ids], dim=1
                            )
                        else:
                            full_input = prompt_ids
                        clean_out = model(
                            input_ids=full_input,
                            use_cache=True,
                            return_dict=True,
                        )
                        past_key_values = clean_out.past_key_values
                        logits = clean_out.logits[:, -1, :]

                        # 4. Re-sample with </thinking> suppressed
                        close_tag_ids = tokenizer.encode("</thinking>", add_special_tokens=False)
                        if close_tag_ids:
                            logits[0, close_tag_ids[0]] = float('-inf')

                        next_token_id = self._cognitive_sample(
                            logits, signal, temperature, top_p, generated_tokens
                        )

                        logger.info(
                            f"[METIS] Anti-Lazy: Re-sampled with clean KV cache"
                        )

                        # Fall through to normal append below
                    else:
                        # Thinking block closed naturally
                        is_thinking = False
                        if vis_buffer:
                            for t, s in vis_buffer:
                                if self._on_token:
                                    self._on_token(t, s)
                            vis_buffer.clear()
                        logger.info(
                            f"[METIS] Thinking block closed naturally "
                            f"at {tokens_in_block} tokens"
                        )

            generated_tokens.append(next_token_id)

            # ─── Streaming callback (Buffered) ───
            if self._on_token is not None and next_token_id != tokenizer.eos_token_id:
                token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
                
                try:
                    if is_thinking:
                        # Thinking mode: buffer output until we're sure it's not a premature closing tag
                        vis_buffer.append((token_text, signal))
                        # Keep buffer at 20 chars (enough to contain </thinking>)
                        # Flush safe head when buffer exceeds limit
                        full_buf = "".join(x[0] for x in vis_buffer)
                        while len(full_buf) > 20:
                            t, s = vis_buffer.pop(0)
                            self._on_token(t, s)
                            full_buf = "".join(x[0] for x in vis_buffer)
                    else:
                        # Non-thinking mode (or just ended): flush buffer first
                        while vis_buffer:
                            t, s = vis_buffer.pop(0)
                            self._on_token(t, s)
                        self._on_token(token_text, signal)
                except Exception:
                    pass

            if next_token_id == tokenizer.eos_token_id:
                break

            input_ids = torch.tensor(
                [[next_token_id]], device=model.device
            )

        # Ensure thinking block is closed if still open (max_tokens exhausted / EOS)
        if is_thinking:
            close_tag = "\n</thinking>\n"
            close_ids = tokenizer.encode(close_tag, add_special_tokens=False)
            for tid in close_ids:
                generated_tokens.append(tid)
            if self._on_token is not None:
                self._on_token(
                    close_tag,
                    CognitiveSignal(
                        decision=Decision.DEEP,
                        introspection="[Thinking End: budget]",
                    ),
                )

        # End METIS session
        self._metis.end_session()

        # --- Statistics ---
        n_signals = len(signals)
        n_fast = sum(1 for s in signals if s.decision == Decision.FAST)
        n_deep = sum(1 for s in signals if s.decision == Decision.DEEP)
        system1_ratio = n_fast / max(n_signals, 1)
        system2_ratio = n_deep / max(n_signals, 1)
        avg_entropy = (
            sum(s.semantic_entropy for s in signals) / n_signals
            if n_signals > 0 else 0.0
        )
        avg_confidence = (
            sum(s.confidence for s in signals) / n_signals
            if n_signals > 0 else 0.0
        )
        uncertainty = self._metis.get_uncertainty_score()

        # Final cognitive state
        final_signal = signals[-1] if signals else CognitiveSignal()

        # --- Build raw answer ---
        if was_refused:
            raw_text = self._refuse_message
            thinking_text = ""
        elif cot_injected and cot_inject_token_idx > 0:
            # CoT was dynamically injected mid-generation.
            # Answer = tokens BEFORE injection point (the partial answer).
            # Everything after injection = thinking process (discard from output).
            answer_tokens = generated_tokens[:cot_inject_token_idx]
            thinking_tokens = generated_tokens[cot_inject_token_idx:]
            raw_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            thinking_text = tokenizer.decode(thinking_tokens, skip_special_tokens=True)
            # Clean any stray tags from the answer portion
            raw_text = re.sub(r'</?thinking[^>]*>', '', raw_text).strip()
        else:
            # think=ON from start, or no thinking: use regex splitting
            full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            raw_text, thinking_text = self._split_thinking(full_text)

        # --- System 2 verification (Kuhn et al. SE) ---
        se_result, system2_triggered, was_verified, raw_text = (
            self._run_system2_verification(
                enable_system2, was_refused, system2_ratio, uncertainty,
                prompt, se_n_samples, chat_template, raw_text,
            )
        )

        # --- Metacognitive introspection ---
        meta_judgment = self._metis.introspect(se_result)

        # --- G4: Hallucination self-correction ---
        hallucination_corrected, raw_text = self._run_hallucination_correction(
            was_refused, meta_judgment, prompt, max_tokens,
            chat_template, avg_confidence, raw_text, model, tokenizer,
        )

        # --- Boundary behavior: Hedge annotation ---
        was_hedged = False
        final_text = raw_text

        # Boundary intervention ratio: fraction of non-GENERATE tokens
        n_boundary = sum(
            1 for s in signals
            if s.boundary_action != BoundaryAction.GENERATE
        )
        boundary_ratio = n_boundary / max(n_signals, 1)

        # Hedge requires strong convergent evidence:
        #   1. CUSUM HEDGE event: strong standalone (boundary guard explicitly fired)
        #   2. SE uncertain + moderate CUSUM: two independent signals agree
        #   CUSUM accumulation alone does NOT trigger hedge — it only gates SE verification
        #   (handled upstream in _run_system2_verification).
        se_uncertain = se_result is not None and se_result.is_uncertain
        should_hedge = (
            not was_refused
            and (
                final_signal.boundary_action == BoundaryAction.HEDGE
                or (se_uncertain and uncertainty >= self._system2_uncertainty_score * 0.5)
            )
        )

        if should_hedge:
            was_hedged = True

        # Metacognitive regulation: supplement actions based on introspection
        if not was_hedged and not was_refused:
            regulation = self._metis.regulate(meta_judgment)
            if regulation["should_hedge"]:
                was_hedged = True

        # Apply text prefix/suffix only if explicitly provided (non-empty)
        if was_hedged and (self._hedge_prefix or self._hedge_suffix):
            final_text = self._hedge_prefix + raw_text + self._hedge_suffix

        # --- Introspection summary ---
        introspection_parts = self._build_introspection(
            was_refused=was_refused,
            was_verified=was_verified,
            was_hedged=was_hedged,
            cot_injected=cot_injected,
            hallucination_corrected=hallucination_corrected,
            final_signal=final_signal,
            se_result=se_result,
            meta_judgment=meta_judgment,
            seek_results=seek_results,
        )

        latency = (time.perf_counter() - start_time) * 1000

        return InferenceResult(
            text=final_text,
            thinking_text=thinking_text,
            tokens_generated=len(generated_tokens),
            latency_ms=latency,
            final_decision=final_signal.decision,
            final_epistemic_state=final_signal.epistemic_state,
            final_boundary_action=final_signal.boundary_action,
            avg_token_entropy=avg_entropy,
            avg_confidence=avg_confidence,
            uncertainty_score=uncertainty,
            semantic_entropy_result=se_result,
            system1_ratio=system1_ratio,
            system2_ratio=system2_ratio,
            system2_triggered=system2_triggered,
            boundary_interventions=boundary_interventions,
            was_hedged=was_hedged,
            was_refused=was_refused,
            was_verified=was_verified,
            introspection="; ".join(introspection_parts),
        )

    @staticmethod
    def _detect_repetition_hybrid(
        tokens: List[int], max_window: int
    ) -> Tuple[int, float]:
        """
        Hybrid repetition detection:
        - Short windows (<32): Positional Fuzzy Match (Threshold 0.9)
          Prevents false positives on valid math steps (x=y+1 vs x=y+2)
        - Long windows (>=32): Jaccard Similarity (Threshold 0.6)
          Catches semantic loops and rephrasing (paragraph level)
        
        Returns: (length, score)
        """
        # Rust native fast path (~10-50x speedup)
        if _HAS_NATIVE_REP:
            return _native_rep_detect(tokens, max_window)

        n = len(tokens)
        
        # 1. Check Long Semantic Loops (Jaccard)
        # Dense scan for Jaccard to catch arbitrary loop lengths
        # Step size 4 is granular enough for bag-of-words
        # Range: [32, max_window]
        # NOTE: min window=32. Shorter windows have extremely high false-positive
        # rate on CJK text due to shared functional characters (的/是/了/在).
        for w in range(32, max_window + 1, 4):
            if n < 2 * w:
                continue
            
            set_a = set(tokens[n - 2 * w : n - w])
            set_b = set(tokens[n - w : n])
            
            if not set_a or not set_b:
                continue
                
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            jaccard = intersection / union
            
            # Threshold 0.7: requires genuine token-set overlap, not just shared
            # functional characters. Previous 0.45 caused massive false positives
            # on normal Chinese mathematical reasoning text.
            if jaccard >= 0.7:
                logger.debug(f"[METIS] Jaccard match: w={w}, score={jaccard:.2f}")
                return w, jaccard

        # 2. Check Short Exact/Near-Exact Loops (Positional)
        # Dense scan downwards from 15 to 4 (very short exact loops)
        for w in range(min(max_window, 15), 3, -1):
            if n < 2 * w:
                continue
                
            matches = 0
            for i in range(w):
                if tokens[n - 2 * w + i] == tokens[n - w + i]:
                    matches += 1
            
            score = matches / w
            if score >= 0.9: # Strict for short windows
                return w, score
                
        return 0, 0.0

    @staticmethod
    def _split_thinking(text: str) -> tuple:
        """
        Separate <thinking>...</thinking> blocks from final answer.

        Returns:
            (answer_text, thinking_text)
            - answer_text: text with all thinking blocks removed, cleaned up
            - thinking_text: concatenated content of all thinking blocks
        """
        # Match all <thinking>...</thinking> blocks (greedy, DOTALL)
        pattern = re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL)
        thinking_parts = pattern.findall(text)
        thinking_text = "\n".join(t.strip() for t in thinking_parts if t.strip())

        # Remove thinking blocks from output
        answer = pattern.sub('', text)
        # Also remove orphaned opening/closing tags (incomplete blocks)
        answer = re.sub(r'</?thinking>', '', answer)
        # Remove incomplete tags: <thinking (no >) or </thinking (no >)
        # Complete tags already removed above, so only fragments remain
        answer = re.sub(r'</?thinking\b[^>]*', '', answer)
        # Clean up excess whitespace left behind
        answer = re.sub(r'\n{3,}', '\n\n', answer).strip()

        return answer, thinking_text

    # Adaptive repetition penalty constants
    _REP_PENALTY_BASE = 1.2    # Base penalty (mild, for confident tokens)
    _REP_PENALTY_MAX = 1.5     # Max penalty (strong, for uncertain/looping tokens)
    _REP_PENALTY_WINDOW = 128  # Only penalize tokens in last N generated

    def _cognitive_sample(
        self,
        logits: torch.Tensor,
        signal: CognitiveSignal,
        base_temperature: float,
        base_top_p: float,
        generated_tokens: Optional[List[int]] = None,
    ) -> int:
        """
        Cognitive-aware sampling — METIS core intervention point.

        Dynamically adjusts sampling strategy based on cognitive signals:
        - FAST (System 1): greedy → output best token directly when confident
        - NORMAL: user settings → standard sampling
        - DEEP (System 2): raise temp + expand top_p → increase exploration

        Also applies:
        - Entropy-aware logit modulation (contrastive-decoding-like sharpening)
        - Repetition penalty on recently generated tokens
        """
        # -- 1. Decision-driven adaptive temperature --
        # METIS autonomously controls sampling strategy.
        # The model's cognitive state determines exploration level,
        # not a user-provided base temperature.
        if signal.decision == Decision.FAST:
            # System 1: high certainty, greedy — no exploration needed
            temperature = 0.0
            top_p = 1.0
        elif signal.decision == Decision.DEEP:
            # System 2: genuine uncertainty, autonomous exploration
            # Floor 0.4 ensures meaningful sampling even when base_temp=0.
            # Scale with z_score: higher uncertainty → more exploration (cap 0.8).
            z_boost = min(max(signal.z_score - 1.0, 0.0) * 0.1, 0.4)
            temperature = max(0.4 + z_boost, base_temperature)
            top_p = min(base_top_p + 0.1, 0.95)
        else:
            # NORMAL: use base settings (greedy if base=0, else user's choice)
            temperature = base_temperature
            top_p = base_top_p

        # -- 2. Adaptive repetition penalty --
        # Penalty scales with cognitive state:
        #   - Confident (low z): mild penalty (1.2) — don't disrupt valid repetition
        #   - Uncertain (high z): strong penalty (up to 1.5) — prevent loops
        #   - Rising momentum: extra +0.05 (entropy accelerating → approaching chaos)
        if generated_tokens:
            recent = generated_tokens[-self._REP_PENALTY_WINDOW:]
            unique_tokens = set(recent)
            if unique_tokens:
                # Adaptive penalty: base + z-driven boost + momentum boost
                z_boost = 0.1 * min(max(signal.z_score, 0.0), 3.0)
                mom_boost = 0.05 if signal.entropy_momentum > 0 else 0.0
                penalty = min(
                    self._REP_PENALTY_BASE + z_boost + mom_boost,
                    self._REP_PENALTY_MAX,
                )
                token_ids = torch.tensor(list(unique_tokens), device=logits.device)
                scores = logits[..., token_ids]
                scores = torch.where(
                    scores > 0,
                    scores / penalty,
                    scores * penalty,
                )
                logits[..., token_ids] = scores

        # -- 3. Entropy-aware logit modulation --
        # When z-score high + confidence low: flat distribution, likely to sample unreliable tokens
        # Adaptive sharpening reduces random walk
        # SKIP at temperature=0: greedy decoding should be deterministic.
        # Sharpening can flip argmax on quantized models due to float precision.
        if temperature > 0 and signal.z_score > 1.0 and signal.confidence < 0.5:
            sharpness = 1.0 + 0.15 * min(signal.z_score - 1.0, 3.0)
            logits = logits * sharpness

        # -- 4. Sampling --
        if temperature <= 0:
            return logits.argmax(dim=-1).item()

        logits = logits / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                logits, descending=True, dim=-1
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            indices_to_remove = mask.scatter(
                -1, sorted_indices, mask
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    @torch.no_grad()
    def _generate_simple(
        self,
        model: nn.Module,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        chat_template: bool = True,
    ) -> Dict[str, Any]:
        """
        Lightweight greedy generation — for hallucination self-correction verification.

        Does not start METIS monitoring (avoids recursion), pure generation + basic stats.
        Returns: {"text": str, "avg_confidence": float}
        """
        if chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        # Save prompt IDs for KV cache regeneration if needed
        prompt_ids = input_ids
        past_key_values = None
        generated = []
        confidences = []

        for _ in range(max_tokens):
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            probs = torch.softmax(logits.float(), dim=-1)
            conf = probs.max().item()
            confidences.append(conf)

            next_id = logits.argmax(dim=-1).item()
            generated.append(next_id)

            if next_id == tokenizer.eos_token_id:
                break
            input_ids = torch.tensor([[next_id]], device=model.device)

        return {
            "text": tokenizer.decode(generated, skip_special_tokens=True),
            "avg_confidence": (
                sum(confidences) / len(confidences) if confidences else 0.0
            ),
        }

    def _run_system2_verification(
        self,
        enable_system2: bool,
        was_refused: bool,
        system2_ratio: float,
        uncertainty: float,
        prompt: str,
        se_n_samples: Optional[int],
        chat_template: bool,
        raw_text: str,
    ) -> Tuple[Optional[SemanticEntropyResult], bool, bool, str]:
        """
        System 2 semantic entropy verification (Kuhn et al.)

        Returns:
            (se_result, system2_triggered, was_verified, raw_text)
        """
        se_result: Optional[SemanticEntropyResult] = None
        system2_triggered = False
        was_verified = False

        should_verify = (
            enable_system2
            and not was_refused
            and (
                system2_ratio >= self._system2_deep_ratio
                or uncertainty >= self._system2_uncertainty_score
            )
        )

        if should_verify:
            system2_triggered = True
            logger.info(
                f"[METIS] System 2 triggered: "
                f"deep_ratio={system2_ratio:.2f}, "
                f"uncertainty={uncertainty:.2f}"
            )
            try:
                se_result = self._metis.evaluate_semantic_entropy(
                    prompt=prompt,
                    n_samples=se_n_samples,
                    chat_template=chat_template,
                )
                was_verified = True

                if not se_result.is_uncertain and se_result.majority_answer:
                    raw_text = se_result.majority_answer
                    logger.info(
                        f"[METIS] SE verified: "
                        f"SE={se_result.semantic_entropy:.2f}, "
                        f"clusters={se_result.n_clusters}, "
                        f"using majority answer"
                    )
                elif se_result.is_uncertain:
                    self._metis.record_se_gap(
                        query=prompt,
                        semantic_entropy=se_result.semantic_entropy,
                        n_clusters=se_result.n_clusters,
                        n_samples=se_result.n_samples,
                    )
            except Exception as e:
                logger.warning(f"[METIS] System 2 SE failed: {e}")

        return se_result, system2_triggered, was_verified, raw_text

    def _run_hallucination_correction(
        self,
        was_refused: bool,
        meta_judgment: MetaJudgment,
        prompt: str,
        max_tokens: int,
        chat_template: bool,
        avg_confidence: float,
        raw_text: str,
        model: nn.Module,
        tokenizer: Any,
    ) -> Tuple[bool, str]:
        """
        G4: Intelligent single-pass hallucination correction (Draft-Critique-Refine, non-recursive).

        Differences from old Simple Retry:
        1. Verification prompt guides model to scrutinize original answer before re-answering
        2. Strict token budget: correction cap = min(max_correction_tokens, max_tokens)
           Prevents System 2 latency blowup
        3. Stricter adoption criteria: not just confidence, but requires correction version
           confidence to be significantly higher than original (> 10% relative improvement)

        Returns:
            (hallucination_corrected, raw_text)
        """
        hallucination_corrected = False
        if (
            not was_refused
            and meta_judgment.suggested_action in ("verify", "abort")
        ):
            logger.info(
                f"[METIS] Hallucination self-correction triggered: "
                f"risk={meta_judgment.hallucination_risk:.2f}"
            )
            try:
                # Draft-Critique prompt: guide model to scrutinize and re-answer
                verify_prompt = (
                    f"Question: {prompt}\n\n"
                    f"Previous answer: {raw_text[:300]}\n\n"
                    f"Please scrutinize the above answer for factual errors or uncertain statements, "
                    f"then provide a more accurate answer. If unsure, state so explicitly."
                )
                # Strict token budget: correction should not exceed original length
                correction_budget = min(
                    self._max_correction_tokens, max_tokens
                )
                retry_result = self._generate_simple(
                    model, tokenizer, verify_prompt,
                    max_tokens=correction_budget, temperature=0.0,
                    chat_template=chat_template,
                )
                retry_conf = retry_result["avg_confidence"]
                # Adoption criteria: confidence must improve significantly (> 10% relative)
                # Avoid false adoption from minor fluctuations
                confidence_threshold = avg_confidence * 1.1 if avg_confidence > 0 else 0.1
                if retry_conf > confidence_threshold:
                    raw_text = retry_result["text"]
                    hallucination_corrected = True
                    logger.info(
                        "[METIS] Self-correction accepted: "
                        f"retry_conf={retry_conf:.2f} "
                        f"> threshold={confidence_threshold:.2f} "
                        f"(original={avg_confidence:.2f})"
                    )
                else:
                    logger.info(
                        "[METIS] Self-correction rejected: "
                        f"retry_conf={retry_conf:.2f} "
                        f"<= threshold={confidence_threshold:.2f}"
                    )
            except Exception as e:
                logger.warning(f"[METIS] Self-correction failed: {e}")

        return hallucination_corrected, raw_text

    @staticmethod
    def _build_introspection(
        *,
        was_refused: bool,
        was_verified: bool,
        was_hedged: bool,
        cot_injected: bool,
        hallucination_corrected: bool,
        final_signal: CognitiveSignal,
        se_result: Optional[SemanticEntropyResult],
        meta_judgment: MetaJudgment,
        seek_results: List[str],
    ) -> List[str]:
        """Build introspection summary string list"""
        parts: List[str] = []
        if was_refused:
            parts.append(f"Knowledge boundary refused: {final_signal.introspection}")
        if was_verified and se_result:
            parts.append(
                f"System 2 verified: SE={se_result.semantic_entropy:.2f} bits, "
                f"{se_result.n_clusters} semantic clusters, "
                f"{'uncertain' if se_result.is_uncertain else 'certain'}"
            )
        if was_hedged:
            parts.append("Answer hedged with uncertainty note")
        if cot_injected:
            parts.append("CoT reasoning chain injected")
        if hallucination_corrected:
            parts.append("Hallucination self-corrected via verification re-generation")
        if seek_results:
            parts.append(f"External retrieval triggered {len(seek_results)} times")
        if meta_judgment.reasoning:
            parts.append(
                f"Metacognition: {meta_judgment.suggested_action} "
                f"({meta_judgment.reasoning})"
            )
        return parts
