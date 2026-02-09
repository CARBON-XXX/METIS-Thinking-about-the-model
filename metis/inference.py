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

import torch
import torch.nn as nn

from .metis import Metis
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
        system2_uncertainty_score: float = 0.3,
        # Boundary behavior
        hedge_prefix: str = "",
        hedge_suffix: str = "\n\n(Note: I am not entirely sure about this answer. Please verify with other sources.)",
        refuse_message: str = "I apologize, but this question is beyond my reliable knowledge base. I cannot provide a certain answer.",
        # REFUSE strategy
        refuse_grace_period: int = 8,
        refuse_consecutive_threshold: int = 3,
        # Self-correction budget control
        max_correction_tokens: int = 96,
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
        self._cot_manager = CoTManager()
        self._on_seek = on_seek
        self._on_token = on_token

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
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
            if use_thinking_protocol:
                system_content = (
                    "You are a deep thinking AI. Before answering, you must engage in a comprehensive, "
                    "fluid stream-of-consciousness internal monologue inside <thinking>...</thinking> tags. "
                    "Explore the problem from multiple angles, question your assumptions, and verify your logic step-by-step. "
                    "Do not use bullet points for your thinking; instead, write in a natural, flowing narrative. "
                    "Only when you have a solid conclusion should you close the thinking tag and output the final answer."
                )
                # Check if messages already has system, if so prepend/replace, else insert
                messages.insert(0, {"role": "system", "content": system_content})

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
            
        # Thinking Protocol injection (Force start)
        if use_thinking_protocol:
            # Check if text already ends with assistant marker, append <thinking>
            # Qwen/ChatML usually ends with "assistant\n"
            text += "<thinking>\n"

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
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
        cot_strategies_used: List[CoTStrategy] = []
        seek_results: List[str] = []
        
        # Thinking Protocol state tracking
        is_thinking = use_thinking_protocol
        thinking_start_step = 0 if is_thinking else -1
        min_thinking_tokens = 64  # Minimum thinking length (Anti-Lazy)

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

            # --- G3: Dynamic CoT Injection (CoTManager-driven) ---
            # CoTManager selects the most appropriate strategy based on signal characteristics:
            #   REFLECTION -> model oscillating at knowledge boundary
            #   DECOMPOSITION -> sustained deep reasoning, complex problem
            #   CLARIFICATION -> conceptual ambiguity, high semantic diversity
            #   STANDARD -> generic high entropy
            # Also manages cooldown and injection count limits to prevent runaway latency
            self._cot_manager.observe(signal)

            if self._cot_manager.should_inject():
                strategy = self._cot_manager.select_strategy(signal)
                gen_context = tokenizer.decode(generated_tokens[-30:], skip_special_tokens=True) if generated_tokens else ""
                cot_text = self._cot_manager.get_prompt(strategy, context=gen_context)
                
                # Dynamic Thinking: if not currently in a thinking block, open one
                if not is_thinking:
                    think_open = "\n<thinking>\n"
                    think_open_ids = tokenizer.encode(think_open, add_special_tokens=False)
                    for tid in think_open_ids:
                        generated_tokens.append(tid)
                    if self._on_token is not None:
                        open_signal = CognitiveSignal(decision=Decision.DEEP, introspection="[Thinking Start]")
                        self._on_token(think_open, open_signal)
                    # Feed into model
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
                    logger.info(f"[METIS] Dynamic Thinking: Opening block for CoT: {strategy.value}")
                else:
                    # Already thinking, reset budget to extend reasoning
                    thinking_start_step = step
                    logger.info(f"[METIS] Dynamic Thinking: Extending budget for CoT: {strategy.value}")
                
                self._cot_manager.record_injection(strategy)
                cot_injected = True
                cot_strategies_used.append(strategy)

                cot_ids = tokenizer.encode(cot_text, add_special_tokens=False)
                
                # Pre-calculate signal for CoT tokens
                cot_signal = CognitiveSignal(
                    semantic_entropy=signal.semantic_entropy,
                    confidence=signal.confidence,
                    decision=Decision.DEEP,
                    boundary_action=BoundaryAction.GENERATE,
                    introspection=f"[CoT {strategy.value}]",
                    z_score=signal.z_score,
                )

                for cid in cot_ids:
                    generated_tokens.append(cid)
                    # Stream individual tokens for organic UX
                    if self._on_token is not None:
                        token_text = tokenizer.decode([cid], skip_special_tokens=True)
                        try:
                            # CoT injection displayed directly, not buffered (system-forced)
                            self._on_token(token_text, cot_signal)
                            time.sleep(0.02)
                        except Exception:
                            pass
                
                # Feed CoT tokens into model to update KV cache
                cot_input = torch.tensor([cot_ids], device=model.device)
                cot_out = model(
                    input_ids=cot_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = cot_out.past_key_values
                
                # CRITICAL FIX: Update logits!
                # After CoT injection, the next token should be predicted based on the last CoT token,
                # not the original logits before injection. Otherwise CoT has no effect on the current decision.
                logits = cot_out.logits[:, -1, :]
                
                logger.info(
                    f"[METIS] CoT injected at step {step}: "
                    f"strategy={strategy.value}, "
                    f"injection #{len(cot_strategies_used)}"
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

            # --- Token sampling (cognitive-aware) ---
            next_token_id = self._cognitive_sample(
                logits, signal, temperature, top_p
            )
            
            # --- Anti-Lazy Thinking: enforce deep reasoning ---
            if is_thinking:
                # Check if model is trying to close the thinking block
                # Only decode last few tokens to check for closing tag
                check_window = 12
                recent_ids = generated_tokens[-check_window:] + [next_token_id]
                recent_text = tokenizer.decode(recent_ids, skip_special_tokens=True)
                
                if "</thinking>" in recent_text:
                    tokens_in_block = step - thinking_start_step
                    if tokens_in_block < min_thinking_tokens:
                        logger.info(f"[METIS] Anti-Lazy: Rejected premature closing at {tokens_in_block} tokens")
                        
                        # 1. Rollback: Remove ALL closing tag tokens from generated_tokens
                        # The tag '</thinking>' may span multiple tokens (e.g., '</', 'thinking', '>')
                        # next_token_id hasn't been appended yet, so we only clean generated_tokens
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
                        
                        # 2. Clear Visualizer Buffer (Hide the rejected tag)
                        vis_buffer.clear()

                        # 3. Inject Continuation
                        continuations = [
                            "Wait, I should double check that calculation. ",
                            "Specifically, I need to verify ",
                            "Furthermore, considering the edge cases, ",
                            "Let me re-evaluate the previous step. ",
                        ]
                        import random
                        continuation = continuations[step % len(continuations)]
                        cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
                        
                        # Inject continuation tokens
                        cont_input = torch.tensor([cont_ids], device=model.device)
                        cont_out = model(
                            input_ids=cont_input,
                            past_key_values=past_key_values, # Dirty KV but okay
                            use_cache=True,
                            return_dict=True,
                        )
                        past_key_values = cont_out.past_key_values
                        logits = cont_out.logits[:, -1, :]
                        
                        logger.info(
                            f"[METIS] Injected continuation at step {step}: "
                            f"{continuation}"
                        )
                        
                        # Add to visualizer
                        cont_signal = CognitiveSignal(decision=Decision.DEEP, introspection="[Anti-Lazy Extension]")
                        for cid in cont_ids:
                            generated_tokens.append(cid)
                            if self._on_token:
                                txt = tokenizer.decode([cid], skip_special_tokens=True)
                                self._on_token(txt, cont_signal)
                                time.sleep(0.02)
                        
                        # Loop continue with last token of continuation
                        next_token_id = cont_ids[-1]
                        input_ids = torch.tensor([[next_token_id]], device=model.device)
                        
                        # Skip the normal append at bottom
                        continue 
                    else:
                        # Thinking block closed naturally, ensure </thinking> tag is visible
                        is_thinking = False
                        # Flush vis_buffer tokens (including </thinking>)
                        if vis_buffer:
                            for t, s in vis_buffer:
                                if self._on_token:
                                    self._on_token(t, s)
                            vis_buffer.clear()
                        logger.info(f"[METIS] Thinking block closed naturally at {tokens_in_block} tokens")

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
        else:
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

        should_hedge = (
            not was_refused
            and (
                final_signal.boundary_action == BoundaryAction.HEDGE
                or uncertainty >= self._system2_uncertainty_score
                or (se_result is not None and se_result.is_uncertain)
                or boundary_ratio >= 0.10  # >=10% tokens triggered boundary -> hedge
            )
        )

        if should_hedge:
            was_hedged = True
            final_text = self._hedge_prefix + raw_text + self._hedge_suffix

        # Metacognitive regulation: supplement actions based on introspection
        if not was_hedged and not was_refused:
            regulation = self._metis.regulate(meta_judgment)
            if regulation["should_hedge"]:
                was_hedged = True
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
        # Clean up excess whitespace left behind
        answer = re.sub(r'\n{3,}', '\n\n', answer).strip()

        return answer, thinking_text

    def _cognitive_sample(
        self,
        logits: torch.Tensor,
        signal: CognitiveSignal,
        base_temperature: float,
        base_top_p: float,
    ) -> int:
        """
        Cognitive-aware sampling — METIS core intervention point.

        Dynamically adjusts sampling strategy based on cognitive signals:
        - FAST (System 1): greedy → output best token directly when confident
        - NORMAL: user settings → standard sampling
        - DEEP (System 2): raise temp + expand top_p → increase exploration

        Also applies entropy-aware logit modulation:
        - High confidence (low z): no extra processing
        - Low confidence (high z + low conf): sharpen distribution, suppress long-tail noise
          (similar to contrastive decoding, Li et al. 2022)
        """
        # -- 1. Decision-driven temperature/top_p adjustment --
        if signal.decision == Decision.FAST:
            # System 1: high certainty, greedy
            temperature = 0.0
            top_p = 1.0
        elif signal.decision == Decision.DEEP and base_temperature > 0:
            # System 2: high uncertainty, widen search space
            # Only raise temp when user explicitly set temperature>0, respect greedy intent
            temperature = base_temperature * 1.3
            top_p = min(base_top_p + 0.1, 1.0)
        else:
            temperature = base_temperature
            top_p = base_top_p

        # -- 2. Entropy-aware logit modulation --
        # When z-score high + confidence low: flat distribution, likely to sample unreliable tokens
        # Adaptive sharpening reduces random walk
        if signal.z_score > 1.0 and signal.confidence < 0.5:
            sharpness = 1.0 + 0.15 * min(signal.z_score - 1.0, 3.0)
            logits = logits * sharpness

        # -- 3. Sampling --
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
            and meta_judgment.hallucination_risk > 0.3
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
