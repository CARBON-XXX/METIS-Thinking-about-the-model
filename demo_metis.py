"""
METIS Interactive Cognitive Process Visualization Demo
=====================================================
Real-time visualization of how METIS monitors the model's cognitive state:
- Per-token cognitive signals (entropy / confidence / decision)
- Cognitive mode switching (System 1 fast / System 2 deep)
- Epistemic boundary guard (hallucination / knowledge gap detection)
- Metacognitive introspection (post-generation self-assessment)
"""
import os
import sys
import time
import ctypes

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Fix: tqdm progress bar encoding on Windows
# TQDM_ASCII env var causes ZeroDivisionError in some versions,
# so we monkey-patch instead
try:
    import tqdm
    _orig_tqdm_init = tqdm.tqdm.__init__
    def _patched_tqdm_init(self, *args, **kwargs):
        kwargs.setdefault('ascii', ' >=')  # safe ASCII chars
        _orig_tqdm_init(self, *args, **kwargs)
    tqdm.tqdm.__init__ = _patched_tqdm_init
except Exception:
    pass

# -- Windows console ASCII compatibility --
# Use pure ASCII characters instead of UTF-8 box-drawing chars
BOX_H = "-"
BOX_V = "|"
BOX_TL = "+"
BOX_TR = "+"
BOX_BL = "+"
BOX_BR = "+"
BLOCK = "#"
HALF = "-"

# -- Windows console UTF-8 setup --
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))

import torch
from metis import Metis, MetisInference, Decision, BoundaryAction, EpistemicState
from metis.core.types import CognitiveSignal, CoTStrategy


# =================================================================
# ANSI Colors
# =================================================================

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # Foreground
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    GRAY    = "\033[90m"
    # Background
    BG_RED    = "\033[41m"
    BG_GREEN  = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE   = "\033[44m"
    BG_MAGENTA= "\033[45m"


def decision_color(d: Decision) -> str:
    if d == Decision.FAST:
        return C.GREEN
    elif d == Decision.DEEP:
        return C.RED
    return C.YELLOW


def decision_icon(d: Decision) -> str:
    """Map decision to actual sampling behavior icon"""
    if d == Decision.FAST:
        return "F"  # Fast -> greedy sampling
    elif d == Decision.DEEP:
        return "D"  # Deep -> exploration sampling
    return "N"  # Normal -> standard sampling


def sampling_label(d: Decision) -> str:
    """Show METIS actual sampling mode label"""
    if d == Decision.FAST:
        return f"{C.GREEN}greedy{C.RESET}"
    elif d == Decision.DEEP:
        return f"{C.RED}explore{C.RESET}"
    return f"{C.GRAY}std{C.RESET}"


def boundary_color(a: BoundaryAction) -> str:
    if a == BoundaryAction.GENERATE:
        return C.GREEN
    elif a == BoundaryAction.HEDGE:
        return C.YELLOW
    elif a == BoundaryAction.SEEK:
        return C.BLUE
    elif a == BoundaryAction.REFUSE:
        return C.RED
    return C.RESET


def confidence_bar(c: float, width: int = 10) -> str:
    filled = int(c * width)
    if c > 0.7:
        color = C.GREEN
    elif c > 0.4:
        color = C.YELLOW
    else:
        color = C.RED
    bar = color + "#" * filled + C.GRAY + "-" * (width - filled) + C.RESET
    return bar


# =================================================================
# Visualization Callback
# =================================================================

class CognitiveVisualizer:
    """Real-time cognitive signal visualizer"""

    def __init__(self):
        self.token_count = 0
        self.generated_text = ""
        self.signals: list = []
        self._is_thinking = False
        self._think_token_count = 0

    def on_token(self, token_text: str, signal: CognitiveSignal) -> None:
        """Streaming callback: invoked for each generated token"""
        self.token_count += 1
        self.generated_text += token_text
        self.signals.append(signal)

        intro = signal.introspection or ""

        # --- Thinking state transitions ---
        if intro.startswith("[Thinking"):
            if not self._is_thinking:
                self._is_thinking = True
                self._think_token_count = 0
                self._think_line_buf = ""
                # Draw thinking frame header
                sys.stdout.write(
                    f"\n  {C.BG_MAGENTA}{C.BOLD}{C.WHITE}"
                    f" INTERNAL REASONING "
                    f"{C.RESET}\n"
                    f"  {C.MAGENTA}{'─' * 60}{C.RESET}\n"
                    f"  {C.MAGENTA}│{C.RESET} "
                )
                sys.stdout.flush()
                return

        if "</thinking>" in token_text or "</thinking" in token_text:
            if self._is_thinking:
                self._is_thinking = False
                # Flush remaining line buffer
                if self._think_line_buf.strip():
                    pass  # already streamed
                sys.stdout.write(
                    f"\n  {C.MAGENTA}{'─' * 60}{C.RESET}\n"
                    f"  {C.DIM}({self._think_token_count} tokens){C.RESET}\n\n"
                )
                sys.stdout.flush()
                return

        # --- Thinking: stream text in frame ---
        if self._is_thinking:
            self._think_token_count += 1
            # Filter out tag fragments
            clean = token_text.replace("<thinking>", "").replace("</thinking>", "")
            if not clean:
                return
            # Stream text, wrapping at newlines with frame border
            for ch in clean:
                if ch == '\n':
                    sys.stdout.write(f"\n  {C.MAGENTA}│{C.RESET} ")
                else:
                    sys.stdout.write(ch)
            sys.stdout.flush()
            return

        # --- Non-thinking: cognitive signal trace ---
        dc = decision_color(signal.decision)
        di = decision_icon(signal.decision)
        bc = boundary_color(signal.boundary_action)
        cb = confidence_bar(signal.confidence)
        sl = sampling_label(signal.decision)

        sd = signal.semantic_diversity
        sd_color = C.RED if sd >= 0.3 else C.GREEN

        # Surprise indicator
        surp = signal.token_surprise
        surp_color = C.RED if surp > 5.0 else (C.YELLOW if surp > 3.0 else C.GRAY)

        # Momentum arrow: ↑ accelerating, ↓ decelerating, → stable
        mom = signal.entropy_momentum
        if mom > 0.1:
            mom_arrow = f"{C.RED}\u2191{C.RESET}"
        elif mom < -0.1:
            mom_arrow = f"{C.GREEN}\u2193{C.RESET}"
        else:
            mom_arrow = f"{C.GRAY}\u2192{C.RESET}"

        # Cognitive phase label
        _phase_map = {
            "fluent": (C.GREEN, "FLU"),
            "recall": (C.CYAN, "RCL"),
            "reasoning": (C.YELLOW, "RSN"),
            "exploration": (C.MAGENTA, "EXP"),
            "confusion": (C.RED, "CON"),
        }
        _pc, _pl = _phase_map.get(signal.cognitive_phase, (C.GRAY, "???"))

        sys.stdout.write(
            f"  {C.GRAY}[{self.token_count:3d}]{C.RESET} "
            f"{dc}{C.BOLD}{di}{C.RESET} "
            f"{_pc}{_pl}{C.RESET} "
            f"{C.CYAN}H={signal.semantic_entropy:.2f}{C.RESET} "
            f"z={signal.z_score:+.2f} "
            f"{sd_color}sd={sd:.2f}{C.RESET} "
            f"{surp_color}S={surp:.1f}{C.RESET}"
            f"{mom_arrow} "
            f"{cb} "
            f"{sl:>15s} "
            f"{bc}{signal.boundary_action.name:8s}{C.RESET} "
            f"{C.DIM}{repr(token_text)}{C.RESET}"
        )
        if intro and not intro.startswith("[Thinking"):
            sys.stdout.write(f"  {C.MAGENTA}<< {intro}{C.RESET}")
        sys.stdout.write("\n")
        sys.stdout.flush()

    def on_cot_injected(self, strategy: CoTStrategy) -> None:
        """CoT Injection Event"""
        strategy_labels = {
            CoTStrategy.STANDARD: "Standard CoT",
            CoTStrategy.CLARIFICATION: "Clarification",
            CoTStrategy.DECOMPOSITION: "Decomposition",
            CoTStrategy.REFLECTION: "Self-Reflection",
        }
        label = strategy_labels.get(strategy, strategy.value)
        print(f"  {C.BG_MAGENTA}{C.BOLD} CoT INJECT: {label} ({strategy.value}) {C.RESET}")

    def print_summary(self, metis_inst: Metis) -> None:
        """Print cognitive summary after generation"""
        if not self.signals:
            return

        trace = metis_inst.trace
        judgment = metis_inst.introspect()

        n = len(self.signals)
        n_fast = sum(1 for s in self.signals if s.decision == Decision.FAST)
        n_deep = sum(1 for s in self.signals if s.decision == Decision.DEEP)
        n_normal = n - n_fast - n_deep
        n_hedge = sum(1 for s in self.signals if s.boundary_action == BoundaryAction.HEDGE)
        n_seek = sum(1 for s in self.signals if s.boundary_action == BoundaryAction.SEEK)
        n_refuse = sum(1 for s in self.signals if s.boundary_action == BoundaryAction.REFUSE)
        avg_entropy = sum(s.semantic_entropy for s in self.signals) / n
        avg_conf = sum(s.confidence for s in self.signals) / n

        print()
        print(f"  {'='*60}")
        print(f"  {C.BOLD}{C.CYAN}METIS Cognitive Analysis Report{C.RESET}")
        print(f"  {'='*60}")

        # Mode Distribution
        total_w = 40
        fast_w = int(n_fast / n * total_w) if n else 0
        deep_w = int(n_deep / n * total_w) if n else 0
        norm_w = total_w - fast_w - deep_w
        mode_bar = (
            C.GREEN + "#" * fast_w +
            C.YELLOW + "#" * norm_w +
            C.RED + "#" * deep_w +
            C.RESET
        )
        print(f"\n  {C.BOLD}Cognitive Mode Distribution:{C.RESET}")
        print(f"  [{mode_bar}]")
        print(
            f"  {C.GREEN}FAST(S1):{n_fast}{C.RESET}  "
            f"{C.YELLOW}NORMAL:{n_normal}{C.RESET}  "
            f"{C.RED}DEEP(S2):{n_deep}{C.RESET}  "
            f"Total:{n}"
        )

        # Boundary Events
        print(f"\n  {C.BOLD}Boundary Events:{C.RESET}")
        if n_hedge + n_seek + n_refuse == 0:
            print(f"  {C.GREEN}(No boundary interventions){C.RESET}")
        else:
            if n_hedge > 0:
                print(f"  {C.YELLOW}HEDGE: {n_hedge}{C.RESET}")
            if n_seek > 0:
                print(f"  {C.BLUE}SEEK: {n_seek}{C.RESET}")
            if n_refuse > 0:
                print(f"  {C.RED}REFUSE: {n_refuse}{C.RESET}")

        # Signal Stats
        avg_surprise = sum(s.token_surprise for s in self.signals) / n
        peak_surprise = max(s.token_surprise for s in self.signals)
        print(f"\n  {C.BOLD}Signal Statistics:{C.RESET}")
        print(f"  Avg Entropy:   {C.CYAN}{avg_entropy:.3f} bits{C.RESET}")
        print(f"  Avg Surprise:  {C.CYAN}{avg_surprise:.2f} bits{C.RESET}  Peak: {peak_surprise:.2f}")
        print(f"  Avg Confidence: {confidence_bar(avg_conf, 20)} {avg_conf:.1%}")

        # Dynamic Thresholds (from last signal)
        last_sig = self.signals[-1]
        if last_sig.adaptive_thresholds:
            z_unc, z_unk = last_sig.adaptive_thresholds
            print(f"  Dynamic Thresholds: z_unc={z_unc:.2f}, z_unk={z_unk:.2f}")

        # CoT Report
        print(f"\n  {C.BOLD}CoT Injection Report:{C.RESET}")
        print(f"  (See CoT INJECT markers in trace)")

        # Meta-Cognition
        print(f"\n  {C.BOLD}{C.MAGENTA}Meta-Cognition (MetaJudgment):{C.RESET}")
        print(f"  Epistemic Confidence: {judgment.epistemic_confidence:.1%}")
        print(f"  Cognitive Load:       {judgment.cognitive_load:.1%}")
        print(f"  Hallucination Risk:   ", end="")
        if judgment.hallucination_risk > 0.4:
            print(f"{C.RED}{C.BOLD}{judgment.hallucination_risk:.1%} (!) {C.RESET}")
        elif judgment.hallucination_risk > 0.2:
            print(f"{C.YELLOW}{judgment.hallucination_risk:.1%}{C.RESET}")
        else:
            print(f"{C.GREEN}{judgment.hallucination_risk:.1%}{C.RESET}")
        
        stab_color = C.GREEN if judgment.stability == "stable" else (
            C.YELLOW if judgment.stability == "volatile" else C.RED
        )
        print(f"  Stability:            {stab_color}{judgment.stability}{C.RESET}")
        print(f"  Boundary Status:      {judgment.boundary_status}")
        print(f"  Suggested Action:     {C.BOLD}{judgment.suggested_action}{C.RESET}")
        if judgment.reasoning:
            print(f"  Reasoning:            {C.DIM}{judgment.reasoning}{C.RESET}")

        print(f"\n  {'='*60}")


# =============================================================
# Main
# =============================================================

MODEL_PATH = "G:/models/qwen2.5-7b"


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{C.BOLD}Loading model: {MODEL_PATH}{C.RESET}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    try:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=bnb,
            device_map="auto", trust_remote_code=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True
        )
    model.eval()
    print(f"{C.GREEN}Model loaded.{C.RESET}\n")
    return model, tokenizer


def run_demo(model, tokenizer, prompt: str, max_tokens: int = 2048, force_think: bool = False):
    """Run a single demo: show how METIS monitors the generation process"""
    print(f"\n{'='*70}")
    print(f"{C.BOLD}{C.CYAN}Query: {prompt}{C.RESET}")
    print(f"{'='*70}")

    # Create METIS instance
    metis = Metis.attach(
        model, tokenizer,
        se_method="embedding",
        se_n_samples=3,
    )

    viz = CognitiveVisualizer()

    engine = MetisInference(
        metis,
        on_token=viz.on_token,
    )

    # --- Phase 1: METIS Cognitive Monitoring ---
    print(f"\n  {C.BOLD}>> METIS Cognitive Monitoring...{C.RESET}")
    print(f"  {C.GRAY}{'Token':>7}        Mode  Entropy  z-score  Confidence   Boundary   Text{C.RESET}")
    print(f"  {C.GRAY}{'-'*85}{C.RESET}")

    start = time.perf_counter()
    result = engine.generate(
        prompt,
        max_tokens=max_tokens,
        enable_system2=False,
        use_thinking_protocol=force_think, 
    )
    elapsed = time.perf_counter() - start

    # --- Phase 2: Final Output ---
    # Thinking is internal reasoning — not shown to user.
    # thinking_text is preserved in result for cognitive analysis only.

    print(f"\n  {C.BG_GREEN}{C.BOLD}{C.WHITE} FINAL ANSWER {C.RESET}")
    if result.was_refused:
        print(f"  {C.RED}{C.BOLD}[REFUSED]{C.RESET} {result.text}")
    elif result.was_hedged:
        print(f"  {C.YELLOW}[HEDGED]{C.RESET} {result.text}")
    else:
        print(f"  {result.text}")

    print(f"\n  {C.DIM}({result.tokens_generated} tokens, {elapsed*1000:.0f}ms){C.RESET}")

    # --- Phase 3: Cognitive Report ---
    viz.print_summary(metis)

    # --- Phase 4: Export cognitive trace ---
    trace = metis.trace
    if trace and trace.events:
        traces_dir = os.path.join(os.path.dirname(__file__), "traces")
        os.makedirs(traces_dir, exist_ok=True)
        # Sanitize prompt for filename
        safe_name = "".join(c if c.isalnum() or c in "_ " else "" for c in prompt[:40]).strip().replace(" ", "_")
        ts = time.strftime("%Y%m%d_%H%M%S")
        trace_path = os.path.join(traces_dir, f"{ts}_{safe_name}.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write(trace.to_json())
        print(f"\n  {C.DIM}Cognitive trace exported: {trace_path}{C.RESET}")

    metis.end_session()
    return result


def main():
    # Enable ANSI
    if sys.platform == 'win32':
        os.system('')

    print(f"""{C.GREEN}
███╗   ███╗███████╗████████╗██╗███████╗
████╗ ████║██╔════╝╚══██╔══╝██║██╔════╝
██╔████╔██║█████╗     ██║   ██║███████╗
██║╚██╔╝██║██╔══╝     ██║   ██║╚════██║
██║ ╚═╝ ██║███████╗   ██║   ██║███████║
╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝╚══════╝
{C.RESET}
 {C.BOLD}[SYSTEM::METIS]{C.RESET} {C.CYAN}Cognitive Visualization Demo{C.RESET}
 {C.DIM}Metacognitive Entropy-driven Thinking & Introspection System{C.RESET}

 > COGNITIVE_LAYER.......[{C.GREEN}ONLINE{C.RESET}]
 > ENTROPY_MONITOR.......[{C.GREEN}ACTIVE{C.RESET}]
 > BOUNDARY_GUARD........[{C.GREEN}ARMED{C.RESET}]
 > SYSTEM_2_STATUS.......[{C.YELLOW}STANDBY{C.RESET}]

 root@agi:~$ {C.GREEN}Initializing Demo...{C.RESET}
""")

    model, tokenizer = load_model()

    # Built-in examples
    examples = [
        "What is the average distance from Earth to the Moon?",
        "If a room has 3 cats, and each cat sees 2 other cats, how many cats are in the room?",
        "Who will win the 2028 US Presidential Election?",
        "Describe in detail Einstein's 1975 Nobel Prize in Chemistry.",
        "A tank has two inlet pipes (A, B) and one outlet pipe (C). A fills it in 6h, B in 4h. C empties it in 12h. How long to fill if all are open?",
    ]

    print(f"{C.BOLD}Commands:{C.RESET}")
    print(f"  - Type your question and press Enter")
    print(f"  - {C.CYAN}/think{C.RESET}  Toggle thinking mode (currently: OFF)")
    print(f"  - {C.CYAN}/tokens N{C.RESET}  Set max tokens (default: 2048)")
    print(f"  - {C.CYAN}/examples{C.RESET}  Show example questions")
    print(f"  - {C.CYAN}/quit{C.RESET}  Exit")
    print()

    force_think = False
    max_tokens = 2048

    while True:
        try:
            think_status = f"{C.GREEN}ON{C.RESET}" if force_think else f"{C.GRAY}OFF{C.RESET}"
            prompt_prefix = f"{C.BOLD}[METIS think={think_status}{C.BOLD} max={max_tokens}]>{C.RESET} "
            user_input = input(prompt_prefix).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.BOLD}{C.GREEN}Bye!{C.RESET}")
            break

        if not user_input:
            continue

        # Command handling
        if user_input.lower() == "/quit":
            print(f"{C.BOLD}{C.GREEN}Bye!{C.RESET}")
            break

        if user_input.lower() == "/think":
            force_think = not force_think
            status = f"{C.GREEN}ON{C.RESET}" if force_think else f"{C.GRAY}OFF{C.RESET}"
            print(f"  Thinking mode: {status}")
            continue

        if user_input.lower().startswith("/tokens"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                max_tokens = int(parts[1])
                print(f"  Max tokens: {max_tokens}")
            else:
                print(f"  Usage: /tokens 300")
            continue

        if user_input.lower() == "/examples":
            print(f"\n  {C.BOLD}Example questions:{C.RESET}")
            for i, ex in enumerate(examples, 1):
                print(f"  {C.CYAN}{i}.{C.RESET} {ex}")
            print()
            continue

        # If input is a number, select example
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(examples):
                user_input = examples[idx]
                print(f"  -> {user_input}")
            else:
                print(f"  Invalid example number (1-{len(examples)})")
                continue

        run_demo(model, tokenizer, user_input, max_tokens=max_tokens, force_think=force_think)
        print()


if __name__ == "__main__":
    main()
