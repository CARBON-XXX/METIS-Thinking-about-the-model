#!/usr/bin/env python3
"""
METIS Telemetry REPL — Interactive Cognitive CLI
=================================================
Phase 12: Final Assembly & Telemetry

Provides a rich terminal interface to observe METIS's internal
cognitive process in real-time: entropy probe → epistemic routing
→ optional search → generation → telemetry display.

Usage:
    python tools/run_metis_cli.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── ANSI color codes ──
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[48;5;236m"


MODEL_PATH = str(PROJECT_ROOT / "experiment_output_dpo_balanced" / "metis_dpo_cognitive")

BANNER = f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███╗   ███╗███████╗████████╗██╗███████╗                    ║
║   ████╗ ████║██╔════╝╚══██╔══╝██║██╔════╝                    ║
║   ██╔████╔██║█████╗     ██║   ██║███████╗                    ║
║   ██║╚██╔╝██║██╔══╝     ██║   ██║╚════██║                    ║
║   ██║ ╚═╝ ██║███████╗   ██║   ██║███████║                    ║
║   ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝╚══════╝                    ║
║                                                              ║
║   Metacognitive Engine for Transparent Inference Systems      ║
║   Phase 12: Final Assembly & Telemetry                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
"""

HELP_TEXT = f"""
{C.DIM}Commands:
  Type any question to query METIS.
  /quit or /exit  — Exit the REPL
  /help           — Show this help{C.RESET}
"""


def entropy_bar(h: float, width: int = 20) -> str:
    """Render a visual entropy bar."""
    filled = int(min(h / 2.5, 1.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    if h > 0.8:
        color = C.RED
    elif h > 0.4:
        color = C.YELLOW
    else:
        color = C.GREEN
    return f"{color}{bar}{C.RESET} {h:.4f}"


def route_color(route: str) -> str:
    """Color-code the cognitive route."""
    if "DEEP" in route:
        return f"{C.RED}{C.BOLD}{route}{C.RESET}"
    elif "FAST" in route:
        return f"{C.GREEN}{C.BOLD}{route}{C.RESET}"
    return f"{C.WHITE}{route}{C.RESET}"


def state_color(state: str) -> str:
    """Color-code epistemic state."""
    mapping = {
        "known": C.GREEN,
        "likely": C.CYAN,
        "uncertain": C.YELLOW,
        "unknown": C.RED,
    }
    color = mapping.get(state, C.WHITE)
    return f"{color}{C.BOLD}[{state.upper()}]{C.RESET}"


def print_telemetry(resp) -> None:
    """Pretty-print the full METIS telemetry panel."""
    print()
    print(f"  {C.DIM}{'─' * 58}{C.RESET}")
    print(f"  {C.BOLD}METIS TELEMETRY{C.RESET}")
    print(f"  {C.DIM}{'─' * 58}{C.RESET}")

    # Semantic Entropy
    print(f"  {C.BLUE}𝓗 Semantic Entropy:{C.RESET}  {entropy_bar(resp.semantic_entropy)}")

    # Epistemic State
    print(f"  {C.YELLOW}Epistemic State:{C.RESET}     {state_color(resp.epistemic_state)}")

    # Clusters
    print(f"  {C.CYAN}Clusters:{C.RESET}            "
          f"{resp.n_clusters} (sizes={resp.cluster_sizes})")

    # Search (if triggered)
    if resp.searched:
        print(f"  {C.MAGENTA}Search Query:{C.RESET}       "
              f"\"{resp.search_query}\"")
        ctx_preview = resp.retrieved_context[:120].replace('\n', ' ')
        print(f"  {C.MAGENTA}Retrieved:{C.RESET}          "
              f"{ctx_preview}...")

    # Cognitive Route
    print(f"  {C.BOLD}Cognitive Route:{C.RESET}     {route_color(resp.cognitive_route)}")

    # Thinking
    if resp.thinking_text:
        repair_tag = f" {C.YELLOW}(auto-repaired){C.RESET}" if resp.thinking_repaired else ""
        print(f"  {C.DIM}Thinking:{C.RESET}           "
              f"present ({len(resp.thinking_text)} chars){repair_tag}")
        # Show first 200 chars of thinking
        preview = resp.thinking_text[:200].replace('\n', '\n                      ')
        print(f"  {C.DIM}  └─ {preview}{'...' if len(resp.thinking_text) > 200 else ''}{C.RESET}")

    # Trajectory
    print(f"  {C.BOLD}Trajectory:{C.RESET}         {resp.trajectory}")

    # Stats
    print(f"  {C.DIM}Tokens: {resp.tokens_generated}  |  "
          f"Latency: {resp.latency_ms:.0f}ms{C.RESET}")

    print(f"  {C.DIM}{'─' * 58}{C.RESET}")

    # Final Answer
    print(f"\n  {C.BOLD}{C.WHITE}Answer:{C.RESET}")
    for line in resp.final_answer.strip().split('\n'):
        print(f"  {line}")
    print()


def main() -> None:
    print(BANNER)
    print(f"  {C.DIM}Loading model from: {MODEL_PATH}{C.RESET}")
    print(f"  {C.DIM}This may take ~90 seconds...{C.RESET}\n")

    t0 = time.time()
    from metis import Metis
    from metis.cognitive.metacognition import MetacognitiveOrchestrator

    metis = Metis.from_pretrained(MODEL_PATH)
    orch = MetacognitiveOrchestrator(metis)
    load_time = time.time() - t0

    print(f"  {C.GREEN}✓ METIS online in {load_time:.1f}s{C.RESET}")
    print(HELP_TEXT)

    while True:
        try:
            prompt = input(f"  {C.CYAN}{C.BOLD}You ▸{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {C.DIM}Goodbye.{C.RESET}")
            break

        if not prompt:
            continue
        if prompt.lower() in ("/quit", "/exit", "quit", "exit"):
            print(f"  {C.DIM}Goodbye.{C.RESET}")
            break
        if prompt.lower() == "/help":
            print(HELP_TEXT)
            continue

        print(f"\n  {C.DIM}Processing...{C.RESET}")
        resp = orch.process_query(prompt)
        print_telemetry(resp)


if __name__ == "__main__":
    main()
