#!/usr/bin/env python3
"""
METIS Interactive Chat — 轻量交互式推理
========================================
直接调用 generate_cognitive()，零额外开销。
实时展示认知路由 (FAST/DEEP) 和 <thinking> 推理过程。

Usage:
    python tools/chat_metis.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── ANSI ──
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
    ITALIC  = "\033[3m"


MODEL_PATH = str(PROJECT_ROOT / "experiment_output_dpo_balanced" / "metis_dpo_cognitive")

BANNER = f"""
{C.CYAN}{C.BOLD}┌──────────────────────────────────────────┐
│  METIS Chat — Cognitive Routing Engine   │
│  Type a question. Watch it think.        │
│  /quit to exit  /raw to toggle raw mode  │
└──────────────────────────────────────────┘{C.RESET}
"""


def route_badge(route: str) -> str:
    if "DEEP" in route:
        return f"{C.RED}{C.BOLD}[DEEP]{C.RESET}"
    elif route == "FAST":
        return f"{C.GREEN}{C.BOLD}[FAST]{C.RESET}"
    else:
        return f"{C.GREEN}[FAST·implicit]{C.RESET}"


def print_thinking(text: str, repaired: bool) -> None:
    """Pretty-print the thinking block."""
    tag = f" {C.YELLOW}(auto-repaired){C.RESET}" if repaired else ""
    print(f"\n  {C.MAGENTA}{C.BOLD}💭 Thinking{tag}{C.RESET}")
    print(f"  {C.DIM}{'─' * 50}{C.RESET}")
    for line in text.strip().split("\n"):
        print(f"  {C.DIM}{C.ITALIC}{line}{C.RESET}")
    print(f"  {C.DIM}{'─' * 50}{C.RESET}")


def print_answer(text: str) -> None:
    print(f"\n  {C.BOLD}{C.WHITE}Answer:{C.RESET}")
    for line in text.strip().split("\n"):
        print(f"  {line}")


def main() -> None:
    print(BANNER)
    print(f"  {C.DIM}Loading model: {MODEL_PATH}{C.RESET}")
    print(f"  {C.DIM}Please wait ~90s...{C.RESET}\n")

    t0 = time.time()
    from metis import Metis, MetisInference
    metis = Metis.from_pretrained(MODEL_PATH)
    engine = MetisInference(metis)
    print(f"  {C.GREEN}✓ METIS ready in {time.time() - t0:.1f}s{C.RESET}\n")

    show_raw = False
    history_count = 0

    while True:
        try:
            prompt = input(f"  {C.CYAN}{C.BOLD}You ▸{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {C.DIM}再见。{C.RESET}")
            break

        if not prompt:
            continue
        if prompt.lower() in ("/quit", "/exit", "quit", "exit"):
            print(f"  {C.DIM}再见。{C.RESET}")
            break
        if prompt.lower() == "/raw":
            show_raw = not show_raw
            print(f"  {C.DIM}Raw mode: {'ON' if show_raw else 'OFF'}{C.RESET}")
            continue
        if prompt.lower() == "/help":
            print(BANNER)
            continue

        history_count += 1
        print(f"\n  {C.DIM}Generating...{C.RESET}")

        result = engine.generate_cognitive(prompt)

        # ── Route badge ──
        print(f"\n  {C.DIM}#{history_count}{C.RESET}  "
              f"Route: {route_badge(result.cognitive_route)}  "
              f"{C.DIM}tokens={result.tokens_generated}  "
              f"latency={result.latency_ms:.0f}ms{C.RESET}")

        # ── Thinking block (if DEEP) ──
        if result.thinking_text:
            print_thinking(result.thinking_text, result.thinking_repaired)

        # ── Answer ──
        print_answer(result.text)

        # ── Raw output (optional) ──
        if show_raw:
            print(f"\n  {C.DIM}── raw ──{C.RESET}")
            print(f"  {C.DIM}{result.raw_output[:500]}{C.RESET}")
            print(f"  {C.DIM}── end raw ──{C.RESET}")

        print()


if __name__ == "__main__":
    main()
