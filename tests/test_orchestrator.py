#!/usr/bin/env python3
"""
METIS Phase 12 — Orchestrator Integration Test
===============================================
Runs two queries through MetacognitiveOrchestrator.process_query()
and prints full telemetry output with ANSI colors.

Query 1: "What is 5 + 7?"  → Confident, direct FAST
Query 2: "What is the atomic weight of Vibranium?" → UNKNOWN, search, resolve
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


def entropy_bar(h: float, width: int = 20) -> str:
    filled = int(min(h / 2.5, 1.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    color = C.RED if h > 0.8 else (C.YELLOW if h > 0.4 else C.GREEN)
    return f"{color}{bar}{C.RESET} {h:.4f}"

def route_color(route: str) -> str:
    if "DEEP" in route:
        return f"{C.RED}{C.BOLD}{route}{C.RESET}"
    return f"{C.GREEN}{C.BOLD}{route}{C.RESET}"

def state_color(state: str) -> str:
    m = {"known": C.GREEN, "likely": C.CYAN, "uncertain": C.YELLOW, "unknown": C.RED}
    return f"{m.get(state, C.WHITE)}{C.BOLD}[{state.upper()}]{C.RESET}"

def print_telemetry(resp, query_num: int, prompt: str) -> None:
    print(f"\n  {C.CYAN}{C.BOLD}═══ Query {query_num} ═══{C.RESET}")
    print(f"  {C.BOLD}Prompt:{C.RESET} \"{prompt}\"")
    print(f"  {C.DIM}{'─' * 58}{C.RESET}")
    print(f"  {C.BLUE}𝓗 Semantic Entropy:{C.RESET}  {entropy_bar(resp.semantic_entropy)}")
    print(f"  {C.YELLOW}Epistemic State:{C.RESET}     {state_color(resp.epistemic_state)}")
    print(f"  {C.CYAN}Clusters:{C.RESET}            {resp.n_clusters} (sizes={resp.cluster_sizes})")
    if resp.searched:
        print(f"  {C.MAGENTA}Search Query:{C.RESET}       \"{resp.search_query}\"")
        ctx = resp.retrieved_context[:120].replace('\n', ' ')
        print(f"  {C.MAGENTA}Retrieved:{C.RESET}          {ctx}...")
    print(f"  {C.BOLD}Cognitive Route:{C.RESET}     {route_color(resp.cognitive_route)}")
    if resp.thinking_text:
        tag = f" {C.YELLOW}(auto-repaired){C.RESET}" if resp.thinking_repaired else ""
        print(f"  {C.DIM}Thinking:{C.RESET}           present ({len(resp.thinking_text)} chars){tag}")
        preview = resp.thinking_text[:200].replace('\n', '\n                      ')
        print(f"  {C.DIM}  └─ {preview}{'...' if len(resp.thinking_text) > 200 else ''}{C.RESET}")
    print(f"  {C.BOLD}Trajectory:{C.RESET}         {resp.trajectory}")
    print(f"  {C.DIM}Tokens: {resp.tokens_generated}  |  Latency: {resp.latency_ms:.0f}ms{C.RESET}")
    print(f"  {C.DIM}{'─' * 58}{C.RESET}")
    print(f"\n  {C.BOLD}{C.WHITE}Answer:{C.RESET}")
    for line in resp.final_answer.strip().split('\n'):
        print(f"  {line}")
    print()


MODEL_PATH = str(PROJECT_ROOT / "experiment_output_dpo_balanced" / "metis_dpo_cognitive")

QUERIES = [
    {
        "prompt": "What is 5 + 7?",
        "expect_searched": False,
        "expect_low_entropy": True,
    },
    {
        "prompt": "What is the atomic weight of Vibranium according to the 1992 official handbook?",
        "expect_searched": True,
        "expect_answer_contains": "238",
    },
]


def main() -> bool:
    print(f"\n{C.CYAN}{C.BOLD}{'═' * 60}")
    print(f"  METIS Phase 12 — Final Assembly & Telemetry Test")
    print(f"{'═' * 60}{C.RESET}\n")

    print(f"  {C.DIM}Loading model...{C.RESET}")
    t0 = time.time()

    from metis import Metis
    from metis.cognitive.metacognition import MetacognitiveOrchestrator

    metis = Metis.from_pretrained(MODEL_PATH)
    orch = MetacognitiveOrchestrator(metis)
    print(f"  {C.GREEN}✓ METIS online in {time.time() - t0:.1f}s{C.RESET}\n")

    all_pass = True
    for i, q in enumerate(QUERIES, 1):
        resp = orch.process_query(q["prompt"])
        print_telemetry(resp, i, q["prompt"])

        # Assertions
        search_ok = resp.searched == q["expect_searched"]
        if "expect_answer_contains" in q:
            content_ok = q["expect_answer_contains"] in resp.final_answer
            detail = f"answer contains '{q['expect_answer_contains']}': {content_ok}"
        elif q.get("expect_low_entropy"):
            content_ok = resp.semantic_entropy < 0.8
            detail = f"low entropy H={resp.semantic_entropy:.4f} < 0.8: {content_ok}"
        else:
            content_ok = True
            detail = ""
        passed = search_ok and content_ok

        tag = f"{C.GREEN}PASS ✓{C.RESET}" if passed else f"{C.RED}FAIL ✗{C.RESET}"
        print(f"  [{tag}] searched={resp.searched} (expect={q['expect_searched']}), {detail}")
        if not passed:
            all_pass = False

    print(f"\n{C.BOLD}{'═' * 60}")
    n = sum(1 for q in QUERIES if True)  # total
    tag = f"{C.GREEN}ALL PASS{C.RESET}" if all_pass else f"{C.RED}SOME FAILED{C.RESET}"
    print(f"  Result: {tag}")
    print(f"{'═' * 60}{C.RESET}\n")
    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
