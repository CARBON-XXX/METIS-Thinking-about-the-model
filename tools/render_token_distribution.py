#!/usr/bin/env python3
"""
Phase 24.2: Token Distribution Proof — Bimodal Evidence for METIS Dynamic Routing.

Defends against the "short-answer cheating" hypothesis by mathematically proving
that METIS produces a BIMODAL token distribution (FAST vs DEEP routes), not a
uniformly truncated one.

Output: paper/figures/fig2_bimodal_distribution.pdf
"""
from __future__ import annotations

import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
MODEL_PATH = "experiment_output_dpo_balanced/metis_dpo_cognitive"
OUTPUT_FIG = PROJECT_ROOT / "paper" / "figures" / "fig2_bimodal_distribution.pdf"
OUTPUT_DATA = PROJECT_ROOT / "paper" / "data" / "token_distribution.json"

logger = logging.getLogger("metis.phase24.2")


# ═══════════════════════════════════════════════════════════════════
# Dataset (identical to Phase 24.1 V2)
# ═══════════════════════════════════════════════════════════════════

def _build_mixed_dataset(n: int = 100) -> List[Dict[str, Any]]:
    """Identical dataset builder from Phase 24.1 for reproducibility."""
    random.seed(SEED)
    questions: List[Dict[str, Any]] = []

    math_templates = [
        ("A store sells apples for ${p} each. If someone buys {n} apples and pays with a ${t} bill, how much change?",
         lambda p, n, t: str(t - p * n), {"p": (1, 5), "n": (2, 10), "t": (20, 50)}),
        ("A train travels at {s} km/h for {h} hours. Distance in km?",
         lambda s, h: str(s * h), {"s": (40, 120), "h": (1, 5)}),
        ("A rectangle has length {l}cm and width {w}cm. What is its area?",
         lambda l, w: str(l * w), {"l": (3, 20), "w": (2, 15)}),
        ("{a} + {b} × {c} = ?",
         lambda a, b, c: str(a + b * c), {"a": (1, 50), "b": (2, 10), "c": (3, 12)}),
        ("If {n} people split a ${t} bill equally, how much does each pay?",
         lambda n, t: str(round(t / n, 2)), {"n": (2, 8), "t": (40, 200)}),
    ]

    for i in range(50):
        tmpl, fn, ranges = math_templates[i % len(math_templates)]
        params = {k: random.randint(*v) for k, v in ranges.items()}
        q = tmpl.format(**params)
        a = fn(**params)
        questions.append({
            "question": q, "answer": a, "type": "math",
            "difficulty": "complex" if i % 3 == 0 else "simple",
        })

    qa_items = [
        ("What is the chemical symbol for gold?", "Au"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What planet is known as the Red Planet?", "Mars"),
        ("What is the boiling point of water in Celsius?", "100"),
        ("What is the largest organ in the human body?", "skin"),
        ("In what year did World War II end?", "1945"),
        ("What is the speed of light in km/s approximately?", "300000"),
        ("What gas do plants absorb from the atmosphere?", "carbon dioxide"),
        ("How many sides does a hexagon have?", "6"),
        ("What is the capital of Japan?", "Tokyo"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the smallest prime number?", "2"),
        ("What element has the atomic number 1?", "Hydrogen"),
        ("How many continents are there?", "7"),
        ("What is the longest river in the world?", "Nile"),
        ("What is the freezing point of water in Fahrenheit?", "32"),
        ("Who discovered penicillin?", "Fleming"),
        ("What is the square root of 144?", "12"),
        ("What is the chemical formula for table salt?", "NaCl"),
        ("Which planet is closest to the Sun?", "Mercury"),
        ("What is the powerhouse of the cell?", "mitochondria"),
        ("How many bones in the adult human body?", "206"),
        ("What is the chemical symbol for iron?", "Fe"),
        ("Who developed the theory of relativity?", "Einstein"),
        ("What is the largest ocean?", "Pacific"),
    ]
    for i in range(50):
        q, a = qa_items[i % len(qa_items)]
        questions.append({"question": q, "answer": a, "type": "qa", "difficulty": "simple"})

    random.shuffle(questions)
    return questions[:n]


def _fmt(question: str, sys: Optional[str] = None) -> str:
    """ChatML format (identical to Phase 24.1)."""
    p = []
    if sys:
        p.append(f"<|im_start|>system\n{sys}<|im_end|>")
    p.append(f"<|im_start|>user\n{question}<|im_end|>")
    p.append("<|im_start|>assistant\n")
    return "\n".join(p)


# ═══════════════════════════════════════════════════════════════════
# Step 1: Generate per-request token lengths via vLLM
# ═══════════════════════════════════════════════════════════════════

def collect_token_lengths(model_path: str) -> Tuple[List[int], List[str], List[Dict[str, Any]]]:
    """
    Run METIS Dynamic prompts through vLLM and collect per-request token counts.
    Returns: (token_lengths, route_labels, per_request_details)
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")

    from vllm import LLM, SamplingParams

    logger.info(f"  vLLM init: {model_path}")
    llm = LLM(
        model=model_path, trust_remote_code=True, dtype="bfloat16",
        gpu_memory_utilization=0.85, max_num_seqs=256, seed=SEED, enforce_eager=True,
    )
    logger.info("  vLLM engine ready")

    dataset = _build_mixed_dataset(100)
    metis_sys = (
        "You are a precise AI. For complex questions, think step by step inside "
        "<thinking>...</thinking> tags before answering. For simple factual "
        "questions, answer directly without thinking tags."
    )

    logger.info("  Generating 100 METIS Dynamic prompts (deterministic, temp=0)")
    t0 = time.time()
    outputs = llm.generate(
        [_fmt(it["question"], sys=metis_sys) for it in dataset],
        SamplingParams(max_tokens=512, temperature=0),
    )
    wall = time.time() - t0
    logger.info(f"  Generation complete: {wall:.1f}s")

    token_lengths: List[int] = []
    route_labels: List[str] = []
    details: List[Dict[str, Any]] = []

    for o, it in zip(outputs, dataset):
        txt = o.outputs[0].text
        nt = len(o.outputs[0].token_ids)
        token_lengths.append(nt)

        # Route classification (same logic as Phase 24.1 S4)
        if "<thinking>" in txt.lower():
            route = "DEEP"
        elif nt < 30:
            route = "FAST"
        else:
            route = "NORMAL"
        route_labels.append(route)

        details.append({
            "question": it["question"],
            "type": it["type"],
            "difficulty": it["difficulty"],
            "token_count": nt,
            "route": route,
            "has_thinking_tags": "<thinking>" in txt.lower(),
            "response_preview": txt[:120],
        })

    del llm
    gc.collect()
    return token_lengths, route_labels, details


# ═══════════════════════════════════════════════════════════════════
# Step 2: Mathematical bimodality proof via Gaussian Mixture Model
# ═══════════════════════════════════════════════════════════════════

def prove_bimodality(token_lengths: List[int]) -> Dict[str, Any]:
    """
    Fit a 2-component GMM and compute bimodality metrics.
    Returns proof dict with GMM parameters, BIC comparison, and Hartigan's dip test.
    """
    from sklearn.mixture import GaussianMixture

    X = np.array(token_lengths).reshape(-1, 1)

    # Fit k=1 and k=2 GMMs, compare via BIC (lower = better)
    gmm1 = GaussianMixture(n_components=1, random_state=SEED).fit(X)
    gmm2 = GaussianMixture(n_components=2, random_state=SEED).fit(X)

    bic1 = gmm1.bic(X)
    bic2 = gmm2.bic(X)
    delta_bic = bic1 - bic2  # positive → 2-component is better

    # Extract GMM-2 parameters
    means = gmm2.means_.flatten()
    stds = np.sqrt(gmm2.covariances_.flatten())
    weights = gmm2.weights_.flatten()

    # Sort by mean (peak1 = lower tokens, peak2 = higher tokens)
    order = np.argsort(means)
    means = means[order]
    stds = stds[order]
    weights = weights[order]

    # Bimodality Coefficient (BC)
    # BC = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)*(n-3)))
    # BC > 0.555 → bimodal
    from scipy.stats import skew, kurtosis as scipy_kurtosis
    n = len(token_lengths)
    s = skew(token_lengths)
    k = scipy_kurtosis(token_lengths, fisher=False)  # excess=False → Pearson's
    bc = (s ** 2 + 1) / k
    bimodal_by_bc = bc > 0.555

    # Hartigan's Dip Test (if diptest is available)
    dip_stat, dip_p = None, None
    try:
        import diptest
        dip_stat, dip_p = diptest.diptest(np.array(token_lengths, dtype=np.float64))
    except ImportError:
        logger.warning("  diptest not installed — skipping Hartigan's dip test")

    proof = {
        "gmm_2_component": {
            "peak_1_mean": round(float(means[0]), 1),
            "peak_1_std": round(float(stds[0]), 1),
            "peak_1_weight": round(float(weights[0]), 4),
            "peak_2_mean": round(float(means[1]), 1),
            "peak_2_std": round(float(stds[1]), 1),
            "peak_2_weight": round(float(weights[1]), 4),
        },
        "bic_comparison": {
            "bic_unimodal": round(float(bic1), 2),
            "bic_bimodal": round(float(bic2), 2),
            "delta_bic": round(float(delta_bic), 2),
            "bimodal_preferred": delta_bic > 0,
            "note": "ΔBIC > 10 is 'very strong' evidence (Kass & Raftery, 1995)",
        },
        "bimodality_coefficient": {
            "BC": round(float(bc), 4),
            "threshold": 0.555,
            "bimodal": bimodal_by_bc,
        },
        "hartigans_dip_test": {
            "dip_statistic": round(float(dip_stat), 6) if dip_stat is not None else None,
            "p_value": round(float(dip_p), 6) if dip_p is not None else None,
            "reject_unimodal": dip_p < 0.05 if dip_p is not None else None,
        },
        "descriptive_stats": {
            "n": n,
            "mean": round(float(np.mean(token_lengths)), 1),
            "median": round(float(np.median(token_lengths)), 1),
            "std": round(float(np.std(token_lengths)), 1),
            "min": int(np.min(token_lengths)),
            "max": int(np.max(token_lengths)),
        },
    }

    logger.info(f"  GMM Peak 1: μ={means[0]:.1f} σ={stds[0]:.1f} w={weights[0]:.2f}")
    logger.info(f"  GMM Peak 2: μ={means[1]:.1f} σ={stds[1]:.1f} w={weights[1]:.2f}")
    logger.info(f"  ΔBIC = {delta_bic:.1f} (bimodal preferred: {delta_bic > 0})")
    logger.info(f"  BC = {bc:.4f} (bimodal: {bimodal_by_bc})")
    if dip_p is not None:
        logger.info(f"  Dip test: D={dip_stat:.6f}, p={dip_p:.6f} (reject unimodal: {dip_p < 0.05})")

    return proof


# ═══════════════════════════════════════════════════════════════════
# Step 3: Render publication-quality figure
# ═══════════════════════════════════════════════════════════════════

def render_figure(
    token_lengths: List[int],
    route_labels: List[str],
    proof: Dict[str, Any],
    output_path: Path,
) -> None:
    """Render KDE + Histogram with bimodal annotations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import norm

    sns.set_theme(style="whitegrid", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(10, 6))

    data = np.array(token_lengths)
    fast_mask = np.array([r == "FAST" for r in route_labels])
    deep_mask = np.array([r in ("DEEP", "NORMAL") for r in route_labels])

    # Histogram (stacked by route)
    bins = np.linspace(0, max(token_lengths) + 20, 40)
    ax.hist(data[fast_mask], bins=bins, alpha=0.5, color="#2196F3",
            label="FAST Route", edgecolor="white", linewidth=0.5)
    ax.hist(data[deep_mask], bins=bins, alpha=0.5, color="#FF5722",
            label="DEEP/NORMAL Route", edgecolor="white", linewidth=0.5)

    # KDE overlay (full distribution)
    sns.kdeplot(data, ax=ax, color="black", linewidth=2.5, label="KDE (all requests)")

    # GMM component curves
    gmm = proof["gmm_2_component"]
    x_range = np.linspace(0, max(token_lengths) + 50, 500)

    # Component 1 (FAST)
    y1 = gmm["peak_1_weight"] * norm.pdf(x_range, gmm["peak_1_mean"], gmm["peak_1_std"])
    y1_scaled = y1 * len(data) * (bins[1] - bins[0])
    ax.plot(x_range, y1_scaled, "--", color="#1565C0", linewidth=1.8,
            label=f"GMM Component 1 (μ={gmm['peak_1_mean']:.0f})")

    # Component 2 (DEEP)
    y2 = gmm["peak_2_weight"] * norm.pdf(x_range, gmm["peak_2_mean"], gmm["peak_2_std"])
    y2_scaled = y2 * len(data) * (bins[1] - bins[0])
    ax.plot(x_range, y2_scaled, "--", color="#BF360C", linewidth=1.8,
            label=f"GMM Component 2 (μ={gmm['peak_2_mean']:.0f})")

    # Peak annotations
    peak1_y = max(y1_scaled) * 0.85
    peak2_y = max(y2_scaled) * 0.85

    ax.annotate(
        f"FAST Route\n(Low Entropy)\nμ={gmm['peak_1_mean']:.0f} tokens",
        xy=(gmm["peak_1_mean"], peak1_y),
        xytext=(gmm["peak_1_mean"] + 40, peak1_y + 2),
        fontsize=11, fontweight="bold", color="#1565C0",
        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#1565C0", alpha=0.9),
    )

    ax.annotate(
        f"DEEP Route\n(High Entropy)\nμ={gmm['peak_2_mean']:.0f} tokens",
        xy=(gmm["peak_2_mean"], peak2_y),
        xytext=(gmm["peak_2_mean"] + 50, peak2_y + 2),
        fontsize=11, fontweight="bold", color="#BF360C",
        arrowprops=dict(arrowstyle="->", color="#BF360C", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FBE9E7", edgecolor="#BF360C", alpha=0.9),
    )

    # Statistical proof text box
    bic = proof["bic_comparison"]
    bc = proof["bimodality_coefficient"]
    dip = proof["hartigans_dip_test"]

    stats_text = (
        f"Bimodality Proof\n"
        f"────────────────\n"
        f"ΔBIC = {bic['delta_bic']:.1f} (>10 = very strong)\n"
        f"BC = {bc['BC']:.4f} (>{bc['threshold']} = bimodal)\n"
    )
    if dip["dip_statistic"] is not None:
        stats_text += f"Dip p = {dip['p_value']:.4f} (<0.05 = reject H₀)"
    else:
        stats_text += "Dip test: N/A (diptest not installed)"

    ax.text(
        0.97, 0.97, stats_text, transform=ax.transAxes,
        fontsize=9, verticalalignment="top", horizontalalignment="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA",
                  edgecolor="#9E9E9E", alpha=0.95),
    )

    # Labels and title
    ax.set_xlabel("Generated Token Count per Request", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title(
        "METIS Dynamic Routing: Bimodal Token Distribution\n"
        "Evidence Against Short-Answer Cheating Hypothesis",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper center", fontsize=9, ncol=3, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Figure saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 70)
    logger.info("  PHASE 24.2: TOKEN DISTRIBUTION PROOF")
    logger.info("  Defending against short-answer cheating hypothesis")
    logger.info("=" * 70)

    t_start = time.time()

    # Step 1: Collect per-request token lengths
    logger.info("\n[Step 1] Collecting per-request token lengths via vLLM...")
    token_lengths, route_labels, details = collect_token_lengths(
        str(PROJECT_ROOT / MODEL_PATH)
    )

    route_counts = {r: route_labels.count(r) for r in ["FAST", "DEEP", "NORMAL"]}
    logger.info(f"  Routes: {route_counts}")
    logger.info(f"  Token range: {min(token_lengths)} — {max(token_lengths)}")
    logger.info(f"  Mean: {np.mean(token_lengths):.1f}, Median: {np.median(token_lengths):.1f}")

    # Step 2: Mathematical bimodality proof
    logger.info("\n[Step 2] Fitting GMM and computing bimodality metrics...")
    proof = prove_bimodality(token_lengths)

    # Step 3: Render figure
    logger.info("\n[Step 3] Rendering publication figure...")
    render_figure(token_lengths, route_labels, proof, OUTPUT_FIG)

    # Save data
    output_data = {
        "phase": "24.2",
        "title": "Token Distribution Proof — Bimodal Evidence",
        "model": MODEL_PATH,
        "n_requests": len(token_lengths),
        "token_lengths": token_lengths,
        "route_labels": route_labels,
        "route_counts": route_counts,
        "bimodality_proof": proof,
        "per_request_details": details,
        "figure_path": str(OUTPUT_FIG),
        "wall_time_s": round(time.time() - t_start, 2),
    }
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    logger.info(f"  Data saved: {OUTPUT_DATA}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 24.2 — BIMODAL PROOF COMPLETE")
    logger.info("=" * 70)
    gmm = proof["gmm_2_component"]
    bic = proof["bic_comparison"]
    bc = proof["bimodality_coefficient"]
    logger.info(f"  Peak 1 (FAST): μ={gmm['peak_1_mean']:.0f} tok, w={gmm['peak_1_weight']:.2f}")
    logger.info(f"  Peak 2 (DEEP): μ={gmm['peak_2_mean']:.0f} tok, w={gmm['peak_2_weight']:.2f}")
    logger.info(f"  ΔBIC = {bic['delta_bic']:.1f}  BC = {bc['BC']:.4f}")
    logger.info(f"  Verdict: {'BIMODAL ✓' if bic['bimodal_preferred'] and bc['bimodal'] else 'INCONCLUSIVE'}")
    logger.info(f"  Figure: {OUTPUT_FIG}")
    logger.info(f"  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
