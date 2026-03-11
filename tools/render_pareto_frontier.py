#!/usr/bin/env python3
"""
Render Fig 1: Pareto Frontier — Accuracy vs Token Cost for all strategies.
Input:  paper/data/pareto.json
Output: paper/figures/fig1_pareto_frontier.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "paper" / "data" / "pareto.json"
OUTPUT = PROJECT_ROOT / "paper" / "figures" / "fig1_pareto_frontier.pdf"


def main() -> None:
    d = json.load(open(DATA))
    strats = d["strategies"]

    labels = {
        "zero_shot": "Zero-Shot",
        "forced_cot": "Forced CoT",
        "self_consistency_k5": "Self-Consistency (k=5)",
        "metis_dynamic": "METIS Dynamic",
    }
    colors = {
        "zero_shot": "#9E9E9E",
        "forced_cot": "#2196F3",
        "self_consistency_k5": "#FF9800",
        "metis_dynamic": "#E91E63",
    }
    markers = {
        "zero_shot": "s",
        "forced_cot": "^",
        "self_consistency_k5": "D",
        "metis_dynamic": "*",
    }
    sizes = {
        "zero_shot": 120,
        "forced_cot": 120,
        "self_consistency_k5": 120,
        "metis_dynamic": 280,
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    xs, ys = [], []
    for key in ["zero_shot", "forced_cot", "self_consistency_k5", "metis_dynamic"]:
        s = strats[key]
        x = s["avg_tokens"]
        y = s["accuracy"] * 100
        xs.append(x)
        ys.append(y)
        ax.scatter(
            x, y,
            c=colors[key], marker=markers[key], s=sizes[key],
            edgecolors="black", linewidths=0.8, zorder=5,
            label=f'{labels[key]}  ({y:.0f}%, {x:.0f} tok)',
        )

    # Pareto frontier line (connect dominated-free points)
    pareto_pts = sorted(zip(xs, ys), key=lambda p: p[0])
    frontier_x, frontier_y = [pareto_pts[0][0]], [pareto_pts[0][1]]
    max_y = pareto_pts[0][1]
    for px, py in pareto_pts[1:]:
        if py >= max_y:
            frontier_x.append(px)
            frontier_y.append(py)
            max_y = py
    ax.plot(frontier_x, frontier_y, "--", color="#E91E63", alpha=0.4, linewidth=1.5, zorder=2)

    # Highlight METIS as Pareto-optimal
    metis = strats["metis_dynamic"]
    ax.annotate(
        "Pareto-optimal\n(highest accuracy,\nlowest token cost)",
        xy=(metis["avg_tokens"], metis["accuracy"] * 100),
        xytext=(metis["avg_tokens"] + 120, metis["accuracy"] * 100 - 5),
        fontsize=9, fontweight="bold", color="#880E4F",
        arrowprops=dict(arrowstyle="->", color="#880E4F", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FCE4EC", edgecolor="#880E4F", alpha=0.9),
    )

    # Efficiency iso-lines (accuracy/kTok)
    for eff in [5, 20, 50]:
        x_iso = np.linspace(1, 900, 200)
        y_iso = eff * x_iso / 1000 * 100  # convert to percentage scale
        valid = y_iso <= 100
        ax.plot(x_iso[valid], y_iso[valid], ":", color="#BDBDBD", linewidth=0.8, alpha=0.6)
        # Label the iso-line
        idx = np.searchsorted(y_iso[valid], 60) if 60 < y_iso[valid][-1] else len(y_iso[valid]) - 1
        idx = min(idx, len(x_iso[valid]) - 1)
        if idx > 0:
            ax.text(x_iso[valid][idx], y_iso[valid][idx] + 1.5,
                    f"{eff} acc/kTok", fontsize=7, color="#9E9E9E", rotation=15)

    ax.set_xlabel("Average Generated Tokens per Request", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Pareto Frontier: Accuracy vs. Computational Cost\nMETIS Dynamic Routing vs. Baseline Strategies",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(55, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
