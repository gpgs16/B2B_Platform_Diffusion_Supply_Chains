"""
Analysis for combined H2 figure: 2×2 grid.

  Top row:    Seeding sweep for C11, C5  (A in target, B in target, A elsewhere)
  Bottom row: Cost gap sweep for C11, C5

Reads: h2_combined_seeding.json, h2_combined_costgap.json
Saves: fig_h2_combined.png
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

BLUE = "#2196F3"
RED = "#E53935"
GREEN = "#43A047"

TARGET_COMMUNITIES = [11, 5]
SEEDING_PCTS = [2, 5, 10, 15, 30]
COST_GAPS = [0, 1, 2, 3, 5]


def load(filename):
    with open(os.path.join(OUT_DIR, filename)) as f:
        return json.load(f)


def _extract_shares(results, target_cid):
    """Extract A_in_target, B_in_target, A_elsewhere from a list of run results."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None

    a_target_vals = []
    b_target_vals = []
    a_elsewhere_vals = []

    for r in valid:
        shares = r["final_community_shares"]
        sizes = r.get("community_sizes", {})
        t_key = str(target_cid)
        t_shares = shares.get(t_key, {})

        # A and B fraction in target community
        a_target_vals.append(t_shares.get("A", 0))
        b_target_vals.append(t_shares.get("B", 0))

        # A elsewhere: weighted average of A's share across all non-target communities
        total_firms_elsewhere = 0
        weighted_a_elsewhere = 0.0
        for cid_str, c_shares in shares.items():
            if cid_str == t_key:
                continue
            c_size = sizes.get(cid_str, sizes.get(int(cid_str), 1))
            total_firms_elsewhere += c_size
            weighted_a_elsewhere += c_shares.get("A", 0) * c_size

        if total_firms_elsewhere > 0:
            a_elsewhere_vals.append(weighted_a_elsewhere / total_firms_elsewhere)
        else:
            a_elsewhere_vals.append(0.0)

    def stats(vals):
        m = np.mean(vals)
        ci = 1.96 * np.std(vals) / np.sqrt(len(vals))
        return m, ci

    return {
        "a_target": stats(a_target_vals),
        "b_target": stats(b_target_vals),
        "a_elsewhere": stats(a_elsewhere_vals),
    }


def main():
    h2a = load("h2_combined_seeding.json")
    h2b = load("h2_combined_costgap.json")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── Top row: Seeding Sweep ────────────────────────────────────
    for col, target in enumerate(TARGET_COMMUNITIES):
        ax = axes[0, col]
        a_t_m, a_t_ci = [], []
        b_t_m, b_t_ci = [], []
        a_e_m, a_e_ci = [], []
        labels = []

        for pct in SEEDING_PCTS:
            key = f"C{target}_pct{pct}"
            results = h2a.get(key, [])
            s = _extract_shares(results, target)
            if s is None:
                continue
            a_t_m.append(s["a_target"][0]);  a_t_ci.append(s["a_target"][1])
            b_t_m.append(s["b_target"][0]);  b_t_ci.append(s["b_target"][1])
            a_e_m.append(s["a_elsewhere"][0]); a_e_ci.append(s["a_elsewhere"][1])
            labels.append(f"{pct}%")

        x = np.arange(len(labels))
        w = 0.25
        ax.bar(x - w, a_t_m, w, color=BLUE, alpha=0.85, edgecolor="white",
               yerr=a_t_ci, capsize=3, label=f"A in C{target}")
        ax.bar(x, b_t_m, w, color=RED, alpha=0.85, edgecolor="white",
               yerr=b_t_ci, capsize=3, label=f"B in C{target}")
        ax.bar(x + w, a_e_m, w, color=GREEN, alpha=0.85, edgecolor="white",
               yerr=a_e_ci, capsize=3, label="A elsewhere")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("A's Seeding Fraction in Community")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Seeding Sweep — Community {target}")
        ax.legend(fontsize=8, loc="upper left")

    axes[0, 0].set_ylabel("Platform Adoption Share")

    # ── Bottom row: Cost Gap Sweep ────────────────────────────────
    for col, target in enumerate(TARGET_COMMUNITIES):
        ax = axes[1, col]
        a_t_m, a_t_ci = [], []
        b_t_m, b_t_ci = [], []
        a_e_m, a_e_ci = [], []
        labels = []

        for gap in COST_GAPS:
            key = f"C{target}_gap{gap}pp"
            results = h2b.get(key, [])
            s = _extract_shares(results, target)
            if s is None:
                continue
            a_t_m.append(s["a_target"][0]);  a_t_ci.append(s["a_target"][1])
            b_t_m.append(s["b_target"][0]);  b_t_ci.append(s["b_target"][1])
            a_e_m.append(s["a_elsewhere"][0]); a_e_ci.append(s["a_elsewhere"][1])
            labels.append(f"{gap}pp")

        x = np.arange(len(labels))
        w = 0.25
        ax.bar(x - w, a_t_m, w, color=BLUE, alpha=0.85, edgecolor="white",
               yerr=a_t_ci, capsize=3, label=f"A in C{target}")
        ax.bar(x, b_t_m, w, color=RED, alpha=0.85, edgecolor="white",
               yerr=b_t_ci, capsize=3, label=f"B in C{target}")
        ax.bar(x + w, a_e_m, w, color=GREEN, alpha=0.85, edgecolor="white",
               yerr=a_e_ci, capsize=3, label="A elsewhere")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("B's Cost Advantage (pp)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Cost Gap Sweep — Community {target}")
        ax.legend(fontsize=8, loc="upper right")

    axes[1, 0].set_ylabel("Platform Adoption Share")

    fig.suptitle(
        "H2: Strategic Seeding Lock-In\n"
        "Top: Seeding fraction sweep (B has 2pp cost advantage)  |  "
        "Bottom: Cost gap sweep (A seeded at 15%)",
        fontsize=13, y=1.03,
    )
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_h2_combined.png")
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
