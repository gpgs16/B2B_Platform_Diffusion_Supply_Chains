"""
Analysis script for final report.
Generates focused figures and summary statistics.

Sensitivity experiments S1–S18 + Hypotheses H1, H2a/b, H3.
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

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.figsize": (8, 5),
})

BLUE = "#2196F3"
RED = "#E53935"
GREEN = "#43A047"
ORANGE = "#FB8C00"
GRAY = "#9E9E9E"
DARK = "#212121"
LIGHT_BG = "#FAFAFA"
PURPLE = "#7B1FA2"

ROI_HORIZON = 12  # Matches BASELINE in run_final.py


def load(filename):
    with open(os.path.join(OUT_DIR, filename)) as f:
        return json.load(f)


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def extract_adoption(data, sort_fn):
    """Extract sorted (level, mean, ci) from sensitivity data."""
    items = []
    for key, results in data.items():
        valid = [r for r in results if "error" not in r]
        total = [r["final_adoption_a"] + r["final_adoption_b"] for r in valid]
        m = np.mean(total)
        ci = 1.96 * np.std(total) / np.sqrt(len(total))
        items.append((sort_fn(key), m, ci))
    items.sort(key=lambda x: x[0])
    return zip(*items)


def extract_hhi(data, sort_fn):
    """Extract sorted (level, mean_hhi, ci_hhi) from sensitivity data."""
    items = []
    for key, results in data.items():
        valid = [r for r in results if "error" not in r]
        hhis = [r["final_hhi"] for r in valid]
        m = np.mean(hhis)
        ci = 1.96 * np.std(hhis) / np.sqrt(len(hhis))
        items.append((sort_fn(key), m, ci))
    items.sort(key=lambda x: x[0])
    return zip(*items)


# ── All 18 SA experiment definitions ──────────────────────────────────
# Each entry: (label, filename, sort_fn, xlabel, baseline_value, baseline_label)
SA_EXPERIMENTS = [
    ("Platform Cost %",     "s1_cost_level.json",        lambda k: int(k.split("_")[1]),                       "Platform Cost (% of Bilateral)", 30, "30%"),
    ("ROI Horizon",         "s2_roi_horizon.json",       lambda k: int(k.split("_")[1].replace("q", "")),      "ROI Payback Horizon (Q)",        12, "12Q"),
    ("Initial Seeding",     "s3_seeding.json",           lambda k: int(k.split("_")[1].replace("pct", "")),    "Initial Seeding (%)",             5, "5%"),
    ("Cooldown",            "s4_cooldown.json",           lambda k: int(k.split("_")[1].replace("q", "")),      "Cooldown (Q)",                    2, "2Q"),
    ("PO Cost",             "s5_po_cost.json",            lambda k: int(k.split("_")[1]),                       "PO Cost ($)",                   180, "$180"),
    ("Invoice Cost",        "s6_invoice_cost.json",       lambda k: float(k.split("_")[1]),                     "Invoice Cost ($)",              9.5, "$9.50"),
    ("Search Cost",         "s7_search_cost.json",        lambda k: int(k.split("_")[1].replace("k", "")),     "Search Cost ($K)",               15, "$15K"),
    ("Negotiation Cost",    "s8_negotiation_cost.json",   lambda k: int(k.split("_")[1].replace("k", "")),     "Negotiation Cost ($K)",           8, "$8K"),
    ("Audit Cost",          "s9_audit_cost.json",         lambda k: int(k.split("_")[1].replace("k", "")),     "Audit Cost ($K)",                30, "$30K"),
    ("Onboarding Cost",     "s10_onboarding_cost.json",   lambda k: int(k.split("_")[1].replace("k", "")),     "Onboarding Cost ($K)",           20, "$20K"),
    ("Strategic Hrs",       "s11_strategic_hours.json",   lambda k: int(k.split("_")[1].replace("h", "")),     "Strategic Mgmt Hours/Q",         312, "312h"),
    ("Non-Strategic Hrs",   "s12_nonstrat_hours.json",    lambda k: int(k.split("_")[1].replace("h", "")),     "Non-Strategic Mgmt Hours/Q",     104, "104h"),
    ("Manager Mult",        "s13_mgr_mult.json",          lambda k: float(k.split("_")[1].replace("x", "")),   "Manager Wage Multiplier",       2.0, "2.0×"),
    ("Scale Reference",     "s14_admin_scale_ref.json",   lambda k: float(k.split("_")[1].replace("k", "")),   "Admin Scale Reference ($K)",    5.0, "$5K"),
    ("PO/Inv Cap",          "s15_cap_po_invoice.json",    lambda k: float(k.split("_")[2]),                     "PO/Invoice Scale Cap",          3.0, "3.0"),
    ("Audit Cap",           "s16_cap_audit.json",         lambda k: float(k.split("_")[2]),                     "Audit Scale Cap",               2.0, "2.0"),
    ("Onboarding Cap",      "s17_cap_onboarding.json",    lambda k: float(k.split("_")[2]),                     "Onboarding Scale Cap",          1.5, "1.5"),
    ("Setup Cost",          "s18_setup_cost.json",        lambda k: int(k.split("_")[1].replace("k", "")),     "Platform Setup Cost ($K)",       50, "$50K"),
]


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Sensitivity Ranking (18 parameters)
# ══════════════════════════════════════════════════════════════════════

def fig_sensitivity_ranking():
    """Horizontal bar chart of parameter sensitivity (adoption range) — 18 params."""
    labels, ranges = [], []
    for label, fname, sort_fn, *_ in SA_EXPERIMENTS:
        try:
            data = load(fname)
        except FileNotFoundError:
            continue
        _, means, _ = extract_adoption(data, sort_fn)
        means = list(means)
        labels.append(label)
        ranges.append(max(means) - min(means))

    # Sort by range
    order = np.argsort(ranges)[::-1]
    labels = [labels[i] for i in order]
    ranges = [ranges[i] for i in order]

    colors = []
    for r in ranges:
        if r > 0.10:
            colors.append(RED)
        elif r > 0.01:
            colors.append(ORANGE)
        else:
            colors.append(GREEN)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.35)))
    y = range(n)
    bars = ax.barh(y, [r * 100 for r in ranges], color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Adoption Range (percentage points)")
    ax.set_title("Parameter Sensitivity Ranking (18 OFAT experiments)")
    ax.invert_yaxis()

    for i, (bar, r) in enumerate(zip(bars, ranges)):
        ax.text(bar.get_width() + 0.3, i, f"{r*100:.1f}pp",
                va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(ranges) * 100 * 1.25 if ranges else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig("fig1_sensitivity_ranking.png")


# ══════════════════════════════════════════════════════════════════════
# Generic dual-axis sensitivity figure (adoption + HHI)
# ══════════════════════════════════════════════════════════════════════

def _fig_sensitivity_dual(sa_index: int, fig_name: str, title: str,
                          ylim_adopt=(20, 90), ylim_hhi=(0.4, 1.1)):
    """Generic dual-axis plot for a single SA experiment."""
    label, fname, sort_fn, xlabel, baseline, bl_label = SA_EXPERIMENTS[sa_index][:6]
    data = load(fname)
    x, means, cis = extract_adoption(data, sort_fn)
    x, means, cis = list(x), list(means), list(cis)
    _, hhi_means, hhi_cis = extract_hhi(data, sort_fn)
    hhi_means, hhi_cis = list(hhi_means), list(hhi_cis)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Baseline line
    ax1.axvline(baseline, color=RED, linestyle="--", alpha=0.7, label=f"Baseline: {bl_label}")

    ax1.plot(x, [m * 100 for m in means], "o-", color=BLUE, linewidth=2, markersize=8,
             label="Total Adoption")
    ax1.fill_between(x, [(m - c) * 100 for m, c in zip(means, cis)],
                     [(m + c) * 100 for m, c in zip(means, cis)],
                     alpha=0.15, color=BLUE)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Total Platform Adoption (%)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_ylim(*ylim_adopt)
    ax1.grid(True, alpha=0.3)

    # HHI on right axis
    ax2 = ax1.twinx()
    ax2.plot(x, hhi_means, "s--", color=PURPLE, linewidth=2, markersize=7, label="HHI")
    ax2.fill_between(x, [m - c for m, c in zip(hhi_means, hhi_cis)],
                     [m + c for m, c in zip(hhi_means, hhi_cis)],
                     alpha=0.1, color=PURPLE)
    ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_ylabel("HHI (1.0 = monopoly)", color=PURPLE)
    ax2.tick_params(axis="y", labelcolor=PURPLE)
    ax2.set_ylim(*ylim_hhi)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    ax1.set_title(title)
    savefig(fig_name)


# ── Figures 2–5: Retained SA (S1–S4) ─────────────────────────────────

def fig_cost_sensitivity():
    _fig_sensitivity_dual(0, "fig2_cost_sensitivity.png",
                          "S1: Platform Cost Sensitivity — Coexistence Holds Across Full Range",
                          ylim_adopt=(20, 90))

def fig_roi_sensitivity():
    _fig_sensitivity_dual(1, "fig3_roi_sensitivity.png",
                          "S2: ROI Horizon Sensitivity — Coexistence Holds Across Full Range",
                          ylim_adopt=(35, 95))

def fig_seeding_sensitivity():
    _fig_sensitivity_dual(2, "fig_s3_seeding.png",
                          "S3: Initial Seeding Sensitivity",
                          ylim_adopt=(40, 75))

def fig_cooldown_sensitivity():
    """S4: Cooldown sensitivity with churn overlay."""
    data = load("s4_cooldown.json")
    sort_fn = lambda k: int(k.split("_")[1].replace("q", ""))
    x, means, cis = extract_adoption(data, sort_fn)
    x, means, cis = list(x), list(means), list(cis)

    # Extract total leaves (churn)
    leaves_items = []
    for key, results in data.items():
        valid = [r for r in results if "error" not in r]
        total_leaves = [r.get("total_leaves", 0) for r in valid]
        leaves_items.append((sort_fn(key), np.mean(total_leaves), 1.96 * np.std(total_leaves) / np.sqrt(len(total_leaves))))
    leaves_items.sort(key=lambda x_: x_[0])
    _, leave_means, leave_cis = zip(*leaves_items)
    leave_means, leave_cis = list(leave_means), list(leave_cis)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.axvline(2, color=RED, linestyle="--", alpha=0.7, label="Baseline: 2Q")
    ax1.plot(x, [m * 100 for m in means], "o-", color=BLUE, linewidth=2, markersize=8,
             label="Total Adoption")
    ax1.fill_between(x, [(m - c) * 100 for m, c in zip(means, cis)],
                     [(m + c) * 100 for m, c in zip(means, cis)],
                     alpha=0.15, color=BLUE)

    ax1.set_xlabel("Cooldown Period (Quarters)")
    ax1.set_ylabel("Total Platform Adoption (%)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_ylim(45, 65)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, leave_means, "s--", color=ORANGE, linewidth=2, markersize=7, label="Total Leaves (churn)")
    ax2.fill_between(x, [m - c for m, c in zip(leave_means, leave_cis)],
                     [m + c for m, c in zip(leave_means, leave_cis)],
                     alpha=0.1, color=ORANGE)
    ax2.set_ylabel("Total Firm Departures (40Q)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax1.set_title("S4: Cooldown Period Sensitivity")
    savefig("fig_s4_cooldown.png")


# ── Figures S5–S18: New SA group figures ──────────────────────────────

def fig_sa_group_a():
    """Group A: Transaction costs (S5 PO, S6 Invoice)."""
    _fig_sensitivity_dual(4, "fig_s5_po_cost.png",
                          "S5: PO Cost Sensitivity", ylim_adopt=(40, 75))
    _fig_sensitivity_dual(5, "fig_s6_invoice_cost.png",
                          "S6: Invoice Cost Sensitivity", ylim_adopt=(40, 75))

def fig_sa_group_b():
    """Group B: Relationship costs (S7–S10)."""
    _fig_sensitivity_dual(6, "fig_s7_search_cost.png",
                          "S7: Search Cost Sensitivity", ylim_adopt=(40, 75))
    _fig_sensitivity_dual(7, "fig_s8_negotiation_cost.png",
                          "S8: Negotiation Cost Sensitivity", ylim_adopt=(40, 75))
    _fig_sensitivity_dual(8, "fig_s9_audit_cost.png",
                          "S9: Audit Cost Sensitivity", ylim_adopt=(40, 75))
    _fig_sensitivity_dual(9, "fig_s10_onboarding_cost.png",
                          "S10: Onboarding Cost Sensitivity", ylim_adopt=(40, 75))

def fig_sa_group_c():
    """Group C: Management hours (S11–S13)."""
    _fig_sensitivity_dual(10, "fig_s11_strategic_hours.png",
                          "S11: Strategic Management Hours Sensitivity", ylim_adopt=(40, 75))
    _fig_sensitivity_dual(11, "fig_s12_nonstrat_hours.png",
                          "S12: Non-Strategic Management Hours Sensitivity", ylim_adopt=(40, 75))
    _fig_sensitivity_dual(12, "fig_s13_mgr_mult.png",
                          "S13: Manager Wage Multiplier Sensitivity", ylim_adopt=(40, 75))

def fig_sa_group_d():
    """Group D: Admin scaling (S14–S17)."""
    _fig_sensitivity_dual(13, "fig_s14_admin_scale_ref.png",
                          "S14: Admin Scale Reference Sensitivity", ylim_adopt=(40, 75))
    _fig_sensitivity_dual(14, "fig_s15_cap_po_invoice.png",
                          "S15: PO/Invoice Scale Cap Sensitivity", ylim_adopt=(40, 75))
    _fig_sensitivity_dual(15, "fig_s16_cap_audit.png",
                          "S16: Audit Scale Cap Sensitivity", ylim_adopt=(40, 75))
    _fig_sensitivity_dual(16, "fig_s17_cap_onboarding.png",
                          "S17: Onboarding Scale Cap Sensitivity", ylim_adopt=(40, 75))

def fig_sa_group_e():
    """Group E: Setup cost (S18)."""
    _fig_sensitivity_dual(17, "fig_s18_setup_cost.png",
                          "S18: Platform Setup Cost Sensitivity", ylim_adopt=(40, 75))


def fig_diagnostic_cost_breakdown():
    """Bar chart showing setup cost vs. total horizon savings per community."""
    diag = load("diagnostic.json")
    cs = diag["community_summary"]

    cids = sorted(cs.keys(), key=lambda k: int(k))
    communities = [f"C{c}" for c in cids]

    setup_costs = [cs[c]["avg_setup_cost"] for c in cids]
    horizon_savings = [cs[c]["avg_total_savings_q"] * ROI_HORIZON for c in cids]
    adoption_rates = [cs[c]["adoption_rate"] for c in cids]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(communities))
    width = 0.35

    bars_setup = ax.bar(x - width / 2, setup_costs, width, color=RED, alpha=0.8,
                        label="Setup Cost (one-time)", edgecolor="white")
    bars_savings = ax.bar(x + width / 2, horizon_savings, width, color=GREEN, alpha=0.8,
                          label=f"Total Savings ({ROI_HORIZON}Q horizon)", edgecolor="white")

    # Annotate adoption rates
    for i, (xi, rate) in enumerate(zip(x, adoption_rates)):
        ax.text(xi, max(setup_costs[i], horizon_savings[i]) + 500,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(communities)
    ax.set_ylabel("USD")
    ax.set_title("Setup Cost vs. Total Horizon Savings per Community (adoption rate annotated)")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add secondary info: wage index
    ax2 = ax.twinx()
    wages = [cs[c]["avg_wage_index"] for c in cids]
    ax2.plot(x, wages, "D--", color=ORANGE, markersize=6, linewidth=1.5,
             label="Wage Index", alpha=0.8)
    ax2.set_ylabel("Wage Index", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper left")

    savefig("fig4_diagnostic_cost_breakdown.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE: H1 — Coexistence Community Shares
# ══════════════════════════════════════════════════════════════════════

def fig_h1_coexistence():
    """Stacked bar showing A/B/bilateral split by community from H1."""
    h1 = load("h1_coexistence.json")
    valid = [r for r in h1 if "error" not in r]

    # Aggregate community shares across seeds
    comm_a, comm_b = {}, {}
    for r in valid:
        shares = r["final_community_shares"]
        for cid, platforms in shares.items():
            if cid not in comm_a:
                comm_a[cid] = []
                comm_b[cid] = []
            comm_a[cid].append(platforms.get("A", 0))
            comm_b[cid].append(platforms.get("B", 0))

    cids = sorted(comm_a.keys(), key=lambda k: int(k))
    labels = [f"C{c}" for c in cids]
    a_means = [np.mean(comm_a[c]) for c in cids]
    b_means = [np.mean(comm_b[c]) for c in cids]
    bilateral_means = [max(0, 1.0 - a - b) for a, b in zip(a_means, b_means)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))

    ax.bar(x, a_means, color=BLUE, alpha=0.8, label="Platform A", edgecolor="white")
    ax.bar(x, b_means, bottom=a_means, color=RED, alpha=0.8, label="Platform B", edgecolor="white")
    ax.bar(x, bilateral_means, bottom=[a + b for a, b in zip(a_means, b_means)],
           color=GRAY, alpha=0.4, label="Bilateral", edgecolor="white")

    # Annotate total adoption above each bar
    for i, (a, b, bi) in enumerate(zip(a_means, b_means, bilateral_means)):
        total_adopt = a + b
        ax.text(i, a + b + bi + 0.02, f"{total_adopt:.0%}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of Community Firms")
    ax.set_title("H1: Community-Level Platform Shares (gray = bilateral retention)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add HHI annotation
    hhis = [r["final_hhi"] for r in valid]
    ax.text(0.02, 0.95, f"Mean HHI = {np.mean(hhis):.2f} ± {np.std(hhis):.2f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT_BG, edgecolor=GRAY))

    savefig("fig5_h1_coexistence.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE: H2a — Seeding Sweep
# ══════════════════════════════════════════════════════════════════════

def fig_h2a_seeding_sweep():
    """H2a: Platform A's and B's share in target community vs seeding fraction."""
    h2a = load("h2a_seeding_sweep.json")

    targets = [11, 5, 4]
    seedings = [1, 2, 5, 10, 15]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for idx, target in enumerate(targets):
        ax = axes[idx]
        a_target_means, a_target_cis = [], []
        b_target_means, b_target_cis = [], []
        seed_labels = []

        for pct in seedings:
            key = f"C{target}_pct{pct}"
            results = h2a.get(key, [])
            valid = [r for r in results if "error" not in r]
            if not valid:
                continue

            a_in_target = []
            b_in_target = []
            for r in valid:
                shares = r["final_community_shares"]
                t_shares = shares.get(str(target), {})
                t_a = t_shares.get("A", 0)
                t_b = t_shares.get("B", 0)
                t_total = t_a + t_b + t_shares.get("bilateral", 0)
                a_in_target.append(t_a / max(t_total, 0.001))
                b_in_target.append(t_b / max(t_total, 0.001))

            a_target_means.append(np.mean(a_in_target))
            a_target_cis.append(1.96 * np.std(a_in_target) / np.sqrt(len(a_in_target)))
            b_target_means.append(np.mean(b_in_target))
            b_target_cis.append(1.96 * np.std(b_in_target) / np.sqrt(len(b_in_target)))
            seed_labels.append(f"{pct}%")

        x = np.arange(len(seed_labels))
        width = 0.35
        ax.bar(x - width / 2, a_target_means, width, color=BLUE, alpha=0.8,
               label=f"A in C{target}", edgecolor="white",
               yerr=a_target_cis, capsize=3)
        ax.bar(x + width / 2, b_target_means, width, color=RED, alpha=0.8,
               label=f"B in C{target}", edgecolor="white",
               yerr=b_target_cis, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(seed_labels)
        ax.set_xlabel("Seeding Fraction")
        ax.set_title(f"Target: C{target}")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("Share of Adopters in Target Community")
    fig.suptitle("H2a: Seeding Sweep — A (seeded) vs B (2pp cheaper) in Target Community", fontsize=13, y=1.02)
    plt.tight_layout()
    savefig("fig6_h2a_seeding_sweep.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE: H2b — Cost Gap Sweep
# ══════════════════════════════════════════════════════════════════════

def fig_h2b_cost_gap_sweep():
    """H2b: Platform A's share vs B's cost advantage."""
    h2b = load("h2b_cost_gap_sweep.json")

    targets = [11, 5, 4]
    gaps = [0, 1, 2, 3, 5]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for idx, target in enumerate(targets):
        ax = axes[idx]
        a_target_means, a_target_cis = [], []
        b_target_means = []
        gap_labels = []

        for gap in gaps:
            key = f"C{target}_gap{gap}pp"
            results = h2b.get(key, [])
            valid = [r for r in results if "error" not in r]
            if not valid:
                continue

            a_vals, b_vals = [], []
            for r in valid:
                shares = r["final_community_shares"]
                t_shares = shares.get(str(target), {})
                a_vals.append(t_shares.get("A", 0))
                b_vals.append(t_shares.get("B", 0))

            a_target_means.append(np.mean(a_vals))
            a_target_cis.append(1.96 * np.std(a_vals) / np.sqrt(len(a_vals)))
            b_target_means.append(np.mean(b_vals))
            gap_labels.append(f"{gap}pp")

        x = np.arange(len(gap_labels))
        width = 0.35
        ax.bar(x - width / 2, a_target_means, width, color=BLUE, alpha=0.8,
               label="A (seeded)", edgecolor="white", yerr=a_target_cis, capsize=3)
        ax.bar(x + width / 2, b_target_means, width, color=RED, alpha=0.8,
               label="B (cheaper)", edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(gap_labels)
        ax.set_xlabel("B's Cost Advantage (pp)")
        ax.set_title(f"Target: C{target}")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Platform Share in Target Community")
    fig.suptitle("H2b: Cost Gap Sweep — Seeding vs Price Competition", fontsize=13, y=1.02)
    plt.tight_layout()
    savefig("fig7_h2b_cost_gap_sweep.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE: H3 — Disruption Adoption Trajectories
# ══════════════════════════════════════════════════════════════════════

def fig_h3_disruption():
    """H3: Adoption trajectories with and without disruptions."""
    h3 = load("h3_disruption.json")

    scenarios = {
        "no_disruption": ("No Disruption", GRAY, "-"),
        "disruption_freq20_n3": ("Freq=20Q, N=3", BLUE, "-"),
        "disruption_freq20_n5": ("Freq=20Q, N=5", ORANGE, "--"),
        "disruption_freq10_n3": ("Freq=10Q, N=3", RED, "-"),
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, (label, color, ls) in scenarios.items():
        results = h3.get(name, [])
        valid = [r for r in results if "error" not in r]
        if not valid:
            continue

        # Build mean trajectory
        max_q = max(len(r["time_series"]) for r in valid)
        trajectories = np.full((len(valid), max_q), np.nan)
        for i, r in enumerate(valid):
            ts = r["time_series"]
            for q, rec in enumerate(ts):
                trajectories[i, q] = rec["Adoption_Rate_A"] + rec["Adoption_Rate_B"]

        mean_traj = np.nanmean(trajectories, axis=0)
        quarters = np.arange(len(mean_traj))

        ax.plot(quarters, mean_traj * 100, color=color, linestyle=ls,
                linewidth=2, label=label)

        # Final annotation — stagger vertically to avoid overlap
        final_mean = np.mean([r["final_adoption_a"] + r["final_adoption_b"] for r in valid])
        # Offset map: shift Freq=20Q,N=3 up and Freq=20Q,N=5 down
        y_offsets = {
            "no_disruption": 0,
            "disruption_freq20_n3": 8,
            "disruption_freq20_n5": -8,
            "disruption_freq10_n3": 0,
        }
        y_off = y_offsets.get(name, 0)
        ax.annotate(f"{final_mean * 100:.1f}%",
                    xy=(quarters[-1], mean_traj[-1] * 100),
                    xytext=(5, y_off), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold", va="center")

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Total Platform Adoption (%)")
    ax.set_title("H3: Disruption-Driven Adoption Acceleration")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    savefig("fig8_h3_disruption.png")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY STATS (for tables in report)
# ══════════════════════════════════════════════════════════════════════

def compute_summary():
    """Compute and save all summary statistics for the report."""
    summary = {}

    # Sensitivity ranges for all 18 experiments
    sens = {}
    for label, fname, sort_fn, *_ in SA_EXPERIMENTS:
        base = fname.replace(".json", "")
        try:
            data = load(fname)
        except FileNotFoundError:
            continue
        _, means, _ = extract_adoption(data, sort_fn)
        means = list(means)
        _, hhi_means, _ = extract_hhi(data, sort_fn)
        hhi_means = list(hhi_means)
        sens[base] = {
            "label": label,
            "min_adoption": round(min(means), 3),
            "max_adoption": round(max(means), 3),
            "range_pp": round((max(means) - min(means)) * 100, 1),
            "min_hhi": round(min(hhi_means), 3),
            "max_hhi": round(max(hhi_means), 3),
        }
    summary["sensitivity"] = sens

    # H1 summary
    h1 = load("h1_coexistence.json")
    valid = [r for r in h1 if "error" not in r]
    summary["h1"] = {
        "n_runs": len(valid),
        "adoption_a": f"{np.mean([r['final_adoption_a'] for r in valid]):.3f} ± {np.std([r['final_adoption_a'] for r in valid]):.3f}",
        "adoption_b": f"{np.mean([r['final_adoption_b'] for r in valid]):.3f} ± {np.std([r['final_adoption_b'] for r in valid]):.3f}",
        "total_adoption": f"{np.mean([r['final_adoption_a']+r['final_adoption_b'] for r in valid]):.3f} ± {np.std([r['final_adoption_a']+r['final_adoption_b'] for r in valid]):.3f}",
        "hhi": f"{np.mean([r['final_hhi'] for r in valid]):.3f} ± {np.std([r['final_hhi'] for r in valid]):.3f}",
    }

    # H2a summary
    h2a = load("h2a_seeding_sweep.json")
    h2a_summary = {}
    for key, results in h2a.items():
        valid = [r for r in results if "error" not in r]
        if valid:
            h2a_summary[key] = {
                "adoption_a": round(np.mean([r["final_adoption_a"] for r in valid]), 3),
                "adoption_b": round(np.mean([r["final_adoption_b"] for r in valid]), 3),
                "hhi": round(np.mean([r["final_hhi"] for r in valid]), 3),
            }
    summary["h2a_seeding_sweep"] = h2a_summary

    # H2b summary
    h2b = load("h2b_cost_gap_sweep.json")
    h2b_summary = {}
    for key, results in h2b.items():
        valid = [r for r in results if "error" not in r]
        if valid:
            h2b_summary[key] = {
                "adoption_a": round(np.mean([r["final_adoption_a"] for r in valid]), 3),
                "adoption_b": round(np.mean([r["final_adoption_b"] for r in valid]), 3),
                "hhi": round(np.mean([r["final_hhi"] for r in valid]), 3),
            }
    summary["h2b_cost_gap_sweep"] = h2b_summary

    # H3 summary
    h3 = load("h3_disruption.json")
    h3_summary = {}
    for key, results in h3.items():
        valid = [r for r in results if "error" not in r]
        if valid:
            total = [r["final_adoption_a"] + r["final_adoption_b"] for r in valid]
            h3_summary[key] = {
                "total_adoption": f"{np.mean(total):.3f} ± {np.std(total):.3f}",
            }
    summary["h3"] = h3_summary

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {os.path.join(OUT_DIR, 'summary.json')}")

    return summary


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures...")
    fig_sensitivity_ranking()
    fig_cost_sensitivity()
    fig_roi_sensitivity()
    fig_seeding_sensitivity()
    fig_cooldown_sensitivity()
    fig_sa_group_a()
    fig_sa_group_b()
    fig_sa_group_c()
    fig_sa_group_d()
    fig_sa_group_e()
    fig_diagnostic_cost_breakdown()
    fig_h1_coexistence()
    fig_h2a_seeding_sweep()
    fig_h2b_cost_gap_sweep()
    fig_h3_disruption()
    print("\nComputing summary statistics...")
    compute_summary()
    print("\nDone!")
