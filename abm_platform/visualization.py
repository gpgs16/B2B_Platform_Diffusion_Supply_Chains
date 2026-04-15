"""
Visualization: time-series charts (matplotlib) and force-directed
network graph (pyvis) for the procurement platform ABM.

Updated for binary channel allocation and platform economics.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

if TYPE_CHECKING:
    from abm_platform.model import SupplyChainModel


OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "abm_output",
)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Time-series charts ───────────────────────────────────────────────

def plot_adoption_curves(model: "SupplyChainModel", save: bool = True):
    df = model.datacollector.get_model_vars_dataframe()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Quarter"], df["Adoption_Rate_A"], "r-o", label="Platform A", markersize=4)
    ax.plot(df["Quarter"], df["Adoption_Rate_B"], "g-s", label="Platform B", markersize=4)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Adoption Rate (fraction of firms)")
    ax.set_title("Platform Adoption Curves")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if save:
        ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, "adoption_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_market_share(model: "SupplyChainModel", save: bool = True):
    df = model.datacollector.get_model_vars_dataframe()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Quarter"], df["Platform_A_Share"], "r-o", label="Platform A", markersize=4)
    ax.plot(df["Quarter"], df["Platform_B_Share"], "g-s", label="Platform B", markersize=4)
    bilateral = 1.0 - df["Platform_A_Share"] - df["Platform_B_Share"]
    ax.plot(df["Quarter"], bilateral, "b-^", label="Bilateral", markersize=4)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Market Share (fraction of trade volume)")
    ax.set_title("Channel Market Share Over Time")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if save:
        ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, "market_share.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_platform_cash(model: "SupplyChainModel", save: bool = True):
    """Plot cumulative platform profit over time."""
    df = model.datacollector.get_model_vars_dataframe()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Quarter"], df["Platform_A_Profit"].cumsum() / 1e3, "r-o", label="Platform A", markersize=4)
    ax.plot(df["Quarter"], df["Platform_B_Profit"].cumsum() / 1e3, "g-s", label="Platform B", markersize=4)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5, label="Break-even")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Cumulative Profit (USD thousands)")
    ax.set_title("Platform Cumulative Profit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save:
        ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, "platform_cash.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_trade_volume(model: "SupplyChainModel", save: bool = True):
    df = model.datacollector.get_model_vars_dataframe()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Quarter"], df["Total_Trade_Volume"] / 1e6, "k-o", label="Total Trade", markersize=4)
    ax.plot(df["Quarter"], df["Platform_A_Volume"] / 1e6, "r-s", label="Platform A", markersize=4)
    ax.plot(df["Quarter"], df["Platform_B_Volume"] / 1e6, "g-^", label="Platform B", markersize=4)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Trade Volume (USD millions)")
    ax.set_title("Trade Volume Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save:
        ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, "trade_volume.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_supplier_switches(model: "SupplyChainModel", save: bool = True):
    df = model.datacollector.get_model_vars_dataframe()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["Quarter"], df["Supplier_Switches"], color="#F4B400", alpha=0.8)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Supplier Switches")
    ax.set_title("Supplier Switches Per Quarter")
    ax.grid(True, alpha=0.3, axis="y")
    if save:
        ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, "supplier_switches.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_membership_flow(model: "SupplyChainModel", save: bool = True):
    """Firms joining / leaving each platform per quarter."""
    df = model.datacollector.get_model_vars_dataframe()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    ax1.bar(df["Quarter"], df["Firms_Joined_A"], color="#EA4335", alpha=0.7, label="Joined")
    ax1.bar(df["Quarter"], -df["Firms_Left_A"], color="#EA433580", alpha=0.5, label="Left")
    ax1.set_title("Platform A Membership Flow")
    ax1.set_xlabel("Quarter")
    ax1.set_ylabel("Firms")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(df["Quarter"], df["Firms_Joined_B"], color="#34A853", alpha=0.7, label="Joined")
    ax2.bar(df["Quarter"], -df["Firms_Left_B"], color="#34A85380", alpha=0.5, label="Left")
    ax2.set_title("Platform B Membership Flow")
    ax2.set_xlabel("Quarter")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, "membership_flow.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ── Force-directed network ───────────────────────────────────────────

_CHANNEL_COLORS = {
    "bilateral":  "#4285F4",  # blue
    "platform_a": "#EA4335",  # red
    "platform_b": "#34A853",  # green
}


def _node_color(agent) -> str:
    """Compute node colour based on platform membership."""
    from abm_platform.agents.company import CompanyAgent
    if not isinstance(agent, CompanyAgent):
        return "#888888"
    if "A" in agent.platform_memberships:
        return "#EA4335"  # red — Platform A
    if "B" in agent.platform_memberships:
        return "#34A853"  # green — Platform B
    return "#4285F4"  # blue — no platform


def build_network_html(
    model: "SupplyChainModel",
    filename: str = "supply_network.html",
) -> str:
    """Build an interactive pyvis force-directed graph and save as HTML."""
    ensure_output_dir()
    G = model.supply_graph

    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#333333",
    )
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.04,
    )

    for fid in G.nodes:
        agent = model.firm_agents.get(fid)
        if agent is None:
            continue

        color = _node_color(agent)
        total_demand = sum(agent.quarterly_demand.values()) if agent.quarterly_demand else 1
        size = max(8, min(40, 5 + 5 * (total_demand ** 0.3)))

        label = agent.firm_name[:25] if agent.firm_name else str(fid)
        role = "OEM" if agent.is_oem else f"Tier {agent.tier_depth}"

        # In-degree (number of suppliers) from the graph
        in_deg = G.in_degree(fid)

        # Last join quarter per platform
        join_parts = []
        for pid in sorted(agent.join_quarter):
            join_parts.append(f"{pid} Q{agent.join_quarter[pid]}")
        join_str = ", ".join(join_parts) if join_parts else "never"

        lines = [
            agent.firm_name,
            f"{role} | C{agent.community_id} | in-degree {in_deg}",
            f"Joined: {join_str}",
        ]
        if agent.join_reason:
            lines.append(agent.join_reason)
        title = '<br>'.join(lines)

        net.add_node(
            fid,
            label=label,
            title=title,
            color=color,
            size=size,
            shape="dot" if not agent.is_oem else "star",
        )

    for src, tgt, data in G.edges(data=True):
        tgt_agent = model.firm_agents.get(tgt)
        if tgt_agent is None:
            continue

        channel = tgt_agent.edge_channels.get(src, "bilateral")
        edge_color = _CHANNEL_COLORS.get(channel, "#cccccc")
        width = 1 if channel == "bilateral" else 3

        products = data.get("products", [])
        title = f"Channel: {channel}<br>Products: {', '.join(products[:5])}"
        if len(products) > 5:
            title += f" (+{len(products) - 5} more)"

        net.add_edge(src, tgt, color=edge_color, width=width, title=title, arrows="to")

    outpath = os.path.join(OUTPUT_DIR, filename)
    net.save_graph(outpath)
    return outpath


# ── Summary dashboard ────────────────────────────────────────────────

def generate_all_plots(model: "SupplyChainModel"):
    """Generate all visualisation outputs."""
    ensure_output_dir()
    plot_adoption_curves(model)
    plot_market_share(model)
    plot_platform_cash(model)
    plot_trade_volume(model)
    plot_supplier_switches(model)
    plot_membership_flow(model)
    html_path = build_network_html(model)
    print(f"Plots saved to {OUTPUT_DIR}/")
    print(f"Network graph: {html_path}")
