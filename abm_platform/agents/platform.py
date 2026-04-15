"""
PlatformAgent — represents a B2B procurement platform.

Two instances (A and B) compete for member firms.  A platform tracks
member firms, trade volume routed through it, market share, and
commission revenue.  Platform operational costs use a scale-economy
model: OpEx = fixed + var × n × 1/(1+ln(n)).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mesa import Agent

if TYPE_CHECKING:
    from abm_platform.model import SupplyChainModel


class PlatformAgent(Agent):
    """Mesa agent for a B2B procurement platform."""

    def __init__(
        self,
        model: "SupplyChainModel",
        *,
        platform_id: str,
    ):
        super().__init__(model)
        self.platform_id = platform_id

        # Members (immediate — no setup delay)
        self.member_firms: set[int] = set()

        # Quarter metrics
        self.quarter_commission: float = 0.0
        self.market_share: float = 0.0
        self.trade_volume: float = 0.0
        self.quarter_opex: float = 0.0
        self.quarter_profit: float = 0.0

        # Quarter-level join/leave counters
        self.quarter_joined: int = 0
        self.quarter_left: int = 0

    # ── Public API ───────────────────────────────────────────────

    def add_member(self, firm_id: int):
        """Add *firm_id* to the platform (immediate, no delay)."""
        if firm_id in self.member_firms:
            return
        self.member_firms.add(firm_id)
        self.quarter_joined += 1

    def remove_member(self, firm_id: int):
        self.member_firms.discard(firm_id)
        self.quarter_left += 1

    def is_active_member(self, firm_id: int) -> bool:
        """True if firm is a member."""
        return firm_id in self.member_firms

    @property
    def member_count(self) -> int:
        return len(self.member_firms)

    @property
    def total_count(self) -> int:
        return len(self.member_firms)

    # ── Step ─────────────────────────────────────────────────────

    def step(self):
        """Called once per quarter after all CompanyAgents have stepped."""
        self._compute_trade_volume()
        self._compute_market_share()
        self._compute_opex()

    def _compute_trade_volume(self):
        """Sum trade volume and compute commission revenue (tiered per-edge)."""
        from abm_platform.config import tiered_commission_rate

        cfg = self.model.config
        pid = self.platform_id
        channel_str = f"platform_{pid.lower()}"

        tiers = cfg.commission_tiers_a if pid == "A" else cfg.commission_tiers_b

        total_volume = 0.0
        total_commission = 0.0

        for agent in self.model.firm_agents.values():
            for sup_id, channel in agent.edge_channels.items():
                if channel != channel_str:
                    continue
                sup_agent = self.model.firm_agents.get(sup_id)
                if sup_agent is None:
                    continue

                graph = self.model.supply_graph
                if not graph.has_edge(sup_id, agent.firm_id):
                    continue
                edge = graph[sup_id][agent.firm_id]

                edge_material = 0.0
                for product in edge.get("products", []):
                    qty = agent.quarterly_demand.get(product, 0.0)
                    if qty <= 0:
                        raw_qty = sup_agent.quarterly_demand.get(product, 0.0)
                        if raw_qty <= 0:
                            continue
                        n_downstream = sum(
                            1 for _, _, d in graph.out_edges(sup_id, data=True)
                            if product in d.get("products", [])
                        )
                        qty = raw_qty / max(n_downstream, 1)
                    if qty <= 0:
                        continue
                    from abm_platform.environment.network import get_unit_cost
                    unit_price = get_unit_cost(
                        product, sup_agent.country, self.model.loaded_data
                    )
                    edge_material += unit_price * qty

                total_volume += edge_material
                if edge_material > 0:
                    # Layer A: tiered rate based on edge material cost
                    rate = tiered_commission_rate(edge_material, tiers)
                    total_commission += edge_material * rate

        self.trade_volume = total_volume
        self.quarter_commission = total_commission

    def _compute_market_share(self):
        """Platform's share of total supply chain trade volume."""
        grand_total = self.model.total_trade_volume
        self.market_share = self.trade_volume / grand_total if grand_total > 0 else 0.0

    def _compute_opex(self):
        """Compute platform operational costs with scale economies."""
        cfg = self.model.config
        n = self.member_count
        if n <= 0:
            self.quarter_opex = cfg.platform_fixed_opex
        else:
            scale_factor = 1.0 / (1.0 + math.log(n))
            self.quarter_opex = (
                cfg.platform_fixed_opex
                + cfg.platform_var_opex_per_member * n * scale_factor
            )
        self.quarter_profit = self.quarter_commission - self.quarter_opex

    def reset_quarter(self):
        """Reset per-quarter accumulators."""
        self.quarter_commission = 0.0
        self.trade_volume = 0.0
        self.quarter_opex = 0.0
        self.quarter_profit = 0.0
        self.quarter_joined = 0
        self.quarter_left = 0
