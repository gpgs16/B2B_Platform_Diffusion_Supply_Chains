"""
CompanyAgent — represents every firm (OEM and suppliers) in the supply chain.

Key design changes from earlier version:
  - Binary channel allocation: if both parties on same platform ⇒ 100 % platform
  - Quality cost simplified: order (1 + defect_rate) × Q instead of separate cost
  - Lead time is a selection criterion for suppliers
  - Regional vs global supplier discovery (bilateral = regional only)
  - Profit-based platform join/leave decisions (ROI over payback horizon)
  - Onboarding time for new suppliers and platforms
  - Companies can leave or switch platforms
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mesa import Agent

from abm_platform.config import (
    COUNTRY_TO_REGION,
    HOURLY_COMPENSATION_USD,
    MARGIN_BY_REGION_TIER,
    CHINESE_OEM_MARGIN,
    R_HOLDING_QUARTERLY,
    SIGMA_DEMAND_FRAC,
    Z_SERVICE,
    volume_discount_multiplier,
    tiered_commission_rate,
    admin_scale_factor,
)
from abm_platform.environment.transport import (
    get_lead_time,
    transport_cost,
)
from abm_platform.environment.network import (
    get_unit_cost,
    import_duty_rate,
    get_moq,
)

if TYPE_CHECKING:
    from abm_platform.model import SupplyChainModel

# Default weight per unit in tons (used when real data unavailable)
DEFAULT_UNIT_WEIGHT_TONS = 0.005  # 5 kg average auto part


class CompanyAgent(Agent):
    """Mesa agent for a firm in the automotive supply chain."""

    def __init__(
        self,
        model: "SupplyChainModel",
        *,
        firm_id: int,
        firm_name: str,
        country: str,
        lat: float,
        lon: float,
        is_oem: bool,
        tier_depth: int,
        is_top_supplier: bool,
        certifications: list[str],
        product_catalog: list[str],
        defect_rate: float,
        is_eco_friendly: bool = False,
    ):
        super().__init__(model)
        self.firm_id = firm_id
        self.firm_name = firm_name
        self.country = country
        self.lat = lat
        self.lon = lon
        self.is_oem = is_oem
        self.tier_depth = tier_depth
        self.is_top_supplier = is_top_supplier
        self.certifications = certifications
        self.product_catalog = product_catalog
        self.defect_rate = defect_rate
        self.is_eco_friendly = is_eco_friendly

        # ── Runtime state ────────────────────────────────────────
        # Binary channel per supplier edge: {supplier_id: channel_str}
        # channel_str is one of: "bilateral", "platform_a", "platform_b"
        self.edge_channels: dict[int, str] = {}

        # Demand this quarter: {product: quantity}
        self.quarterly_demand: dict[str, float] = {}

        # Max capacity (infinite — no supplier rejection)
        self.max_capacity: float = float('inf')
        self.current_load: float = 0.0

        # Accumulated quarter cost / revenue for data collection
        self.quarter_cost: float = 0.0
        self.quarter_revenue: float = 0.0

        # Platform membership: set of platform_ids ("A", "B")
        self.platform_memberships: set[str] = set()

        # Pending supplier onboarding: {sup_id: quarter_when_ready}
        self.pending_onboarding: dict[int, int] = {}
        # Products waiting for new supplier onboarding: {product: quarter_when_ready}
        self.pending_delivery_products: dict[str, int] = {}

        # Track current quarter (set by model before step)
        self._quarter: int = 0

        # Louvain community id (set by model init, -1 = unassigned)
        self.community_id: int = -1

        # Adoption reason (for hover tooltip)
        self.join_reason: str = ""

        # Track when firm joined each platform
        self.join_quarter: dict[str, int] = {}  # {platform_id: quarter}

        # Detailed adoption log: list of dicts recording every join/leave event
        self.adoption_log: list[dict] = []

        # ── Willingness-to-Cooperate (WTC) state ────────────────
        self.wtc: dict[str, float] = {}               # current smoothed WTC per platform
        self.consecutive_quarters: dict[str, int] = {} # quarters on platform consecutively
        self.last_wtc: dict[str, float] = {}           # preserved WTC when leaving (for rejoin drag)
        self._last_leave_quarter: int = -999           # prevents immediate rejoin after leaving

    # ── Wage helpers ─────────────────────────────────────────────

    @property
    def _wage_staff(self) -> float:
        return HOURLY_COMPENSATION_USD.get(self.country, 7.0)

    @property
    def _wage_manager(self) -> float:
        base = HOURLY_COMPENSATION_USD.get(self.country, 7.0)
        return base * self.model.config.procurement_mgr_mult

    @property
    def _wage_index(self) -> float:
        """Ratio of local wage to US wage (for scaling flat-USD costs)."""
        local = HOURLY_COMPENSATION_USD.get(self.country, 7.0)
        us = HOURLY_COMPENSATION_USD["USA"]
        return local / us

    @property
    def margin(self) -> float:
        if self.is_oem:
            return CHINESE_OEM_MARGIN
        region_tiers = MARGIN_BY_REGION_TIER.get(self.country)
        if region_tiers:
            return region_tiers.get(self.tier_depth, region_tiers.get(1, 0.045))
        return 0.045

    # ── Channel determination (binary) ───────────────────────────

    def determine_channel(self, supplier_id: int) -> str:
        """
        Determine the procurement channel for a buyer-supplier edge.
        If both are active members of the same platform: 100 % platform.
        Otherwise: bilateral.
        """
        sup_agent = self.model.firm_agents.get(supplier_id)
        if sup_agent is None:
            return "bilateral"

        shared_platforms = []
        for pid in self.platform_memberships:
            plat = self.model.platform_agents.get(pid)
            if plat and plat.is_active_member(supplier_id):
                shared_platforms.append(pid)

        if not shared_platforms:
            return "bilateral"

        return f"platform_{shared_platforms[0].lower()}"

    # ── Cost calculations ────────────────────────────────────────

    def _supplier_level_cost(
        self, supplier: "CompanyAgent", is_new: bool, channel: str = "bilateral",
        material_cost: float = 0.0,
    ) -> float:
        """SLC: Relationship management + onboarding + negotiation + conditional audit.

        Scaling rules (per user-specified restructuring):
        - Management hours: full log-scale (no cap)
        - Onboarding: log-scaled, capped at admin_scale_cap_onboarding
        - Negotiation: full log-scale (no cap)
        - Audit: only for new suppliers with material≥threshold; capped scale
        """
        cfg = self.model.config
        is_strategic = supplier.is_top_supplier
        h_mgmt = cfg.h_mgmt_strategic if is_strategic else cfg.h_mgmt_non_strategic

        # Platform reduces management hours
        if channel == "platform_a":
            h_mgmt *= cfg.platform_mgmt_pct_a
        elif channel == "platform_b":
            h_mgmt *= cfg.platform_mgmt_pct_b

        # Scale management hours by edge value (full log-scaled, no cap)
        scale = admin_scale_factor(material_cost, cfg.admin_scale_reference)
        slc = self._wage_manager * h_mgmt * scale

        if is_new:
            # Onboarding (log-scaled, capped)
            onb_scale = min(scale, cfg.admin_scale_cap_onboarding)
            slc += cfg.onboarding_cost * self._wage_index * onb_scale

            # Negotiation cost (log-scaled, no cap; platform reduces base)
            if channel == "bilateral":
                slc += cfg.bilateral_negotiation_cost * self._wage_index * scale
            elif channel == "platform_a":
                slc += cfg.bilateral_negotiation_cost * cfg.platform_negotiation_pct_a * self._wage_index * scale
            else:
                slc += cfg.bilateral_negotiation_cost * cfg.platform_negotiation_pct_b * self._wage_index * scale

            # Conditional audit: only when material ≥ threshold
            if material_cost >= cfg.audit_material_threshold:
                audit_scale = min(scale, cfg.admin_scale_cap_audit)
                slc += cfg.audit_cost * audit_scale
        return slc

    def _order_level_cost(self, n_orders: int, channel: str,
                          material_cost: float = 0.0) -> float:
        """OLC: Per-order processing + invoicing. PO/invoice scale capped at 1.5×."""
        cfg = self.model.config
        wi = self._wage_index
        if channel == "platform_a":
            po = cfg.bilateral_po_cost * cfg.platform_po_pct_a
            inv = cfg.bilateral_invoice_cost * cfg.platform_invoice_pct_a
        elif channel == "platform_b":
            po = cfg.bilateral_po_cost * cfg.platform_po_pct_b
            inv = cfg.bilateral_invoice_cost * cfg.platform_invoice_pct_b
        else:
            po = cfg.bilateral_po_cost
            inv = cfg.bilateral_invoice_cost
        scale = min(
            admin_scale_factor(material_cost, cfg.admin_scale_reference),
            cfg.admin_scale_cap_po_invoice,
        )
        return n_orders * (po + inv) * wi * scale

    def _seller_admin_cost(self, n_orders: int, channel: str,
                           material_cost: float = 0.0) -> float:
        """Seller-side admin cost: mirrors _order_level_cost. PO/invoice scale capped at 1.5×."""
        cfg = self.model.config
        wi = self._wage_index
        if channel == "platform_a":
            po = cfg.bilateral_po_cost * cfg.platform_po_pct_a
            inv = cfg.bilateral_invoice_cost * cfg.platform_invoice_pct_a
        elif channel == "platform_b":
            po = cfg.bilateral_po_cost * cfg.platform_po_pct_b
            inv = cfg.bilateral_invoice_cost * cfg.platform_invoice_pct_b
        else:
            po = cfg.bilateral_po_cost
            inv = cfg.bilateral_invoice_cost
        scale = min(
            admin_scale_factor(material_cost, cfg.admin_scale_reference),
            cfg.admin_scale_cap_po_invoice,
        )
        return n_orders * (po + inv) * wi * scale

    def _mgmt_hours_saving(self, supplier: "CompanyAgent",
                           material_cost: float, channel_str: str) -> float:
        """Return ONLY the management-hours cost difference (bilateral − platform).

        Used for partial-digitization credit on non-platform partners,
        where PO/invoice savings are zero but mgmt improvement still
        applies one-sided.
        """
        cfg = self.model.config
        is_strategic = supplier.is_top_supplier
        h_mgmt = cfg.h_mgmt_strategic if is_strategic else cfg.h_mgmt_non_strategic
        scale = admin_scale_factor(material_cost, cfg.admin_scale_reference)
        mgmt_bi = self._wage_manager * h_mgmt * scale
        if channel_str == "platform_a":
            h_mgmt_pl = h_mgmt * cfg.platform_mgmt_pct_a
        else:
            h_mgmt_pl = h_mgmt * cfg.platform_mgmt_pct_b
        mgmt_pl = self._wage_manager * h_mgmt_pl * scale
        return mgmt_bi - mgmt_pl

    def _seller_mgmt_saving(self, material_cost: float,
                            channel_str: str) -> float:
        """Seller-side management-hours saving (bilateral − platform).

        Sellers don't have per-partner mgmt hours in the model; the
        seller-side saving from management overhead is approximated as
        the difference in management hours cost for a non-strategic
        supplier relationship (mirrors buyer-side logic).
        """
        cfg = self.model.config
        h_mgmt = cfg.h_mgmt_non_strategic
        scale = admin_scale_factor(material_cost, cfg.admin_scale_reference)
        mgmt_bi = self._wage_manager * h_mgmt * scale
        if channel_str == "platform_a":
            h_mgmt_pl = h_mgmt * cfg.platform_mgmt_pct_a
        else:
            h_mgmt_pl = h_mgmt * cfg.platform_mgmt_pct_b
        mgmt_pl = self._wage_manager * h_mgmt_pl * scale
        return mgmt_bi - mgmt_pl

    def _update_wtc(self, pid: str) -> None:
        """
        Update Willingness-to-Cooperate for platform *pid*.
        Called once per quarter for each active membership.

        Formula: w = (1/(t+1)) × benefit_t + (t/(t+1)) × wtc_prev
        where t = consecutive quarters on platform.

        benefit_t = realized savings on edges routed through the platform
        (buyer-side cost reduction + seller-side admin savings).
        """
        cfg = self.model.config
        graph = self.model.supply_graph
        channel = f"platform_{pid.lower()}"

        benefit_t = 0.0

        # Buyer-side savings: platform vs bilateral for edges routed through platform
        for sup_id in graph.predecessors(self.firm_id):
            if self.edge_channels.get(sup_id) != channel:
                continue
            sup_agent = self.model.firm_agents.get(sup_id)
            if sup_agent is None:
                continue
            edge = graph[sup_id][self.firm_id]
            for product in edge.get("products", []):
                qty = self.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    qty = sup_agent.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    continue
                cost_pl = self.total_sc_cost(sup_agent, product, qty, channel)
                cost_bi = self.total_sc_cost(sup_agent, product, qty, "bilateral")
                benefit_t += cost_bi - cost_pl

        # Seller-side admin savings from buyer edges on platform
        for buyer_id in graph.successors(self.firm_id):
            buyer_agent = self.model.firm_agents.get(buyer_id)
            if buyer_agent is None:
                continue
            if pid not in buyer_agent.platform_memberships:
                continue
            edge = graph[self.firm_id][buyer_id]
            for product in edge.get("products", []):
                qty = buyer_agent.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    qty = self.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    continue
                up = get_unit_cost(product, self.country, self.model.loaded_data)
                edge_mat = up * qty
                # Seller admin savings (PO + invoice + mgmt)
                n_orders = min(max(1, int(math.ceil(qty / cfg.order_batch_size))), cfg.max_orders_per_quarter)
                admin_bi = self._seller_admin_cost(n_orders, "bilateral",
                                                   material_cost=edge_mat)
                admin_pl = self._seller_admin_cost(n_orders, channel,
                                                   material_cost=edge_mat)
                benefit_t += admin_bi - admin_pl

        # Exponential smoothing
        t = self.consecutive_quarters.get(pid, 0)
        wtc_prev = self.wtc.get(pid, 0.0)
        weight_new = 1.0 / (t + 1)
        self.wtc[pid] = weight_new * benefit_t + (1.0 - weight_new) * wtc_prev
        self.consecutive_quarters[pid] = t + 1

    def _transport_cost(
        self, supplier: "CompanyAgent", quantity: float
    ) -> float:
        return transport_cost(
            supplier.lat, supplier.lon, supplier.country,
            self.lat, self.lon, self.country,
            DEFAULT_UNIT_WEIGHT_TONS, quantity,
        )

    def _duty_cost(
        self, supplier: "CompanyAgent", unit_price: float, quantity: float
    ) -> float:
        rate = import_duty_rate(supplier.country, self.country)
        return rate * unit_price * quantity

    def _inventory_holding_cost(
        self,
        supplier: "CompanyAgent",
        unit_price: float,
        quantity: float,
    ) -> float:
        lt = get_lead_time(
            supplier.country, self.country,
            lat1=supplier.lat, lon1=supplier.lon,
            lat2=self.lat, lon2=self.lon,
        )
        q_cycle = quantity / 2.0
        sigma = SIGMA_DEMAND_FRAC * quantity
        q_safety = Z_SERVICE * sigma * math.sqrt(lt)
        return R_HOLDING_QUARTERLY * unit_price * (q_cycle + q_safety)

    def _lead_time_weeks(self, supplier: "CompanyAgent") -> float:
        """Lead time in weeks (used as selection criterion)."""
        return get_lead_time(
            supplier.country, self.country,
            lat1=supplier.lat, lon1=supplier.lon,
            lat2=self.lat, lon2=self.lon,
        )

    def total_sc_cost(
        self,
        supplier: "CompanyAgent",
        product: str,
        quantity: float,
        channel: str,
        is_new: bool = False,
    ) -> float:
        """
        Total supply chain cost for procuring *quantity* of *product*
        from *supplier* via *channel*.

        Quality cost is handled by inflating quantity by (1 + defect_rate)
        to represent excess ordering needed to cover defective parts.
        """
        cfg = self.model.config
        data = self.model.loaded_data
        unit_price = get_unit_cost(product, supplier.country, data)

        moq = get_moq(product, data)
        disc = volume_discount_multiplier(int(quantity), moq)
        unit_price *= disc

        # Inflate quantity to cover defective parts (simplified quality cost)
        effective_qty = quantity * (1.0 + supplier.defect_rate)

        material_cost = unit_price * effective_qty

        slc = self._supplier_level_cost(supplier, is_new, channel,
                                         material_cost=material_cost)
        n_orders = min(max(1, int(math.ceil(quantity / cfg.order_batch_size))), cfg.max_orders_per_quarter)
        olc = self._order_level_cost(n_orders, channel,
                                     material_cost=material_cost)
        tc = self._transport_cost(supplier, effective_qty)
        duty = self._duty_cost(supplier, unit_price, effective_qty)
        ihc = self._inventory_holding_cost(supplier, unit_price, effective_qty)

        # Tiered commission (no growth discount)
        platform_fee = 0.0
        if channel.startswith("platform_"):
            cfg = self.model.config
            pid = channel[-1].upper()  # "a" or "b"
            tiers = cfg.commission_tiers_a if pid == "A" else cfg.commission_tiers_b
            rate = tiered_commission_rate(material_cost, tiers)
            platform_fee = material_cost * rate

        return material_cost + slc + olc + tc + duty + ihc + platform_fee

    # ── Step behaviour ───────────────────────────────────────────

    def step(self):
        """Called once per quarter by the model scheduler."""
        # Skip all trade activity if this firm is disrupted
        if self.model.is_firm_disrupted(self.firm_id):
            return

        # 1. Complete any pending onboarding
        self._complete_onboarding()

        # 2. Update binary channel assignments for all supplier edges
        self._assign_channels()

        # 3. Accumulate costs (buyer side) and revenue (seller side)
        self._compute_quarter_cost()
        self._compute_quarter_revenue()

        # 3b. Update WTC for all active platform memberships
        for pid in list(self.platform_memberships):
            self._update_wtc(pid)

        # 4. Try supplier switches via platform (global) or bilateral (regional)
        self._try_supplier_switch()

    # ── Channel assignment (binary) ──────────────────────────────

    def _assign_channels(self):
        """Set binary channel for every supplier edge."""
        graph = self.model.supply_graph
        for sup_id in graph.predecessors(self.firm_id):
            self.edge_channels[sup_id] = self.determine_channel(sup_id)

    # ── Onboarding ───────────────────────────────────────────────

    def _complete_onboarding(self):
        """Activate suppliers whose onboarding period has elapsed."""
        completed = [
            sid for sid, ready_q in list(self.pending_onboarding.items())
            if self._quarter >= ready_q
        ]
        for sid in completed:
            self.pending_onboarding.pop(sid)
        # Also clear product delivery blocks
        done_prods = [
            p for p, ready_q in list(self.pending_delivery_products.items())
            if self._quarter >= ready_q
        ]
        for p in done_prods:
            self.pending_delivery_products.pop(p)

    # ── Cost accumulation ────────────────────────────────────────

    def _compute_quarter_cost(self):
        """Compute total procurement cost for this quarter."""
        graph = self.model.supply_graph
        for sup_id in graph.predecessors(self.firm_id):
            sup_agent = self.model.firm_agents.get(sup_id)
            if sup_agent is None:
                continue

            # Skip disrupted or onboarding suppliers
            if self.model.is_firm_disrupted(sup_id):
                continue
            if sup_id in self.pending_onboarding:
                continue

            channel = self.edge_channels.get(sup_id, "bilateral")
            edge_data = graph[sup_id][self.firm_id]
            products = edge_data.get("products", [])
            if not products:
                continue

            for product in products:
                qty = self.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    qty = sup_agent.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    continue

                cost = self.total_sc_cost(sup_agent, product, qty, channel)
                self.quarter_cost += cost

    def _compute_quarter_revenue(self):
        """Estimate revenue this firm earns as a supplier (seller side).

        Products for which a new supplier is still onboarding are skipped
        (the firm cannot deliver what it cannot source).
        """
        cfg = self.model.config
        graph = self.model.supply_graph
        rev = 0.0
        for buyer_id in graph.successors(self.firm_id):
            buyer_agent = self.model.firm_agents.get(buyer_id)
            if buyer_agent is None:
                continue
            # Determine channel from buyer's perspective
            buyer_channel = "bilateral"
            if buyer_agent:
                buyer_channel = buyer_agent.edge_channels.get(self.firm_id, "bilateral")
            edge = graph[self.firm_id][buyer_id]
            for product in edge.get("products", []):
                # Skip products waiting on supplier onboarding
                if product in self.pending_delivery_products:
                    continue
                qty = buyer_agent.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    qty = self.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    continue
                unit_price = get_unit_cost(product, self.country, self.model.loaded_data)
                edge_mat = unit_price * qty
                rev += edge_mat
                # Deduct seller-side admin cost (lower on platform)
                n_orders = min(max(1, int(math.ceil(qty / cfg.order_batch_size))), cfg.max_orders_per_quarter)
                rev -= self._seller_admin_cost(n_orders, buyer_channel,
                                               material_cost=edge_mat)
        self.quarter_revenue = rev

    # ── Supplier switching ───────────────────────────────────────

    def _try_supplier_switch(self):
        """
        Search for better suppliers using ROI-based switching.
        - On a platform: search globally among platform members.
        - Bilateral only: search regionally (same region only).

        Switching cost = onboarding + audit + mgmt_hours * wage.
        ROI = (annual_savings * horizon - switch_cost) / switch_cost.
        If ROI > 0 → switch deterministically.
        """
        graph = self.model.supply_graph
        cfg = self.model.config

        for product, qty in self.quarterly_demand.items():
            if qty <= 0:
                continue

            # Current supplier(s) for this product
            current_suppliers = []
            for sup_id in graph.predecessors(self.firm_id):
                if sup_id in self.pending_onboarding:
                    continue
                edge = graph[sup_id][self.firm_id]
                if product in edge.get("products", []):
                    current_suppliers.append(sup_id)

            if not current_suppliers:
                continue

            current_sup_id = current_suppliers[0]
            current_agent = self.model.firm_agents.get(current_sup_id)
            if current_agent is None:
                continue

            current_channel = self.edge_channels.get(current_sup_id, "bilateral")
            current_cost = self.total_sc_cost(
                current_agent, product, qty, current_channel
            )
            current_lt = self._lead_time_weeks(current_agent)

            best_alt_id = None
            best_alt_cost = current_cost
            best_alt_lt = current_lt
            best_alt_channel = current_channel

            # Search via platforms (global visibility)
            for plat_id in self.platform_memberships:
                platform_agent = self.model.platform_agents.get(plat_id)
                if platform_agent is None:
                    continue
                channel = f"platform_{plat_id.lower()}"

                for alt_id in platform_agent.member_firms:
                    if alt_id == current_sup_id or alt_id == self.firm_id:
                        continue
                    alt_agent = self.model.firm_agents.get(alt_id)
                    if alt_agent is None or product not in alt_agent.product_catalog:
                        continue

                    alt_cost = self.total_sc_cost(
                        alt_agent, product, qty, channel, is_new=True
                    )
                    alt_lt = self._lead_time_weeks(alt_agent)

                    if alt_cost < best_alt_cost or (
                        alt_cost == best_alt_cost and alt_lt < best_alt_lt
                    ):
                        best_alt_cost = alt_cost
                        best_alt_lt = alt_lt
                        best_alt_id = alt_id
                        best_alt_channel = channel

            # Search bilaterally — regional only unless bilateral_global_search enabled
            my_region = COUNTRY_TO_REGION.get(self.country)
            for alt_id, alt_agent in self.model.firm_agents.items():
                if alt_id == current_sup_id or alt_id == self.firm_id:
                    continue
                if alt_agent.is_oem or product not in alt_agent.product_catalog:
                    continue
                if not cfg.bilateral_global_search:
                    alt_region = COUNTRY_TO_REGION.get(alt_agent.country)
                    if alt_region != my_region:
                        continue  # bilateral = regional visibility only

                alt_cost = self.total_sc_cost(
                    alt_agent, product, qty, "bilateral", is_new=True
                )
                alt_lt = self._lead_time_weeks(alt_agent)

                if alt_cost < best_alt_cost or (
                    alt_cost == best_alt_cost and alt_lt < best_alt_lt
                ):
                    best_alt_cost = alt_cost
                    best_alt_lt = alt_lt
                    best_alt_id = alt_id
                    best_alt_channel = "bilateral"

            if best_alt_id is None:
                continue

            # ROI-based switching decision
            quarterly_saving = current_cost - best_alt_cost
            if quarterly_saving <= 0:
                continue
            annual_savings = quarterly_saving * 4

            # Switching cost: onboarding + mgmt hours + negotiation + conditional audit
            wi = self._wage_index
            best_alt_agent = self.model.firm_agents.get(best_alt_id)
            switch_cost = (
                cfg.onboarding_cost * wi
                + cfg.h_mgmt_strategic * self._wage_manager
            )
            # Negotiation cost (always incurred for new supplier)
            if best_alt_channel == "bilateral":
                switch_cost += cfg.bilateral_negotiation_cost * wi
            elif best_alt_channel == "platform_a":
                switch_cost += cfg.bilateral_negotiation_cost * cfg.platform_negotiation_pct_a * wi
            else:
                switch_cost += cfg.bilateral_negotiation_cost * cfg.platform_negotiation_pct_b * wi
            # Conditional audit: only for high-value suppliers
            if best_alt_agent is not None:
                up = get_unit_cost(product, best_alt_agent.country, self.model.loaded_data)
                mat_est = up * qty * (1.0 + best_alt_agent.defect_rate)
                if mat_est >= cfg.audit_material_threshold:
                    scale = min(
                        admin_scale_factor(mat_est, cfg.admin_scale_reference),
                        cfg.admin_scale_cap_audit,
                    )
                    switch_cost += cfg.audit_cost * scale
            # Bilateral search adds search cost (log-scaled, no cap)
            if best_alt_channel == "bilateral":
                search_scale = admin_scale_factor(mat_est, cfg.admin_scale_reference)
                switch_cost += cfg.bilateral_search_cost * wi * search_scale

            horizon_years = cfg.roi_horizon_quarters / 4.0
            roi = (annual_savings * horizon_years - switch_cost) / max(switch_cost, 1.0)
            if roi > 0:
                self._execute_switch(
                    current_sup_id, best_alt_id, product, best_alt_channel
                )

    def _execute_switch(
        self,
        old_sup_id: int,
        new_sup_id: int,
        product: str,
        channel: str,
    ):
        """
        Switch from old supplier to new supplier for *product*.
        Removes product from old edge, adds product to new edge.
        New supplier goes through onboarding if configured.
        """
        graph = self.model.supply_graph

        # Remove product from old supplier edge
        if graph.has_edge(old_sup_id, self.firm_id):
            old_products = graph[old_sup_id][self.firm_id].get("products", [])
            if product in old_products:
                old_products.remove(product)
            # Remove edge entirely if no products remain
            if not old_products:
                graph.remove_edge(old_sup_id, self.firm_id)
                self.edge_channels.pop(old_sup_id, None)

        # Add new edge / product
        if not graph.has_edge(new_sup_id, self.firm_id):
            graph.add_edge(
                new_sup_id, self.firm_id,
                products=[product],
                relation_ids=[],
                source_tier=self.tier_depth,
            )
        else:
            if product not in graph[new_sup_id][self.firm_id]["products"]:
                graph[new_sup_id][self.firm_id]["products"].append(product)

        # Set channel for new edge
        self.edge_channels[new_sup_id] = channel

        # Onboarding delay: new supplier + product blocked until ready
        cfg = self.model.config
        delay = (cfg.onboarding_delay_platform if channel.startswith("platform_")
                 else cfg.onboarding_delay_bilateral)
        if delay > 0:
            ready_q = self._quarter + delay
            self.pending_onboarding[new_sup_id] = ready_q
            self.pending_delivery_products[product] = ready_q

        self.model._quarter_switches += 1

    # ── Platform join / leave / switch decisions ─────────────────

    def evaluate_platform_membership(self):
        """
        Called by the model during the platform-growth phase.
        Evaluates whether to join, leave, or switch platforms
        based on profitability over the ROI horizon.

        No multihoming: a firm can be on at most one platform.

        Decision order for firms already on a platform (past cooldown):
          1. Compare current platform value with (other platform value − setup cost).
             If another platform is strictly better → switch (regardless of WTC sign).
          2. If no better platform exists and WTC < 0 → leave entirely.
          3. Otherwise → stay.
        """
        rng = self.model.random
        cfg = self.model.config
        horizon = cfg.roi_horizon_quarters

        # 1. Existing memberships: evaluate switching or leaving
        for pid in list(self.platform_memberships):
            plat = self.model.platform_agents.get(pid)
            if not plat:
                continue
            if self.consecutive_quarters.get(pid, 0) < cfg.platform_cooldown_quarters:
                continue

            # Compare current platform with all others.
            # Current platform: no setup cost (already paid).
            # Other platform: incurs setup cost → discourages rampant switching.
            _, current_info = self._compute_join_roi(pid, plat, horizon)
            stay_value = self._total_savings_from_info(current_info) * horizon

            best_other_pid = None
            best_switch_value = stay_value  # must strictly beat staying
            best_other_info = None

            for other_pid, other_plat in self.model.platform_agents.items():
                if other_pid == pid:
                    continue
                _, other_info = self._compute_join_roi(
                    other_pid, other_plat, horizon
                )
                setup_cost = cfg.platform_setup_cost * self._wage_index
                switch_value = (
                    self._total_savings_from_info(other_info) * horizon
                    - setup_cost
                )
                if switch_value > best_switch_value and switch_value > 0:
                    best_switch_value = switch_value
                    best_other_pid = other_pid
                    best_other_info = other_info

            if best_other_pid is not None:
                # Better platform exists and is profitable → switch
                self._do_leave(pid, plat)
                self._do_join(best_other_pid, best_other_info)
            elif self.wtc.get(pid, 0.0) <= cfg.wtc_leave_threshold:
                # No better platform and WTC zero or negative → leave entirely
                self._do_leave(pid, plat)
            elif stay_value < 0:
                # Current-quarter savings negative (platform costlier than
                # bilateral right now) even though smoothed WTC may still be
                # positive from historical inertia → leave.
                self._do_leave(pid, plat)

        # 2. If on no platform, compare all and join the BEST one
        if self.platform_memberships:
            return

        # Rejoin cooldown: must wait cooldown quarters after leaving
        quarters_since_leave = self._quarter - self._last_leave_quarter
        if quarters_since_leave < cfg.platform_cooldown_quarters:
            return

        best_pid = None
        best_roi = 0.0
        best_info = None

        for pid, plat in self.model.platform_agents.items():
            roi, info = self._compute_join_roi(pid, plat, horizon)
            if roi > best_roi:
                best_roi = roi
                best_pid = pid
                best_info = info

        if best_pid is not None:
            self._do_join(best_pid, best_info)

    def _compute_join_roi(self, pid, plat, horizon):
        """
        Compute the ROI for joining platform *pid* without side effects.
        Returns (roi, info_dict) where info_dict has savings components."""
        cfg = self.model.config
        graph = self.model.supply_graph
        channel_str = f"platform_{pid.lower()}"

        suppliers = list(graph.predecessors(self.firm_id))
        buyers = list(graph.successors(self.firm_id))
        is_leaf_supplier = len(suppliers) == 0 and len(buyers) > 0
        is_pure_buyer = self.is_oem or (len(buyers) == 0 and len(suppliers) > 0)

        # --- 1. Buyer savings ---
        # Same-platform suppliers only: full savings (bilateral − platform).
        # Non-platform suppliers: zero (PO/invoice require two-way exchange;
        # one-sided mgmt-hours credit removed — negligible per sensitivity analysis).
        buyer_savings_per_q = 0.0
        if not is_leaf_supplier:
            for sup_id in suppliers:
                sup_agent = self.model.firm_agents.get(sup_id)
                if sup_agent is None:
                    continue
                edge = graph[sup_id][self.firm_id]
                for product in edge.get("products", []):
                    qty = self.quarterly_demand.get(product, 0.0)
                    if qty <= 0:
                        qty = sup_agent.quarterly_demand.get(product, 0.0)
                    if qty <= 0:
                        continue
                    if plat.is_active_member(sup_id):
                        cost_bi = self.total_sc_cost(sup_agent, product, qty, "bilateral")
                        cost_pl = self.total_sc_cost(sup_agent, product, qty, channel_str)
                        buyer_savings_per_q += cost_bi - cost_pl

        # --- 2. Seller admin savings (skip for OEMs / pure buyers) ---
        # Same-platform buyers only: full PO+invoice+mgmt savings.
        # Non-platform buyers: zero (PO/invoice require two-way exchange;
        # one-sided mgmt-hours credit and revenue-risk display removed).
        seller_admin_per_q = 0.0
        if not is_pure_buyer:
            for buyer_id in buyers:
                buyer_agent = self.model.firm_agents.get(buyer_id)
                if buyer_agent is None:
                    continue
                edge = graph[self.firm_id][buyer_id]
                for product in edge.get("products", []):
                    qty = buyer_agent.quarterly_demand.get(product, 0.0)
                    if qty <= 0:
                        qty = self.quarterly_demand.get(product, 0.0)
                    if qty <= 0:
                        continue
                    up = get_unit_cost(product, self.country, self.model.loaded_data)
                    edge_revenue = up * qty

                    if pid in buyer_agent.platform_memberships:
                        n_orders = min(max(1, int(math.ceil(qty / cfg.order_batch_size))), cfg.max_orders_per_quarter)
                        admin_bi = self._seller_admin_cost(n_orders, "bilateral",
                                                           material_cost=edge_revenue)
                        admin_pl = self._seller_admin_cost(n_orders, channel_str,
                                                           material_cost=edge_revenue)
                        seller_admin_per_q += admin_bi - admin_pl

        seller_value_per_q = seller_admin_per_q

        # --- 2b. Discovery & visibility value ---
        # Also track which sourced products have alternatives on this platform
        # (used to weight search savings below).
        products_covered = set()

        # Buyers: count platform members who offer products I need
        discovery_value_per_q = 0.0
        if not is_leaf_supplier:
            my_products = set(self.quarterly_demand.keys())
            for alt_id in plat.member_firms:
                if alt_id == self.firm_id or alt_id in set(suppliers):
                    continue
                alt_agent = self.model.firm_agents.get(alt_id)
                if alt_agent is None:
                    continue
                overlap = my_products & set(alt_agent.product_catalog)
                if overlap:
                    products_covered |= overlap
                    for prod in overlap:
                        qty = self.quarterly_demand.get(prod, 0.0)
                        if qty <= 0:
                            continue
                        up = get_unit_cost(prod, alt_agent.country, self.model.loaded_data)
                        discovery_value_per_q += up * qty * cfg.platform_discovery_premium_pct

        # Sellers: count platform buyers who might need my products (visibility)
        visibility_value_per_q = 0.0
        if not is_pure_buyer:
            for alt_id in plat.member_firms:
                if alt_id == self.firm_id or alt_id in set(buyers):
                    continue
                alt_agent = self.model.firm_agents.get(alt_id)
                if alt_agent is None:
                    continue
                for prod in self.product_catalog:
                    qty = alt_agent.quarterly_demand.get(prod, 0.0)
                    if qty > 0:
                        up = get_unit_cost(prod, self.country, self.model.loaded_data)
                        visibility_value_per_q += up * qty * cfg.platform_visibility_premium_pct

        # --- 3. Search cost savings (weighted by platform coverage) ---
        # Search savings are log-scaled per supplier edge and weighted by
        # whether the platform covers the firm's sourced products.
        search_savings_per_q = 0.0
        if not is_leaf_supplier:
            my_products = set(self.quarterly_demand.keys())
            coverage_ratio = (len(products_covered) / max(len(my_products), 1)
                              if my_products else 0.0)
            p_need_search = 0.05
            search_pct = (
                cfg.platform_search_pct_a if pid == "A"
                else cfg.platform_search_pct_b
            )
            # Per-edge: search saving scales with edge material value
            for sup_id in suppliers:
                sup_agent = self.model.firm_agents.get(sup_id)
                if sup_agent is None:
                    continue
                edge = graph[sup_id][self.firm_id]
                edge_mat = 0.0
                for product in edge.get("products", []):
                    qty = self.quarterly_demand.get(product, 0.0)
                    if qty <= 0:
                        qty = sup_agent.quarterly_demand.get(product, 0.0)
                    if qty > 0:
                        up = get_unit_cost(product, sup_agent.country,
                                           self.model.loaded_data)
                        edge_mat += up * qty
                if edge_mat <= 0:
                    continue
                s_scale = admin_scale_factor(edge_mat, cfg.admin_scale_reference)
                search_savings_per_q += (
                    p_need_search
                    * (cfg.bilateral_search_cost - cfg.bilateral_search_cost * search_pct)
                    * self._wage_index
                    * s_scale
                    * coverage_ratio
                )

        # ROI is based on realized transaction savings by default:
        # buyer-side cost savings + seller-side admin savings.
        # During active disruptions, speculative values (search, discovery,
        # visibility) are added because firms actively need alternative
        # supplier discovery and supply-chain resilience.
        total_savings_per_q = buyer_savings_per_q + seller_admin_per_q
        disruption_active = getattr(self.model, '_disruption_active', False)
        if disruption_active:
            total_savings_per_q += (
                search_savings_per_q
                + discovery_value_per_q
                + visibility_value_per_q
            )

        setup_cost = cfg.platform_setup_cost * self._wage_index
        roi = (total_savings_per_q * horizon - setup_cost) / max(setup_cost, 1.0)

        info = {
            "pid": pid,
            "plat": plat,
            "roi": roi,
            "buyer_savings_per_q": buyer_savings_per_q,
            "seller_value_per_q": seller_value_per_q,
            "seller_admin_per_q": seller_admin_per_q,
            "search_savings_per_q": search_savings_per_q,
            "discovery_value_per_q": discovery_value_per_q,
            "visibility_value_per_q": visibility_value_per_q,
        }
        return roi, info

    def _do_join(self, pid, info):
        """Actually join platform *pid* using pre-computed info."""
        plat = info["plat"]
        roi = info["roi"]
        buyer_savings_per_q = info["buyer_savings_per_q"]
        seller_admin = info.get("seller_admin_per_q", 0.0)
        search_savings_per_q = info["search_savings_per_q"]
        discovery_value_per_q = info.get("discovery_value_per_q", 0.0)
        visibility_value_per_q = info.get("visibility_value_per_q", 0.0)

        plat.add_member(self.firm_id)
        self.platform_memberships.add(pid)
        self.join_quarter[pid] = self._quarter
        # Initialize WTC state (rejoin: load last_wtc as prior)
        if pid in self.last_wtc:
            self.wtc[pid] = self.last_wtc[pid]
            self.consecutive_quarters[pid] = 1  # 50/50 weight at first update
        else:
            self.wtc[pid] = 0.0
            self.consecutive_quarters[pid] = 0  # first-time: 100% immediate benefit
        # Record readable reason for tooltip
        parts = []
        if buyer_savings_per_q > 0:
            parts.append(f"buying saving ${buyer_savings_per_q:,.0f}/q")
        if seller_admin > 0:
            parts.append(f"seller admin saving ${seller_admin:,.0f}/q")
        if search_savings_per_q > 0:
            parts.append(f"search saving ${search_savings_per_q:,.0f}/q")
        if discovery_value_per_q > 0:
            parts.append(f"discovery ${discovery_value_per_q:,.0f}/q")
        if visibility_value_per_q > 0:
            parts.append(f"visibility ${visibility_value_per_q:,.0f}/q")
        parts.append(f"ROI {roi:.1f}x")
        self.join_reason = f"Joined {pid}: " + " | ".join(parts)

        # --- Adoption log entry ---
        direct_partners = set(self.model.supply_graph.predecessors(self.firm_id)) | set(self.model.supply_graph.successors(self.firm_id))
        partners_on = sum(1 for p in direct_partners if plat.is_active_member(p))
        self.adoption_log.append({
            "quarter": self._quarter,
            "action": "join",
            "platform": pid,
            "roi": round(roi, 2),
            "buyer_savings_q": round(buyer_savings_per_q, 0),
            "seller_value_q": round(seller_admin, 0),
            "search_savings_q": round(search_savings_per_q, 0),
            "discovery_q": round(discovery_value_per_q, 0),
            "visibility_q": round(visibility_value_per_q, 0),
            "partners_on_platform": partners_on,
            "total_partners": len(direct_partners),
        })

    def _do_leave(self, pid, plat):
        """Execute platform departure — shared by WTC-based leaving and switching."""
        # Preserve WTC history for rejoin drag
        current_wtc = self.wtc.get(pid, 0.0)
        self.last_wtc[pid] = current_wtc

        # --- Adoption log entry ---
        self.adoption_log.append({
            "quarter": self._quarter,
            "action": "leave",
            "platform": pid,
            "wtc": round(current_wtc, 0),
        })

        # Clear active WTC state
        self.wtc.pop(pid, None)
        self.consecutive_quarters.pop(pid, None)

        channel = f"platform_{pid.lower()}"
        plat.remove_member(self.firm_id)
        self.platform_memberships.discard(pid)
        self.join_quarter.pop(pid, None)
        self.join_reason = ""
        self._last_leave_quarter = self._quarter  # enforce rejoin cooldown
        for sup_id, ch in list(self.edge_channels.items()):
            if ch == channel:
                self.edge_channels[sup_id] = "bilateral"

    def _total_savings_from_info(self, info):
        """Sum per-quarter savings from info dict.

        Realized savings (buyer + seller admin) always included.
        During active disruptions, speculative values are added.
        """
        total = info["buyer_savings_per_q"] + info.get("seller_admin_per_q", 0.0)
        if getattr(self.model, '_disruption_active', False):
            total += (
                info.get("search_savings_per_q", 0.0)
                + info.get("discovery_value_per_q", 0.0)
                + info.get("visibility_value_per_q", 0.0)
            )
        return total
