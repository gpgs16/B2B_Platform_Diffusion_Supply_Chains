"""
SupplyChainModel — Mesa Model orchestrating quarterly simulation steps.

Lifecycle per quarter:
  1. Generate seasonal OEM demand & cascade upstream
  2. Each CompanyAgent: assign channels, compute costs, attempt switches
  3. PlatformAgents: compute trade volume, market share
  4. Platform growth: non-members evaluate joining (profit-based);
     members evaluate leaving/switching
  5. Check termination (max quarters)
  6. DataCollector records metrics
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
from mesa import Model
from mesa.datacollection import DataCollector

from abm_platform.agents.company import CompanyAgent
from abm_platform.agents.platform import PlatformAgent
from abm_platform.config import ExperimentConfig, SEASONAL_WEIGHTS, COUNTRY_TO_REGION
from abm_platform.data.loader import LoadedData, load_all
from abm_platform.environment.network import (
    build_supply_graph,
    cascade_demand,
    get_unit_cost,
)

# ── DataCollector helper functions ───────────────────────────────────

def _platform_a_share(m): return m.platform_agents["A"].market_share if "A" in m.platform_agents else 0.0
def _platform_b_share(m): return m.platform_agents["B"].market_share if "B" in m.platform_agents else 0.0
def _adoption_a(m):
    pa = m.platform_agents.get("A")
    return pa.total_count / max(len(m.firm_agents), 1) if pa else 0.0
def _adoption_b(m):
    pb = m.platform_agents.get("B")
    return pb.total_count / max(len(m.firm_agents), 1) if pb else 0.0

def _vol_a(m): return m.platform_agents["A"].trade_volume if "A" in m.platform_agents else 0.0
def _vol_b(m): return m.platform_agents["B"].trade_volume if "B" in m.platform_agents else 0.0
def _comm_a(m): return m.platform_agents["A"].quarter_commission if "A" in m.platform_agents else 0.0
def _comm_b(m): return m.platform_agents["B"].quarter_commission if "B" in m.platform_agents else 0.0
def _total_trade(m): return m.total_trade_volume
def _total_plat_vol(m): return sum(p.trade_volume for p in m.platform_agents.values())
def _switches(m): return m._quarter_switches
def _profit_a(m): return m.platform_agents["A"].quarter_profit if "A" in m.platform_agents else 0.0
def _profit_b(m): return m.platform_agents["B"].quarter_profit if "B" in m.platform_agents else 0.0
def _opex_a(m): return m.platform_agents["A"].quarter_opex if "A" in m.platform_agents else 0.0
def _opex_b(m): return m.platform_agents["B"].quarter_opex if "B" in m.platform_agents else 0.0
def _joined_a(m): return m.platform_agents["A"].quarter_joined if "A" in m.platform_agents else 0
def _left_a(m): return m.platform_agents["A"].quarter_left if "A" in m.platform_agents else 0
def _joined_b(m): return m.platform_agents["B"].quarter_joined if "B" in m.platform_agents else 0
def _left_b(m): return m.platform_agents["B"].quarter_left if "B" in m.platform_agents else 0


def _hhi(m):
    """Herfindahl index for platform market concentration."""
    shares = []
    for p in m.platform_agents.values():
        if p.market_share > 0:
            shares.append(p.market_share)
    bilateral = 1.0 - sum(shares)
    if bilateral > 0:
        shares.append(bilateral)
    return sum(s ** 2 for s in shares) if shares else 1.0


def _eff_rate_a(m):
    """Effective commission rate for Platform A (commission / volume)."""
    pa = m.platform_agents.get("A")
    if pa and pa.trade_volume > 0:
        return pa.quarter_commission / pa.trade_volume
    return 0.0

def _eff_rate_b(m):
    """Effective commission rate for Platform B (commission / volume)."""
    pb = m.platform_agents.get("B")
    if pb and pb.trade_volume > 0:
        return pb.quarter_commission / pb.trade_volume
    return 0.0

def _disruption_active(m):
    return 1 if m._disruption_active else 0

def _disrupted_firms(m):
    return len(m._disrupted_firm_ids)

def _cascade_failures(m):
    return m._cascade_failures

def _platform_edge_pct_a(m):
    """Fraction of supply-chain edges routed through Platform A."""
    total = platform_a = 0
    for agent in m.firm_agents.values():
        for ch in agent.edge_channels.values():
            total += 1
            if ch == "platform_a":
                platform_a += 1
    return platform_a / max(total, 1)

def _platform_edge_pct_b(m):
    """Fraction of supply-chain edges routed through Platform B."""
    total = platform_b = 0
    for agent in m.firm_agents.values():
        for ch in agent.edge_channels.values():
            total += 1
            if ch == "platform_b":
                platform_b += 1
    return platform_b / max(total, 1)

def _steady_state(m):
    return 1 if m._steady_state_quarter is not None else 0

def _steady_state_quarter(m):
    return m._steady_state_quarter if m._steady_state_quarter is not None else 0


def _community_adoption(m):
    """Per-Louvain-community adoption rates as a dict."""
    community_total = defaultdict(int)
    community_adopted = defaultdict(int)
    for agent in m.firm_agents.values():
        cid = agent.community_id
        if cid < 0:
            continue
        community_total[cid] += 1
        if agent.platform_memberships:
            community_adopted[cid] += 1
    return {
        cid: community_adopted[cid] / max(community_total[cid], 1)
        for cid in community_total
    }


def _community_platform_shares(m):
    """Per-Louvain-community platform share breakdown.

    Returns dict[cid → {A, B, bilateral, dominance_index,
    trade_A, trade_B, trade_bilateral}].
    dominance_index = |share_A - share_B| (0=even, 1=monopoly).
    trade_* = fraction of intra-community edges routed via that channel.
    """
    community_a = defaultdict(int)
    community_b = defaultdict(int)
    community_total = defaultdict(int)
    # Trade-based: count intra-community edges by channel
    trade_a = defaultdict(int)
    trade_b = defaultdict(int)
    trade_total = defaultdict(int)

    for agent in m.firm_agents.values():
        cid = agent.community_id
        if cid < 0:
            continue
        community_total[cid] += 1
        if "A" in agent.platform_memberships:
            community_a[cid] += 1
        elif "B" in agent.platform_memberships:
            community_b[cid] += 1

    # Count intra-community edge channels
    for buyer_id, agent in m.firm_agents.items():
        cid = agent.community_id
        if cid < 0:
            continue
        for sup_id, ch in agent.edge_channels.items():
            sup_agent = m.firm_agents.get(sup_id)
            if sup_agent is None or sup_agent.community_id != cid:
                continue
            trade_total[cid] += 1
            if ch == "platform_a":
                trade_a[cid] += 1
            elif ch == "platform_b":
                trade_b[cid] += 1

    result = {}
    for cid in community_total:
        n = max(community_total[cid], 1)
        share_a = community_a[cid] / n
        share_b = community_b[cid] / n
        te = max(trade_total[cid], 1)
        result[cid] = {
            "A": share_a,
            "B": share_b,
            "bilateral": 1.0 - share_a - share_b,
            "dominance_index": abs(share_a - share_b),
            "trade_A": trade_a[cid] / te,
            "trade_B": trade_b[cid] / te,
            "trade_bilateral": 1.0 - trade_a[cid] / te - trade_b[cid] / te,
        }
    return result


class SupplyChainModel(Model):
    """Mesa ABM for B2B procurement platform diffusion."""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        loaded_data: LoadedData | None = None,
        **kwargs,
    ):
        cfg = config or ExperimentConfig()
        kwargs.setdefault("seed", cfg.seed)
        super().__init__(**kwargs)
        self.config = cfg
        self.loaded_data = loaded_data or load_all()
        self._quarter = 0
        self._quarter_switches = 0
        self.termination_reason: str | None = None

        # Disruption state
        self._disruption_active = False
        self._disrupted_firm_ids: set[int] = set()       # firms currently unavailable
        self._disruption_target_ids: list[int] = []      # pre-selected targets
        self._cascade_failures: int = 0                  # unfulfilled demands this quarter

        # Steady-state detection
        self._consecutive_stable_quarters: int = 0
        self._steady_state_quarter: int | None = None

        # ── Build graph ──────────────────────────────────────────
        self.supply_graph = build_supply_graph(
            self.loaded_data.firms, self.loaded_data.relations
        )

        # ── Create CompanyAgents ─────────────────────────────────
        self.firm_agents: dict[int, CompanyAgent] = {}
        for fid, frec in self.loaded_data.firms.items():
            agent = CompanyAgent(
                self,
                firm_id=fid,
                firm_name=frec.firm_name,
                country=frec.country,
                lat=frec.lat,
                lon=frec.lon,
                is_oem=frec.is_oem,
                tier_depth=frec.tier_depth,
                is_top_supplier=frec.is_top_supplier,
                certifications=frec.certifications,
                product_catalog=frec.product_catalog,
                defect_rate=frec.defect_rate,
                is_eco_friendly=frec.is_eco_friendly,
            )
            self.firm_agents[fid] = agent

        # ── Detect communities (Louvain) ──────────────────────────
        self._detect_communities()


        # ── Create PlatformAgents ────────────────────────────────
        self.platform_agents: dict[str, PlatformAgent] = {}
        self.platform_agents["A"] = PlatformAgent(
            self,
            platform_id="A",
        )
        self.platform_agents["B"] = PlatformAgent(
            self,
            platform_id="B",
        )

        # ── Initial platform adoption ────────────────────────────
        self._seed_platform_adoption()

        # ── Identify sole-source suppliers (cannot be disrupted) ────
        self._sole_source_firm_ids = self._compute_sole_source_firms()

        # ── Identify critical nodes for disruption ────────────────
        self._identify_critical_nodes()

        # ── Initialize channel assignments ───────────────────────
        for fid, agent in self.firm_agents.items():
            for sup_id in self.supply_graph.predecessors(fid):
                agent.edge_channels[sup_id] = agent.determine_channel(sup_id)

        # ── Trade volume tracker ─────────────────────────────────
        self.total_trade_volume: float = 0.0

        # ── DataCollector ────────────────────────────────────────
        self.datacollector = DataCollector(
            model_reporters={
                "Quarter": lambda m: m._quarter,
                "Platform_A_Share": _platform_a_share,
                "Platform_B_Share": _platform_b_share,
                "Adoption_Rate_A": _adoption_a,
                "Adoption_Rate_B": _adoption_b,
                "Platform_A_Volume": _vol_a,
                "Platform_B_Volume": _vol_b,
                "Total_Platform_Volume": _total_plat_vol,
                "Total_Trade_Volume": _total_trade,
                "Supplier_Switches": _switches,
                "Platform_A_Fee_Revenue": _comm_a,
                "Platform_B_Fee_Revenue": _comm_b,
                "Platform_A_Profit": _profit_a,
                "Platform_B_Profit": _profit_b,
                "Platform_A_OpEx": _opex_a,
                "Platform_B_OpEx": _opex_b,
                "Firms_Joined_A": _joined_a,
                "Firms_Left_A": _left_a,
                "Firms_Joined_B": _joined_b,
                "Firms_Left_B": _left_b,
                "HHI": _hhi,
                "Effective_Rate_A": _eff_rate_a,
                "Effective_Rate_B": _eff_rate_b,
                "Platform_Edge_Pct_A": _platform_edge_pct_a,
                "Platform_Edge_Pct_B": _platform_edge_pct_b,
                "Disruption_Active": _disruption_active,
                "Disrupted_Firms": _disrupted_firms,
                "Cascade_Failures": _cascade_failures,
                "Steady_State": _steady_state,
                "Steady_State_Quarter": _steady_state_quarter,
                "Community_Adoption": _community_adoption,
                "Community_Platform_Shares": _community_platform_shares,
            },
        )

    # ── Seeding ──────────────────────────────────────────────────

    def _seed_platform_adoption(self, platform_id: str | None = None):
        """
        Seed firms onto platforms at model init.

        When ``community_seeding`` is True, each platform whose
        ``platform_X_seed_community`` is ≥ 0 is seeded into that
        community (fraction controlled by ``platform_X_community_seed_pct``).
        Platforms with seed_community == -1 fall back to random seeding.

        When ``community_seeding`` is False (default), both platforms
        are seeded randomly into ``initial_adoption_frac`` of firms,
        preferring non-overlapping sets.
        """
        eligible = [
            fid for fid in self.firm_agents
            if list(self.supply_graph.predecessors(fid))
            or list(self.supply_graph.successors(fid))
        ]

        cfg = self.config
        frac = cfg.initial_adoption_frac
        n_total = max(1, int(len(self.firm_agents) * frac))

        def _community_pool(target_cid, seed_pct):
            """Return firm IDs from a specific community."""
            comm_firms = [
                f for f in eligible
                if self.firm_agents[f].community_id == target_cid
            ]
            self.random.shuffle(comm_firms)
            n = max(1, int(len(comm_firms) * seed_pct))
            return comm_firms[:n]

        # --- Platform A ---
        if (cfg.community_seeding
                and cfg.platform_a_seed_community >= 0
                and cfg.platform_a_seed_community in self.communities):
            pool_a = _community_pool(cfg.platform_a_seed_community,
                                     cfg.platform_a_community_seed_pct)
        else:
            self.random.shuffle(eligible)
            pool_a = eligible[:n_total]

        # --- Platform B ---
        if (cfg.community_seeding
                and cfg.platform_b_seed_community >= 0
                and cfg.platform_b_seed_community in self.communities):
            pool_b = _community_pool(cfg.platform_b_seed_community,
                                     cfg.platform_b_community_seed_pct)
        else:
            # Random: prefer non-A firms to reduce overlap
            remaining = [f for f in eligible if f not in set(pool_a)]
            self.random.shuffle(remaining)
            pool_b = remaining[:n_total]
            if not pool_b:
                pool_b = eligible[n_total:2 * n_total]

        for fid in pool_a:
            self.platform_agents["A"].add_member(fid)
            agent = self.firm_agents[fid]
            agent.platform_memberships.add("A")
            agent.join_quarter["A"] = 0
            agent.wtc["A"] = 0.0
            agent.consecutive_quarters["A"] = 0
            agent.adoption_log.append({
                "quarter": 0, "action": "seed", "platform": "A",
            })

        for fid in pool_b:
            self.platform_agents["B"].add_member(fid)
            agent = self.firm_agents[fid]
            agent.platform_memberships.add("B")
            agent.join_quarter["B"] = 0
            agent.wtc["B"] = 0.0
            agent.consecutive_quarters["B"] = 0
            agent.adoption_log.append({
                "quarter": 0, "action": "seed", "platform": "B",
            })

    # ── Louvain community detection ─────────────────────────────

    def _detect_communities(self):
        """
        Partition firms into communities using the Louvain algorithm on
        the undirected supply graph.  Each firm gets a ``community_id``
        attribute (int >= 0).
        """
        from networkx.algorithms.community import louvain_communities

        undirected = self.supply_graph.to_undirected()
        partitions = louvain_communities(undirected, seed=self.config.seed)
        self.communities: dict[int, set[int]] = {}
        for idx, comm in enumerate(partitions):
            self.communities[idx] = comm
            for fid in comm:
                agent = self.firm_agents.get(fid)
                if agent:
                    agent.community_id = idx

    # ── Bell-curve annual growth ──────────────────────────────────

    def _cumulative_growth(self, year: int) -> float:
        """Return compound growth factor up to *year* (0-indexed).

        Each year's rate is drawn once from N(mean, std) clipped to [min, max].
        Rates are cached so the same year always returns the same value.
        """
        if not hasattr(self, "_yearly_growth_rates"):
            self._yearly_growth_rates: dict[int, float] = {}
        cfg = self.config
        factor = 1.0
        for y in range(year):
            if y not in self._yearly_growth_rates:
                rate = self.random.gauss(cfg.annual_growth_rate, cfg.growth_rate_std)
                rate = max(cfg.growth_rate_min, min(cfg.growth_rate_max, rate))
                self._yearly_growth_rates[y] = rate
            factor *= (1.0 + self._yearly_growth_rates[y])
        return factor

    # ── Disruption: sole-source identification ──────────────────

    def _compute_sole_source_firms(self) -> set[int]:
        """Return firm IDs that are the sole global supplier for any product."""
        product_suppliers: dict[str, set[int]] = {}
        for u, _v, data in self.supply_graph.edges(data=True):
            for prod in data.get("products", []):
                product_suppliers.setdefault(prod, set()).add(u)
        sole_source = set()
        for prod, sups in product_suppliers.items():
            if len(sups) == 1:
                sole_source.update(sups)
        return sole_source

    # ── Disruption: critical node identification ─────────────────

    def _identify_critical_nodes(self):
        """
        Pre-select the top-N most critical non-OEM supplier nodes for
        disruption using betweenness centrality.
        Excludes sole-source suppliers (they cannot be disrupted).
        """
        cfg = self.config
        n = cfg.disruption_n_firms

        # Legacy single-target override
        if isinstance(cfg.disruption_target, int):
            self._disruption_target_ids = [cfg.disruption_target]
            return

        bc = nx.betweenness_centrality(self.supply_graph)
        scored = [
            (fid, score)
            for fid, score in bc.items()
            if (agent := self.firm_agents.get(fid))
            and not agent.is_oem
            and fid not in self._sole_source_firm_ids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        self._disruption_target_ids = [fid for fid, _ in scored[:n]]

    def _apply_disruption(self):
        """Mark disruption target firms as unavailable for trade."""
        self._disrupted_firm_ids = set(self._disruption_target_ids)
        self._disruption_active = True

    def _restore_disruption(self):
        """Restore all disrupted firms to normal operation."""
        self._disrupted_firm_ids.clear()
        self._disruption_active = False

    def _can_find_alt_supplier(self, buyer_agent: CompanyAgent, product: str) -> bool:
        """Check if buyer can find an alternative supplier during disruption.

        Platform members: search globally among all non-disrupted platform members.
        Bilateral-only firms: search regionally (or globally if bilateral_global_search).
        """
        buyer_country = buyer_agent.country
        buyer_region = COUNTRY_TO_REGION.get(buyer_country)

        # Platform members: can find any platform member globally
        for pid in buyer_agent.platform_memberships:
            plat = self.platform_agents.get(pid)
            if plat is None:
                continue
            for alt_id in plat.member_firms:
                if alt_id == buyer_agent.firm_id or alt_id in self._disrupted_firm_ids:
                    continue
                alt_agent = self.firm_agents.get(alt_id)
                if alt_agent and product in alt_agent.product_catalog:
                    return True

        # Bilateral: regional search only (unless bilateral_global_search enabled)
        for alt_id, alt_agent in self.firm_agents.items():
            if alt_id == buyer_agent.firm_id or alt_id in self._disrupted_firm_ids:
                continue
            if alt_agent.is_oem or product not in alt_agent.product_catalog:
                continue
            if not self.config.bilateral_global_search:
                alt_region = COUNTRY_TO_REGION.get(alt_agent.country)
                if alt_region != buyer_region:
                    continue
            return True

        return False

    def is_firm_disrupted(self, firm_id: int) -> bool:
        """Check if a firm is currently unavailable due to disruption."""
        return firm_id in self._disrupted_firm_ids

    # ── Step ─────────────────────────────────────────────────────

    def step(self):
        """Advance the simulation by one quarter."""
        self._quarter += 1
        self._quarter_switches = 0

        # 1. Reset per-quarter accumulators
        for agent in self.firm_agents.values():
            agent.quarter_cost = 0.0
            agent.quarter_revenue = 0.0
            agent._quarter = self._quarter
        for p in self.platform_agents.values():
            p.reset_quarter()

        # 2. Disruption: recurring events based on frequency
        cfg = self.config
        if cfg.disruption_enabled and self._disruption_target_ids:
            warmup = 10  # disruptions only start after 10 quarters
            freq = max(1, cfg.disruption_frequency)
            dur = cfg.disruption_duration
            # Is this quarter the start of ANY disruption cycle?
            if self._quarter >= warmup:
                offset = (self._quarter - warmup) % freq
                if offset == 0 and not self._disruption_active:
                    self._apply_disruption()
                elif self._disruption_active and offset >= dur:
                    self._restore_disruption()

        # 3. Generate OEM demand and cascade (with bell-curve annual growth)
        q_idx = (self._quarter - 1) % 4
        seasonal = self.config.seasonal_weights[q_idx]
        year = (self._quarter - 1) // 4
        growth_factor = self._cumulative_growth(year)
        oem_demand = {}
        for oem_id, annual_vol in self.config.annual_volume.items():
            oem_demand[oem_id] = annual_vol * seasonal * growth_factor

        demands = cascade_demand(
            self.supply_graph, self.loaded_data.bom, oem_demand,
        )
        # Apply disruption: zero-out demand for disrupted firms and count failures
        self._cascade_failures = 0
        for fid, prod_demands in demands.items():
            agent = self.firm_agents.get(fid)
            if agent:
                if fid in self._disrupted_firm_ids:
                    # Disrupted firm can't fulfil any demand — cascade failure
                    self._cascade_failures += len(prod_demands)
                    agent.quarterly_demand = {}
                    agent.current_load = 0.0
                else:
                    # Check if any supplier of this firm is disrupted
                    adjusted = {}
                    for prod, qty in prod_demands.items():
                        suppliers_for_prod = [
                            sid for sid in self.supply_graph.predecessors(fid)
                            if prod in self.supply_graph[sid][fid].get("products", [])
                        ]
                        disrupted_suppliers = [
                            sid for sid in suppliers_for_prod
                            if sid in self._disrupted_firm_ids
                        ]
                        if disrupted_suppliers and len(disrupted_suppliers) == len(suppliers_for_prod):
                            # All current suppliers disrupted — can the firm find an alternative?
                            if self._can_find_alt_supplier(agent, prod):
                                adjusted[prod] = qty  # platform or global bilateral found alt
                            else:
                                self._cascade_failures += 1
                                adjusted[prod] = 0.0
                        else:
                            adjusted[prod] = qty
                    agent.quarterly_demand = adjusted
                    agent.current_load = sum(adjusted.values())

        # 4. Estimate total trade volume (for market share denominator)
        self.total_trade_volume = self._estimate_total_trade()

        # 5. Company agents step: assign channels, compute costs, attempt switches
        company_agents = list(self.firm_agents.values())
        self.random.shuffle(company_agents)
        for agent in company_agents:
            agent.step()

        # 6. Platform growth: all firms evaluate join/leave/switch
        self._platform_growth()

        # 7b. Steady-state detection: 8 consecutive quarters with zero membership changes
        #     Skip counting during any disruption window + 2 recovery quarters.
        if self._steady_state_quarter is None:
            cfg = self.config
            skip = False
            if cfg.disruption_enabled and self._quarter >= 10:
                freq = max(1, cfg.disruption_frequency)
                dur = cfg.disruption_duration
                offset = (self._quarter - 10) % freq
                # Skip during disruption (0..dur-1) plus 2 recovery quarters
                if offset < dur + 2:
                    skip = True

            if skip:
                # Reset counter — disruption quarters don't count
                self._consecutive_stable_quarters = 0
            else:
                total_changes = sum(
                    p.quarter_joined + p.quarter_left
                    for p in self.platform_agents.values()
                )
                if total_changes == 0:
                    self._consecutive_stable_quarters += 1
                    if self._consecutive_stable_quarters >= 8:
                        self._steady_state_quarter = self._quarter
                else:
                    self._consecutive_stable_quarters = 0

        # 7. Re-assign channels to reflect membership changes from growth
        for agent in self.firm_agents.values():
            agent._assign_channels()

        # 8. Compute platform trade volume/commission with final channels
        for p in self.platform_agents.values():
            p.step()

        # 9. Collect data
        self.datacollector.collect(self)

    def _estimate_total_trade(self) -> float:
        """Edge-based estimate of total quarterly trade value."""
        total = 0.0
        graph = self.supply_graph
        for u, v, data in graph.edges(data=True):
            sup_agent = self.firm_agents.get(u)
            buyer_agent = self.firm_agents.get(v)
            if sup_agent is None or buyer_agent is None:
                continue
            for product in data.get("products", []):
                qty = buyer_agent.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    raw_qty = sup_agent.quarterly_demand.get(product, 0.0)
                    if raw_qty <= 0:
                        continue
                    n_downstream = sum(
                        1 for _, _, d in graph.out_edges(u, data=True)
                        if product in d.get("products", [])
                    )
                    qty = raw_qty / max(n_downstream, 1)
                if qty <= 0:
                    continue
                unit = get_unit_cost(product, sup_agent.country, self.loaded_data)
                total += unit * qty
        return total

    def _platform_growth(self):
        """Every firm evaluates platform membership each quarter."""
        agents = list(self.firm_agents.values())
        self.random.shuffle(agents)
        for agent in agents:
            agent.evaluate_platform_membership()

    def check_termination(self) -> bool:
        """Check whether the simulation should stop."""
        if self._quarter >= self.config.n_quarters:
            self.termination_reason = f"Max quarters reached ({self.config.n_quarters})"
            return True
        es = self.config.early_stop_quarters
        if es > 0 and self._consecutive_stable_quarters >= es:
            self.termination_reason = (
                f"Early stop: {es} consecutive stable quarters (Q{self._quarter})"
            )
            return True
        return False

    def run_model(self):
        """Run the simulation until a termination condition is met."""
        while not self.check_termination():
            self.step()
