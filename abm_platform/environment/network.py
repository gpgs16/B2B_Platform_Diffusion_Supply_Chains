"""
Supply chain network: NetworkX digraph built from relation data,
demand cascade logic, product-cost lookups, and capacity management.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

from abm_platform.config import (
    IMPORT_DUTY_RATE,
    snap_to_moq_tier,
    volume_discount_multiplier,
)
from abm_platform.data.loader import (
    BOMEntry,
    FirmRecord,
    LoadedData,
    RelationRecord,
)


def build_supply_graph(
    firms: dict[int, FirmRecord],
    relations: list[RelationRecord],
) -> nx.DiGraph:
    """
    Build directed supply chain graph.
    Edges point from supplier -> buyer (product flows downstream).
    Each edge carries product list and tier info.
    """
    G = nx.DiGraph()

    for fid, firm in firms.items():
        G.add_node(
            fid,
            firm_name=firm.firm_name,
            country=firm.country,
            lat=firm.lat,
            lon=firm.lon,
            is_oem=firm.is_oem,
            tier_depth=firm.tier_depth,
            is_top_supplier=firm.is_top_supplier,
        )

    for rel in relations:
        src = rel.source_firm_id
        tgt = rel.target_firm_id
        if src not in firms or tgt not in firms:
            continue
        if G.has_edge(src, tgt):
            G[src][tgt]["products"].extend(rel.products)
            G[src][tgt]["relation_ids"].append(rel.relation_id)
        else:
            G.add_edge(
                src,
                tgt,
                products=list(rel.products),
                relation_ids=[rel.relation_id],
                source_tier=rel.source_tier,
            )
    return G


# ── Unit cost ────────────────────────────────────────────────────────

def get_unit_cost(
    product: str,
    supplier_country: str,
    data: LoadedData,
) -> float:
    """
    Best-effort unit cost (USD) for *product* from *supplier_country*.

    Priority:
      1. Country-specific mid cost from Excel Sheet 5
      2. China mid cost from china_product_costs.csv (as fallback)
      3. Heuristic $50 (last resort)
    """
    cc_map = data.country_costs.get(product)
    if cc_map:
        entry = cc_map.get(supplier_country)
        if entry:
            return entry.cost_mid
        china = cc_map.get("China")
        if china:
            return china.cost_mid

    china_cost = data.china_costs.get(product)
    if china_cost:
        return china_cost.cost_mid

    return 50.0  # last resort


# ── MOQ ──────────────────────────────────────────────────────────────

def get_moq(product: str, data: LoadedData) -> int:
    """Minimum order quantity for *product*, snapped to standard tiers."""
    cc = data.china_costs.get(product)
    raw = float(cc.min_order_qty) if cc else 1.0
    return snap_to_moq_tier(int(raw))


# ── Import duty ──────────────────────────────────────────────────────

def import_duty_rate(exporter: str, importer: str) -> float:
    """Import duty rate for the given country pair."""
    pair = (exporter, importer)
    if pair in IMPORT_DUTY_RATE:
        return IMPORT_DUTY_RATE[pair]
    if exporter == importer:
        return 0.0
    return 0.05


# ── Capacity ─────────────────────────────────────────────────────────

def derive_max_capacity(
    firm_id: int,
    bom_data: dict[int, list[BOMEntry]],
    annual_volumes: dict[int, int],
) -> float:
    """
    Returns infinite capacity — suppliers never reject orders.
    Kept as a function for interface compatibility.
    """
    return float('inf')


# ── Demand cascade ───────────────────────────────────────────────────

def cascade_demand(
    graph: nx.DiGraph,
    bom_data: dict[int, list[BOMEntry]],
    oem_demand: dict[int, float],
) -> dict[int, dict[str, float]]:
    """
    Propagate OEM quarterly vehicle demand through the supply chain.

    Returns {firm_id: {product: demand_quantity}}.
    Each supplier's demand = sum(downstream buyer demand * qty_per_vehicle)
    across all paths.
    """
    bom_index: dict[int, dict[tuple[int, str], float]] = {}
    for oem_id, entries in bom_data.items():
        for e in entries:
            bom_index.setdefault(oem_id, {})[(e.supplier_firm_id, e.product)] = e.quantity_per_vehicle

    demand: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for oem_id, vehicle_qty in oem_demand.items():
        oem_bom = bom_index.get(oem_id, {})
        for (fid, product), qty_per_vehicle in oem_bom.items():
            demand[fid][product] += vehicle_qty * qty_per_vehicle

    return dict(demand)
