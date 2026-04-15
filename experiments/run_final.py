"""
Unified experiment runner for final report.

Phases:
  Phase 1 — Bilateral cost diagnostic (per-firm ROI at Q5)
  Phase 2 — Sensitivity analysis (18 OFAT experiments, 100 seeds each)
  Phase 3 — Hypothesis experiments (H1 coexistence, H2 lock-in, H3 disruption)

Usage:
  python run_final.py                # run everything
  python run_final.py --phase 1      # diagnostic only
  python run_final.py --phase 2      # sensitivity only
  python run_final.py --phase 3      # hypothesis only
"""

import argparse
import json
import math
import os
import sys
import time

# Allow running from project root or from within experiments/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields as dc_fields

import numpy as np

from abm_platform.config import ExperimentConfig
from abm_platform.data.loader import load_all
from abm_platform.model import SupplyChainModel
from abm_platform.agents.company import (
    tiered_commission_rate,
    admin_scale_factor,
    get_unit_cost,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = 100
N_QUARTERS = 40
SEEDS = list(range(1, N_SEEDS + 1))
MAX_WORKERS = min(os.cpu_count() or 4, 10)

# ── Baseline overrides ────────────────────────────────────────────────
# Cost %: uniform 30% across all categories
# Seeding: 5%, Cooldown: 2Q, ROI: 12Q, Setup: $50K (Shivlab 2024)
BASELINE = dict(
    platform_search_pct_a=0.30, platform_search_pct_b=0.30,
    platform_po_pct_a=0.30, platform_po_pct_b=0.30,
    platform_invoice_pct_a=0.30, platform_invoice_pct_b=0.30,
    platform_mgmt_pct_a=0.30, platform_mgmt_pct_b=0.30,
    platform_negotiation_pct_a=0.30, platform_negotiation_pct_b=0.30,
    initial_adoption_frac=0.05,
    platform_cooldown_quarters=2,
    roi_horizon_quarters=12,
    platform_setup_cost=50_000.0,
    bilateral_search_cost=15_000.0,
    bilateral_negotiation_cost=8_000.0,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_config_dict(**overrides) -> dict:
    merged = {**BASELINE, **overrides}
    cfg = ExperimentConfig(**merged)
    d = {}
    for f in dc_fields(cfg):
        val = getattr(cfg, f.name)
        if isinstance(val, dict):
            d[f.name] = dict(val)
        elif isinstance(val, list):
            d[f.name] = list(val)
        else:
            d[f.name] = val
    return d


def _run_single(config_dict: dict) -> dict:
    """Run one simulation and return summary + time series."""
    from abm_platform.config import ExperimentConfig
    from abm_platform.data.loader import load_all
    from abm_platform.model import SupplyChainModel

    data = load_all()
    config = ExperimentConfig(**config_dict)
    model = SupplyChainModel(config=config, loaded_data=data)
    model.run_model()
    df = model.datacollector.get_model_vars_dataframe()

    community_sizes = {cid: len(firms) for cid, firms in model.communities.items()}

    ts_records = []
    for _, row in df.iterrows():
        rec = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, dict):
                rec[col] = json.loads(json.dumps(val, default=str))
            elif isinstance(val, (np.integer,)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating,)):
                rec[col] = float(val)
            else:
                rec[col] = val
        ts_records.append(rec)

    last = df.iloc[-1]
    return {
        "seed": config.seed,
        "n_quarters_run": len(df),
        "termination_reason": model.termination_reason,
        "steady_state_quarter": int(last.get("Steady_State_Quarter", 0)),
        "final_adoption_a": float(last["Adoption_Rate_A"]),
        "final_adoption_b": float(last["Adoption_Rate_B"]),
        "final_share_a": float(last["Platform_A_Share"]),
        "final_share_b": float(last["Platform_B_Share"]),
        "final_hhi": float(last["HHI"]),
        "final_profit_a": float(last["Platform_A_Profit"]),
        "final_profit_b": float(last["Platform_B_Profit"]),
        "community_sizes": community_sizes,
        "final_community_shares": last["Community_Platform_Shares"],
        "time_series": ts_records,
    }


def run_batch(configs: list[dict], label: str) -> list[dict]:
    results = []
    total = len(configs)
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  {label}: {total} runs, {MAX_WORKERS} workers")
    print(f"{'='*60}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_single, c): i for i, c in enumerate(configs)}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                results.append(result)
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - t0
                    print(f"  [{done}/{total}] {elapsed:.0f}s elapsed")
            except Exception as e:
                idx = futures[future]
                print(f"  ERROR on run {idx}: {e}")
                results.append({"error": str(e), "seed": configs[idx].get("seed")})

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s ({elapsed/total:.1f}s/run avg)")
    return results


def save_json(data, filename):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: BILATERAL COST DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════

def run_diagnostic():
    """Per-firm ROI decomposition at Q5, with deeper cost breakdown."""
    print("\n" + "█" * 60)
    print("  PHASE 1: Per-Firm Cost Decomposition Diagnostic")
    print("█" * 60)

    data = load_all()
    config = ExperimentConfig(**{**BASELINE, **dict(
        n_quarters=N_QUARTERS, seed=42,
        community_seeding=False, disruption_enabled=False)})
    model = SupplyChainModel(config=config, loaded_data=data)

    for _ in range(5):
        model.step()

    cfg = config
    graph = model.supply_graph
    firm_records = []

    for fid, agent in model.firm_agents.items():
        cid = agent.community_id
        country = agent.country
        is_oem = agent.is_oem
        platform = next(iter(agent.platform_memberships)) if agent.platform_memberships else None

        suppliers = list(graph.predecessors(fid))
        buyers = list(graph.successors(fid))

        buy_edge_count = 0
        buy_material_total = 0.0
        buy_commission_total = 0.0
        buy_bilateral_admin = 0.0
        buy_platform_admin = 0.0

        for sup_id in suppliers:
            sup_agent = model.firm_agents.get(sup_id)
            if sup_agent is None:
                continue
            edge = graph[sup_id][fid]
            for product in edge.get("products", []):
                qty = agent.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    qty = sup_agent.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    continue
                buy_edge_count += 1
                up = get_unit_cost(product, sup_agent.country, data)
                mat = up * qty
                buy_material_total += mat
                tier_rate = tiered_commission_rate(mat, cfg.commission_tiers_a)
                buy_commission_total += mat * tier_rate
                scale = admin_scale_factor(mat, cfg.admin_scale_reference)
                n_orders = min(max(1, int(math.ceil(qty / cfg.order_batch_size))),
                               cfg.max_orders_per_quarter)
                po_scale = min(scale, cfg.admin_scale_cap_po_invoice)
                wage = agent._wage_index
                bi_po = n_orders * cfg.bilateral_po_cost * wage * po_scale
                bi_inv = n_orders * cfg.bilateral_invoice_cost * wage * po_scale
                buy_bilateral_admin += bi_po + bi_inv
                pl_po = n_orders * cfg.bilateral_po_cost * cfg.platform_po_pct_a * wage * po_scale
                pl_inv = n_orders * cfg.bilateral_invoice_cost * cfg.platform_invoice_pct_a * wage * po_scale
                buy_platform_admin += pl_po + pl_inv

        sell_edge_count = 0
        sell_material_total = 0.0
        for buyer_id in buyers:
            buyer_agent = model.firm_agents.get(buyer_id)
            if buyer_agent is None:
                continue
            edge = graph[fid][buyer_id]
            for product in edge.get("products", []):
                qty = buyer_agent.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    qty = agent.quarterly_demand.get(product, 0.0)
                if qty <= 0:
                    continue
                sell_edge_count += 1
                up = get_unit_cost(product, agent.country, data)
                sell_material_total += up * qty

        # Use model's own ROI computation
        best_roi = None
        best_info = {}
        for pid in ["A", "B"]:
            plat = model.platform_agents.get(pid)
            if plat is None:
                continue
            roi, info = agent._compute_join_roi(pid, plat, cfg.roi_horizon_quarters)
            if best_roi is None or roi > best_roi:
                best_roi = roi
                best_info = info

        setup_cost = cfg.platform_setup_cost * agent._wage_index
        total_savings_q = best_info.get("buyer_savings_per_q", 0) + best_info.get("seller_admin_per_q", 0)

        rec = {
            "firm_id": fid,
            "community": cid,
            "country": country,
            "is_oem": is_oem,
            "current_platform": platform,
            "buy_edges": buy_edge_count,
            "sell_edges": sell_edge_count,
            "total_edges": buy_edge_count + sell_edge_count,
            "buy_material_q": round(buy_material_total, 0),
            "sell_material_q": round(sell_material_total, 0),
            "total_material_q": round(buy_material_total + sell_material_total, 0),
            "buy_commission_q": round(buy_commission_total, 2),
            "buy_bilateral_admin_q": round(buy_bilateral_admin, 2),
            "buy_platform_admin_q": round(buy_platform_admin, 2),
            "buy_admin_savings_q": round(buy_bilateral_admin - buy_platform_admin, 2),
            "wage_index": round(agent._wage_index, 3),
            "roi": best_roi,
            "buyer_savings_q": best_info.get("buyer_savings_per_q", 0),
            "seller_admin_q": best_info.get("seller_admin_per_q", 0),
            "seller_risk_q": best_info.get("seller_revenue_risk_per_q", 0),
            "search_savings_q": best_info.get("search_savings_per_q", 0),
            "total_savings_q": total_savings_q,
            "setup_cost": setup_cost,
            "savings_to_setup_ratio": round(total_savings_q * cfg.roi_horizon_quarters / max(setup_cost, 1), 3),
        }
        firm_records.append(rec)

    # Aggregate by community
    community_agg = defaultdict(lambda: {
        "firms": 0, "adopted": 0, "bilateral": 0,
        "total_material": 0.0, "total_commission": 0.0,
        "total_admin_savings": 0.0, "total_setup_cost": 0.0,
        "positive_roi": 0, "rois": [], "edges": [],
        "buy_materials": [], "wages": [], "countries": defaultdict(int),
        "total_savings_qs": [], "setup_costs": [],
    })

    for r in firm_records:
        cid = r["community"]
        c = community_agg[cid]
        c["firms"] += 1
        if r["current_platform"]:
            c["adopted"] += 1
        else:
            c["bilateral"] += 1
        c["total_material"] += r["total_material_q"]
        c["total_commission"] += r["buy_commission_q"]
        c["total_admin_savings"] += r["buy_admin_savings_q"]
        c["total_setup_cost"] += r.get("setup_cost", 0)
        roi = r.get("roi")
        if roi is not None:
            c["rois"].append(roi)
            if roi > 0:
                c["positive_roi"] += 1
        c["edges"].append(r["total_edges"])
        c["buy_materials"].append(r["buy_material_q"])
        c["wages"].append(r["wage_index"])
        c["total_savings_qs"].append(r["total_savings_q"])
        c["setup_costs"].append(r["setup_cost"])
        c["countries"][r["country"]] += 1

    community_summary = {}
    for cid, c in sorted(community_agg.items()):
        community_summary[cid] = {
            "firms": c["firms"],
            "adopted": c["adopted"],
            "bilateral": c["bilateral"],
            "adoption_rate": round(c["adopted"] / max(c["firms"], 1), 3),
            "avg_material_per_firm": round(c["total_material"] / max(c["firms"], 1), 0),
            "avg_commission_per_firm": round(c["total_commission"] / max(c["firms"], 1), 2),
            "avg_admin_savings": round(c["total_admin_savings"] / max(c["firms"], 1), 2),
            "avg_setup_cost": round(np.mean(c["setup_costs"]), 0),
            "avg_total_savings_q": round(np.mean(c["total_savings_qs"]), 2),
            "avg_roi": round(np.mean(c["rois"]), 3) if c["rois"] else None,
            "pct_positive_roi": round(c["positive_roi"] / max(c["firms"], 1), 3),
            "avg_edges": round(np.mean(c["edges"]), 1),
            "avg_wage_index": round(np.mean(c["wages"]), 3),
            "countries": dict(c["countries"]),
        }

    result = {
        "description": "Per-firm ROI decomposition at Q5 (seed=42, baseline config)",
        "n_firms": len(firm_records),
        "firm_records": firm_records,
        "community_summary": community_summary,
    }

    save_json(result, "diagnostic.json")

    print(f"\n  {'Comm':>4} {'N':>3} {'Adpt%':>5} {'AvgEdge':>7} {'AvgWage':>7} "
          f"{'AvgSavQ':>8} {'AvgSetup':>8} {'AvgROI':>7} {'%ROI>0':>6}")
    for cid in sorted(community_summary.keys()):
        cs = community_summary[cid]
        print(f"  C{cid:>2} {cs['firms']:>3} {cs['adoption_rate']:>5.0%} "
              f"{cs['avg_edges']:>7.1f} {cs['avg_wage_index']:>7.3f} "
              f"{cs['avg_total_savings_q']:>8,.0f} {cs['avg_setup_cost']:>8,.0f} "
              f"{cs['avg_roi'] or 0:>7.2f} {cs['pct_positive_roi']:>6.0%}")

    return result


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: SENSITIVITY ANALYSIS (18 OFAT Experiments)
# ══════════════════════════════════════════════════════════════════════

def run_sensitivity():
    """Run all 18 OFAT sensitivity experiments."""
    print("\n" + "█" * 60)
    print("  PHASE 2: OFAT Sensitivity Analysis (18 experiments)")
    print("█" * 60)

    experiments = {}

    # --- S1: Platform Cost Level (uniform) ---
    print("\n--- S1: Platform Cost Level ---")
    s1_levels = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    s1_results = {}
    for level in s1_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                platform_search_pct_a=level, platform_search_pct_b=level,
                platform_po_pct_a=level, platform_po_pct_b=level,
                platform_invoice_pct_a=level, platform_invoice_pct_b=level,
                platform_mgmt_pct_a=level, platform_mgmt_pct_b=level,
                platform_negotiation_pct_a=level, platform_negotiation_pct_b=level,
            ))
        s1_results[f"pct_{int(level*100)}"] = run_batch(configs, f"S1: cost={int(level*100)}%")
    experiments["s1_cost_level"] = s1_results
    save_json(s1_results, "s1_cost_level.json")

    # --- S2: ROI Horizon ---
    print("\n--- S2: ROI Horizon ---")
    s2_levels = [4, 6, 8, 10, 12, 16, 20]
    s2_results = {}
    for level in s2_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                roi_horizon_quarters=level,
            ))
        s2_results[f"horizon_{level}q"] = run_batch(configs, f"S2: horizon={level}Q")
    experiments["s2_roi_horizon"] = s2_results
    save_json(s2_results, "s2_roi_horizon.json")

    # --- S3: Initial Seeding ---
    print("\n--- S3: Initial Seeding ---")
    s3_levels = [0.02, 0.05, 0.10, 0.14, 0.20]
    s3_results = {}
    for level in s3_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                initial_adoption_frac=level,
            ))
        s3_results[f"seed_{int(level*100)}pct"] = run_batch(configs, f"S3: init={int(level*100)}%")
    experiments["s3_seeding"] = s3_results
    save_json(s3_results, "s3_seeding.json")

    # --- S4: Cooldown Period ---
    print("\n--- S4: Cooldown Period ---")
    s4_levels = [0, 2, 4, 6, 8, 12]
    s4_results = {}
    for level in s4_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                platform_cooldown_quarters=level,
            ))
        s4_results[f"cool_{level}q"] = run_batch(configs, f"S4: cooldown={level}Q")
    experiments["s4_cooldown"] = s4_results
    save_json(s4_results, "s4_cooldown.json")

    # --- S5: PO Cost (Group A: Transaction) ---
    print("\n--- S5: PO Cost ---")
    s5_levels = [90, 135, 180, 270, 360]
    s5_results = {}
    for level in s5_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                bilateral_po_cost=float(level),
            ))
        s5_results[f"po_{level}"] = run_batch(configs, f"S5: PO=${level}")
    experiments["s5_po_cost"] = s5_results
    save_json(s5_results, "s5_po_cost.json")

    # --- S6: Invoice Cost (Group A: Transaction) ---
    print("\n--- S6: Invoice Cost ---")
    s6_levels = [5.0, 7.0, 9.5, 13.0, 19.0]
    s6_results = {}
    for level in s6_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                bilateral_invoice_cost=level,
            ))
        s6_results[f"inv_{level}"] = run_batch(configs, f"S6: invoice=${level}")
    experiments["s6_invoice_cost"] = s6_results
    save_json(s6_results, "s6_invoice_cost.json")

    # --- S7: Search Cost (Group B: Relationship) ---
    print("\n--- S7: Search Cost ---")
    s7_levels = [5_000, 10_000, 15_000, 25_000, 40_000]
    s7_results = {}
    for level in s7_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                bilateral_search_cost=float(level),
            ))
        s7_results[f"search_{level//1000}k"] = run_batch(configs, f"S7: search=${level//1000}K")
    experiments["s7_search_cost"] = s7_results
    save_json(s7_results, "s7_search_cost.json")

    # --- S8: Negotiation Cost (Group B: Relationship) ---
    print("\n--- S8: Negotiation Cost ---")
    s8_levels = [3_000, 5_000, 8_000, 12_000, 20_000]
    s8_results = {}
    for level in s8_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                bilateral_negotiation_cost=float(level),
            ))
        s8_results[f"neg_{level//1000}k"] = run_batch(configs, f"S8: negotiation=${level//1000}K")
    experiments["s8_negotiation_cost"] = s8_results
    save_json(s8_results, "s8_negotiation_cost.json")

    # --- S9: Audit Cost (Group B: Relationship) ---
    print("\n--- S9: Audit Cost ---")
    s9_levels = [10_000, 20_000, 30_000, 50_000, 75_000]
    s9_results = {}
    for level in s9_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                audit_cost=float(level),
            ))
        s9_results[f"audit_{level//1000}k"] = run_batch(configs, f"S9: audit=${level//1000}K")
    experiments["s9_audit_cost"] = s9_results
    save_json(s9_results, "s9_audit_cost.json")

    # --- S10: Onboarding Cost (Group B: Relationship) ---
    print("\n--- S10: Onboarding Cost ---")
    s10_levels = [10_000, 15_000, 20_000, 30_000, 50_000]
    s10_results = {}
    for level in s10_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                onboarding_cost=float(level),
            ))
        s10_results[f"onb_{level//1000}k"] = run_batch(configs, f"S10: onboarding=${level//1000}K")
    experiments["s10_onboarding_cost"] = s10_results
    save_json(s10_results, "s10_onboarding_cost.json")

    # --- S11: Strategic Management Hours (Group C: Management) ---
    print("\n--- S11: Strategic Mgmt Hours ---")
    s11_levels = [104.0, 156.0, 208.0, 312.0, 416.0, 520.0]
    s11_results = {}
    for level in s11_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                h_mgmt_strategic=level,
            ))
        s11_results[f"strat_{int(level)}h"] = run_batch(configs, f"S11: strategic={int(level)}h")
    experiments["s11_strategic_hours"] = s11_results
    save_json(s11_results, "s11_strategic_hours.json")

    # --- S12: Non-Strategic Management Hours (Group C: Management) ---
    print("\n--- S12: Non-Strategic Mgmt Hours ---")
    s12_levels = [26.0, 52.0, 78.0, 104.0, 156.0, 208.0]
    s12_results = {}
    for level in s12_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                h_mgmt_non_strategic=level,
            ))
        s12_results[f"nonstrat_{int(level)}h"] = run_batch(configs, f"S12: non-strategic={int(level)}h")
    experiments["s12_nonstrat_hours"] = s12_results
    save_json(s12_results, "s12_nonstrat_hours.json")

    # --- S13: Procurement Manager Multiplier (Group C: Management) ---
    print("\n--- S13: Manager Multiplier ---")
    s13_levels = [1.0, 1.5, 2.0, 2.5, 3.0]
    s13_results = {}
    for level in s13_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                procurement_mgr_mult=level,
            ))
        s13_results[f"mgr_{level}x"] = run_batch(configs, f"S13: mgr_mult={level}x")
    experiments["s13_mgr_mult"] = s13_results
    save_json(s13_results, "s13_mgr_mult.json")

    # --- S14: Admin Scale Reference (Group D: Admin Scaling) ---
    print("\n--- S14: Admin Scale Reference ---")
    s14_levels = [1_000.0, 2_500.0, 5_000.0, 10_000.0, 25_000.0]
    s14_results = {}
    for level in s14_levels:
        label = f"{level/1000:.1f}k" if level < 10_000 else f"{int(level//1000)}k"
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                admin_scale_reference=level,
            ))
        s14_results[f"ref_{label}"] = run_batch(configs, f"S14: scale_ref=${label}")
    experiments["s14_admin_scale_ref"] = s14_results
    save_json(s14_results, "s14_admin_scale_ref.json")

    # --- S15: PO/Invoice Scale Cap (Group D: Admin Scaling) ---
    print("\n--- S15: PO/Invoice Scale Cap ---")
    s15_levels = [1.5, 2.0, 3.0, 5.0, 10.0]
    s15_results = {}
    for level in s15_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                admin_scale_cap_po_invoice=level,
            ))
        s15_results[f"cap_poi_{level}"] = run_batch(configs, f"S15: PO/inv cap={level}")
    experiments["s15_cap_po_invoice"] = s15_results
    save_json(s15_results, "s15_cap_po_invoice.json")

    # --- S16: Audit Scale Cap (Group D: Admin Scaling) ---
    print("\n--- S16: Audit Scale Cap ---")
    s16_levels = [1.0, 1.5, 2.0, 3.0, 5.0]
    s16_results = {}
    for level in s16_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                admin_scale_cap_audit=level,
            ))
        s16_results[f"cap_aud_{level}"] = run_batch(configs, f"S16: audit cap={level}")
    experiments["s16_cap_audit"] = s16_results
    save_json(s16_results, "s16_cap_audit.json")

    # --- S17: Onboarding Scale Cap (Group D: Admin Scaling) ---
    print("\n--- S17: Onboarding Scale Cap ---")
    s17_levels = [1.0, 1.25, 1.5, 2.0, 3.0]
    s17_results = {}
    for level in s17_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                admin_scale_cap_onboarding=level,
            ))
        s17_results[f"cap_onb_{level}"] = run_batch(configs, f"S17: onboarding cap={level}")
    experiments["s17_cap_onboarding"] = s17_results
    save_json(s17_results, "s17_cap_onboarding.json")

    # --- S18: Platform Setup Cost (Group E: Setup) ---
    print("\n--- S18: Platform Setup Cost ---")
    s18_levels = [20_000, 35_000, 50_000, 75_000, 100_000]
    s18_results = {}
    for level in s18_levels:
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False, disruption_enabled=False,
                platform_setup_cost=float(level),
            ))
        s18_results[f"setup_{level//1000}k"] = run_batch(configs, f"S18: setup=${level//1000}K")
    experiments["s18_setup_cost"] = s18_results
    save_json(s18_results, "s18_setup_cost.json")

    return experiments


# ══════════════════════════════════════════════════════════════════════
# PHASE 3: HYPOTHESIS EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════

H2_TARGET_COMMUNITIES = [11, 5, 4]
H2_SEEDING_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.15]
H2_COST_GAPS_PP = [0.00, 0.01, 0.02, 0.03, 0.05]


def run_hypothesis_experiments():
    """Run H1, H2, H3."""
    print("\n" + "█" * 60)
    print("  PHASE 3: Hypothesis Experiments")
    print("█" * 60)

    # --- H1: Coexistence (random seeding, symmetric platforms) ---
    print("\n--- H1: Coexistence ---")
    h1_configs = []
    for seed in SEEDS:
        h1_configs.append(_make_config_dict(
            n_quarters=N_QUARTERS, seed=seed,
            community_seeding=False, disruption_enabled=False,
        ))
    h1_results = run_batch(h1_configs, "H1: Coexistence")
    save_json(h1_results, "h1_coexistence.json")

    # Quick summary
    valid = [r for r in h1_results if "error" not in r]
    if valid:
        aa = [r["final_adoption_a"] for r in valid]
        ab = [r["final_adoption_b"] for r in valid]
        hhi = [r["final_hhi"] for r in valid]
        print(f"  H1: A={np.mean(aa):.3f}±{np.std(aa):.3f}  "
              f"B={np.mean(ab):.3f}±{np.std(ab):.3f}  "
              f"HHI={np.mean(hhi):.3f}±{np.std(hhi):.3f}")

    # --- H2a: Strategic Seeding Lock-In — Seeding Sweep ---
    # B has fixed 2pp cost advantage; vary seeding fraction
    print("\n--- H2a: Seeding Sweep ---")
    h2a_results = {}
    for target_cid in H2_TARGET_COMMUNITIES:
        for seed_pct in H2_SEEDING_FRACTIONS:
            treatment = f"C{target_cid}_pct{int(seed_pct*100)}"
            configs = []
            for seed in SEEDS:
                configs.append(_make_config_dict(
                    n_quarters=N_QUARTERS, seed=seed,
                    community_seeding=True,
                    platform_a_seed_community=target_cid,
                    platform_a_community_seed_pct=seed_pct,
                    platform_b_seed_community=-1,
                    platform_b_community_seed_pct=1.0,
                    disruption_enabled=False,
                    # B gets 2pp cost advantage on all categories
                    platform_search_pct_b=0.28,
                    platform_po_pct_b=0.28,
                    platform_invoice_pct_b=0.28,
                    platform_mgmt_pct_b=0.28,
                    platform_negotiation_pct_b=0.28,
                ))
            h2a_results[treatment] = run_batch(configs, f"H2a: {treatment}")

    save_json(h2a_results, "h2a_seeding_sweep.json")

    for t, results in h2a_results.items():
        valid = [r for r in results if "error" not in r]
        if valid:
            aa = [r["final_adoption_a"] for r in valid]
            ab = [r["final_adoption_b"] for r in valid]
            print(f"  {t}: A={np.mean(aa):.3f}  B={np.mean(ab):.3f}")

    # --- H2b: Strategic Seeding Lock-In — Cost Gap Sweep ---
    # Fixed high seeding (15%); vary cost gap
    print("\n--- H2b: Cost Gap Sweep ---")
    h2b_results = {}
    for target_cid in H2_TARGET_COMMUNITIES:
        for gap_pp in H2_COST_GAPS_PP:
            treatment = f"C{target_cid}_gap{int(gap_pp*100)}pp"
            configs = []
            b_pct = 0.30 - gap_pp  # B's cost advantage
            for seed in SEEDS:
                configs.append(_make_config_dict(
                    n_quarters=N_QUARTERS, seed=seed,
                    community_seeding=True,
                    platform_a_seed_community=target_cid,
                    platform_a_community_seed_pct=0.15,
                    platform_b_seed_community=-1,
                    platform_b_community_seed_pct=1.0,
                    disruption_enabled=False,
                    platform_search_pct_b=b_pct,
                    platform_po_pct_b=b_pct,
                    platform_invoice_pct_b=b_pct,
                    platform_mgmt_pct_b=b_pct,
                    platform_negotiation_pct_b=b_pct,
                ))
            h2b_results[treatment] = run_batch(configs, f"H2b: {treatment}")

    save_json(h2b_results, "h2b_cost_gap_sweep.json")

    for t, results in h2b_results.items():
        valid = [r for r in results if "error" not in r]
        if valid:
            aa = [r["final_adoption_a"] for r in valid]
            ab = [r["final_adoption_b"] for r in valid]
            print(f"  {t}: A={np.mean(aa):.3f}  B={np.mean(ab):.3f}")

    # --- H3: Disruption ---
    print("\n--- H3: Disruption-Driven Adoption ---")
    h3_scenarios = {
        "no_disruption": dict(disruption_enabled=False),
        "disruption_freq20_n3": dict(disruption_enabled=True, bilateral_global_search=True, disruption_frequency=20, disruption_n_firms=3),
        "disruption_freq20_n5": dict(disruption_enabled=True, bilateral_global_search=True, disruption_frequency=20, disruption_n_firms=5),
        "disruption_freq10_n3": dict(disruption_enabled=True, bilateral_global_search=True, disruption_frequency=10, disruption_n_firms=3),
    }

    h3_results = {}
    for name, params in h3_scenarios.items():
        configs = []
        for seed in SEEDS:
            configs.append(_make_config_dict(
                n_quarters=N_QUARTERS, seed=seed,
                community_seeding=False,
                **params,
            ))
        h3_results[name] = run_batch(configs, f"H3: {name}")

    save_json(h3_results, "h3_disruption.json")

    for name, results in h3_results.items():
        valid = [r for r in results if "error" not in r]
        if valid:
            total = [r["final_adoption_a"] + r["final_adoption_b"] for r in valid]
            print(f"  {name}: total adoption = {np.mean(total):.3f} ± {np.std(total):.3f}")

    return h1_results, h2a_results, h2b_results, h3_results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None)
    args = parser.parse_args()

    t_start = time.time()

    if args.phase is None or args.phase == 1:
        run_diagnostic()
    if args.phase is None or args.phase == 2:
        run_sensitivity()
    if args.phase is None or args.phase == 3:
        run_hypothesis_experiments()

    print(f"\n{'='*60}")
    print(f"  All phases completed in {time.time() - t_start:.0f}s")
    print(f"  Results saved to: {OUT_DIR}")
    print(f"{'='*60}")
