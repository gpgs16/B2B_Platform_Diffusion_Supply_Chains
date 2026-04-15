"""
Combined H2 experiment: seeding sweep + cost gap sweep for 2 communities.

Runs ONLY the H2 experiments needed for the combined figure (fig 6+7).
Output: h2_combined_seeding.json, h2_combined_costgap.json
"""

import argparse
import json
import math
import os
import sys
import time

# Allow running from project root or from within experiments/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields as dc_fields

import numpy as np

from abm_platform.config import ExperimentConfig

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = 100
N_QUARTERS = 40
SEEDS = list(range(1, N_SEEDS + 1))
MAX_WORKERS = min(os.cpu_count() or 4, 10)

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

# ── Experiment parameters ─────────────────────────────────────────────
TARGET_COMMUNITIES = [11, 5]
SEEDING_FRACTIONS = [0.02, 0.05, 0.10, 0.15, 0.30]
COST_GAPS_PP = [0.00, 0.01, 0.02, 0.03, 0.05]


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
    from abm_platform.config import ExperimentConfig
    from abm_platform.data.loader import load_all
    from abm_platform.model import SupplyChainModel

    data = load_all()
    config = ExperimentConfig(**config_dict)
    model = SupplyChainModel(config=config, loaded_data=data)
    model.run_model()
    df = model.datacollector.get_model_vars_dataframe()

    community_sizes = {cid: len(firms) for cid, firms in model.communities.items()}
    last = df.iloc[-1]
    return {
        "seed": config.seed,
        "n_quarters_run": len(df),
        "final_adoption_a": float(last["Adoption_Rate_A"]),
        "final_adoption_b": float(last["Adoption_Rate_B"]),
        "community_sizes": community_sizes,
        "final_community_shares": last["Community_Platform_Shares"],
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


def main():
    # ── H2a: Seeding Sweep ────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  H2 Combined: Seeding Sweep")
    print("█" * 60)

    h2a_results = {}
    for target_cid in TARGET_COMMUNITIES:
        for seed_pct in SEEDING_FRACTIONS:
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

    save_json(h2a_results, "h2_combined_seeding.json")

    for t, results in h2a_results.items():
        valid = [r for r in results if "error" not in r]
        if valid:
            aa = [r["final_adoption_a"] for r in valid]
            ab = [r["final_adoption_b"] for r in valid]
            print(f"  {t}: A={np.mean(aa):.3f}  B={np.mean(ab):.3f}")

    # ── H2b: Cost Gap Sweep ───────────────────────────────────────
    print("\n" + "█" * 60)
    print("  H2 Combined: Cost Gap Sweep")
    print("█" * 60)

    h2b_results = {}
    for target_cid in TARGET_COMMUNITIES:
        for gap_pp in COST_GAPS_PP:
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

    save_json(h2b_results, "h2_combined_costgap.json")

    for t, results in h2b_results.items():
        valid = [r for r in results if "error" not in r]
        if valid:
            aa = [r["final_adoption_a"] for r in valid]
            ab = [r["final_adoption_b"] for r in valid]
            print(f"  {t}: A={np.mean(aa):.3f}  B={np.mean(ab):.3f}")

    print("\n✓ H2 combined experiments complete.")
    print(f"  Seeding: {len(TARGET_COMMUNITIES) * len(SEEDING_FRACTIONS)} treatments × {N_SEEDS} seeds")
    print(f"  CostGap: {len(TARGET_COMMUNITIES) * len(COST_GAPS_PP)} treatments × {N_SEEDS} seeds")


if __name__ == "__main__":
    main()
