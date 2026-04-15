"""
Entry point for the B2B Procurement Platform Diffusion ABM.

Usage:
    python -m abm_platform.run                       # default 40 quarters
    python -m abm_platform.run --quarters 20
    python -m abm_platform.run --seed 99
"""

from __future__ import annotations

import argparse
import time

from abm_platform.config import ExperimentConfig
from abm_platform.data.loader import load_all
from abm_platform.model import SupplyChainModel
from abm_platform.visualization import generate_all_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the B2B procurement platform diffusion ABM."
    )
    p.add_argument("--quarters", type=int, default=40)
    p.add_argument("--search-pct-a", type=float, default=0.30, help="A search cost as fraction of bilateral")
    p.add_argument("--search-pct-b", type=float, default=0.30, help="B search cost as fraction of bilateral")
    p.add_argument("--initial-adoption", type=float, default=0.05)
    p.add_argument("--roi-horizon", type=int, default=12)
    p.add_argument("--disruption-quarter", type=int, default=0, help="Quarter to trigger disruption (0=none)")
    p.add_argument("--disruption-duration", type=int, default=2)
    p.add_argument("--cooldown", type=int, default=2, help="Quarters before leave evaluation")
    p.add_argument("--wtc-threshold", type=float, default=0.0, help="WTC leave threshold ($)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-viz", action="store_true", help="Skip visualization output")
    return p.parse_args()


def main():
    args = parse_args()

    config = ExperimentConfig(
        n_quarters=args.quarters,
        platform_search_pct_a=args.search_pct_a,
        platform_search_pct_b=args.search_pct_b,
        initial_adoption_frac=args.initial_adoption,
        roi_horizon_quarters=args.roi_horizon,
        disruption_quarter=args.disruption_quarter,
        disruption_duration=args.disruption_duration,
        platform_cooldown_quarters=args.cooldown,
        wtc_leave_threshold=args.wtc_threshold,
        seed=args.seed,
    )

    print("Loading data...")
    data = load_all()
    print(f"  Firms: {len(data.firms)}")
    print(f"  Relations: {len(data.relations)}")
    print(f"  BOM entries: {sum(len(v) for v in data.bom.values())}")
    print(f"  China products: {len(data.china_costs)}")
    print(f"  Country costs: {len(data.country_costs)} products")

    print(f"\nInitialising model (seed={config.seed})...")
    model = SupplyChainModel(config=config, loaded_data=data, rng=config.seed)
    print(f"  CompanyAgents: {len(model.firm_agents)}")
    print(f"  Louvain communities: {len(model.communities)}")
    print(f"  Platform A members: {model.platform_agents['A'].member_count}")
    print(f"  Platform B members: {model.platform_agents['B'].member_count}")
    print(f"  Supply graph: {model.supply_graph.number_of_nodes()} nodes, "
          f"{model.supply_graph.number_of_edges()} edges")
    if model._disruption_target_id is not None:
        target = model.firm_agents.get(model._disruption_target_id)
        tname = target.firm_name if target else "?"
        print(f"  Disruption target: {model._disruption_target_id} ({tname})")

    print(f"\nRunning (max {config.n_quarters} quarters)...")
    t0 = time.time()
    model.run_model()
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Termination: {model.termination_reason}")

    df = model.datacollector.get_model_vars_dataframe()
    last = df.iloc[-1]
    print(f"\n{'='*55}")
    print(f"RESULTS (Quarter {int(last['Quarter'])})")
    print(f"{'='*55}")
    print(f"  Platform A adoption: {last['Adoption_Rate_A']:.1%}")
    print(f"  Platform B adoption: {last['Adoption_Rate_B']:.1%}")
    print(f"  Platform A market share: {last['Platform_A_Share']:.1%}")
    print(f"  Platform B market share: {last['Platform_B_Share']:.1%}")
    print(f"  Platform A edge %: {last.get('Platform_Edge_Pct_A', 0):.1%}")
    print(f"  Platform B edge %: {last.get('Platform_Edge_Pct_B', 0):.1%}")
    print(f"  Total trade volume: ${last['Total_Trade_Volume']:,.0f}")
    print(f"  Platform A volume: ${last.get('Platform_A_Volume', 0):,.0f}")
    print(f"  Supplier switches: {int(last['Supplier_Switches'])}")
    print(f"  Platform A profit: ${last.get('Platform_A_Profit', 0):,.0f}")
    print(f"  Platform B profit: ${last.get('Platform_B_Profit', 0):,.0f}")
    print(f"  Platform A OpEx:   ${last.get('Platform_A_OpEx', 0):,.0f}")
    print(f"  Platform B OpEx:   ${last.get('Platform_B_OpEx', 0):,.0f}")
    print(f"  Effective rate A:  {last.get('Effective_Rate_A', 0):.4%}")
    print(f"  Effective rate B:  {last.get('Effective_Rate_B', 0):.4%}")
    ss_q = last.get('Steady_State_Quarter', 0)
    print(f"  Steady state:     {'Q' + str(int(ss_q)) if ss_q else 'Not reached'}")

    if not args.no_viz:
        print("\nGenerating visualizations...")
        generate_all_plots(model)

    return model


if __name__ == "__main__":
    main()
