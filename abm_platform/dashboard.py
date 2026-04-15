"""
Streamlit interactive dashboard for the B2B Procurement Platform ABM.

Features:
  - Log-scaled admin costs with per-component caps
  - Realized-savings-only adoption (buyer + seller admin)
  - Battery community auto-detection for Platform B seeding
  - Disruption events with demand cascade
  - Disruption-conditional speculative values
  - 9-tier degressive commission (calibrated for volume-discounted edges)

Launch:  streamlit run abm_platform/dashboard.py
"""

from __future__ import annotations

import sys
import os
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="B2B Platform ABM",
    page_icon="\U0001f3ed",
    layout="wide",
)

from abm_platform.config import ExperimentConfig
from abm_platform.data.loader import load_all, LoadedData
from abm_platform.model import SupplyChainModel
from abm_platform.visualization import build_network_html


# -- Cached data loading --------------------------------------------------

@st.cache_resource(show_spinner="Loading supply-chain data \u2026")
def get_data() -> LoadedData:
    return load_all()


# -- Session State Initialisation ------------------------------------------

def _init_state():
    defaults = {
        "model": None,
        "df": None,
        "running": False,
        "paused": False,
        "step_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ===========================================================================
# Defaults
# ===========================================================================

_DEFAULTS = dict(
    n_quarters=40, seed_val=42,
    search_pct_a=0.30, search_pct_b=0.30,
    po_pct_a=0.30, po_pct_b=0.30,
    invoice_pct_a=0.30, invoice_pct_b=0.30,
    mgmt_pct_a=0.30, mgmt_pct_b=0.30,
    bilateral_search=15_000, bilateral_po=180, bilateral_invoice=9.5,
    platform_setup=50_000, onboarding=20_000, audit=30_000,
    fixed_opex=20_000, var_opex=140,
    initial_adoption=0.05, roi_horizon=12,
    discovery_premium=0.020, visibility_premium=0.045,
    vol_oem1=2_500, vol_oem2=32_000, vol_oem3=50_000, vol_oem4=15_000,
    annual_growth=0.035,
    community_seeding=False,
    platform_a_community_seed_pct=1.0,
    platform_b_community_seed_pct=1.0,
    disruption_enabled=False, disruption_frequency=20,
    disruption_n_firms=3,
    admin_scale_reference=5_000,
    admin_scale_cap_po_invoice=3.0,
    admin_scale_cap_audit=2.0,
    admin_scale_cap_onboarding=1.5,
    bilateral_negotiation=8_000,
    platform_negotiation_pct_a=0.30,
    platform_negotiation_pct_b=0.30,
    audit_material_threshold=40_000,
    bilateral_global_search=False,
    onboarding_delay_bilateral=0,
    onboarding_delay_platform=0,
    order_batch_size=1_000,
    max_orders_per_quarter=12,
    # Commission tier defaults (must match config.py)
    # Tiers 1–4: Optuna V14 trial #115 (rounded). Tiers 5–9: original conservative values.
    ct1_rate=6.0, ct2_rate=1.0, ct3_rate=0.10, ct4_rate=0.010,
    ct5_rate=0.005, ct6_rate=0.002, ct7_rate=0.001, ct8_rate=0.0005,
    ct9_rate=0.0002,
)


# -- Sidebar ---------------------------------------------------------------

st.sidebar.title("Simulation Parameters")
st.sidebar.divider()

# ── Simulation ────────────────────────────────────────────────────────
st.sidebar.header("Simulation")
n_quarters = st.sidebar.number_input("Max quarters", 4, 400, _DEFAULTS["n_quarters"], step=4)
seed = st.sidebar.number_input("Random seed", 0, 9999, _DEFAULTS["seed_val"])
early_stop = st.sidebar.checkbox(
    "Early stop (8 stable quarters)",
    value=False,
    help="Stop the simulation when there are no platform joins or leaves "
         "for 8 consecutive quarters (ignoring disruption windows).",
)

# ── Platform Cost Reductions ─────────────────────────────────────────
with st.sidebar.expander("Platform Cost Reductions", expanded=True):
    st.caption("Platform costs as fraction of bilateral costs (symmetric A=B).")
    po_pct_a = st.slider(
        "PO cost (% of bilateral)", 0.05, 1.0, _DEFAULTS["po_pct_a"], 0.01,
        help="Platform PO cost = bilateral PO × this fraction",
    )
    po_pct_b = po_pct_a
    invoice_pct_a = st.slider(
        "Invoice cost (% of bilateral)", 0.05, 1.0, _DEFAULTS["invoice_pct_a"], 0.01,
        help="Platform invoice cost = bilateral invoice × this fraction",
    )
    invoice_pct_b = invoice_pct_a
    mgmt_pct_a = st.slider(
        "Mgmt hours (% of bilateral)", 0.05, 1.0, _DEFAULTS["mgmt_pct_a"], 0.01,
        help="Platform mgmt cost = bilateral × this fraction (lower = cheaper)",
    )
    mgmt_pct_b = mgmt_pct_a
    platform_negotiation_pct_a = st.slider(
        "Negotiation (% of bilateral)", 0.05, 1.0, _DEFAULTS["platform_negotiation_pct_a"], 0.01,
        help="Platform negotiation cost as fraction of bilateral negotiation cost.",
    )
    platform_negotiation_pct_b = platform_negotiation_pct_a
    search_pct_a = st.slider(
        "Search cost (% of bilateral)", 0.05, 1.0, _DEFAULTS["search_pct_a"], 0.01,
        help="Platform search = bilateral × this fraction",
    )
    search_pct_b = search_pct_a
# ── Bilateral Cost Baselines ──────────────────────────────────────────
with st.sidebar.expander("Bilateral Cost Baselines ($)"):
    bilateral_search = st.number_input(
        "Search cost", 1000, 50_000, _DEFAULTS["bilateral_search"], step=1000,
    )
    bilateral_po = st.number_input(
        "PO cost", 50, 1000, _DEFAULTS["bilateral_po"], step=10,
    )
    bilateral_invoice = st.number_input(
        "Invoice cost", 1.0, 50.0, float(_DEFAULTS["bilateral_invoice"]), step=0.5,
    )
    bilateral_negotiation = st.number_input(
        "Negotiation cost", 1_000, 50_000, _DEFAULTS["bilateral_negotiation"],
        step=1_000,
        help="One-time cost per new supplier relationship.",
    )
    platform_setup = st.number_input(
        "Platform setup cost", 5000, 100_000, _DEFAULTS["platform_setup"], step=5000,
    )
    onboarding = st.number_input(
        "Onboarding cost", 5000, 100_000, _DEFAULTS["onboarding"], step=5000,
    )
    audit = st.number_input(
        "Audit cost", 5000, 100_000, _DEFAULTS["audit"], step=5000,
    )
    audit_material_threshold = st.number_input(
        "Audit material threshold", 10_000, 1_000_000, _DEFAULTS["audit_material_threshold"],
        step=10_000,
        help="Audit triggered only for new suppliers with material value ≥ this.",
    )

# ── Admin Cost Scaling ────────────────────────────────────────────────
with st.sidebar.expander("Admin Cost Scaling"):
    st.caption("Log-scaled admin effort: max(1, 1 + log₁₀(mat / ref))")
    admin_scale_reference = st.number_input(
        "Reference ($)", 1_000, 1_000_000, _DEFAULTS["admin_scale_reference"],
        step=5_000,
        help="$30K ref → a $300K edge has 2× admin, $3M has 3×.",
    )
    admin_scale_cap_po_invoice = st.slider(
        "PO/Invoice cap (×)", 1.0, 5.0, _DEFAULTS["admin_scale_cap_po_invoice"], 0.1,
    )
    admin_scale_cap_audit = st.slider(
        "Audit cap (×)", 1.0, 5.0, _DEFAULTS["admin_scale_cap_audit"], 0.1,
    )
    admin_scale_cap_onboarding = st.slider(
        "Onboarding cap (×)", 1.0, 5.0, _DEFAULTS["admin_scale_cap_onboarding"], 0.1,
    )

# ── Platform OpEx ─────────────────────────────────────────────────────
with st.sidebar.expander("Platform OpEx"):
    fixed_opex = st.number_input(
        "Fixed OpEx ($/quarter)", 10_000, 5_000_000, _DEFAULTS["fixed_opex"], step=10_000,
    )
    var_opex = st.number_input(
        "Variable OpEx ($/member/q)", 10, 2000, _DEFAULTS["var_opex"], step=50,
    )

# ── Adoption ──────────────────────────────────────────────────────────
st.sidebar.header("Adoption")
initial_adoption = st.sidebar.slider(
    "Initial adoption fraction", 0.01, 0.30, _DEFAULTS["initial_adoption"], 0.01,
)
roi_horizon = st.sidebar.slider(
    "ROI horizon (quarters)", 2, 16, _DEFAULTS["roi_horizon"],
    help="Payback horizon for join/switch ROI. ROI = (realized savings × horizon − setup) / setup.",
)

# ── Speculative Values (Disruption-Conditional) ──────────────────────
with st.sidebar.expander("Speculative Values (disruption-only)"):
    st.caption(
        "These values are computed for diagnostics and logged in adoption records. "
        "They do NOT drive join/leave decisions under normal conditions. "
        "When disruptions are active, they are included in the ROI calculation."
    )
    discovery_premium = st.slider(
        "Discovery premium (%)", 0.0, 5.0, _DEFAULTS["discovery_premium"] * 100, 0.1,
        help="Buyer value for alternative suppliers on platform (% of trade value).",
    ) / 100
    visibility_premium = st.slider(
        "Visibility premium (%)", 0.0, 5.0, _DEFAULTS["visibility_premium"] * 100, 0.1,
        help="Seller value for being visible to new buyers on platform (% of trade value).",
    ) / 100

# ── Order Batching ────────────────────────────────────────────────────
with st.sidebar.expander("Order Batching"):
    order_batch_size = st.number_input(
        "Units per PO batch", 500, 50_000, _DEFAULTS["order_batch_size"], step=500,
        help="Smaller batches → more orders → larger PO/invoice admin savings.",
    )
    max_orders_per_quarter = st.number_input(
        "Max orders per quarter", 1, 20, _DEFAULTS["max_orders_per_quarter"], step=1,
    )

# ── OEM Annual Volumes ────────────────────────────────────────────────
with st.sidebar.expander("OEM Annual Volumes"):
    vol_oem1 = st.number_input("OEM1: ", 500, 100_000, _DEFAULTS["vol_oem1"], step=500)
    vol_oem2 = st.number_input("OEM2: ", 5_000, 1_000_000, _DEFAULTS["vol_oem2"], step=5_000)
    vol_oem3 = st.number_input("OEM3: ", 1_000, 500_000, _DEFAULTS["vol_oem3"], step=5_000)
    vol_oem4 = st.number_input("OEM4: ", 1_000, 200_000, _DEFAULTS["vol_oem4"], step=1_000)
    annual_growth = st.slider(
        "Annual demand growth (%)", -7.0, 14.0, _DEFAULTS["annual_growth"] * 100, 0.5,
        help="Drawn from N(mean, σ=3.5pp) each year. Bounds ±3σ.",
    ) / 100

# ── Onboarding Delay ──────────────────────────────────────────────────
with st.sidebar.expander("Onboarding Delay"):
    onboarding_delay_bilateral = st.number_input(
        "Bilateral (quarters)", 0, 8, _DEFAULTS["onboarding_delay_bilateral"], step=1,
        help="Quarters until a new bilateral supplier can deliver. 0 = immediate.",
    )
    onboarding_delay_platform = st.number_input(
        "Platform (quarters)", 0, 8, _DEFAULTS["onboarding_delay_platform"], step=1,
        help="Quarters until a new platform supplier can deliver. 0 = immediate.",
    )

# Community selector
_COMMUNITY_LABELS = {
    -1: "Random",
    0: "C0 — Manufacturing Process",
    1: "C1 — Electric Powertrain ⚡",
    2: "C2 — Mixed (Thermal/Body)",
    3: "C3 — Mixed (Electronics)",
    4: "C4 — Chassis",
    5: "C5 — Body & Interior",
    6: "C6 — Mixed (Small)",
    7: "C7 — General Parts",
    8: "C8 — Mixed (Small)",
    9: "C9 — Mixed (Small)",
    10: "C10 — Drivetrain",
    11: "C11 — ICE Powertrain",
    12: "C12 — Mixed (Small)",
}
_community_options = list(_COMMUNITY_LABELS.keys())
_community_format = lambda x: _COMMUNITY_LABELS.get(x, f"C{x}")

with st.sidebar.expander("Community Targets", expanded=False):
    community_seeding = st.checkbox(
        "Seed platforms into specific communities",
        value=_DEFAULTS["community_seeding"],
        help="When checked, seed one or both platforms into chosen Louvain communities at t=0. "
             "Unchecked = random seeding for both.",
    )
    platform_a_seed_community = st.selectbox(
        "A target community",
        options=_community_options,
        index=0,
        format_func=_community_format,
        disabled=not community_seeding,
    )
    platform_a_community_seed_pct = st.slider(
        "A community seed (%)", 0.10, 1.0, _DEFAULTS["platform_a_community_seed_pct"], 0.05,
        disabled=(not community_seeding) or (platform_a_seed_community == -1),
    )
    platform_b_seed_community = st.selectbox(
        "B target community",
        options=_community_options,
        index=0,
        format_func=_community_format,
        disabled=not community_seeding,
    )
    platform_b_community_seed_pct = st.slider(
        "B community seed (%)", 0.10, 1.0, _DEFAULTS["platform_b_community_seed_pct"], 0.05,
        disabled=(not community_seeding) or (platform_b_seed_community == -1),
    )
st.sidebar.header("Disruption")
disruption_enabled = st.sidebar.checkbox(
    "Enable disruptions",
    value=_DEFAULTS["disruption_enabled"],
    help="Recurring disruptions start after Q10. Platform members gain speculative value bonuses during disruptions.",
)
disruption_n_firms = st.sidebar.slider(
    "Firms disrupted per event", 1, 20, _DEFAULTS["disruption_n_firms"],
    disabled=not disruption_enabled,
)
disruption_frequency = st.sidebar.slider(
    "Disruption frequency (every N quarters)", 5, 100, _DEFAULTS["disruption_frequency"], step=5,
    disabled=not disruption_enabled,
)
bilateral_global_search = st.sidebar.checkbox(
    "Bilateral firms: global search",
    value=_DEFAULTS["bilateral_global_search"],
    help="Bilateral-only firms can search globally (but pay full search cost).",
)

# ── Stability (WTC & Cooldown) ────────────────────────────────────────
with st.sidebar.expander("Stability (WTC & Cooldown)"):
    cooldown_quarters = st.slider(
        "Cooldown (quarters)", 1, 16, 6,
        help="Minimum quarters before a firm can evaluate leaving.",
    )
    wtc_threshold = st.number_input(
        "WTC leave threshold ($/q)", 0.0, 50_000.0, 0.0, step=100.0,
        help="Firms leave when smoothed WTC drops below this. $0 = leave only when net negative.",
    )

# ── Commission Tiers ──────────────────────────────────────────────────
with st.sidebar.expander("Commission Tiers"):
    st.caption(
        "9-tier degressive rates shared by both platforms. "
    )
    tier_1_thresh = st.number_input("Tier 1 threshold ($)", 0, 10_000_000_000, 0, step=10_000, key="t1t")
    tier_1_rate = st.number_input("Tier 1 rate (%)", 0.0, 20.0, _DEFAULTS["ct1_rate"], step=0.1, key="t1r") / 100
    tier_2_thresh = st.number_input("Tier 2 threshold ($)", 0, 10_000_000_000, 10_000, step=10_000, key="t2t")
    tier_2_rate = st.number_input("Tier 2 rate (%)", 0.0, 20.0, _DEFAULTS["ct2_rate"], step=0.01, format="%.2f", key="t2r") / 100
    tier_3_thresh = st.number_input("Tier 3 threshold ($)", 0, 10_000_000_000, 100_000, step=10_000, key="t3t")
    tier_3_rate = st.number_input("Tier 3 rate (%)", 0.0, 20.0, _DEFAULTS["ct3_rate"], step=0.01, format="%.2f", key="t3r") / 100
    tier_4_thresh = st.number_input("Tier 4 threshold ($)", 0, 10_000_000_000, 1_000_000, step=100_000, key="t4t")
    tier_4_rate = st.number_input("Tier 4 rate (%)", 0.0, 20.0, _DEFAULTS["ct4_rate"], step=0.001, format="%.3f", key="t4r") / 100
    tier_5_thresh = st.number_input("Tier 5 threshold ($)", 0, 10_000_000_000, 5_000_000, step=1_000_000, key="t5t")
    tier_5_rate = st.number_input("Tier 5 rate (%)", 0.0, 20.0, _DEFAULTS["ct5_rate"], step=0.001, format="%.3f", key="t5r") / 100
    tier_6_thresh = st.number_input("Tier 6 threshold ($)", 0, 10_000_000_000, 10_000_000, step=1_000_000, key="t6t")
    tier_6_rate = st.number_input("Tier 6 rate (%)", 0.0, 20.0, _DEFAULTS["ct6_rate"], step=0.001, format="%.3f", key="t6r") / 100
    tier_7_thresh = st.number_input("Tier 7 threshold ($)", 0, 10_000_000_000, 50_000_000, step=10_000_000, key="t7t")
    tier_7_rate = st.number_input("Tier 7 rate (%)", 0.0, 20.0, _DEFAULTS["ct7_rate"], step=0.0001, format="%.4f", key="t7r") / 100
    tier_8_thresh = st.number_input("Tier 8 threshold ($)", 0, 10_000_000_000, 200_000_000, step=50_000_000, key="t8t")
    tier_8_rate = st.number_input("Tier 8 rate (%)", 0.0, 20.0, _DEFAULTS["ct8_rate"], step=0.0001, format="%.4f", key="t8r") / 100
    tier_9_thresh = st.number_input("Tier 9 threshold ($)", 0, 10_000_000_000, 1_000_000_000, step=100_000_000, key="t9t")
    tier_9_rate = st.number_input("Tier 9 rate (%)", 0.0, 20.0, _DEFAULTS["ct9_rate"], step=0.0001, format="%.4f", key="t9r") / 100

st.sidebar.divider()

# ── Playback ──────────────────────────────────────────────────────────
st.sidebar.header("Playback")
step_mode = st.sidebar.checkbox("Step-by-step mode", value=False)
speed = st.sidebar.number_input("Playback speed (sec/quarter)", min_value=0.1, max_value=30.0,
    value=1.0, step=0.1, format="%.1f", disabled=not step_mode)

st.sidebar.divider()


# -- Action buttons --------------------------------------------------------

col_run, col_pause, col_reset = st.sidebar.columns(3)
run_clicked = col_run.button("\u25b6 Run", type="primary", use_container_width=True)
pause_clicked = col_pause.button("\u23f8 Pause", use_container_width=True,
    disabled=not st.session_state.running)
reset_clicked = col_reset.button("\U0001f504 Reset", use_container_width=True)


# -- Main area -------------------------------------------------------------

st.title("B2B Procurement Platform Diffusion \u2014 ABM Dashboard")


def _build_config() -> ExperimentConfig:
    shared_tiers = [
        (tier_1_thresh, tier_1_rate),
        (tier_2_thresh, tier_2_rate),
        (tier_3_thresh, tier_3_rate),
        (tier_4_thresh, tier_4_rate),
        (tier_5_thresh, tier_5_rate),
        (tier_6_thresh, tier_6_rate),
        (tier_7_thresh, tier_7_rate),
        (tier_8_thresh, tier_8_rate),
        (tier_9_thresh, tier_9_rate),
    ]
    return ExperimentConfig(
        n_quarters=n_quarters,
        platform_search_pct_a=search_pct_a,
        platform_search_pct_b=search_pct_b,
        platform_po_pct_a=po_pct_a,
        platform_po_pct_b=po_pct_b,
        platform_invoice_pct_a=invoice_pct_a,
        platform_invoice_pct_b=invoice_pct_b,
        platform_mgmt_pct_a=mgmt_pct_a,
        platform_mgmt_pct_b=mgmt_pct_b,
        bilateral_search_cost=float(bilateral_search),
        bilateral_po_cost=float(bilateral_po),
        bilateral_invoice_cost=float(bilateral_invoice),
        platform_setup_cost=float(platform_setup),
        onboarding_cost=float(onboarding),
        audit_cost=float(audit),
        platform_fixed_opex=float(fixed_opex),
        platform_var_opex_per_member=float(var_opex),
        initial_adoption_frac=initial_adoption,
        annual_growth_rate=annual_growth,
        community_seeding=community_seeding,
        platform_a_seed_community=int(platform_a_seed_community),
        platform_a_community_seed_pct=platform_a_community_seed_pct,
        platform_b_seed_community=int(platform_b_seed_community),
        platform_b_community_seed_pct=platform_b_community_seed_pct,
        disruption_enabled=disruption_enabled,
        disruption_frequency=disruption_frequency,
        disruption_n_firms=disruption_n_firms,
        platform_cooldown_quarters=cooldown_quarters,
        wtc_leave_threshold=float(wtc_threshold),
        platform_discovery_premium_pct=discovery_premium,
        platform_visibility_premium_pct=visibility_premium,
        admin_scale_reference=float(admin_scale_reference),
        admin_scale_cap_po_invoice=float(admin_scale_cap_po_invoice),
        admin_scale_cap_audit=float(admin_scale_cap_audit),
        admin_scale_cap_onboarding=float(admin_scale_cap_onboarding),
        bilateral_negotiation_cost=float(bilateral_negotiation),
        platform_negotiation_pct_a=float(platform_negotiation_pct_a),
        platform_negotiation_pct_b=float(platform_negotiation_pct_b),
        audit_material_threshold=float(audit_material_threshold),
        bilateral_global_search=bilateral_global_search,
        onboarding_delay_bilateral=int(onboarding_delay_bilateral),
        onboarding_delay_platform=int(onboarding_delay_platform),
        order_batch_size=int(order_batch_size),
        max_orders_per_quarter=int(max_orders_per_quarter),
        commission_tiers_a=shared_tiers,
        commission_tiers_b=shared_tiers,
        annual_volume={
            2000682: vol_oem1,
            2000766: vol_oem2,
            2000621: vol_oem3,
            2000772: vol_oem4,
        },
        roi_horizon_quarters=roi_horizon,
        early_stop_quarters=8 if early_stop else 0,
        seed=seed,
    )


def _create_model() -> SupplyChainModel:
    config = _build_config()
    data = get_data()
    return SupplyChainModel(config=config, loaded_data=data)


# -- Handle button actions -------------------------------------------------

if reset_clicked:
    st.session_state.model = None
    st.session_state.df = None
    st.session_state.running = False
    st.session_state.paused = False
    st.rerun()

if pause_clicked:
    st.session_state.paused = True
    st.session_state.running = False

if run_clicked:
    st.session_state.paused = False
    st.session_state.running = True

    if st.session_state.model is None:
        st.session_state.model = _create_model()

    if not step_mode:
        with st.spinner("Running simulation \u2026"):
            st.session_state.model.run_model()
        st.session_state.df = st.session_state.model.datacollector.get_model_vars_dataframe()
        st.session_state.running = False
        reason = st.session_state.model.termination_reason or "Completed"
        st.toast(f"Simulation ended: {reason}", icon="\u2705")
        st.rerun()


# -- Auto-advance (step mode) ---------------------------------------------

model = st.session_state.model
if model is not None and st.session_state.running and step_mode:
    if not model.check_termination():
        model.step()
        st.session_state.df = model.datacollector.get_model_vars_dataframe()
    else:
        st.session_state.running = False
        if model.termination_reason:
            st.toast(f"Simulation ended: {model.termination_reason}", icon="\U0001f3c1")


# -- Display results -------------------------------------------------------

model = st.session_state.model
df = st.session_state.df

if model is None or df is None or df.empty:
    st.info("Set parameters in the sidebar and click **\u25b6 Run** to begin.")
    st.stop()

# -- Status ----------------------------------------------------------------

last = df.iloc[-1]
q = int(last["Quarter"])
if model.termination_reason:
    st.success(f"**Q{q}/{n_quarters}** \u2014 {model.termination_reason}")
elif st.session_state.running:
    st.info(f"**Q{q}/{n_quarters}** \u2014 Running\u2026")
elif st.session_state.paused:
    st.warning(f"**Q{q}/{n_quarters}** \u2014 Paused")
else:
    st.info(f"**Q{q}/{n_quarters}**")

# -- KPI cards -------------------------------------------------------------

c1, c2, c3 = st.columns(3)
c1.metric("A Market Share", f"{last['Platform_A_Share']:.1%}")
c2.metric("B Market Share", f"{last['Platform_B_Share']:.1%}")
hhi = last.get("HHI", 0)
c3.metric("HHI", f"{hhi:.3f}")

c4, c5, c6 = st.columns(3)
c4.metric("A Profit/Q", f"${last.get('Platform_A_Profit', 0):,.0f}")
c5.metric("B Profit/Q", f"${last.get('Platform_B_Profit', 0):,.0f}")
total_adopted = sum(1 for fa in model.firm_agents.values() if fa.platform_memberships)
c6.metric("Firms Adopted", f"{total_adopted}/{len(model.firm_agents)}")

# -- Tabs ------------------------------------------------------------------

tab_net, tab_charts, tab_data = st.tabs(["🔗 Network", "📊 Charts", "📋 Raw Data"])

# -- Supply Chain Network --------------------------------------------------

with tab_net:
    lcol1, lcol2, lcol3, lcol4 = st.columns(4)
    lcol1.markdown("\U0001f534 **Platform A**")
    lcol2.markdown("\U0001f7e2 **Platform B**")
    lcol3.markdown("\U0001f535 **No Platform / Bilateral**")
    lcol4.markdown("\u2b50 **OEM (star shape)**")

    html_path = build_network_html(model, filename="dashboard_network.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=700, scrolling=True)

# -- Charts ----------------------------------------------------------------

with tab_charts:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["Quarter"], df["Platform_A_Share"] * 100, "r-o", label="Platform A", ms=4)
    ax.plot(df["Quarter"], df["Platform_B_Share"] * 100, "g-s", label="Platform B", ms=4)
    bilateral = (1.0 - df["Platform_A_Share"] - df["Platform_B_Share"]) * 100
    ax.plot(df["Quarter"], bilateral, "b--", label="Bilateral", alpha=0.5)
    ax.set_xlabel("Quarter"); ax.set_ylabel("Market Share (%)")
    ax.set_title("Market Share Over Time")
    ax.legend(fontsize=8); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["Quarter"], df["Total_Trade_Volume"] / 1e6, "k-o", label="Total Trade", ms=4)
        ax.plot(df["Quarter"], df["Platform_A_Volume"] / 1e6, "r-s", label="Platform A", ms=4)
        ax.plot(df["Quarter"], df["Platform_B_Volume"] / 1e6, "g-^", label="Platform B", ms=4)
        ax.set_xlabel("Quarter"); ax.set_ylabel("USD millions")
        ax.set_title("Trade Volume Over Time")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close(fig)

    with r2c2:
        fig, ax = plt.subplots(figsize=(7, 4))
        profit_a = df.get("Platform_A_Profit", pd.Series([0]*len(df)))
        profit_b = df.get("Platform_B_Profit", pd.Series([0]*len(df)))
        opex_a = df.get("Platform_A_OpEx", pd.Series([0]*len(df)))
        opex_b = df.get("Platform_B_OpEx", pd.Series([0]*len(df)))
        ax.plot(df["Quarter"], profit_a / 1e3, "r-o", label="A Profit", ms=4)
        ax.plot(df["Quarter"], profit_b / 1e3, "g-s", label="B Profit", ms=4)
        ax.plot(df["Quarter"], opex_a / 1e3, "r--", label="A OpEx", alpha=0.5)
        ax.plot(df["Quarter"], opex_b / 1e3, "g--", label="B OpEx", alpha=0.5)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("Quarter"); ax.set_ylabel("USD thousands")
        ax.set_title("Platform Profit & OpEx")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close(fig)

    # Effective commission rate
    if "Effective_Rate_A" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 3.5))
        ax.plot(df["Quarter"], df["Effective_Rate_A"] * 100, "r-o", label="Platform A", ms=4)
        ax.plot(df["Quarter"], df["Effective_Rate_B"] * 100, "g-s", label="Platform B", ms=4)
        ax.set_xlabel("Quarter"); ax.set_ylabel("Effective Rate (%)")
        ax.set_title("Effective Commission Rate Over Time")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close(fig)

    # Community bar chart
    comm_shares = last.get("Community_Platform_Shares", {})
    if comm_shares:
        cids = sorted(comm_shares.keys())
        comm_a = [comm_shares[c]["A"] for c in cids]
        comm_b = [comm_shares[c]["B"] for c in cids]
        comm_bi = [comm_shares[c]["bilateral"] for c in cids]

        comm_sizes = {}
        for agent in model.firm_agents.values():
            cid = agent.community_id
            if cid >= 0:
                comm_sizes[cid] = comm_sizes.get(cid, 0) + 1

        labels = [
            f"C{c} ({comm_sizes.get(c, 0)})"
            for c in cids
        ]

        fig, ax = plt.subplots(figsize=(14, 5))
        x = range(len(cids))
        ax.bar(x, comm_a, color="#EA4335", label="Platform A")
        ax.bar(x, comm_b, bottom=comm_a, color="#34A853", label="Platform B")
        ax.bar(x, comm_bi, bottom=[a + b for a, b in zip(comm_a, comm_b)],
               color="#4285F4", alpha=0.5, label="Bilateral")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Share")
        ax.set_title("Platform Share by Community")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        # Community dominance table
        dom_rows = []
        for c in cids:
            s = comm_shares[c]
            dom_rows.append({
                "Community": f"C{c}",
                "Firms": comm_sizes.get(c, 0),
                "A %": f"{s['A']:.0%}",
                "B %": f"{s['B']:.0%}",
                "Bilateral %": f"{s['bilateral']:.0%}",
                "Dominance": f"{s.get('dominance_index', 0):.2f}",
                "Trade A %": f"{s.get('trade_A', 0):.0%}",
                "Trade B %": f"{s.get('trade_B', 0):.0%}",
                "Trade Bilateral %": f"{s.get('trade_bilateral', 0):.0%}",
            })
        st.subheader("Community Dominance Index & Trade Routing")
        st.dataframe(pd.DataFrame(dom_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No community data available yet.")

    # Disruption chart
    if "Cascade_Failures" in df.columns and df["Cascade_Failures"].sum() > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.5))
        ax1.bar(df["Quarter"], df["Disrupted_Firms"], color="#FF6D00", alpha=0.8)
        ax1.set_xlabel("Quarter"); ax1.set_ylabel("Firms")
        ax1.set_title("Disrupted Firms (Unavailable)")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(df["Quarter"], df["Cascade_Failures"], color="#D50000", alpha=0.8)
        ax2.set_xlabel("Quarter"); ax2.set_ylabel("Failures")
        ax2.set_title("Demand Cascade Failures")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # Supplier switches
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.bar(df["Quarter"], df["Supplier_Switches"], color="#F4B400", alpha=0.8)
    ax.set_xlabel("Quarter"); ax.set_ylabel("Switches")
    ax.set_title("Supplier Switches Per Quarter")
    ax.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig); plt.close(fig)

    # Membership flow
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.5), sharey=True)
    ax1.bar(df["Quarter"], df["Firms_Joined_A"], color="#EA4335", alpha=0.7, label="Joined")
    ax1.bar(df["Quarter"], -df["Firms_Left_A"], color="#EA433580", alpha=0.5, label="Left")
    ax1.set_title("Platform A \u2014 Members"); ax1.set_xlabel("Quarter")
    ax1.set_ylabel("Firms"); ax1.legend(); ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(df["Quarter"], df["Firms_Joined_B"], color="#34A853", alpha=0.7, label="Joined")
    ax2.bar(df["Quarter"], -df["Firms_Left_B"], color="#34A85380", alpha=0.5, label="Left")
    ax2.set_title("Platform B \u2014 Members"); ax2.set_xlabel("Quarter")
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

# -- Raw Data --------------------------------------------------------------

with tab_data:
    st.subheader("Model-Level Time Series")
    st.dataframe(df, use_container_width=True)

    st.subheader("Firm Summary (Final Quarter)")
    G = model.supply_graph
    firm_rows = []
    for fid, agent in model.firm_agents.items():
        role = "OEM" if agent.is_oem else f"Tier {agent.tier_depth}"
        platforms = ", ".join(sorted(agent.platform_memberships)) or "None"

        # Buyer-side channels (this firm buys from suppliers)
        channels = list(agent.edge_channels.values())
        # Seller-side channels (this firm sells to buyers)
        for buyer_id in G.successors(fid):
            buyer_agent = model.firm_agents.get(buyer_id)
            if buyer_agent is not None:
                ch = buyer_agent.edge_channels.get(fid, "bilateral")
                channels.append(ch)

        n = max(len(channels), 1)
        bi_pct = channels.count("bilateral") / n
        pa_pct = channels.count("platform_a") / n
        pb_pct = channels.count("platform_b") / n

        firm_rows.append({
            "Firm ID": fid,
            "Name": agent.firm_name,
            "Country": agent.country,
            "Role": role,
            "Community": agent.community_id,
            "In-degree": G.in_degree(fid),
            "Out-degree": G.out_degree(fid),
            "Platforms": platforms,
            "Bilateral %": f"{bi_pct:.0%}",
            "Platform A %": f"{pa_pct:.0%}",
            "Platform B %": f"{pb_pct:.0%}",
            "Products": len(agent.product_catalog),
        })

    firm_df = pd.DataFrame(firm_rows).sort_values(["Community", "Role"])
    st.dataframe(firm_df, use_container_width=True, hide_index=True)

    # Central Nodes (top-20 by in-degree)
    st.subheader("Central Nodes \u2014 Adoption Logs")
    st.caption("Top 20 firms by in-degree. Shows every join/leave/seed event.")
    in_degrees = {fid: G.in_degree(fid) for fid in model.firm_agents}
    top_nodes = sorted(in_degrees, key=in_degrees.get, reverse=True)[:20]
    for fid in top_nodes:
        agent = model.firm_agents[fid]
        plat_str = ", ".join(sorted(agent.platform_memberships)) or "None"
        partners_total = in_degrees[fid] + G.out_degree(fid)
        partners_on_a = sum(1 for p in set(G.predecessors(fid)) | set(G.successors(fid))
                            if "A" in model.firm_agents.get(p, agent).platform_memberships)
        partners_on_b = sum(1 for p in set(G.predecessors(fid)) | set(G.successors(fid))
                            if "B" in model.firm_agents.get(p, agent).platform_memberships)
        st.markdown(
            f"**{agent.firm_name}** (ID {fid}) \u2014 in-deg {in_degrees[fid]}, "
            f"C{agent.community_id}, platform: {plat_str}, "
            f"partners on A: {partners_on_a}/{partners_total}, "
            f"B: {partners_on_b}/{partners_total}"
        )
        if agent.adoption_log:
            log_df = pd.DataFrame(agent.adoption_log)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            st.caption("  No adoption events (stayed bilateral the entire simulation).")

    # Full Adoption Log
    st.subheader("Full Adoption Log (All Firms)")
    all_log_rows = []
    for fid, agent in model.firm_agents.items():
        for entry in agent.adoption_log:
            row = {"Firm ID": fid, "Name": agent.firm_name,
                   "Community": agent.community_id,
                   "In-degree": G.in_degree(fid)}
            row.update(entry)
            all_log_rows.append(row)
    if all_log_rows:
        all_log_df = pd.DataFrame(all_log_rows).sort_values(["quarter", "Firm ID"])
        st.dataframe(all_log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No adoption events recorded.")

# -- Auto-rerun for step mode ----------------------------------------------

if st.session_state.running and step_mode:
    time.sleep(speed)
    st.rerun()
