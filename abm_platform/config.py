"""
Configuration for the B2B Procurement Platform Diffusion ABM.

All hard-coded values cite their sources with URLs where available.
Experiment parameters are tuneable through ExperimentConfig.
"""

from dataclasses import dataclass, field

# ======================================================================
# Source-backed constants
# ======================================================================

# --- Seasonal demand (quarterly share of annual volume) ---------------
# Source: MARKLINES https://www.marklines.com/en/statistics/flash_sales/automotive-sales-in-china-by-month-2023
# years 2023, 2024, 2025
SEASONAL_WEIGHTS = [0.21, 0.24, 0.25, 0.30]  # Q1, Q2, Q3, Q4

# --- Annual production volume -----------------------------------------
DEFAULT_ANNUAL_VOLUME = {
    2000682: 2_500,
    2000766: 32_000,
    2000621: 50_000,
    2000772: 15_000,
}

# --- Annual demand growth rate ----------------------------------------
# Source: MARKLINES https://www.marklines.com/en/statistics/flash_sales/automotive-sales-in-china-by-month-2023
ANNUAL_GROWTH_RATE = 0.035

# --- Certification -> Industry Performance Mapping -----------------------------
# Sources:
#   Covaci & Țîțu (2025). "Impact of Process and Supplier Audits on Product Reliability"
#   Țîțu & Pop (2025). "Integrated Quality Management for Automotive Services"
#   Ostadi, Aghdasi & Kazemzadeh (2010). "The impact of ISO/TS 16949 on automotive industries"

QUALITY_METRICS = {
    "uncertified": {
        "rework_rate": 0.078,      # Baseline rework rate of 7.8%
        "defect_index": 1.0,       # Normalized baseline
        "recall_index": 1.0        # Normalized baseline
    },
    "ISO9001": {
        # ISO 9001 serves as the foundational process-oriented model
        "rework_rate": 0.026,      # Decreased to 2.6% when integrated with operational principles
        "defect_index": 0.90,      # General improvement in manufacturing outcomes
        "recall_index": 0.95       
    },
    "IATF16949": {
        # Automotive-specific standard with mandatory high-frequency auditing
        "rework_rate": 0.026,      
        "defect_reduction": 0.25,  # Up to 25% fewer defects vs standard cycles
        "recall_reduction": 0.30,  # 30% reduction in recall incidents
        "efficiency_gain": 0.16    # Technician efficiency increase from 89% to 105%
    }
}


# --- Regional & Tier-based EBIT margins (Verified March 2026) ----------
# Sources: 
#   - CPCA (Dec 2025): China Auto Industry Avg Margin: 4.4%
#   - Roland Berger/Lazard (2025 Study): Europe (3.6%), China (5.7%), S. Korea (3.4%)
#   - S&P Global (Dec 2025): North American Supplier EBIT: 6.2%
# -----------------------------------------------------------------------

# Tier 0 is specifically for Chinese OEMs per user requirement
CHINESE_OEM_MARGIN = 0.044  

MARGIN_BY_REGION_TIER = {
    "China":     {1: 0.057, 2: 0.048, 3: 0.040},  # High EV penetration support
    "India":     {1: 0.062, 2: 0.055, 3: 0.045},  # Strongest growth region
    "USA":       {1: 0.062, 2: 0.050, 3: 0.042},  # Resilience in truck/SUV segments
    "Mexico":    {1: 0.062, 2: 0.050, 3: 0.042},  # Near-shoring benefit from USA
    "Japan":     {1: 0.042, 2: 0.038, 3: 0.035},  # Stagnant ICE volumes
    "South Korea": {1: 0.034, 2: 0.032, 3: 0.030}, # High labor/energy pressure
    "Germany":   {1: 0.036, 2: 0.032, 3: 0.030},  # High energy/transformation costs
    "Spain":     {1: 0.036, 2: 0.032, 3: 0.030},  # Similar to Germany/EU avg
    "Thailand":  {1: 0.050, 2: 0.045, 3: 0.040},  # Hub for Chinese EV expansion
    "Malaysia":  {1: 0.050, 2: 0.045, 3: 0.040},  # Growing electronics/semiconductor hub
}

# --- Procurement staff wages ------------------------------------------
# Source: ERI SalaryExpert (erieri.com / salaryexpert.com) — Procurement Officer
#   role, survey data from employers and employees, 2024–2025.
#   Cross-checked against Glassdoor and Salary.com where available.
#   Hourly rates derived from annual salary ÷ 2,080 hrs, converted to USD
#   at 2024 exchange rates.
#
#   China   (CNY ~7.1/USD):  erieri.com/salary/job/procurement-officer/china
#   India   (INR ~83/USD):   salaryexpert.com/salary/job/procurement-officer/india
#   Japan   (JPY ~150/USD):  erieri.com/salary/job/procurement-specialist/japan
#   Germany (EUR ~1.09/USD): erieri.com/salary/job/procurement-officer/germany
#   Korea   (KRW ~1330/USD): erieri.com/salary/job/procurement-manager/south-korea
#   USA:                     salary.com + erieri.com/salary/job/procurement-officer/united-states
#   Mexico  (MXN ~17/USD):   glassdoor.com — Procurement Specialist, Mexico City
#   Malaysia(MYR ~4.7/USD):  salaryexpert.com/salary/job/procurement-officer/malaysia
#   Thailand(THB ~36/USD):   glassdoor.com — Procurement Officer, Thailand
#   Spain   (EUR ~1.09/USD): payscale.com/research/ES/Job=Procurement_Specialist/Salary
#                            + erieri.com/salary/job/procurement-analyst/spain/madrid
#
HOURLY_COMPENSATION_USD = {
    "China":    17.0,   # ~¥250k/yr ÷ 2080 ÷ 7.1  (ERI SalaryExpert, procurement officer)
    "Japan":    22.0,   # ~¥5.6m/yr ÷ 2080 ÷ 150   (ERI, procurement officer; note: JPY weakness depresses USD figure)
    "India":     6.0,   # ~₹1.04m/yr ÷ 2080 ÷ 83   (ERI SalaryExpert, procurement officer)
    "Germany":  26.0,   # ~€53.7k/yr ÷ 2080 ÷ 1.09 (ERI SalaryExpert, procurement officer)
    "Korea":    36.0,   # ~₩100m/yr ÷ 2080 ÷ 1330  (ERI, procurement manager; officer data limited)
    "USA":      33.0,   # $68.9k/yr ÷ 2080          (ERI SalaryExpert + salary.com, procurement officer)
    "Mexico":    7.5,   # ~MXN 276k/yr ÷ 2080 ÷ 17  (Glassdoor, procurement specialist Mexico City)
    "Malaysia":  8.5,   # ~MYR 82.9k/yr ÷ 2080 ÷ 4.7 (ERI SalaryExpert, procurement officer)
    "Thailand":  5.0,   # ~THB 390k/yr ÷ 2080 ÷ 36  (Glassdoor, procurement officer Thailand — limited sample)
    "Spain":    17.0,   # ~€32k/yr ÷ 2080 ÷ 1.09    (PayScale + ERI Madrid procurement analyst)
}
PROCUREMENT_MGR_MULT   = 2.0   # manager wage = hourly_comp * 2.0

# --- Import duty rate matrix ------------------------------------------
# Source:
#   WTO Tariff Profiles 2024
#     https://www.wto.org/english/res_e/statis_e/statis_e.htm
#   UNCTAD TRAINS Database
#     https://trainsonline.unctad.org/
_DUTY_DEFAULT = 0.05  # fallback 5 %
IMPORT_DUTY_RATE: dict[tuple[str, str], float] = {}
for c in HOURLY_COMPENSATION_USD:
    IMPORT_DUTY_RATE[(c, c)] = 0.0
for a in ("China", "Thailand", "Malaysia"):
    for b in ("China", "Thailand", "Malaysia"):
        IMPORT_DUTY_RATE[(a, b)] = 0.025  # RCEP/ACFTA ~0-5 %
    IMPORT_DUTY_RATE[(a, a)] = 0.0
for pair in [("China", "Japan"), ("Japan", "China")]:
    IMPORT_DUTY_RATE[pair] = 0.015
for pair in [("China", "Korea"), ("Korea", "China")]:
    IMPORT_DUTY_RATE[pair] = 0.025
for pair in [("Germany", "Spain"), ("Spain", "Germany")]:
    IMPORT_DUTY_RATE[pair] = 0.0    # Intra-EU
for pair in [("China", "Germany"), ("Germany", "China"),
             ("China", "Spain"),   ("Spain", "China")]:
    IMPORT_DUTY_RATE[pair] = 0.07
for pair in [("China", "USA"), ("USA", "China")]:
    IMPORT_DUTY_RATE[pair] = 0.05
for pair in [("China", "Mexico"), ("Mexico", "China")]:
    IMPORT_DUTY_RATE[pair] = 0.10
for pair in [("China", "India"), ("India", "China")]:
    IMPORT_DUTY_RATE[pair] = 0.11

# --- Transportation ---------------------------------------------------
# Source:
#   IRU Road Transport Pricing 2024
#     https://www.iru.org/resources/iru-library
FREIGHT_RATE_ROAD     = 0.07   # USD / ton-km (domestic / same-continent)
FREIGHT_RATE_SEA      = 0.02   # USD / ton-km (cross-ocean effective)
FREIGHT_RATE_RAIL     = 0.04   # USD / ton-km (cross-border continental)

# Country pairs separated by an ocean (need sea freight)
OCEAN_PAIRS = {
    frozenset({"China", "Germany"}), frozenset({"China", "Spain"}),
    frozenset({"China", "USA"}),     frozenset({"China", "Mexico"}),
    frozenset({"Japan", "Germany"}), frozenset({"Japan", "Spain"}),
    frozenset({"Japan", "USA"}),     frozenset({"Japan", "Mexico"}),
    frozenset({"Japan", "India"}),   frozenset({"Korea", "Germany"}),
    frozenset({"Korea", "Spain"}),   frozenset({"Korea", "USA"}),
    frozenset({"Korea", "Mexico"}),  frozenset({"Korea", "India"}),
    frozenset({"India", "Germany"}), frozenset({"India", "Spain"}),
    frozenset({"India", "USA"}),     frozenset({"India", "Mexico"}),
    frozenset({"India", "Japan"}),   frozenset({"India", "Korea"}),
}

# --- Supply-chain cost model constants --------------------------------
#
# SOURCES:
#   [1] ERP / platform setup costs: Odoo / SME ERP benchmark
#       https://shivlab.com/blog/erp-implementation-cost-breakdown-sme-guide/
#   [2] Inventory holding cost: industry consensus 15–30% annual
#       https://www.finaleinventory.com/accounting-and-inventory-software/holding-cost
#       https://www.latentview.com/blog/inventory-holding-costs-in-global-connected-supply-chains/

# ── Supplier audit (one-time, for uncertified/new strategic suppliers) ──
COST_AUDIT_USD = 15_000

# ── Supplier management (hours per quarter) ──
H_MGMT_STRATEGIC = 312.0      # ~24 h/wk × 13 wks/qtr
H_MGMT_NON_STRATEGIC = 104.0  # ~8 h/wk × 13 wks/qtr

# ── Supplier onboarding (one-time cost + elapsed time) ──
COST_ONBOARDING_USD = 30_000

# ── Per-order cost (USD; scaled by country wage index internally) ──
COST_PER_PO_BILATERAL_USD = 220.0

# ── Per-invoice cost (USD; scaled by country wage index internally) ──
COST_PER_INVOICE_BILATERAL_USD = 9.40

# ── Supplier search (monetary cost + elapsed time) ──
COST_SEARCH_BILATERAL_USD = 15_000

# ── Platform setup (one-time per firm) ──
# Full on-premise ERP integration would be $50K–$150K+.
COST_PLATFORM_SETUP_USD = 50_000

# One-time negotiation cost for establishing a new bilateral supplier relationship.
COST_NEGOTIATION_BILATERAL_USD = 8_000

# ── Inventory holding ──
# Industry consensus: 15–30% annual holding cost rate. [2]
# 25% annual (= 6.25% quarterly) is well-supported for manufacturing/
R_HOLDING_QUARTERLY = 0.0625  # 25% annual ÷ 4

# Safety-stock parameters — standard O.R. / inventory theory values. (Degraeve et al., 2005)
# Z = 1.645 for 95% service level (standard normal); rounded to 1.65.
# sigma = 20% of Q is a common moderate-variability assumption.
Z_SERVICE = 1.65              # 95% service level (standard normal)
SIGMA_DEMAND_FRAC = 0.20      # σ = 0.20 × Q (moderate demand variability)

# ---------------------------------------------------------------------------
# Volume discount via Wright's Law / Experience Curve
# ---------------------------------------------------------------------------
# The unit-cost reduction with quantity is modelled as a power law derived
# from Wright's Law (T.P. Wright, 1936, "Factors Affecting the Cost of
# Airplanes", J. Aeronautical Sciences 3(4):122-128,
# https://doi.org/10.2514/8.155).
#
# Standard form:  C(Q) = C_ref * (Q / Q_ref)^b
#   where  b = log(LR) / log(2)   and LR is the "learning rate"
#
# The *discount multiplier* (price relative to the reference quantity) is:
#   discount(Q) = (Q / Q_ref)^b,   clipped to [max_discount, 1.0]
#
# Parameter calibration:
#   LR = 0.90  (90 % learning curve, i.e. −10 % cost per doubling of Q)
#   This corresponds to b ≈ −0.152.
#
#   Empirical basis:
#     • Wong (2013) fitted supplier price quotations for 17 critical parts and
#       found learning rates of 70–95 %; the published benchmark for purchased
#       parts is 85–88 %.  A 90 % curve sits in the conservative / mid-range,
#       appropriate for moderately automated automotive Tier-2 components.
#       DOI: 10.1155/2013/584762
#     • Experience-curve b-values of 0.75–0.90 are widely reported across
#       manufacturing industries (Wright's Law, Wikipedia / BCG experience
#       curve literature).
#     • For highly automated commodity parts use LR = 0.95 (b ≈ −0.074).
#     • For labour-intensive sub-assemblies use LR = 0.85 (b ≈ −0.234).
#
#   Maximum discount is capped so the price floor is 70 % of the reference
#   price (i.e. a 30 % maximum discount), consistent with the upper bound of
#   learning-curve effects reported in the procurement literature.

import math

LEARNING_RATE = 0.90                        # LR: fraction of cost retained per doubling
VOLUME_DISCOUNT_B = math.log(LEARNING_RATE) / math.log(2)  # ≈ −0.152
VOLUME_DISCOUNT_MAX = 0.30                  # maximum discount: never below 70 % of ref price

def volume_discount_multiplier(Q: int, Q_ref: int) -> float:
    """
    Return the price multiplier for order quantity Q relative to Q_ref.

    Multiplier = (Q / Q_ref)^b, clipped to [1 - VOLUME_DISCOUNT_MAX, 1.0].

    Sources:
      Wright (1936) DOI 10.2514/8.155
      Wong (2013)   DOI 10.1155/2013/584762
    """
    if Q <= 0 or Q_ref <= 0:
        return 1.0
    raw = (Q / Q_ref) ** VOLUME_DISCOUNT_B
    return max(1.0 - VOLUME_DISCOUNT_MAX, min(1.0, raw))

# ---------------------------------------------------------------------------
# MOQ tiers — automotive supplier market
# ---------------------------------------------------------------------------
MOQ_TIERS = [50, 500, 2_000, 5_000, 25_000, 50_000]

def snap_to_moq_tier(raw_moq: int) -> int:
    """Return the smallest MOQ tier >= raw_moq, or raw_moq if above all tiers."""
    for tier in MOQ_TIERS:
        if tier >= raw_moq:
            return tier
    return raw_moq          # above highest tier — use negotiated value as-is

# --- Regional search (bilateral visibility) ---------------------------
# Bilateral trade: firms can only discover suppliers in their own region.
# Platforms provide global visibility across all regions.
REGIONS = {
    "East Asia":      {"China", "Japan", "Korea"},
    "South Asia":     {"India"},
    "Southeast Asia": {"Malaysia", "Thailand"},
    "Europe":         {"Germany", "Spain"},
    "Americas":       {"USA", "Mexico"},
}

COUNTRY_TO_REGION: dict[str, str] = {}
for _region, _countries in REGIONS.items():
    for _c in _countries:
        COUNTRY_TO_REGION[_c] = _region

# --- Platform economics -----------------------------------------------
# Per-Transaction Tiered Commission: discrete tiers based on order
# material value.  Degressive rates
#   effective_commission = material × tier_rate(material)


def tiered_commission_rate(material_cost: float, tiers: list[tuple[float, float]]) -> float:
    """Return the commission rate for *material_cost* from a sorted tier list.

    Tiers are (threshold, rate) tuples sorted ascending by threshold.
    The rate from the highest tier whose threshold ≤ material_cost applies.
    """
    rate = tiers[0][1] if tiers else 0.0
    for threshold, tier_rate in tiers:
        if material_cost >= threshold:
            rate = tier_rate
        else:
            break
    return rate


def admin_scale_factor(material_cost: float, reference: float = 10_000.0) -> float:
    """Log-scaled admin effort multiplier for edge-level costs.

    Large contracts incur proportionally more admin work (more POs,
    approvals, compliance checks, etc.).  The scaling is logarithmic
    so that a $10 M contract is ~4× the admin of a $10 K contract,
    not 1 000× depending on the reference value.

    Formula:  max(1.0, 1 + log10(material_cost / reference))
    """
    if material_cost <= 0 or reference <= 0:
        return 1.0
    ratio = material_cost / reference
    if ratio <= 1.0:
        return 1.0
    return 1.0 + math.log10(ratio)


# ======================================================================
# Experiment parameters (knobs)
# ======================================================================

@dataclass
class ExperimentConfig:
    """Tuneable simulation parameters."""

    # Simulation horizon (max quarters; may terminate earlier)
    n_quarters: int = 40

    # ── Tiered commission model ──────────────────────────────────
    # Per-transaction tiered rate — list of (threshold, rate) tuples.
    # Orders are charged the rate of the tier whose threshold they exceed.
    # Tiers must be sorted ascending by threshold.
    commission_tiers_a: list = field(default_factory=lambda: [
        (0,             0.060),    # <$10K        → 6.000%   
        (10_000,        0.010),    # $10K–$100K   → 1.000%    
        (100_000,       0.0010),   # $100K–$1M    → 0.100%    
        (1_000_000,     0.00010),  # $1M–$5M      → 0.010%    
        (5_000_000,     0.00005),  # $5M–$10M     → 0.005%    
        (10_000_000,    0.00002),  # $10M–$50M    → 0.002%   
        (50_000_000,    0.00001),  # $50M–$200M   → 0.001%   
        (200_000_000,   0.000005), # $200M–$1B    → 0.0005%   
        (1_000_000_000, 0.000002), # >$1B         → 0.0002%   
    ])
    commission_tiers_b: list = field(default_factory=lambda: [
        (0,             0.060),
        (10_000,        0.010),
        (100_000,       0.0010),
        (1_000_000,     0.00010),
        (5_000_000,     0.00005),
        (10_000_000,    0.00002),
        (50_000_000,    0.00001),
        (200_000_000,   0.000005),
        (1_000_000_000, 0.000002),
    ])

    # ── Platform search cost as % of bilateral ────────────────────
    platform_search_pct_a: float = 0.30
    platform_search_pct_b: float = 0.30

    # ── PO & invoice costs as % of bilateral ─────────────────────
    platform_po_pct_a: float = 0.30
    platform_po_pct_b: float = 0.30
    platform_invoice_pct_a: float = 0.30
    platform_invoice_pct_b: float = 0.30

    # ── Supplier management hours on platform (% of bilateral) ────
    platform_mgmt_pct_a: float = 0.30
    platform_mgmt_pct_b: float = 0.30

    # ── Bilateral cost constants (exposed for dashboard tuning) ──
    bilateral_search_cost: float = 15_000.0
    bilateral_po_cost: float = 180.0
    bilateral_invoice_cost: float = 9.5
    platform_setup_cost: float = 50_000.0
    onboarding_cost: float = 20_000.0
    audit_cost: float = 30_000.0

    # ── Management hours & wage multiplier ────────────────────
    h_mgmt_strategic: float = 312.0       # ~24 h/wk × 13 wks/qtr
    h_mgmt_non_strategic: float = 104.0   # ~8 h/wk × 13 wks/qtr
    procurement_mgr_mult: float = 2.0     # manager wage = hourly_comp × 2.0

    # ── Onboarding delay (quarters until new supplier can deliver) ──
    # 0 = immediate delivery (default).  If > 0, the buying firm cannot
    # deliver products that depend on the new supplier during the delay.
    onboarding_delay_bilateral: int = 0
    onboarding_delay_platform: int = 0

    # ── Platform operational costs (scale economies) ─────────────
    # OpEx = fixed_cost_per_q + variable_cost_per_member × n × 1/(1+ln(n))
    platform_fixed_opex: float = 20_000.0
    platform_var_opex_per_member: float = 140.0

    # Initial adoption fraction (% of firms seeded onto each platform at t=0)
    initial_adoption_frac: float = 0.05

    # Annual demand growth rate — bell-curve distribution
    # Each simulated year draws from N(mean, std) clipped to [min, max].
    # N(3.5%, σ=3.5pp) places ±3σ bounds exactly at −7% and +14%.
    annual_growth_rate: float = ANNUAL_GROWTH_RATE
    growth_rate_std: float = 0.035
    growth_rate_min: float = -0.07
    growth_rate_max: float = 0.14

    # ── Community seeding ─────────────────────────────────────────
    # When True, seed platforms into specific Louvain communities at t=0.
    # When False, both platforms are seeded randomly (default).
    community_seeding: bool = False

    # Target community for each platform (-1 = random).
    platform_a_seed_community: int = -1
    platform_b_seed_community: int = -1

    # Fraction of target community to seed onto each platform.
    platform_a_community_seed_pct: float = 1.0
    platform_b_community_seed_pct: float = 1.0

    # Fixed seed for Louvain community detection so community IDs stay
    # stable across Monte Carlo runs and are decoupled from experiment seed.
    community_detection_seed: int = 42

    # OEM annual volumes (can override defaults)
    annual_volume: dict = field(default_factory=lambda: dict(DEFAULT_ANNUAL_VOLUME))

    # Seasonal weights
    seasonal_weights: list = field(default_factory=lambda: list(SEASONAL_WEIGHTS))

    # ROI payback horizon when firms evaluate join/leave (in quarters)
    roi_horizon_quarters: int = 12

    # ── Discovery & visibility premiums ──────────────────────────
    platform_discovery_premium_pct: float = 0.020
    platform_visibility_premium_pct: float = 0.045



    # ── Disruption events ────────────────────────────────────────
    # Enable disruptions (dashboard checkbox).
    disruption_enabled: bool = False
    # Disruption frequency: how often (in quarters) disruptions recur.
    # First disruption always fires at Q10 (warmup); subsequent at Q10+freq, Q10+2*freq, …
    disruption_frequency: int = 20
    # Duration of each disruption event (constant, not exposed in dashboard).
    disruption_duration: int = 2
    # Number of firms to disrupt (intensity).  Auto-selects the top-N
    # highest betweenness-centrality non-OEM nodes.
    disruption_n_firms: int = 3
    # Legacy single-target override ("auto" or a firm_id int).
    disruption_target: int | str = "auto"

    # ── Willingness-to-Cooperate (WTC) & Cooldown ───────────────
    # Minimum quarters a firm must stay on a platform before it can leave.
    platform_cooldown_quarters: int = 2
    # WTC leave threshold ($/quarter).  Firm leaves when smoothed WTC < 0.
    # With properly calibrated commission tiers, a firm on any edge should
    # have WTC ≥ 0 — leaving only happens when being on the platform is
    # actively costing the firm money.
    wtc_leave_threshold: float = 0.0

    # ── Admin cost log-scaling ───────────────────────────────────
    # Reference value for the log-scaling of admin costs.
    # admin_scale = max(1, 1 + log10(material_cost / reference)).
    admin_scale_reference: float = 5_000.0

    # Cap for PO/invoice admin scaling (mild scaling components).
    admin_scale_cap_po_invoice: float = 3.0
    # Cap for audit cost scaling.
    admin_scale_cap_audit: float = 2.0
    # Cap for onboarding cost scaling.
    admin_scale_cap_onboarding: float = 1.5

    # ── Negotiation costs ────────────────────────────────────────
    # One-time bilateral negotiation cost for new supplier relationship.
    bilateral_negotiation_cost: float = 8_000.0
    # Platform negotiation cost as fraction of bilateral negotiation.
    platform_negotiation_pct_a: float = 0.30
    platform_negotiation_pct_b: float = 0.30

    # ── Conditional audit ────────────────────────────────────────
    # Material value threshold above which audit is triggered for new suppliers.
    audit_material_threshold: float = 40_000.0

    # ── Bilateral global search ──────────────────────────────────
    # When True, bilateral firms can search globally (but pay full search cost).
    bilateral_global_search: bool = False

    # ── Order batch sizing ───────────────────────────────────────
    order_batch_size: int = 1000        
    max_orders_per_quarter: int = 12

    # ── Early-stop ──────────────────────────────────────────────
    # If > 0, stop simulation after N consecutive stable quarters
    # (zero platform joins/leaves).  0 = disabled.
    early_stop_quarters: int = 0

    seed: int | None = 42
