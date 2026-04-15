# B2B Platform Diffusion in Supply Chains

Agent-based model (ABM) for studying competing B2B procurement platforms in automotive supply networks.

This project simulates how firms adopt, leave, or switch platforms under cost, ROI, and disruption dynamics. It supports both interactive exploration (Streamlit dashboard) and large-scale batch experiments for hypothesis testing.

## What This Model Covers

- Two competing procurement platforms (A and B)
- Firm-level adoption decisions with ROI-based join/switch/leave logic
- Supply-chain cost decomposition (search, PO, invoice, management, onboarding, audit, negotiation)
- Tiered commission structures for platforms
- Community-aware seeding (random or targeted)
- Demand propagation across a supplier network
- Disruption scenarios and resilience effects
- Sensitivity analysis and hypothesis experiment pipelines

## Core Modeling Logic (High Level)

Each quarter:

1. Demand is generated and cascaded through the supply graph.
2. Firms compute procurement/revenue effects and update edge channels.
3. Firms evaluate platform membership (join/leave/switch) using ROI and stability rules.
4. Channels are refreshed after membership changes.
5. Platform metrics (trade volume, share, commission, OpEx, profit) are computed.
6. DataCollector stores model metrics for analysis/plots.

Adoption economics are mainly controlled through `ExperimentConfig` in `abm_platform/config.py`.

## Repository Layout

- `abm_platform/`: model package (agents, environment, data loading, dashboard, visualization)
- `experiments/`: experiment runners and analysis scripts
- `lib/`: bundled frontend JS libs used by dashboard/network rendering

Ignored by design in this repository:

- `input_data/`
- `abm_output/`
- `papers/`
- `experiments/FINAL_REPORT.md`

These are intentionally excluded for repository cleanliness and data/output size control.

## Requirements

See `requirements.txt`.

Python version:

- Recommended: Python 3.11+

## Setup

1. Clone repository
2. Create and activate a virtual environment
3. Install dependencies
4. Ensure input datasets are present in `input_data/` (not tracked by git)

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Required Local Data (Not Tracked)

Place all required data files under `input_data/`.

Expected examples include:

- BOM JSON files (e.g., `bom_2000682.json`)
- Cost tables (`china_product_costs.csv`, `automotive_cost_framework.xlsx`)
- Product taxonomy (`Products_limited.csv`)
- OEM graph files under `input_data/oem_subgraphs/`

The loader uses:

- `BASE/input_data/...`
- `BASE/input_data/oem_subgraphs/...`

## Running the Project

### 1) Interactive Dashboard

```bash
streamlit run abm_platform/dashboard.py
```

### 2) CLI Simulation Run

```bash
python -m abm_platform.run --quarters 40 --seed 42
```

Useful flags:

- `--quarters`
- `--seed`
- `--search-pct-a`, `--search-pct-b`
- `--initial-adoption`
- `--roi-horizon`
- `--cooldown`
- `--wtc-threshold`
- `--no-viz`

### 3) Batch Experiments

```bash
python experiments/run_final.py
```

Or per phase:

```bash
python experiments/run_final.py --phase 1
python experiments/run_final.py --phase 2
python experiments/run_final.py --phase 3
```

### 4) Analysis / Figures

```bash
python experiments/analyze_final.py
```

## Reproducibility Notes

- Set fixed seeds for deterministic experiment batches.
- Keep `ExperimentConfig` baselines synchronized between dashboard and experiment scripts.
- Large-scale runs use multiprocessing in `experiments/run_final.py`.

## Common Troubleshooting

- Missing files: verify `input_data/` exists and contains expected CSV/JSON/XLSX files.
- Excel read errors: ensure `openpyxl` is installed.
- Streamlit import issues: install from `requirements.txt` in the active environment.

## Citation / Usage

If you use this model in research, cite the repository and describe:

- commit hash
- parameter baseline
- seed policy
- scenario definitions (H1/H2/H3 or sensitivity set)

## License

No license file is included yet. Add a license before public reuse if needed.
