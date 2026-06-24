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

Large raw-data and output folders are still excluded by design:

- `abm_output/`
- `papers/`
- `experiments/FINAL_REPORT.md`

The sanitized dataset bundle is included under `input_data/synthetic_data/`.

## Requirements

See `requirements.txt`.

Python version:

- Recommended: Python 3.11+

## Setup

1. Clone repository
2. Create and activate a virtual environment
3. Install dependencies
4. Use the synthetic bundle in `input_data/synthetic_data/`

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Synthetic Data

The repository now ships with a sanitized bundle under `input_data/synthetic_data/`.

The loader uses that folder automatically when it exists, so you do not need the original source data to run the model or experiments.

Included files:

- `input_data/synthetic_data/china_product_costs.csv`
- `input_data/synthetic_data/Products_limited.csv`
- `input_data/synthetic_data/automotive_cost_framework.xlsx`
- `input_data/synthetic_data/bom_alpha.json`
- `input_data/synthetic_data/bom_beta.json`
- `input_data/synthetic_data/bom_gamma.json`
- `input_data/synthetic_data/bom_delta.json`
- `input_data/synthetic_data/oem_subgraphs/oem_alpha_firms.csv`
- `input_data/synthetic_data/oem_subgraphs/oem_beta_firms.csv`
- `input_data/synthetic_data/oem_subgraphs/oem_gamma_firms.csv`
- `input_data/synthetic_data/oem_subgraphs/oem_delta_firms.csv`
- `input_data/synthetic_data/oem_subgraphs/oem_alpha_relations.csv`
- `input_data/synthetic_data/oem_subgraphs/oem_beta_relations.csv`
- `input_data/synthetic_data/oem_subgraphs/oem_gamma_relations.csv`
- `input_data/synthetic_data/oem_subgraphs/oem_delta_relations.csv`

The bundle is anonymized and trimmed to the columns the current code uses.

## Workbook Layout

- Sheet `6. Full Cost Results`
	- Read with `header=None`
	- Data expected from row index 3 onward
	- Columns:
		- 0: product name
		- 4..6: China low/high/mid
		- 7..9: India low/high/mid
		- 10..12: Japan low/high/mid
		- 13..15: Korea low/high/mid
		- 16..18: Malaysia low/high/mid
		- 19..21: Thailand low/high/mid
		- 22..24: Germany low/high/mid
		- 25..27: Spain low/high/mid
		- 28..30: Mexico low/high/mid
		- 31..33: USA low/high/mid

- Sheet `4. Cost Multipliers`
	- Read with `header=None`
	- Rows 12..16 expected to hold category multipliers
	- Column 0: category name
	- Columns 1..10: multipliers for countries ordered as:
		`China, India, Japan, Korea, Malaysia, Thailand, Germany, Spain, Mexico, USA`

- Sheet `5. Product Mapping`
	- Read with `header=None`
	- Data expected from row index 1 onward
	- Column 0: product name
	- Column 1: category name

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

- Missing files: verify `input_data/synthetic_data/` exists and contains the sanitized CSV/JSON/XLSX files.
- Excel read errors: ensure `openpyxl` is installed.
- Streamlit import issues: install from `requirements.txt` in the active environment.

## License

MIT License. See `LICENSE`.
