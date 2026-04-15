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

Required files and folders:

- `input_data/china_product_costs.csv`
- `input_data/Products_limited.csv`
- `input_data/automotive_cost_framework.xlsx`
- `input_data/bom_2000682.json`
- `input_data/bom_2000766.json`
- `input_data/bom_2000621.json`
- `input_data/bom_2000772.json`
- `input_data/oem_subgraphs/oem_2000682_firms.csv`
- `input_data/oem_subgraphs/oem_2000766_firms.csv`
- `input_data/oem_subgraphs/oem_2000621_firms.csv`
- `input_data/oem_subgraphs/oem_2000772_firms.csv`
- `input_data/oem_subgraphs/oem_2000682_relations.csv`
- `input_data/oem_subgraphs/oem_2000766_relations.csv`
- `input_data/oem_subgraphs/oem_2000621_relations.csv`
- `input_data/oem_subgraphs/oem_2000772_relations.csv`

### Data Format Specification

This section defines the minimum structure expected by `abm_platform/data/loader.py`.

#### 1) OEM firm files

File pattern:

- `input_data/oem_subgraphs/oem_<OEM_ID>_firms.csv`

Required columns:

- `firm_id` (int)
- `firm_name` (str)
- `nation_name` (str)
- `latitude` (float)
- `longitude` (float)
- `is_oem` (`true`/`false`)
- `tier_depth` (int)
- `is_top_supplier` (`true`/`false`)
- `Certifications` (comma-separated text; e.g., `ISO9001, IATF16949`)
- `product_catalog_common` (semicolon-separated product names)
- `group_name` (str)

Notes:

- Duplicate `firm_id` rows across OEM files are ignored after first load.
- If country/geo is missing for OEM IDs 2000682/2000766/2000621/2000772, defaults are auto-filled.

#### 2) OEM relation files

File pattern:

- `input_data/oem_subgraphs/oem_<OEM_ID>_relations.csv`

Required columns:

- `relation_id` (int)
- `source_firm_id` (int)
- `target_firm_id` (int)
- `source_tier` (int; defaults to 1 if blank)
- `product_name` (semicolon-separated product names)
- `is_conglomerate_supplier` (`true`/`false`)

Notes:

- Duplicate `relation_id` rows across files are ignored after first load.

#### 3) BOM files

File pattern:

- `input_data/bom_<OEM_ID>.json`

Top-level structure:

- JSON object with key `bill_of_materials` (array)

Each node in the recursive tree should include:

- `supplier`: object with `firm_id` (int)
- `product` (str; semicolon-separated values also allowed)
- `quantity_per_vehicle` (number; optional, defaults to 1.0)
- `unit` (str; optional, defaults to `piece`)
- `tier` (int; optional, defaults to 1)
- `inputs` (array of child nodes; optional)

Minimal example:

```json
{
	"bill_of_materials": [
		{
			"supplier": {"firm_id": 12345},
			"product": "Battery Pack",
			"quantity_per_vehicle": 1,
			"unit": "piece",
			"tier": 1,
			"inputs": []
		}
	]
}
```

#### 4) China product cost table

File:

- `input_data/china_product_costs.csv`

Required columns:

- `product_name` (str)
- `cost_low` (float)
- `cost_high` (float)

Optional columns:

- `min_order_quantity` (int; defaults to 1)
- `unit` (str; defaults to `piece`)

Behavior:

- If multiple rows exist for the same `product_name`, loader uses median values.

#### 5) Product taxonomy file

File:

- `input_data/Products_limited.csv`

Required columns:

- `product_name` (str)
- `product_id` (int)

Optional columns:

- `family_name` (str)
- `group_name` (str)
- `is_process` (`true`/`false`; defaults to `false`)

#### 6) Automotive cost framework workbook

File:

- `input_data/automotive_cost_framework.xlsx`

Required sheets and layout:

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

- Missing files: verify `input_data/` exists and contains expected CSV/JSON/XLSX files.
- Excel read errors: ensure `openpyxl` is installed.
- Streamlit import issues: install from `requirements.txt` in the active environment.

## License

MIT License. See `LICENSE`.
