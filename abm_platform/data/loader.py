"""
Data ingestion: loads all CSV / JSON / XLSX files and returns
clean, merged Python structures ready for the ABM.
"""

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from abm_platform.config import (
    QUALITY_METRICS,
)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE, "input_data")
OEM_DIR = os.path.join(DATA_DIR, "oem_subgraphs")


# ── Data classes for loaded records ──────────────────────────────────

@dataclass
class FirmRecord:
    firm_id: int
    firm_name: str
    country: str          # canonical name from data
    lat: float
    lon: float
    is_oem: bool
    tier_depth: int
    is_top_supplier: bool
    certifications: list[str]
    product_catalog: list[str]  # product_catalog_common items
    group_name: str
    defect_rate: float = 0.0
    price_premium: float = 1.0
    is_eco_friendly: bool = False

    def __post_init__(self):
        self.defect_rate = _derive_defect_rate(
            self.certifications, self.is_top_supplier
        )
        self.price_premium = 1.0  # no price premium for certifications
        self.is_eco_friendly = "ISO14001" in self.certifications


@dataclass
class RelationRecord:
    relation_id: int
    source_firm_id: int
    target_firm_id: int
    source_tier: int
    products: list[str]
    is_conglomerate: bool


@dataclass
class BOMEntry:
    """One row from the recursive BOM (any tier)."""
    supplier_firm_id: int
    product: str
    quantity_per_vehicle: float
    unit: str
    tier: int


@dataclass
class ProductCost:
    """Aggregated China cost for a product (median across suppliers)."""
    product_name: str
    cost_low: float
    cost_mid: float
    cost_high: float
    min_order_qty: int
    unit: str


@dataclass
class CountryCost:
    """Per-product, per-country cost triple."""
    product_name: str
    country: str
    cost_low: float
    cost_mid: float
    cost_high: float


# ── Helpers ──────────────────────────────────────────────────────────

def _canon_country(raw: str) -> str:
    """Normalize whitespace in nation_name values."""
    return raw.strip()


def _split_semi(val: str) -> list[str]:
    if not val or not val.strip():
        return []
    return [s.strip() for s in val.split(";") if s.strip()]


def _parse_certs(raw: str) -> list[str]:
    """Extract distinct cert standards from free-text field."""
    if not raw or not raw.strip():
        return []
    tokens = [t.strip().rstrip(",") for t in raw.split(",")]
    out = []
    for t in tokens:
        upper = t.upper()
        if "IATF" in upper or "TS16949" in upper:
            out.append("IATF16949")
        elif "ISO14001" in upper:
            out.append("ISO14001")
        elif "ISO9001" in upper or "ISO 9001" in upper:
            out.append("ISO9001")
        elif "QS9000" in upper or "QS 9000" in upper:
            out.append("QS9000")
        elif "VDA" in upper:
            out.append("VDA6.1")
    return list(dict.fromkeys(out))  # deduplicate, preserve order


def _derive_defect_rate(certs: list[str], is_top: bool) -> float:
    """Derive defect rate from certifications using QUALITY_METRICS.
    IATF16949 supersedes ISO9001 if both present."""
    has_iatf = "IATF16949" in certs
    has_iso = "ISO9001" in certs
    if has_iatf:
        # IATF16949: use defect_reduction from uncertified baseline
        base = QUALITY_METRICS["uncertified"]["rework_rate"]
        reduction = QUALITY_METRICS["IATF16949"]["defect_reduction"]
        rate = base * (1.0 - reduction)
        if is_top:
            rate *= 0.8  # top suppliers perform 20% better
        return rate
    if has_iso:
        rate = QUALITY_METRICS["ISO9001"]["rework_rate"]
        if is_top:
            rate *= 0.8
        return rate
    return QUALITY_METRICS["uncertified"]["rework_rate"]


# ── Loader functions ─────────────────────────────────────────────────

def load_firms() -> dict[int, FirmRecord]:
    """Load & merge firm records from both OEM firm files."""
    firms: dict[int, FirmRecord] = {}
    for fn in [
        os.path.join(OEM_DIR, "oem_2000682_firms.csv"),
        os.path.join(OEM_DIR, "oem_2000766_firms.csv"),
        os.path.join(OEM_DIR, "oem_2000621_firms.csv"),
        os.path.join(OEM_DIR, "oem_2000772_firms.csv"),
    ]:
        with open(fn, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fid = int(row["firm_id"])
                if fid in firms:
                    continue  # already loaded from other OEM file
                country = _canon_country(row.get("nation_name", ""))
                try:
                    lat = float(row["latitude"])
                except (ValueError, KeyError):
                    lat = 0.0
                try:
                    lon = float(row["longitude"])
                except (ValueError, KeyError):
                    lon = 0.0
                firms[fid] = FirmRecord(
                    firm_id=fid,
                    firm_name=row.get("firm_name", ""),
                    country=country,
                    lat=lat,
                    lon=lon,
                    is_oem=row.get("is_oem", "").strip().lower() == "true",
                    tier_depth=int(float(row.get("tier_depth", 0) or 0)),
                    is_top_supplier=row.get("is_top_supplier", "").strip().lower() == "true",
                    certifications=_parse_certs(row.get("Certifications", "")),
                    product_catalog=_split_semi(row.get("product_catalog_common", "")),
                    group_name=row.get("group_name", ""),
                )

    # Both OEMs are Chinese companies; fill in missing geo data
    _OEM_DEFAULTS = {
        2000682: ("China", 25.05, 118.08),   
        2000766: ("China", 24.33, 109.42),  
        2000621: ("China", 29.57, 106.55),  
        2000772: ("China", 43.88, 125.32),   
    }
    for oid, (ctry, lat, lon) in _OEM_DEFAULTS.items():
        if oid in firms and not firms[oid].country:
            firms[oid].country = ctry
            firms[oid].lat = lat
            firms[oid].lon = lon
    return firms


def load_relations() -> list[RelationRecord]:
    """Load supplier relations from both OEM relation files."""
    rels: list[RelationRecord] = []
    seen_ids: set[int] = set()
    for fn in [
        os.path.join(OEM_DIR, "oem_2000682_relations.csv"),
        os.path.join(OEM_DIR, "oem_2000766_relations.csv"),
        os.path.join(OEM_DIR, "oem_2000621_relations.csv"),
        os.path.join(OEM_DIR, "oem_2000772_relations.csv"),
    ]:
        with open(fn, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = int(row["relation_id"])
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                rels.append(
                    RelationRecord(
                        relation_id=rid,
                        source_firm_id=int(row["source_firm_id"]),
                        target_firm_id=int(row["target_firm_id"]),
                        source_tier=int(row.get("source_tier", 1) or 1),
                        products=_split_semi(row.get("product_name", "")),
                        is_conglomerate=row.get("is_conglomerate_supplier", "").strip().lower() == "true",
                    )
                )
    return rels


def load_bom(oem_id: int) -> list[BOMEntry]:
    """Flatten the recursive BOM JSON into a list of BOMEntry."""
    fn = os.path.join(DATA_DIR, f"bom_{oem_id}.json")
    with open(fn, encoding="utf-8") as f:
        data = json.load(f)

    entries: list[BOMEntry] = []

    def _walk(node: dict, parent_qty: float = 1.0):
        fid = node["supplier"]["firm_id"]
        qty_raw = node.get("quantity_per_vehicle")
        qty = float(qty_raw) if qty_raw is not None else 1.0
        # Split semicolon-separated products (consistent with relations loader)
        products = _split_semi(node["product"]) or [node["product"]]
        for prod in products:
            entries.append(
                BOMEntry(
                    supplier_firm_id=fid,
                    product=prod,
                    quantity_per_vehicle=qty,
                    unit=node.get("unit", "piece"),
                    tier=node.get("tier", 1),
                )
            )
        for child in node.get("inputs", []):
            _walk(child, qty)

    for item in data["bill_of_materials"]:
        _walk(item)
    return entries


def load_china_product_costs() -> dict[str, ProductCost]:
    """Load china_product_costs.csv, aggregate by product_name (median)."""
    fn = os.path.join(DATA_DIR, "china_product_costs.csv")
    raw: dict[str, list] = defaultdict(list)
    with open(fn, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["product_name"].strip()
            try:
                lo = float(row["cost_low"])
                hi = float(row["cost_high"])
            except ValueError:
                continue
            moq = int(float(row.get("min_order_quantity", 1) or 1))
            raw[name].append((lo, hi, moq, row.get("unit", "piece").strip()))

    costs: dict[str, ProductCost] = {}
    for name, entries in raw.items():
        lows = [e[0] for e in entries]
        highs = [e[1] for e in entries]
        moqs = [e[2] for e in entries]
        units = [e[3] for e in entries]
        lo_med = sorted(lows)[len(lows) // 2]
        hi_med = sorted(highs)[len(highs) // 2]
        moq_med = sorted(moqs)[len(moqs) // 2]
        costs[name] = ProductCost(
            product_name=name,
            cost_low=lo_med,
            cost_mid=(lo_med + hi_med) / 2,
            cost_high=hi_med,
            min_order_qty=max(moq_med, 1),
            unit=units[0],
        )
    return costs


def load_country_costs() -> dict[str, dict[str, CountryCost]]:
    """
    Parse Sheet 5 of automotive_cost_framework.xlsx.
    Returns {product_name: {country: CountryCost}}.
    """
    fn = os.path.join(DATA_DIR, "automotive_cost_framework.xlsx")
    df = pd.read_excel(fn, sheet_name="6. Full Cost Results", header=None)

    # Column layout (row 2 is the header):
    # 0: Product Name, 1: Category, 2: Unit, 3: Min Order Qty
    # 4-6: China Low/High/Mid
    # Then triplets (Low, High, Mid) for each country at:
    # India(7-9), Japan(10-12), Korea(13-15), Malaysia(16-18),
    # Thailand(19-21), Germany(22-24), Spain(25-27), Mexico(28-30), USA(31-33)
    country_cols = {
        "China":    (4, 5, 6),
        "India":    (7, 8, 9),
        "Japan":    (10, 11, 12),
        "Korea":    (13, 14, 15),
        "Malaysia": (16, 17, 18),
        "Thailand": (19, 20, 21),
        "Germany":  (22, 23, 24),
        "Spain":    (25, 26, 27),
        "Mexico":   (28, 29, 30),
        "USA":      (31, 32, 33),
    }

    result: dict[str, dict[str, CountryCost]] = {}
    # Data rows start at row 3 (0-indexed)
    for idx in range(3, len(df)):
        product_name = df.iloc[idx, 0]
        if pd.isna(product_name) or not str(product_name).strip():
            continue
        product_name = str(product_name).strip()
        result[product_name] = {}
        for country, (lo_col, hi_col, mid_col) in country_cols.items():
            try:
                lo = float(df.iloc[idx, lo_col])
                hi = float(df.iloc[idx, hi_col])
                mid = float(df.iloc[idx, mid_col])
            except (ValueError, TypeError):
                continue
            result[product_name][country] = CountryCost(
                product_name=product_name,
                country=country,
                cost_low=lo,
                cost_mid=mid,
                cost_high=hi,
            )
    return result


def load_category_cost_multipliers() -> dict[str, dict[str, float]]:
    """
    Parse Sheet 6 (Category Summary) to get per-category, per-country
    cost multipliers (China = 1.0).
    Returns {category_name: {country: multiplier}}.
    """
    fn = os.path.join(DATA_DIR, "automotive_cost_framework.xlsx")
    df = pd.read_excel(fn, sheet_name="4. Cost Multipliers", header=None)

    # Multiplier rows start at row 12
    # Row 11 has column headers: Category China India Japan Korea Malaysia Thailand Germany Spain Mexico USA
    countries = ["China", "India", "Japan", "Korea", "Malaysia",
                 "Thailand", "Germany", "Spain", "Mexico", "USA"]
    result: dict[str, dict[str, float]] = {}
    for idx in range(12, 17):
        cat = df.iloc[idx, 0]
        if pd.isna(cat):
            continue
        cat = str(cat).strip()
        mults = {}
        for ci, country in enumerate(countries):
            try:
                mults[country] = float(df.iloc[idx, ci + 1])
            except (ValueError, TypeError):
                mults[country] = 1.0
        result[cat] = mults
    return result


def load_product_category_mapping() -> dict[str, str]:
    """
    Parse Sheet 4 (Product Mapping) → {product_name: category_name}.
    """
    fn = os.path.join(DATA_DIR, "automotive_cost_framework.xlsx")
    df = pd.read_excel(fn, sheet_name="5. Product Mapping", header=None)
    mapping: dict[str, str] = {}
    for idx in range(1, len(df)):
        pname = df.iloc[idx, 0]
        cat = df.iloc[idx, 1]
        if pd.isna(pname) or pd.isna(cat):
            continue
        mapping[str(pname).strip()] = str(cat).strip()
    return mapping


def load_product_taxonomy() -> dict[str, dict]:
    """Load Products_limited.csv → {product_name: {family_name, ...}}."""
    fn = os.path.join(DATA_DIR, "Products_limited.csv")
    taxonomy: dict[str, dict] = {}
    with open(fn, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            taxonomy[row["product_name"].strip()] = {
                "product_id": int(row["product_id"]),
                "family_name": row.get("family_name", ""),
                "group_name": row.get("group_name", ""),
                "is_process": row.get("is_process", "false").lower() == "true",
            }
    return taxonomy


# ── Unified loader ───────────────────────────────────────────────────

@dataclass
class LoadedData:
    firms: dict[int, FirmRecord]
    relations: list[RelationRecord]
    bom: dict[int, list[BOMEntry]]             # {oem_id: entries}
    china_costs: dict[str, ProductCost]
    country_costs: dict[str, dict[str, CountryCost]]
    category_multipliers: dict[str, dict[str, float]]
    product_category_map: dict[str, str]
    product_taxonomy: dict[str, dict]


def load_all() -> LoadedData:
    """One-shot loader returning every dataset the model needs."""
    firms = load_firms()
    relations = load_relations()
    bom = {
        2000682: load_bom(2000682),
        2000766: load_bom(2000766),
        2000621: load_bom(2000621),
        2000772: load_bom(2000772),
    }
    china_costs = load_china_product_costs()
    country_costs = load_country_costs()
    cat_mult = load_category_cost_multipliers()
    prod_cat_map = load_product_category_mapping()
    taxonomy = load_product_taxonomy()

    return LoadedData(
        firms=firms,
        relations=relations,
        bom=bom,
        china_costs=china_costs,
        country_costs=country_costs,
        category_multipliers=cat_mult,
        product_category_map=prod_cat_map,
        product_taxonomy=taxonomy,
    )
