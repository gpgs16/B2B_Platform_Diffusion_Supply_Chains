"""
Transport cost, lead-time, and route-disruption calculations.

Great-circle distance via haversine; freight cost split by mode
(road / rail / sea) based on the OCEAN_PAIRS lookup in config.
Disruption events (e.g. Suez Canal closure) can be passed in to
increase lead time and freight cost on affected routes.

Lead time is distance-based: haversine_km / mode_speed (km/week).
"""

from __future__ import annotations

import math

from abm_platform.config import (
    FREIGHT_RATE_RAIL,
    FREIGHT_RATE_ROAD,
    FREIGHT_RATE_SEA,
    OCEAN_PAIRS,
)

EARTH_RADIUS_KM = 6_371.0

# Speed by transport mode (km per week)
# Source: industry averages
#   Road: ~800 km/day → ~5,600 km/week
#   Rail: ~600 km/day → ~4,200 km/week (China-Europe rail ~10,000 km in 14-18 days)
#   Sea:  ~550 km/day → ~3,850 km/week (container ship ~22 knots avg)
SPEED_ROAD_KM_PER_WEEK = 5_600.0
SPEED_RAIL_KM_PER_WEEK = 4_200.0
SPEED_SEA_KM_PER_WEEK = 3_850.0

# Minimum lead times (weeks) — accounts for port handling, customs, etc.
MIN_LEAD_TIME_DOMESTIC = 0.5
MIN_LEAD_TIME_CROSS_BORDER = 1.0
MIN_LEAD_TIME_OCEAN = 2.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = math.radians(lat2), math.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + (
        math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def transport_mode(country_a: str, country_b: str) -> str:
    """Return 'road', 'rail', or 'sea' for a given country pair."""
    if country_a == country_b:
        return "road"
    pair = frozenset({country_a, country_b})
    if pair in OCEAN_PAIRS:
        return "sea"
    return "rail"  # cross-border same continent


def freight_rate(mode: str) -> float:
    """USD per ton-km for the given mode."""
    if mode == "sea":
        return FREIGHT_RATE_SEA
    if mode == "rail":
        return FREIGHT_RATE_RAIL
    return FREIGHT_RATE_ROAD


def lead_time_weeks(
    mode: str,
    distance_km: float = 0.0,
) -> float:
    """
    Distance-based lead time in weeks.
    Lead time = distance / mode_speed + handling minimum.
    """
    if mode == "sea":
        transit = distance_km / SPEED_SEA_KM_PER_WEEK if distance_km > 0 else 2.0
        return max(transit, MIN_LEAD_TIME_OCEAN)
    if mode == "rail":
        transit = distance_km / SPEED_RAIL_KM_PER_WEEK if distance_km > 0 else 1.5
        return max(transit, MIN_LEAD_TIME_CROSS_BORDER)
    # road / domestic
    transit = distance_km / SPEED_ROAD_KM_PER_WEEK if distance_km > 0 else 0.5
    return max(transit, MIN_LEAD_TIME_DOMESTIC)


def transport_cost(
    lat1: float,
    lon1: float,
    country1: str,
    lat2: float,
    lon2: float,
    country2: str,
    weight_tons: float,
    quantity: float,
) -> float:
    """
    Total transport cost (USD) for shipping *quantity* units of a product
    weighing *weight_tons* per unit between two points.
    """
    dist = haversine_km(lat1, lon1, lat2, lon2)
    mode = transport_mode(country1, country2)
    rate = freight_rate(mode)
    cost = dist * rate * weight_tons * quantity
    return cost


def get_lead_time(
    country1: str,
    country2: str,
    lat1: float = 0.0,
    lon1: float = 0.0,
    lat2: float = 0.0,
    lon2: float = 0.0,
) -> float:
    """Distance-based lead time in weeks for two locations."""
    mode = transport_mode(country1, country2)
    dist = haversine_km(lat1, lon1, lat2, lon2) if (lat1 or lon1 or lat2 or lon2) else 0.0
    return lead_time_weeks(mode, distance_km=dist)
