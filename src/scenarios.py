"""
Demo scenarios for the dashboard.

Until the live weather API is wired in (item 2 on the weekly plan) the map needs
something to show. Each scenario below produces one realistic sensor reading per
district, seeded so that the demo looks identical every time we present it.

The scenarios are shaped by the district's own attributes from districts.csv -
a coastal district gets the cyclone forcing, a steep district gets the orographic
rainfall - so the map does not just show one uniform colour.

We also switch OFF a few river gauges on purpose ("gauge offline"). That is not
laziness: showing that the system still produces a calibrated number and an
honest confidence label for those districts is the point of the whole design.

Owner: Sai Vandith
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import DISTRICTS_CSV, SEED

# districts whose river gauge is "not reporting" in the demo
GAUGE_OFFLINE = {"Wayanad", "Cuddalore", "Chittoor"}

SCENARIOS = {
    "monsoon_depression": "Monsoon depression over the Western Ghats",
    "monsoon_normal": "Ordinary active monsoon day",
    "cyclone_landfall": "Cyclonic storm approaching the east coast",
    "summer_heatwave": "Peak summer heat over the interior districts",
    "calm_winter": "Quiet winter day (baseline)",
}


def _districts() -> pd.DataFrame:
    return pd.read_csv(DISTRICTS_CSV)


def build(scenario: str = "monsoon_depression", seed: int = SEED) -> List[Dict]:
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario '{scenario}'; try {list(SCENARIOS)}")

    df = _districts()
    rng = np.random.default_rng(seed)
    n = len(df)

    steep = (df["slope_deg"] / 30.0).clip(0, 1).to_numpy()      # 0 flat .. 1 steep
    coastal = df["coastal"].to_numpy().astype(float)
    # a Bay of Bengal system does not hit the Malabar coast, so the cyclone
    # scenario has to know which sea a district sits on
    east = (df["coast_side"] == "east").to_numpy().astype(float)
    west = (df["coast_side"] == "west").to_numpy().astype(float)

    if scenario == "monsoon_depression":
        season = "Monsoon"
        # the depression sits on the ghats, so rainfall scales with terrain:
        # the hill districts get hammered, the plains get a wet but survivable day
        rain = 28 + 88 * steep + rng.normal(0, 10, n)
        temp = rng.normal(27.5, 1.2, n)
        hum = 74 + 12 * steep + rng.normal(0, 4, n)
        sst = 28.6 + 0.4 * coastal + rng.normal(0, 0.3, n)
        pdrop = 2.0 + 1.5 * coastal + rng.normal(0, 1.0, n)
        wind = 22 + 14 * coastal + rng.normal(0, 5, n)
        river = 0.40 + 0.48 * steep + 0.08 * coastal + rng.normal(0, 0.06, n)

    elif scenario == "monsoon_normal":
        season = "Monsoon"
        rain = 18 + 26 * steep + rng.normal(0, 8, n)
        temp = rng.normal(29.5, 1.3, n)
        hum = 72 + 6 * steep + rng.normal(0, 5, n)
        sst = 28.4 + rng.normal(0, 0.3, n)
        pdrop = 1.0 + rng.normal(0, 0.8, n)
        wind = 16 + 6 * coastal + rng.normal(0, 4, n)
        river = 0.42 + 0.18 * steep + rng.normal(0, 0.06, n)

    elif scenario == "cyclone_landfall":
        # severe cyclonic storm in the Bay of Bengal, landfall near the AP coast.
        # The east coast takes the storm; the west coast just has a wet day.
        season = "Monsoon"
        rain = 18 + 78 * east + 12 * west + rng.normal(0, 12, n)
        temp = rng.normal(28.0, 1.5, n)
        hum = 66 + 18 * east + 6 * west + rng.normal(0, 4, n)
        sst = 28.0 + 1.4 * east + 0.3 * west + rng.normal(0, 0.25, n)
        pdrop = 1.0 + 7.5 * east + rng.normal(0, 1.0, n)
        wind = 20 + 72 * east + 6 * west + rng.normal(0, 8, n)
        river = 0.38 + 0.28 * east + 0.10 * steep + rng.normal(0, 0.06, n)

    elif scenario == "summer_heatwave":
        # interior districts bake; the coast is cooler and much more humid, which
        # is why the sea breeze keeps Chennai off the heat-wave list even in May
        season = "Summer"
        rain = np.abs(rng.normal(2.5, 2.0, n))
        temp = 42.0 - 5.0 * steep - 2.2 * coastal + rng.normal(0, 0.9, n)
        hum = 32 + 22 * coastal + rng.normal(0, 5, n)
        sst = 28.5 + rng.normal(0, 0.3, n)
        pdrop = np.abs(rng.normal(1.0, 0.8, n))
        wind = 12 + rng.normal(0, 4, n)
        river = 0.22 + rng.normal(0, 0.05, n)

    else:  # calm_winter
        season = "Winter"
        rain = np.abs(rng.normal(4.0, 3.5, n))
        temp = rng.normal(27.0, 1.8, n)
        hum = 58 + rng.normal(0, 7, n)
        sst = 27.0 + rng.normal(0, 0.4, n)
        pdrop = np.abs(rng.normal(0.8, 0.6, n))
        wind = 11 + rng.normal(0, 3, n)
        river = 0.28 + rng.normal(0, 0.05, n)

    readings = []
    for i, row in df.iterrows():
        reading = {
            "district": row["district"],
            "state": row["state"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "population": int(row["population"]),
            "major_river": row["major_river"],
            "nearest_shelter": row["nearest_shelter"],
            "season": season,
            "urbanisation": row["urbanisation"],
            "slope_deg": float(row["slope_deg"]),
            "rainfall_mm": round(float(max(rain[i], 0)), 1),
            "temperature_c": round(float(temp[i]), 1),
            "humidity_pct": round(float(np.clip(hum[i], 15, 100)), 1),
            "sst_c": round(float(sst[i]), 2),
            "pressure_drop_hpa": round(float(max(pdrop[i], 0)), 2),
            "wind_kmph": round(float(max(wind[i], 0)), 1),
            "river_level_frac": round(float(np.clip(river[i], 0.02, 1.6)), 3),
        }
        if row["district"] in GAUGE_OFFLINE:
            reading["river_level_frac"] = None
            reading["notes"] = "river gauge not reporting"
        readings.append(reading)
    return readings


if __name__ == "__main__":
    for name, label in SCENARIOS.items():
        rows = build(name)
        top = max(rows, key=lambda r: r["rainfall_mm"])
        print(f"{name:>20}  {label}")
        print(f"{'':>20}  wettest: {top['district']} {top['rainfall_mm']} mm, "
              f"gauges offline: {sum(1 for r in rows if r['river_level_frac'] is None)}")
