"""
Real observations, and relearning the weather layer from them.

This module exists because of one line of feedback: "you are choosing the existing
preprocessed data?" The honest answer at the time was yes - the CPTs came from our
own simulator. This is the fix.

What it does
------------
Pulls six years of DAILY OBSERVATIONS for the sixteen districts in districts.csv
from three archive services (all free, no key):

    archive-api.open-meteo.com   ERA5 reanalysis: rainfall, max temperature,
                                 mean humidity, max wind, mean surface pressure
    flood-api.open-meteo.com     GloFAS daily river discharge
    marine-api.open-meteo.com    daily mean sea surface temperature

That is roughly 35,000 real district-days. It then re-estimates the CPTs of every
node whose value AND whose parents' values are all observable:

    Season (prior)          from the calendar
    Rainfall | Season        <- real rainfall
    Temperature | Season     <- real daily maxima
    SeaSurfaceTemp | Season  <- real SST, coastal districts only
    Humidity | Rainfall      <- real humidity
    PressureDrop | SST       <- real pressure differences, coastal only
    WindSpeed | PressureDrop <- real wind maxima

What it deliberately does NOT touch
-----------------------------------
SoilMoisture and RiverLevel have a latent parent between them and the data, and
Flood, Landslide, Cyclone, Heatwave and the consequence layer need labelled events
that no public table gives us per district-day. Those CPTs stay as they were -
elicited, then refined on simulated data - and the model card says so. Fitting them
here with no labels would be inventing results.

Why this mattered more than we expected
---------------------------------------
Our simulator drew monsoon rainfall from a gamma with a mean near 66 mm/day.
The real five-year mean daily rainfall for Coimbatore is 3.64 mm. So the simulated
model believed a monsoon day was far wetter than monsoon days actually are, and the
first time we ran the live pipeline the Bayesian surprise gate started flagging
perfectly good readings as implausible - the model was surprised by reality. The
relearned weather layer is what fixes that.

Two models, on purpose
----------------------
  artifacts/learned_cpts.json   every layer from the simulator. Used for the
                                held-out synthetic evaluation, where the test
                                labels come from the same generative process, so
                                the weather layer has to match it.
  artifacts/hybrid_cpts.json    weather layer from real observations, hazard and
                                consequence layers unchanged. Used by the live
                                pipeline. This is the one that faces reality.

Owner: Sai Vandith (ingestion) and Sai Reddy A (re-estimation)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (ARTIFACT_DIR, DATA_DIR, DISTRICTS_CSV, SEED, STATES)
from .discretize import discretize_frame
from .feeds import (FLOOD_URL, MARINE_URL, _get, districts)
from .learn import EQUIVALENT_SAMPLE_SIZE, cpds_from_json, cpds_to_json, save_model
from .networks import build_network_from_cpds, expert_cpds

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

OBSERVED_CSV = DATA_DIR / "observed_weather.csv"
HYBRID_MODEL = ARTIFACT_DIR / "hybrid_cpts.json"
RELEARN_REPORT = ARTIFACT_DIR / "weather_relearn_report.json"

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"

# nodes we are entitled to re-estimate: everything in the weather layer, where the
# node and all of its parents are directly observed
RELEARNED_NODES = ["Season", "Rainfall", "Temperature", "SeaSurfaceTemp",
                   "Humidity", "PressureDrop", "WindSpeed"]

DAILY_VARS = ["precipitation_sum", "temperature_2m_max",
              "relative_humidity_2m_mean", "wind_speed_10m_max",
              "surface_pressure_mean"]


def season_of(date_str: str, coast_side: str = "none") -> str:
    """
    Season from the calendar - but not the same calendar everywhere.

    Our first version used one all-India rule (June to September is the monsoon).
    The real data said otherwise: across fifteen districts, mean daily rainfall came
    out HIGHER in "winter" (4.45 mm) than in "summer" (3.96 mm), which is not a
    plausible climate. The reason is the north-east / retreating monsoon: the Tamil
    Nadu and Andhra coasts get most of their rain in October to December, which our
    rule was labelling winter.

    So east-coast districts have October to December as their monsoon. This is a
    correction we would not have found without real observations, which is rather
    the point.
    """
    month = int(date_str[5:7])
    if coast_side == "east":
        if 10 <= month <= 12:
            return "Monsoon"
        if 6 <= month <= 9:
            return "Monsoon"          # the SW monsoon still reaches the interior
        if 3 <= month <= 5:
            return "Summer"
        return "Winter"
    if 6 <= month <= 9:
        return "Monsoon"
    if 3 <= month <= 5:
        return "Summer"
    return "Winter"


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------
def _archive_weather(lat: float, lon: float) -> pd.DataFrame:
    payload = _get(ARCHIVE_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Kolkata",
    }, timeout=90.0)
    d = payload["daily"]
    df = pd.DataFrame({
        "date": d["time"],
        "rainfall_mm": d["precipitation_sum"],
        "temperature_c": d["temperature_2m_max"],
        "humidity_pct": d["relative_humidity_2m_mean"],
        "wind_kmph": d["wind_speed_10m_max"],
        "pressure_hpa": d["surface_pressure_mean"],
    })
    # 24-hour pressure fall, the quantity the cyclone branch actually models
    df["pressure_drop_hpa"] = (df["pressure_hpa"].shift(1)
                               - df["pressure_hpa"]).clip(lower=0.0)
    return df.drop(columns=["pressure_hpa"])


def _archive_river(lat: float, lon: float) -> Optional[pd.DataFrame]:
    payload = _get(FLOOD_URL, {
        "latitude": lat, "longitude": lon, "daily": "river_discharge",
        "start_date": START_DATE, "end_date": END_DATE,
    }, timeout=90.0)
    d = payload.get("daily", {})
    q = d.get("river_discharge", [])
    if not q or all(v is None for v in q):
        return None                      # no modelled reach here (e.g. Chennai)

    df = pd.DataFrame({"date": d["time"], "discharge": q}).dropna()
    # bankfull = median annual maximum = the 2-year return period
    df["year"] = df["date"].str[:4]
    annual_max = df.groupby("year")["discharge"].max()
    bankfull = float(annual_max.median())
    if bankfull <= 0:
        return None
    df["river_level_frac"] = df["discharge"] / bankfull
    return df[["date", "river_level_frac"]]


def _archive_sst(lat: float, lon: float, coast_side: str) -> Optional[pd.DataFrame]:
    if coast_side == "none":
        return None
    plon = lon - 0.7 if coast_side == "west" else lon + 0.7
    payload = _get(MARINE_URL, {
        "latitude": lat, "longitude": plon,
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": "sea_surface_temperature_mean",
        "timezone": "Asia/Kolkata",
    }, timeout=90.0)
    d = payload.get("daily", {})
    vals = d.get("sea_surface_temperature_mean", [])
    if not vals or all(v is None for v in vals):
        return None
    return pd.DataFrame({"date": d["time"], "sst_c": vals}).dropna()


def build_observed_dataset(force: bool = False,
                           topup: bool = False) -> pd.DataFrame:
    """
    Assemble (and cache) the real district-day table.

    `topup` fetches only the districts missing from the cache and appends them,
    which is what you want after a transient DNS failure has cost you one district
    out of sixteen - refetching six years for the other fifteen to recover one is
    rude to a free service.
    """
    existing = None
    if OBSERVED_CSV.exists() and not force:
        existing = pd.read_csv(OBSERVED_CSV)
        if not topup:
            return existing

    have = set(existing["district"].unique()) if existing is not None else set()
    frames = [existing] if existing is not None else []

    for _, row in districts().iterrows():
        if row["district"] in have:
            continue
        name = row["district"]
        lat, lon = float(row["lat"]), float(row["lon"])
        print(f"  fetching {name} ...", flush=True)
        try:
            wx = _archive_weather(lat, lon)
        except Exception as exc:                       # noqa: BLE001
            print(f"    weather failed: {exc}")
            continue

        try:
            river = _archive_river(lat, lon)
        except Exception as exc:                       # noqa: BLE001
            print(f"    river failed: {exc}")
            river = None
        if river is not None:
            wx = wx.merge(river, on="date", how="left")
        else:
            wx["river_level_frac"] = np.nan

        try:
            sst = _archive_sst(lat, lon, str(row["coast_side"]))
        except Exception as exc:                       # noqa: BLE001
            print(f"    sst failed: {exc}")
            sst = None
        if sst is not None:
            wx = wx.merge(sst, on="date", how="left")
        else:
            wx["sst_c"] = np.nan

        wx["district"] = name
        wx["state"] = row["state"]
        wx["season"] = wx["date"].map(
            lambda d: season_of(d, str(row["coast_side"])))
        wx["slope_deg"] = float(row["slope_deg"])
        wx["urbanisation"] = row["urbanisation"]
        frames.append(wx)
        time.sleep(0.4)                    # be polite to a free service

    if not frames:
        raise RuntimeError("no districts could be fetched")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["district", "date"], keep="last")
    df.to_csv(OBSERVED_CSV, index=False)
    return df


# ---------------------------------------------------------------------------
# re-estimation
# ---------------------------------------------------------------------------
def _expert_prior_table(node: str) -> np.ndarray:
    """The elicited CPT for this node, as (child_states, parent_configs)."""
    cpd = {c.variable: c for c in expert_cpds()}[node]
    return np.asarray(cpd.get_values(), dtype=float)


def _parent_configs(parents: List[str]) -> List[Tuple[str, ...]]:
    import itertools
    return list(itertools.product(*[STATES[p] for p in parents]))


def relearn_node(disc: pd.DataFrame, node: str, parents: List[str],
                 ess: float = EQUIVALENT_SAMPLE_SIZE) -> Tuple[np.ndarray, dict]:
    """
    Dirichlet posterior for one node from real observations, with the elicited CPT
    as the prior mean - exactly the estimator we already use, just pointed at real
    data and restricted to the rows where everything it needs is present.
    """
    prior = _expert_prior_table(node)
    states = STATES[node]
    configs = _parent_configs(parents)

    counts = np.zeros((len(states), max(len(configs), 1)))
    needed = [node] + parents
    rows = disc.dropna(subset=needed)

    if parents:
        index = {cfg: j for j, cfg in enumerate(configs)}
        for _, r in rows.iterrows():
            cfg = tuple(str(r[p]) for p in parents)
            j = index.get(cfg)
            if j is None:
                continue
            counts[states.index(str(r[node])), j] += 1
    else:
        for _, r in rows.iterrows():
            counts[states.index(str(r[node])), 0] += 1

    alpha = prior * ess
    posterior = alpha + counts
    posterior = posterior / posterior.sum(axis=0, keepdims=True)

    info = {
        "node": node,
        "parents": parents,
        "rows_used": int(len(rows)),
        "min_cell_count": int(counts.sum(axis=0).min()) if counts.size else 0,
        "mean_abs_shift": round(float(np.abs(posterior - prior).mean()), 4),
        "max_abs_shift": round(float(np.abs(posterior - prior).max()), 4),
    }
    return posterior, info


def relearn_weather_layer(df: Optional[pd.DataFrame] = None,
                          ess: float = EQUIVALENT_SAMPLE_SIZE) -> dict:
    """
    Rebuild the weather-layer CPTs from real observations and write
    artifacts/hybrid_cpts.json (weather from reality, hazards unchanged).
    """
    from pgmpy.factors.discrete import TabularCPD
    from .learn import LEARNED_MODEL

    df = df if df is not None else build_observed_dataset()
    disc = discretize_frame(df)
    disc["Season"] = df["season"].values
    disc["Urbanisation"] = df["urbanisation"].values

    parents_of = {
        "Season": [], "Rainfall": ["Season"], "Temperature": ["Season"],
        "SeaSurfaceTemp": ["Season"], "Humidity": ["Rainfall"],
        "PressureDrop": ["SeaSurfaceTemp"], "WindSpeed": ["PressureDrop"],
    }

    # start from whatever we already have, then overwrite only the weather layer
    base = json.loads(LEARNED_MODEL.read_text(encoding="utf-8"))
    report = {"rows_total": int(len(df)),
              "districts": int(df["district"].nunique()),
              "window": f"{START_DATE}..{END_DATE}",
              "nodes": []}

    for node in RELEARNED_NODES:
        parents = parents_of[node]
        table, info = relearn_node(disc, node, parents, ess)
        base[node] = {"evidence": parents, "values": table.tolist()}
        report["nodes"].append(info)
        print(f"  {node:<16} rows {info['rows_used']:>6}  "
              f"mean shift {info['mean_abs_shift']:.4f}  "
              f"max {info['max_abs_shift']:.4f}")

    cpds = cpds_from_json(base)
    model = build_network_from_cpds(cpds)          # validates the whole thing
    HYBRID_MODEL.write_text(json.dumps(base, indent=1), encoding="utf-8")

    # the headline comparison: what a monsoon day looks like, before and after
    def rainfall_given(model_json, season):
        spec = model_json["Rainfall"]
        vals = np.asarray(spec["values"], dtype=float)
        j = STATES["Season"].index(season)
        return {s: round(float(vals[i, j]), 4)
                for i, s in enumerate(STATES["Rainfall"])}

    simulated = json.loads(LEARNED_MODEL.read_text(encoding="utf-8"))
    report["monsoon_rainfall_simulated"] = rainfall_given(simulated, "Monsoon")
    report["monsoon_rainfall_observed"] = rainfall_given(base, "Monsoon")
    report["observed_mean_daily_rainfall_mm"] = round(
        float(df["rainfall_mm"].mean()), 2)
    report["observed_monsoon_mean_daily_rainfall_mm"] = round(
        float(df.loc[df["season"] == "Monsoon", "rainfall_mm"].mean()), 2)

    RELEARN_REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def load_hybrid_model():
    """The model the live pipeline should use, falling back gracefully."""
    from .learn import load_model
    if HYBRID_MODEL.exists():
        return build_network_from_cpds(
            cpds_from_json(json.loads(HYBRID_MODEL.read_text(encoding="utf-8"))))
    return load_model()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    force = "--refetch" in sys.argv
    topup = "--topup" in sys.argv

    print("building the real observation table "
          f"({START_DATE} to {END_DATE}, 16 districts)")
    df = build_observed_dataset(force=force, topup=topup)
    print(f"\n{len(df):,} real district-days, "
          f"{df['district'].nunique()} districts")
    print(f"mean daily rainfall overall : {df['rainfall_mm'].mean():.2f} mm")
    print("mean daily rainfall by season:")
    print(df.groupby("season")["rainfall_mm"].agg(["mean", "max", "count"])
          .round(2).to_string())

    print("\nre-estimating the weather layer from real observations")
    report = relearn_weather_layer(df)

    print("\nP(Rainfall | Season = Monsoon)")
    print(f"  simulated model : {report['monsoon_rainfall_simulated']}")
    print(f"  real observations: {report['monsoon_rainfall_observed']}")
    print(f"\nwrote {HYBRID_MODEL.name} and {RELEARN_REPORT.name}")
