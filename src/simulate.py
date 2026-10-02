"""
Simulator that produces our training data (data/events_raw.csv).

WHY SIMULATED DATA - please read before the review
--------------------------------------------------
We could not get a single public table that contains rainfall, humidity, soil
moisture, river gauge level AND a verified flood / landslide label for the same
district on the same day. IMD, NASA POWER, CWC and the EM-DAT disaster archive
each hold one piece of it and joining them properly is a project on its own
(it is item 1 on our weekly plan for the final review).

So for now we generate the data ourselves from a *continuous* physical process
(exponential saturation curves + logistic hazard triggers + noise). Two things
make this honest rather than circular:

  1. The simulator does NOT use the Bayesian network. It works on real-valued
     quantities; the BN only ever sees the discretised version. So the BN has to
     recover the relationship through a lossy 3-bin view of the world - exactly
     the handicap it would face on real gauge data.
  2. The graph structure of the simulator matches our DAG, but none of the
     functional forms do. That means the learned CPTs are a genuine estimation
     result, not the numbers we typed in networks.py handed back to us.

Everything downstream (learning, evaluation, dashboard) reads a CSV, so the day
we finish the IMD + CWC join we drop in the real file and nothing else changes.

Owner: Sai Vandith (simulator) with Sai Reddy A (hazard triggers)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR, RAW_EVENTS_CSV, SEED, DISTRICTS_CSV

SEASONS = ["Winter", "Summer", "Monsoon"]
SEASON_P = [0.33, 0.30, 0.37]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate(n: int = 12000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    districts = pd.read_csv(DISTRICTS_CSV)

    # pick a district for every record - this is where slope / urbanisation and
    # the population figure used by the dashboard come from
    idx = rng.integers(0, len(districts), size=n)
    d = districts.iloc[idx].reset_index(drop=True)

    season = rng.choice(SEASONS, size=n, p=SEASON_P)
    is_winter = season == "Winter"
    is_summer = season == "Summer"
    is_monsoon = season == "Monsoon"

    # ---- rainfall: gamma, because daily rainfall is skewed with a long tail ----
    rain = np.zeros(n)
    rain[is_winter] = rng.gamma(1.2, 6.0, is_winter.sum())
    rain[is_summer] = rng.gamma(1.6, 12.0, is_summer.sum())
    rain[is_monsoon] = rng.gamma(2.2, 30.0, is_monsoon.sum())

    # ---- temperature ----
    temp = np.zeros(n)
    temp[is_winter] = rng.normal(28.5, 2.5, is_winter.sum())
    temp[is_summer] = rng.normal(37.5, 3.2, is_summer.sum())
    temp[is_monsoon] = rng.normal(31.0, 2.0, is_monsoon.sum())

    # ---- sea surface temperature ----
    sst = np.zeros(n)
    sst[is_winter] = rng.normal(27.0, 0.9, is_winter.sum())
    sst[is_summer] = rng.normal(28.2, 0.9, is_summer.sum())
    sst[is_monsoon] = rng.normal(28.6, 0.9, is_monsoon.sum())

    # ---- humidity rises with rain but saturates ----
    hum = 45 + 30 * (1 - np.exp(-rain / 35.0)) + rng.normal(0, 7, n)
    hum = np.clip(hum, 18, 100)

    # ---- a warm sea deepens the 24h pressure fall, which drives the wind ----
    pdrop = np.clip(2.7 * (sst - 27.5) + rng.normal(0, 2.2, n), 0, None)
    wind = np.clip(12 + 4.5 * pdrop + rng.normal(0, 7, n), 0, None)

    # ---- latent physical state ----
    soil = (0.15 + 0.55 * (1 - np.exp(-rain / 50.0))
            + 0.25 * (hum - 45) / 55.0 + rng.normal(0, 0.07, n))
    soil = np.clip(soil, 0.02, 1.0)

    river = (0.25 + 0.45 * (1 - np.exp(-rain / 60.0))
             + 0.50 * (soil - 0.30) + rng.normal(0, 0.09, n))
    river = np.clip(river, 0.02, 1.6)

    slope = np.clip(d["slope_deg"].to_numpy() + rng.normal(0, 1.5, n), 0.5, None)
    urban_high = (d["urbanisation"] == "High").to_numpy().astype(float)

    # ---- hazard triggers (logistic on the physical drivers) ----
    p_flood = _sigmoid(7.0 * (river - 0.85) + 2.5 * (soil - 0.60)
                       + 0.8 * urban_high - 0.40)
    # Landslides: slope is a PREDISPOSING factor, water is the TRIGGER. Our first
    # version added a plain slope term and ended up claiming a 22 % landslide
    # chance for the Nilgiris on a dry January day, which is nonsense. So the
    # slope term is now scaled by how wet the soil is, and slopes below the 12 deg
    # reference are penalised harder than steeper ones are rewarded.
    slope_gain = np.where(slope > 12.0, 0.16, 0.30) * (slope - 12.0)
    wetness = np.clip(soil / 0.60, 0, 1.4)
    p_slide = _sigmoid(-3.00 + 5.0 * (soil - 0.50) + 1.8 * (rain / 100.0 - 0.40)
                       + slope_gain * wetness)
    p_cyc = _sigmoid(0.10 * (wind - 55.0) + 1.4 * (sst - 28.0)
                     + 0.25 * pdrop - 1.20)
    p_heat = _sigmoid(1.10 * (temp - 39.0) - 0.05 * (hum - 50.0) - 0.60)

    flood = rng.binomial(1, p_flood)
    slide = rng.binomial(1, p_slide)
    cyc = rng.binomial(1, p_cyc)
    heat = rng.binomial(1, p_heat)

    # ---- consequence layer: noisy-OR, same idea as in the BN ----
    def noisy_or(links, leak):
        p_no = np.full(n, 1.0 - leak)
        for cause, prob in links:
            p_no = p_no * (1 - prob * cause)
        return rng.binomial(1, 1 - p_no)

    road = noisy_or([(flood, 0.75), (slide, 0.85), (cyc, 0.55)], leak=0.02)
    rescue = noisy_or([(flood, 0.70), (slide, 0.80), (cyc, 0.65)], leak=0.01)

    # hospital demand: ordered logit on rescue + heat wave. The heat-wave weight
    # is large enough on its own to push demand into the top band - heat stroke
    # cases arrive at the PHC whether or not anyone had to be rescued.
    score = 3.4 * rescue + 3.8 * heat + rng.normal(0, 0.5, n)
    demand = np.where(score > 3.6, "High", np.where(score > 1.0, "Medium", "Low"))

    df = pd.DataFrame({
        "district": d["district"],
        "state": d["state"],
        "season": season,
        "rainfall_mm": rain.round(2),
        "temperature_c": temp.round(2),
        "humidity_pct": hum.round(2),
        "sst_c": sst.round(2),
        "pressure_drop_hpa": pdrop.round(2),
        "wind_kmph": wind.round(2),
        "soil_moisture_frac": soil.round(3),
        "river_level_frac": river.round(3),
        "slope_deg": slope.round(1),
        "urbanisation": d["urbanisation"],
        "flood": flood,
        "landslide": slide,
        "cyclone": cyc,
        "heatwave": heat,
        "roadblocked": road,
        "rescueneeded": rescue,
        "hospital_demand": demand,
    })
    return df


def main():
    df = simulate()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_EVENTS_CSV, index=False)
    print(f"wrote {len(df)} records -> {RAW_EVENTS_CSV}")
    print("\nbase rates (how often each hazard actually happens):")
    for h in ["flood", "landslide", "cyclone", "heatwave",
              "roadblocked", "rescueneeded"]:
        print(f"  {h:>13}: {df[h].mean():.3f}")
    print("\nhospital demand mix:")
    print(df["hospital_demand"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
