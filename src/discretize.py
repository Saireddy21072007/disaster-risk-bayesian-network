"""
Turning continuous sensor readings into the discrete states our BN understands.

This is the only place where a number becomes a word. We deliberately kept the
cut-offs in config.BINS so that the mapping is auditable - during the review we
can point at one dictionary and say "these are the IMD bands".

Owner: Jithin Reddy K
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import BINS, STATES

# raw csv column  ->  BN node name
COLUMN_TO_NODE = {
    "rainfall_mm": "Rainfall",
    "temperature_c": "Temperature",
    "humidity_pct": "Humidity",
    "wind_kmph": "WindSpeed",
    "sst_c": "SeaSurfaceTemp",
    "pressure_drop_hpa": "PressureDrop",
    "soil_moisture_frac": "SoilMoisture",
    "river_level_frac": "RiverLevel",
    "slope_deg": "Slope",
}


def bin_value(column: str, value: Optional[float]) -> Optional[str]:
    """One reading -> one state label. Returns None for a missing reading."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    edges, labels = BINS[column]
    idx = int(np.digitize([value], edges)[0])
    return labels[idx]


def discretize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full raw dataframe -> dataframe of state labels, one column per BN node.
    Columns that are already categorical (Season, Urbanisation, the hazard
    labels) are copied across / mapped from 0-1 to No-Yes.
    """
    out = pd.DataFrame(index=df.index)

    for col, node in COLUMN_TO_NODE.items():
        if col in df.columns:
            out[node] = df[col].map(lambda v: bin_value(col, v))

    if "season" in df.columns:
        out["Season"] = df["season"]
    if "urbanisation" in df.columns:
        out["Urbanisation"] = df["urbanisation"]

    for node in ["Flood", "Landslide", "Cyclone", "Heatwave",
                 "RoadBlocked", "RescueNeeded"]:
        col = node.lower()
        if col in df.columns:
            out[node] = df[col].map({0: "No", 1: "Yes", "No": "No", "Yes": "Yes"})

    if "hospital_demand" in df.columns:
        out["HospitalDemand"] = df["hospital_demand"]

    # keep the state order fixed so pgmpy's state_names line up everywhere
    for node in out.columns:
        out[node] = pd.Categorical(out[node], categories=STATES[node])
    return out[[n for n in STATES if n in out.columns]]


def evidence_from_reading(reading: Dict[str, float]) -> Dict[str, str]:
    """
    What the API receives (a dict of raw numbers, possibly incomplete) ->
    what pgmpy's query() wants (a dict of state labels).

    Anything the caller did not send is simply left out of the evidence: the BN
    marginalises over it. That graceful handling of missing sensors is one of the
    main reasons we chose a probabilistic model over a plain classifier.
    """
    evidence: Dict[str, str] = {}
    for col, node in COLUMN_TO_NODE.items():
        if col in reading and reading[col] is not None:
            state = bin_value(col, reading[col])
            if state is not None:
                evidence[node] = state

    # already-categorical inputs
    for key, node in (("season", "Season"), ("urbanisation", "Urbanisation")):
        if reading.get(key) in STATES.get(node, []):
            evidence[node] = reading[key]

    return evidence
