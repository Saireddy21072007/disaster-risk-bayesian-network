"""
Central place for every constant used in the project.

We kept all the state names, discretisation cut-offs and file paths here so that
none of the other modules has to hard-code a threshold. If the domain expert
(or the IMD circular) changes a cut-off, only this file needs an edit.

22AIE301 Probabilistic Reasoning - Project
Group: Sai Reddy A, Rohit Vardhan M, Jithin Reddy K, Sai Vandith
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"          # learned CPTs, plots, metrics
REPORT_DIR = ROOT / "report"

RAW_EVENTS_CSV = DATA_DIR / "events_raw.csv"          # simulated sensor records
DISTRICTS_CSV = DATA_DIR / "districts.csv"            # GIS layer
LEARNED_MODEL = ARTIFACT_DIR / "learned_cpts.json"
METRICS_JSON = ARTIFACT_DIR / "metrics.json"

for _d in (DATA_DIR, ARTIFACT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# random seed - keep the whole project reproducible
# ----------------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------------
# state spaces of every node in the Bayesian network
# The order matters: states are treated as ORDINAL (index 0 = mildest) by the
# CPT builders in networks.py, so please do not shuffle them.
# ----------------------------------------------------------------------------
STATES = {
    # ---- context / static ----
    "Season":         ["Winter", "Summer", "Monsoon"],
    "Slope":          ["Flat", "Moderate", "Steep"],
    "Urbanisation":   ["Low", "High"],
    # ---- weather sensors ----
    "Rainfall":       ["Low", "Moderate", "Heavy"],
    "Temperature":    ["Low", "Normal", "High"],
    "Humidity":       ["Low", "Medium", "High"],
    "SeaSurfaceTemp": ["Normal", "Warm"],
    "PressureDrop":   ["No", "Yes"],
    # four bins, not three: with a single 30-62 kmph bucket an ordinary monsoon
    # depression came out looking 60 % like a cyclone, because the bin averaged
    # over everything up to storm strength
    "WindSpeed":      ["Calm", "Breezy", "Strong", "Gale"],
    # ---- hidden / intermediate physical state ----
    "SoilMoisture":   ["Low", "Medium", "High"],
    "RiverLevel":     ["Low", "Medium", "High"],
    # ---- hazards ----
    "Flood":          ["No", "Yes"],
    "Landslide":      ["No", "Yes"],
    "Cyclone":        ["No", "Yes"],
    "Heatwave":       ["No", "Yes"],
    # ---- consequences ----
    "RoadBlocked":    ["No", "Yes"],
    "RescueNeeded":   ["No", "Yes"],
    "HospitalDemand": ["Low", "Medium", "High"],
}

HAZARDS = ["Flood", "Landslide", "Cyclone", "Heatwave"]
CONSEQUENCES = ["RoadBlocked", "RescueNeeded", "HospitalDemand"]

# Variables a district actually reports to us (what the dashboard can send).
OBSERVABLE = [
    "Season", "Slope", "Urbanisation",
    "Rainfall", "Temperature", "Humidity",
    "WindSpeed", "SeaSurfaceTemp", "PressureDrop",
    "RiverLevel",          # gauge stations - sometimes missing, that is the point
]

# The "safe" state of every observable. Used by the counterfactual part of the
# explanation module ("what if rainfall had only been Low?").
BENIGN_STATE = {
    "Rainfall": "Low",
    "Temperature": "Normal",
    "Humidity": "Low",
    "WindSpeed": "Calm",
    "SeaSurfaceTemp": "Normal",
    "PressureDrop": "No",
    "RiverLevel": "Low",
    "SoilMoisture": "Low",
}

# ----------------------------------------------------------------------------
# discretisation cut-offs
#
# Rainfall bands follow the IMD 24-hour rainfall classification
#   (light < 15.6 mm, moderate 15.6-64.4 mm, heavy >= 64.5 mm).
# Temperature bands follow the IMD heat-wave criterion for the plains (40 C).
# River level is expressed as a fraction of the local danger level, which is how
# CWC publishes gauge readings (warning level ~= 0.9 of danger level).
# ----------------------------------------------------------------------------
BINS = {
    "rainfall_mm":      ([15.6, 64.5], ["Low", "Moderate", "Heavy"]),
    "temperature_c":    ([30.0, 40.0], ["Low", "Normal", "High"]),
    "humidity_pct":     ([55.0, 80.0], ["Low", "Medium", "High"]),
    # 62 kmph is the IMD cut-off for a "cyclonic storm"
    "wind_kmph":        ([20.0, 40.0, 62.0], ["Calm", "Breezy", "Strong", "Gale"]),
    "sst_c":            ([28.0], ["Normal", "Warm"]),                # 28 C = cyclogenesis threshold
    "pressure_drop_hpa": ([4.0], ["No", "Yes"]),                     # 24-h drop
    "soil_moisture_frac": ([0.40, 0.70], ["Low", "Medium", "High"]),
    "river_level_frac": ([0.70, 0.95], ["Low", "Medium", "High"]),
    "slope_deg":        ([10.0, 25.0], ["Flat", "Moderate", "Steep"]),
}

# ----------------------------------------------------------------------------
# risk banding used by the dashboard colours and by decision.py
# ----------------------------------------------------------------------------
RISK_BANDS = [
    (0.75, "High", "#d7301f"),
    (0.40, "Medium", "#fdae61"),
    (0.00, "Low", "#1a9850"),
]


def risk_band(p: float):
    """Return (label, hex colour) for a probability."""
    for cut, label, colour in RISK_BANDS:
        if p >= cut:
            return label, colour
    return "Low", "#1a9850"
