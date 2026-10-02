"""
FastAPI service behind the dashboard.

Endpoint map
    GET  /                     the dashboard page
    GET  /api/scenarios        list of demo scenarios
    GET  /api/overview         one row per district: probabilities + action
                               (light - no explanations, so the map paints fast)
    POST /api/district         full detail for one district: explanations,
                               counterfactuals, value of information, checklist
    POST /api/predict          same thing for a hand-typed reading (the sandbox)
    GET  /api/model            network size, thresholds, d-separation examples
    GET  /api/metrics          the held-out evaluation numbers

The heavy objects (network + junction tree) are built once at import time. On a
laptop the overview call for 16 districts takes well under a second because
RiskEngine memoises on the evidence pattern.

Owner: Sai Reddy A
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import scenarios
from src.config import HAZARDS, METRICS_JSON, risk_band
from src.decision import recommend, switch_points
from src.discretize import evidence_from_reading
from src.engine import InvalidEvidence, get_engine
from src.explain import explain_all, narrate
from src.networks import describe_model

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Multi-hazard disaster prediction (22AIE301)",
              version="1.0",
              description="Bayesian network + XAI + decision network, "
                          "served for the district control room.")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

ENGINE = get_engine()


# ----------------------------------------------------------------------------
# request bodies
# ----------------------------------------------------------------------------
class Reading(BaseModel):
    """Raw sensor values. Every field is optional - that is the whole point."""
    season: Optional[str] = Field(None, description="Winter | Summer | Monsoon")
    urbanisation: Optional[str] = Field(None, description="Low | High")
    slope_deg: Optional[float] = None
    rainfall_mm: Optional[float] = None
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    sst_c: Optional[float] = None
    pressure_drop_hpa: Optional[float] = None
    wind_kmph: Optional[float] = None
    river_level_frac: Optional[float] = None
    population: Optional[int] = None
    district: Optional[str] = None


class DistrictRequest(BaseModel):
    scenario: str = "monsoon_depression"
    district: str


# ----------------------------------------------------------------------------
def _assess(reading: Dict, with_explanations: bool = True) -> Dict:
    evidence = evidence_from_reading(reading)
    try:
        hazards = ENGINE.hazard_probabilities(evidence)
    except InvalidEvidence as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    plan = recommend(ENGINE, evidence, population=reading.get("population"))
    top = plan["dominant_hazard"]
    band, colour = risk_band(hazards[top])

    out = {
        "district": reading.get("district"),
        "state": reading.get("state"),
        "lat": reading.get("lat"),
        "lon": reading.get("lon"),
        "population": reading.get("population"),
        "major_river": reading.get("major_river"),
        "nearest_shelter": reading.get("nearest_shelter"),
        "notes": reading.get("notes"),
        "reading": {k: v for k, v in reading.items()
                    if k not in ("district", "state", "lat", "lon")},
        "evidence": evidence,
        "hazards": {h: round(hazards[h], 4) for h in HAZARDS},
        "consequences": ENGINE.consequence_probabilities(evidence),
        "latent_map": ENGINE.latent_map(evidence),
        "dominant_hazard": top,
        "risk_band": band,
        "colour": colour,
        "recommendation": plan,
        "confidence": ENGINE.confidence(evidence, top),
    }
    if with_explanations:
        out["explanations"] = explain_all(ENGINE, evidence)
        out["headline_explanation"] = out["explanations"][top]["text"]
    else:
        out["headline_explanation"] = narrate(ENGINE, evidence, top)["text"]
    return out


# ----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/scenarios")
def list_scenarios():
    return {"scenarios": [{"id": k, "label": v}
                          for k, v in scenarios.SCENARIOS.items()],
            "gauge_offline": sorted(scenarios.GAUGE_OFFLINE)}


@app.get("/api/overview")
def overview(scenario: str = "monsoon_depression"):
    try:
        readings = scenarios.build(scenario)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    rows = [_assess(r, with_explanations=False) for r in readings]
    rows.sort(key=lambda r: -max(r["hazards"].values()))

    counts = {"High": 0, "Medium": 0, "Low": 0}
    for r in rows:
        counts[r["risk_band"]] += 1
    return {
        "scenario": scenario,
        "label": scenarios.SCENARIOS[scenario],
        "districts": rows,
        "summary": {
            "band_counts": counts,
            "actions": {a: sum(1 for r in rows
                               if r["recommendation"]["action"] == a)
                        for a in ("Monitor", "Advisory", "Prepare", "Evacuate")},
            "expected_people_needing_help": sum(
                r["recommendation"].get("expected_people_needing_help", 0)
                for r in rows),
        },
    }


@app.post("/api/district")
def district_detail(req: DistrictRequest):
    readings = scenarios.build(req.scenario)
    match = next((r for r in readings if r["district"].lower() == req.district.lower()),
                 None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"no district '{req.district}'")
    return _assess(match, with_explanations=True)


@app.post("/api/predict")
def predict(reading: Reading):
    return _assess(reading.model_dump(exclude_none=True), with_explanations=True)


@app.get("/api/model")
def model_info():
    return {
        "model": describe_model(ENGINE.model),
        "nodes": {n: ENGINE.model.get_cpds(n).variables[1:] or ["(root)"]
                  for n in ENGINE.model.nodes()},
        "decision_thresholds": switch_points(),
        "independence_examples": [
            {
                "claim": "Rainfall tells us nothing more about Flood once the "
                         "river gauge is read",
                "query": "Rainfall -/- Flood | RiverLevel, SoilMoisture",
                "d_connected": ENGINE.is_dependent(
                    "Rainfall", "Flood", ["RiverLevel", "SoilMoisture"]),
            },
            {
                "claim": "Flood and Landslide are dependent even with no evidence "
                         "(they share rainfall as a cause)",
                "query": "Flood -- Landslide | {}",
                "d_connected": ENGINE.is_dependent("Flood", "Landslide", []),
            },
            {
                "claim": "Knowing the road is blocked makes flood and landslide "
                         "compete to explain it",
                "query": "Flood -- Landslide | RoadBlocked",
                "d_connected": ENGINE.is_dependent("Flood", "Landslide",
                                                   ["RoadBlocked"]),
            },
        ],
    }


@app.get("/api/metrics")
def metrics():
    if not METRICS_JSON.exists():
        raise HTTPException(status_code=404,
                            detail="run `python -m src.evaluate` first")
    return json.loads(METRICS_JSON.read_text())


@app.get("/api/health")
def health():
    return {"status": "ok", "nodes": ENGINE.model.number_of_nodes()}


# ---------------------------------------------------------------------------
# the live path
#
# Everything above this line runs on the scenario generator and the all-simulated
# model. Everything below runs on real feeds and the hybrid model. They are kept
# separate on purpose - see docs/realtime_pipeline.md.
# ---------------------------------------------------------------------------
class LiveRequest(BaseModel):
    district: str
    mode: str = "snapshot"          # snapshot | live | offline


def _processor(mode: str):
    from src.stream import get_processor
    if mode not in ("live", "snapshot", "offline"):
        raise HTTPException(status_code=422, detail=f"unknown mode '{mode}'")
    return get_processor(mode)


@app.get("/api/live/overview")
def live_overview(mode: str = "snapshot"):
    """
    One pass over every district. `mode=live` hits the public APIs (a few seconds);
    `mode=snapshot` replays the last saved pull, which is what we demo from.
    """
    proc = _processor(mode)
    records = proc.step_all(explain=False)
    rows = [r.to_dict() for r in records]
    rows.sort(key=lambda r: -max(r["hazards"].values()))

    levels = {"Monitor": 0, "Advisory": 0, "Prepare": 0, "Evacuate": 0}
    for r in rows:
        levels[r["alert_level"]] += 1

    return {
        "mode": mode,
        "generated_at": rows[0]["timestamp"] if rows else None,
        "districts": rows,
        "summary": {
            "alert_levels": levels,
            "readings_total": sum(len(r["readings"]) for r in rows),
            "gated": sum(len(r["gate_events"]) for r in rows),
            "model_surprises": sum(
                1 for r in rows for g in r["gate_events"]
                if g.get("verdict") == "model-surprise"),
            "feeds_degraded": sum(1 for r in rows if r["errors"]),
            "expected_people_needing_help": sum(
                r["recommendation"].get("expected_people_needing_help", 0)
                for r in rows),
        },
    }


@app.post("/api/live/district")
def live_district(req: LiveRequest):
    """Full detail for one district, including the explanation and the comparison
    against what hard binning would have concluded."""
    proc = _processor(req.mode)
    from src.feeds import districts as _districts
    df = _districts()
    match = df[df["district"].str.lower() == req.district.lower()]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"no district '{req.district}'")
    return proc.step(match.iloc[0], explain=True).to_dict()


@app.get("/api/live/provenance")
def live_provenance():
    """
    Which parts of the model came from where. This is the answer to "you are
    choosing the existing preprocessed data?" in machine-readable form.
    """
    import json as _json
    from src.observed import (HYBRID_MODEL, OBSERVED_CSV, RELEARN_REPORT,
                              RELEARNED_NODES)
    from src.temporal import _CAL_FILE
    from src.feeds import NWP_MODELS, SOURCE_LABEL

    out = {
        "live_sources": [
            {"service": "api.open-meteo.com", "provides":
             "rainfall, temperature, humidity, wind, surface pressure",
             "models": [SOURCE_LABEL[m] for m in NWP_MODELS]},
            {"service": "flood-api.open-meteo.com",
             "provides": "GloFAS daily river discharge", "models": ["GloFAS"]},
            {"service": "marine-api.open-meteo.com",
             "provides": "sea surface temperature", "models": ["Marine analysis"]},
        ],
        "relearned_from_real_observations": RELEARNED_NODES,
        "still_elicited_or_simulated": [
            "SoilMoisture", "RiverLevel", "Flood", "Landslide", "Cyclone",
            "Heatwave", "RoadBlocked", "RescueNeeded", "HospitalDemand",
        ],
        "why_not_all": "the hazard layer needs labelled events per district-day, "
                       "which no public table provides; the latent hydrology has "
                       "an unobserved parent",
        "observed_table_exists": OBSERVED_CSV.exists(),
        "hybrid_model_exists": HYBRID_MODEL.exists(),
    }
    if RELEARN_REPORT.exists():
        out["relearn_report"] = _json.loads(RELEARN_REPORT.read_text(encoding="utf-8"))
    if _CAL_FILE.exists():
        out["filter_calibration"] = _json.loads(_CAL_FILE.read_text(encoding="utf-8"))
    return out


@app.get("/api/live/audit")
def live_audit(limit: int = 40):
    """Tail of the append-only alert log, so any warning can be reviewed after the
    event together with the evidence that produced it."""
    from src.stream import AUDIT_LOG
    if not AUDIT_LOG.exists():
        return {"entries": []}
    lines = AUDIT_LOG.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines[-max(limit, 1):]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"entries": list(reversed(entries)), "total": len(lines)}
