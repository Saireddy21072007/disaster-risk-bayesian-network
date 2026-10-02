"""
End-to-end tests through the FastAPI layer, using Starlette's TestClient so no
server has to be running.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["nodes"] == 18


def test_dashboard_page_is_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Early Warning Console" in r.text


def test_scenarios_are_listed():
    ids = [s["id"] for s in client.get("/api/scenarios").json()["scenarios"]]
    assert "monsoon_depression" in ids and "calm_winter" in ids


def test_overview_ranks_districts_by_risk():
    r = client.get("/api/overview?scenario=monsoon_depression")
    assert r.status_code == 200
    body = r.json()
    assert len(body["districts"]) == 16
    tops = [max(d["hazards"].values()) for d in body["districts"]]
    assert tops == sorted(tops, reverse=True)
    for d in body["districts"]:
        assert d["recommendation"]["action"] in ("Monitor", "Advisory",
                                                 "Prepare", "Evacuate")


def test_unknown_scenario_is_a_404():
    assert client.get("/api/overview?scenario=alien_invasion").status_code == 404


def test_a_calm_winter_day_needs_no_evacuation():
    body = client.get("/api/overview?scenario=calm_winter").json()
    assert body["summary"]["actions"]["Evacuate"] == 0


def test_the_cyclone_only_hits_the_east_coast():
    """A Bay of Bengal system must not put Kerala on evacuation notice."""
    body = client.get("/api/overview?scenario=cyclone_landfall").json()
    by_name = {d["district"]: d for d in body["districts"]}
    assert by_name["Krishna"]["hazards"]["Cyclone"] > 0.8
    assert by_name["Kozhikode"]["hazards"]["Cyclone"] < 0.3


def test_district_detail_has_an_explanation_for_every_hazard():
    r = client.post("/api/district", json={"scenario": "monsoon_depression",
                                           "district": "Idukki"})
    d = r.json()
    assert set(d["explanations"]) == {"Flood", "Landslide", "Cyclone", "Heatwave"}
    e = d["explanations"][d["dominant_hazard"]]
    assert e["text"] and e["bullets"] and e["attribution"]
    assert d["recommendation"]["checklist"]


def test_offline_gauge_district_still_gets_an_answer():
    """Graceful degradation: Wayanad's river gauge is switched off in the demo."""
    d = client.post("/api/district", json={"scenario": "monsoon_depression",
                                           "district": "Wayanad"}).json()
    assert d["reading"]["river_level_frac"] is None
    assert "RiverLevel" not in d["evidence"]
    assert 0.0 <= d["hazards"]["Flood"] <= 1.0
    # the model should tell us the gauge is the thing worth measuring
    voi = d["explanations"]["Flood"]["value_of_information"]
    assert voi[0]["variable"] == "RiverLevel"


def test_predict_accepts_a_partial_reading():
    r = client.post("/api/predict", json={"season": "Monsoon",
                                          "rainfall_mm": 120.0})
    assert r.status_code == 200
    body = r.json()
    assert body["evidence"] == {"Season": "Monsoon", "Rainfall": "Heavy"}
    assert body["confidence"]["sensors_used"] == 2


def test_predict_rejects_a_nonsense_season():
    r = client.post("/api/predict", json={"season": "Rainy", "rainfall_mm": 10})
    # an unknown season is simply not used as evidence rather than crashing
    assert r.status_code == 200
    assert "Season" not in r.json()["evidence"]


def test_model_info_reports_the_independence_claims():
    info = client.get("/api/model").json()
    claims = {c["query"]: c["d_connected"] for c in info["independence_examples"]}
    assert claims["Rainfall -/- Flood | RiverLevel, SoilMoisture"] is False
    assert claims["Flood -- Landslide | {}"] is True
    assert info["model"]["free_parameters"] < 300
