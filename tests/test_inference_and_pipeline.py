"""
Tests for inference, discretisation, explanations and the decision layer.

The point of these is not coverage for its own sake - each one pins down a claim
we make in the presentation, so if someone edits a weight and the story stops
being true, pytest says so before the review does.
"""

import numpy as np
import pytest

from src.config import BINS, HAZARDS, STATES
from src.decision import (ACTIONS, best_action, evpi, expected_utility,
                          p_emergency, recommend, switch_points)
from src.discretize import bin_value, evidence_from_reading
from src.engine import InvalidEvidence, RiskEngine, entropy
from src.explain import (counterfactuals, evidence_attribution, narrate,
                         value_of_information)
from src.networks import build_expert_network


@pytest.fixture(scope="module")
def eng():
    # the expert network, so the tests do not depend on artifacts/ existing
    return RiskEngine(build_expert_network())


WET = {"Season": "Monsoon", "Rainfall": "Heavy", "Humidity": "High",
       "RiverLevel": "High", "Slope": "Steep", "Urbanisation": "Low"}
DRY = {"Season": "Winter", "Rainfall": "Low", "Humidity": "Low",
       "RiverLevel": "Low", "Slope": "Flat", "Urbanisation": "Low"}


# ---------------------------------------------------------------------------
# discretisation
# ---------------------------------------------------------------------------
def test_imd_rainfall_bands():
    assert bin_value("rainfall_mm", 3.0) == "Low"
    assert bin_value("rainfall_mm", 15.6) == "Moderate"
    assert bin_value("rainfall_mm", 64.5) == "Heavy"
    assert bin_value("rainfall_mm", 200.0) == "Heavy"


def test_missing_reading_gives_no_evidence():
    assert bin_value("rainfall_mm", None) is None
    assert bin_value("rainfall_mm", float("nan")) is None
    ev = evidence_from_reading({"rainfall_mm": 90.0, "river_level_frac": None})
    assert ev == {"Rainfall": "Heavy"}


def test_every_bin_spec_has_one_more_label_than_edges():
    for col, (edges, labels) in BINS.items():
        assert len(labels) == len(edges) + 1, col


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def test_posteriors_are_distributions(eng):
    post = eng.posterior(HAZARDS + ["HospitalDemand"], WET)
    for var, dist in post.items():
        assert set(dist) == set(STATES[var])
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-9)


def test_wet_monsoon_is_riskier_than_dry_winter(eng):
    wet = eng.hazard_probabilities(WET)
    dry = eng.hazard_probabilities(DRY)
    for h in ("Flood", "Landslide"):
        assert wet[h] > dry[h] + 0.3, h
    # ...and the heat wave goes the other way round
    assert eng.probability("Heatwave", "Yes", {"Temperature": "High",
                                               "Humidity": "Low"}) > 0.5


def test_more_evidence_lowers_entropy(eng):
    """Not guaranteed for a single observation, but it must hold on average for
    the flood posterior as we hand over the whole sensor set."""
    partial = entropy(eng.posterior(["Flood"], {"Season": "Monsoon"})["Flood"])
    full = entropy(eng.posterior(["Flood"], WET)["Flood"])
    assert full < partial


def test_evidence_is_validated(eng):
    with pytest.raises(InvalidEvidence):
        eng.hazard_probabilities({"Rainfall": "Torrential"})
    with pytest.raises(InvalidEvidence):
        eng.hazard_probabilities({"NotANode": "Low"})


def test_observed_variable_has_degenerate_posterior(eng):
    post = eng.posterior(["Rainfall"], WET)
    assert post["Rainfall"]["Heavy"] == 1.0


def test_belief_propagation_agrees_with_variable_elimination():
    """Two different exact algorithms must give the same answer - a cheap but
    strong check that we are not misusing either of them."""
    model = build_expert_network()
    ve = RiskEngine(model, method="ve")
    bp = RiskEngine(model, method="bp")
    for h in HAZARDS:
        assert ve.probability(h, "Yes", WET) == pytest.approx(
            bp.probability(h, "Yes", WET), abs=1e-6)


def test_latent_map_reconstructs_saturated_soil(eng):
    m = eng.latent_map({"Rainfall": "Heavy", "Humidity": "High"})
    assert m["SoilMoisture"] == "High"


# ---------------------------------------------------------------------------
# explanations
# ---------------------------------------------------------------------------
def test_attribution_is_signed_and_ranked(eng):
    rows = evidence_attribution(eng, WET, "Flood")
    assert len(rows) == len(WET)
    order = [abs(r["weight_of_evidence_bits"]) for r in rows]
    assert order == sorted(order, reverse=True)
    # the river gauge must be the top driver of flood risk here
    assert rows[0]["variable"] == "RiverLevel"
    assert rows[0]["direction"] == "raises"


def test_urbanisation_low_lowers_flood_risk(eng):
    rows = {r["variable"]: r for r in evidence_attribution(eng, WET, "Flood")}
    assert rows["Urbanisation"]["direction"] == "lowers"


def test_counterfactual_reduces_risk(eng):
    cf = counterfactuals(eng, WET, "Flood", top=2)
    assert cf, "there should be at least one non-benign driver to flip"
    for c in cf:
        assert c["p_counterfactual"] < c["p_actual"]


def test_value_of_information_is_non_negative_and_prefers_the_river(eng):
    partial = {"Season": "Monsoon", "Rainfall": "Heavy"}
    voi = value_of_information(eng, partial, "Flood")
    assert all(v["expected_bits_gained"] >= 0 for v in voi)
    assert voi[0]["variable"] == "RiverLevel"


def test_voi_is_zero_for_something_already_known(eng):
    voi = {v["variable"]: v for v in value_of_information(eng, WET, "Flood")}
    assert "RiverLevel" not in voi          # already observed, so not offered


def test_narrative_mentions_the_number_and_the_driver(eng):
    out = narrate(eng, WET, "Flood")
    assert f"{out['probability']:.0%}" in out["text"]
    assert "river" in out["text"].lower()
    assert out["risk_band"] in ("Low", "Medium", "High")


# ---------------------------------------------------------------------------
# decision layer
# ---------------------------------------------------------------------------
def test_action_is_monotone_in_risk():
    """Higher probability must never lead to a weaker response."""
    severity = {a: i for i, a in enumerate(ACTIONS)}
    last = -1
    for p in np.linspace(0, 1, 101):
        s = severity[best_action(p)]
        assert s >= last, f"response weakened at p={p:.2f}"
        last = s


def test_thresholds_cover_the_unit_interval():
    ranges = switch_points()
    assert set(ranges) <= set(ACTIONS)
    lo = min(r[0] for r in ranges.values())
    hi = max(r[1] for r in ranges.values())
    assert lo == pytest.approx(0.0) and hi == pytest.approx(1.0)


def test_evacuation_threshold_is_derived_not_eighty_percent():
    """Our synopsis said 'evacuate above 80%'. The utility table disagrees, and
    that disagreement is a result we present, so pin it down."""
    ranges = switch_points()
    start_of_evacuation = ranges["Evacuate"][0]
    assert 0.5 < start_of_evacuation < 0.8
    assert abs(start_of_evacuation - 0.80) > 0.05


def test_evpi_is_non_negative_and_vanishes_when_the_answer_is_obvious():
    assert evpi(0.0) == pytest.approx(0.0, abs=1e-9)
    assert evpi(1.0) == pytest.approx(0.0, abs=1e-9)
    assert evpi(0.5) > evpi(0.02)


def test_expected_utility_is_a_proper_mixture():
    for a in ACTIONS:
        for p in (0.0, 0.37, 1.0):
            eu = expected_utility(a, p)
            assert min(expected_utility(a, 0), expected_utility(a, 1)) - 1e-9 <= eu


def test_heatwave_alone_can_trigger_a_response(eng):
    """
    Regression test. Our first decision node only looked at RescueNeeded, so a
    42 C day came back as 'Monitor'. The emergency event is now the union of
    'rescue needed' and 'hospital demand high'.
    """
    heat = {"Season": "Summer", "Temperature": "High", "Humidity": "Low",
            "Rainfall": "Low", "Slope": "Flat", "Urbanisation": "High"}
    assert eng.probability("Heatwave", "Yes", heat) > 0.5
    assert p_emergency(eng, heat) > eng.probability("RescueNeeded", "Yes", heat)
    plan = recommend(eng, heat, population=3_000_000)
    assert plan["action"] != "Monitor"
    assert plan["dominant_hazard"] == "Heatwave"
    assert plan["action_label"] != plan["action"]   # relabelled for heat waves


def test_recommendation_bundle_is_complete(eng):
    plan = recommend(eng, WET, population=1_000_000)
    for key in ("action", "checklist", "expected_utility_table", "evpi",
                "rationale", "p_emergency", "exposed_population"):
        assert key in plan
    assert plan["checklist"]
    assert plan["action"] == "Evacuate"
