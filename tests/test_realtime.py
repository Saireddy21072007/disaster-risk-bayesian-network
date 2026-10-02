"""
Tests for the real-time layer: ingestion, soft evidence, the temporal filter and
the stream pipeline.

None of these touch the network. The feed tests use the saved snapshot, and
everything else works on hand-built inputs. As with the other test files, each test
pins down a claim we make about the method rather than just exercising a line.
"""

import math
from datetime import datetime

import numpy as np
import pytest

from src.config import STATES
from src.engine import RiskEngine, entropy
from src.feeds import Observation, SnapshotFeed, load_snapshot
from src.learn import load_model
from src.measurement import (HYDRO_NODES, SOFT_NODES, SoftEvidenceEngine,
                             bin_likelihood, fuse_observations,
                             hard_evidence_from_fused, observation_node, soften,
                             staleness_weight)
from src.observed import season_of
from src.stream import (DEESCALATE_FRAMES, StreamProcessor, current_season,
                        slope_state)
from src.temporal import (HYDRO_LEAF, JOINT, HydrologyFilter, calibration_offsets,
                         expected_transition, install_filtered_belief,
                         static_hydrology_marginal, stationary)

NOW = datetime(2026, 8, 10, 12, 0, 0)


def _obs(node, column, value, source, unit="", age_minutes=5):
    ts = datetime.now().isoformat()
    o = Observation(node=node, column=column, value=value, source=source,
                    observed_at=ts, fetched_at=ts, unit=unit)
    return o


# ---------------------------------------------------------------------------
# the Gaussian bin likelihood
# ---------------------------------------------------------------------------
def test_bin_likelihood_is_a_distribution():
    lam = bin_likelihood("rainfall_mm", 40.0, 5.0)
    assert len(lam) == len(STATES["Rainfall"])
    assert sum(lam) == pytest.approx(1.0, abs=1e-9)
    assert all(p > 0 for p in lam)


def test_tiny_sigma_recovers_hard_binning():
    """With no measurement error the likelihood should collapse onto one bin."""
    lam = bin_likelihood("rainfall_mm", 100.0, 1e-4)
    assert lam[STATES["Rainfall"].index("Heavy")] > 0.999


def test_a_reading_on_a_bin_edge_splits_its_mass():
    """
    15.6 mm is the IMD light/moderate boundary. A reading of 15.0 with a few mm of
    error must not claim to know which side it is on - this is the whole reason we
    stopped using hard evidence.
    """
    lam = bin_likelihood("rainfall_mm", 15.0, 3.0)
    low = lam[STATES["Rainfall"].index("Low")]
    moderate = lam[STATES["Rainfall"].index("Moderate")]
    assert 0.2 < low < 0.8
    assert 0.2 < moderate < 0.8


def test_wider_sigma_flattens_the_likelihood():
    sharp = bin_likelihood("rainfall_mm", 40.0, 2.0)
    vague = bin_likelihood("rainfall_mm", 40.0, 40.0)
    h_sharp = entropy({str(i): p for i, p in enumerate(sharp)})
    h_vague = entropy({str(i): p for i, p in enumerate(vague)})
    assert h_vague > h_sharp


def test_soften_endpoints():
    lam = [0.9, 0.07, 0.03]
    assert soften(lam, 1.0) == pytest.approx(lam, abs=1e-9)
    assert soften(lam, 0.0) == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-9)


def test_staleness_decays_and_is_variable_specific():
    assert staleness_weight("Rainfall", 0) == pytest.approx(1.0)
    assert staleness_weight("Rainfall", 600) < 1.0
    assert staleness_weight("Rainfall", 600) > staleness_weight("Rainfall", 2000)
    # a pressure drop goes off much faster than a sea surface temperature
    assert (staleness_weight("PressureDrop", 240)
            < staleness_weight("SeaSurfaceTemp", 240))


# ---------------------------------------------------------------------------
# ensemble fusion
# ---------------------------------------------------------------------------
def test_disagreeing_sources_produce_vaguer_evidence():
    """
    Gaussian ensemble dressing: the spread of the members inflates sigma, so three
    models that disagree give softer evidence than three that agree. This is the
    mechanism that makes reliability a measured quantity rather than a constant.
    """
    agree = fuse_observations([
        _obs("Rainfall", "rainfall_mm", 40.0, "gfs_seamless"),
        _obs("Rainfall", "rainfall_mm", 41.0, "ecmwf_ifs025"),
        _obs("Rainfall", "rainfall_mm", 39.5, "icon_seamless"),
    ])["Rainfall"]
    disagree = fuse_observations([
        _obs("Rainfall", "rainfall_mm", 5.0, "gfs_seamless"),
        _obs("Rainfall", "rainfall_mm", 40.0, "ecmwf_ifs025"),
        _obs("Rainfall", "rainfall_mm", 90.0, "icon_seamless"),
    ])["Rainfall"]

    assert disagree.ensemble_spread > agree.ensemble_spread
    assert disagree.sigma_effective > agree.sigma_effective
    assert disagree.sharpness < agree.sharpness
    assert disagree.status == "disputed"
    assert agree.status == "ok"


def test_ensemble_estimate_is_the_mean_of_the_members():
    fused = fuse_observations([
        _obs("Temperature", "temperature_c", 30.0, "gfs_seamless"),
        _obs("Temperature", "temperature_c", 34.0, "ecmwf_ifs025"),
    ])["Temperature"]
    assert fused.estimate == pytest.approx(32.0)
    assert fused.n_sources == 2


def test_single_source_uses_the_instrument_sigma_only():
    fused = fuse_observations([
        _obs("RiverLevel", "river_level_frac", 0.5, "glofas"),
    ])["RiverLevel"]
    assert fused.ensemble_spread == 0.0
    assert fused.sigma_effective == pytest.approx(fused.sigma_instrument)


# ---------------------------------------------------------------------------
# virtual evidence: the construction has to be exactly right
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def soft():
    return SoftEvidenceEngine(load_model())


def test_flat_likelihoods_leave_the_core_model_untouched(soft):
    """
    With every indicator leaf uninformative, the augmented network must agree with
    the core network to machine precision. If it does not, the leaves are leaking
    information and every number downstream is wrong.
    """
    core = RiskEngine(load_model())
    soft.install({})
    for hazard in ("Flood", "Landslide", "Cyclone", "Heatwave"):
        assert (soft.probability(hazard, "Yes", {})
                == pytest.approx(core.probability(hazard, "Yes", {}), abs=1e-9))


def test_one_hot_soft_evidence_equals_hard_evidence(soft):
    """
    Pearl's indicator construction reduces to ordinary conditioning when the
    likelihood is one-hot. This is the test that convinced us the plumbing is right.
    """
    core = RiskEngine(load_model())
    heavy = fuse_observations([
        _obs("Rainfall", "rainfall_mm", 200.0, "gfs_seamless"),
        _obs("Rainfall", "rainfall_mm", 200.0, "ecmwf_ifs025"),
    ])
    heavy["Rainfall"].likelihood = [1e-12, 1e-12, 1.0]      # force one-hot on Heavy

    evidence = soft.install(heavy)
    assert evidence == {observation_node("Rainfall"): "yes"}
    got = soft.probability("Flood", "Yes", evidence)
    want = core.probability("Flood", "Yes", {"Rainfall": "Heavy"})
    assert got == pytest.approx(want, abs=1e-6)


def test_soft_evidence_sits_between_the_two_hard_answers(soft):
    """A reading straddling a bin edge must land between what the two hard
    assertions would have given - not outside them."""
    core = RiskEngine(load_model())
    low = core.probability("Flood", "Yes", {"Rainfall": "Low"})
    moderate = core.probability("Flood", "Yes", {"Rainfall": "Moderate"})

    edge = fuse_observations([
        _obs("Rainfall", "rainfall_mm", 15.5, "gfs_seamless"),
        _obs("Rainfall", "rainfall_mm", 15.7, "ecmwf_ifs025"),
    ])
    evidence = soft.install(edge)
    got = soft.probability("Flood", "Yes", evidence)
    assert min(low, moderate) - 1e-9 <= got <= max(low, moderate) + 1e-9


def test_installing_a_new_reading_clears_the_previous_one(soft):
    """State must not leak between districts."""
    heavy = fuse_observations([_obs("Rainfall", "rainfall_mm", 250.0, "gfs_seamless")])
    soft.install(heavy)
    wet = soft.probability("Flood", "Yes", {observation_node("Rainfall"): "yes"})

    soft.install({})
    dry = soft.probability("Flood", "Yes", {})
    core = RiskEngine(load_model())
    assert dry == pytest.approx(core.probability("Flood", "Yes", {}), abs=1e-9)
    assert wet > dry


def test_surprisal_is_high_for_an_implausible_reading(soft):
    soft.install({})
    ev = {}
    # gale-force wind out of nowhere is a surprise; light wind is not
    gale = bin_likelihood("wind_kmph", 150.0, 4.0)
    calm = bin_likelihood("wind_kmph", 8.0, 4.0)
    assert (soft.surprisal_bits("WindSpeed", gale, ev)
            > soft.surprisal_bits("WindSpeed", calm, ev))


# ---------------------------------------------------------------------------
# the temporal filter
# ---------------------------------------------------------------------------
HEAVY = {"Low": 0.02, "Moderate": 0.10, "Heavy": 0.88}
DRY = {"Low": 0.92, "Moderate": 0.06, "Heavy": 0.02}


def test_rain_accumulates_over_days():
    """
    The whole point of the filter. One heavy day should not saturate a catchment;
    three should. Our static model could not express this and we listed it as a
    limitation at the last review.
    """
    f = HydrologyFilter("t")
    trajectory = []
    for _ in range(3):
        f.step(HEAVY)
        trajectory.append(f.marginals()["RiverLevel"]["High"])

    # strictly rising, and the third day is worth substantially more than the first
    assert trajectory == sorted(trajectory)
    assert trajectory[2] > 1.8 * trajectory[0]
    # and the soil is carrying the memory that makes it happen
    assert f.marginals()["SoilMoisture"]["High"] > 0.3


def test_the_river_recedes_when_the_rain_stops():
    f = HydrologyFilter("t")
    for _ in range(3):
        f.step(HEAVY)
    peak = f.marginals()["RiverLevel"]["High"]
    for _ in range(4):
        f.step(DRY)
    assert f.marginals()["RiverLevel"]["High"] < peak / 2


def test_soil_has_memory():
    """Yesterday's soil is the strongest predictor of today's."""
    wet = HydrologyFilter("wet")
    for _ in range(4):
        wet.step(HEAVY)
    dry = HydrologyFilter("dry")
    for _ in range(4):
        dry.step(DRY)
    # one dry day each: the previously-wet catchment must still be wetter
    wet.step(DRY)
    dry.step(DRY)
    assert (wet.marginals()["SoilMoisture"]["High"]
            > dry.marginals()["SoilMoisture"]["High"])


def test_filter_coasts_when_no_reading_arrives():
    f = HydrologyFilter("t")
    step = f.step(HEAVY, hydro_readings={})
    assert step.coasted is True
    assert step.used_readings == []
    assert sum(step.belief) == pytest.approx(1.0, abs=1e-9)


def test_filter_belief_stays_a_distribution():
    f = HydrologyFilter("t")
    for rain in [HEAVY, DRY, HEAVY, HEAVY, DRY]:
        step = f.step(rain)
        assert sum(step.belief) == pytest.approx(1.0, abs=1e-9)
        assert all(b >= 0 for b in step.belief)


def test_transition_is_calibrated_to_the_static_marginal():
    """
    The filter and the static CPTs are two elicitations of the same system. If their
    resting points disagree, a district with no readings drifts away from the base
    rate for no reason - which is exactly the bug this calibration fixed.
    """
    from src.temporal import climatological_rain
    pi = stationary(expected_transition(climatological_rain()))
    target = static_hydrology_marginal()
    assert np.abs(pi - target).max() < 0.08


def test_a_silent_district_stays_near_the_base_rate():
    """The end-to-end version of the test above."""
    f = HydrologyFilter("silent")
    from src.temporal import climatological_rain
    rain = climatological_rain()
    start = f.marginals()["RiverLevel"]["High"]
    for _ in range(30):
        f.step(rain)
    assert abs(f.marginals()["RiverLevel"]["High"] - start) < 0.06


def test_virtual_evidence_reproduces_the_filter_belief(soft):
    """
    install_filtered_belief divides out the network's own prediction, so the
    network's posterior over the hydrology must come out EQUAL to the filter's
    belief - replacing its memoryless guess rather than compounding with it.
    """
    f = HydrologyFilter("t")
    for _ in range(3):
        f.step(HEAVY)

    evidence = soft.install({})
    evidence = install_filtered_belief(soft, f.belief, evidence)
    assert evidence[HYDRO_LEAF] == "yes"

    joint = soft.joint(["SoilMoisture", "RiverLevel"], evidence)
    got = np.array([float(joint.get_value(SoilMoisture=s, RiverLevel=r))
                    for s, r in JOINT])
    assert np.abs(got - np.asarray(f.belief)).max() < 1e-9


def test_a_wet_filter_raises_the_flood_probability(soft):
    dry_f = HydrologyFilter("d")
    for _ in range(4):
        dry_f.step(DRY)
    wet_f = HydrologyFilter("w")
    for _ in range(4):
        wet_f.step(HEAVY)

    ev = soft.install({})
    dry_p = soft.probability("Flood", "Yes",
                             install_filtered_belief(soft, dry_f.belief, dict(ev)))
    wet_p = soft.probability("Flood", "Yes",
                             install_filtered_belief(soft, wet_f.belief, dict(ev)))
    assert wet_p > dry_p + 0.2


# ---------------------------------------------------------------------------
# season calendar
# ---------------------------------------------------------------------------
def test_east_coast_october_is_monsoon_but_west_coast_october_is_not():
    """
    The retreating monsoon. We only found this because the real data showed
    "winter" wetter than "summer", which is not a climate that exists.
    """
    assert season_of("2024-10-15", "east") == "Monsoon"
    assert season_of("2024-10-15", "west") == "Winter"
    assert season_of("2024-07-15", "west") == "Monsoon"
    assert season_of("2024-04-15", "none") == "Summer"


def test_stream_season_matches_training_season():
    """Training and inference must use the same calendar or the CPT columns line
    up wrongly for every east-coast district."""
    when = datetime(2024, 11, 20)
    for side in ("east", "west", "none"):
        assert current_season(when, side) == season_of("2024-11-20", side)


def test_slope_state_uses_the_configured_bands():
    assert slope_state(2.0) == "Flat"
    assert slope_state(15.0) == "Moderate"
    assert slope_state(29.0) == "Steep"


# ---------------------------------------------------------------------------
# the snapshot feed and the pipeline
# ---------------------------------------------------------------------------
def _have_snapshot() -> bool:
    return len(load_snapshot()) > 0


snapshot_needed = pytest.mark.skipif(
    not _have_snapshot(),
    reason="no artifacts/live_snapshot.json; run python -m src.feeds first")


@snapshot_needed
def test_snapshot_feed_returns_real_readings():
    feed = SnapshotFeed()
    from src.feeds import districts
    row = districts().iloc[0]
    got = feed.fetch_district(row)
    assert got.district == row["district"]
    assert got.observations
    for o in got.observations:
        assert o.node in STATES
        assert isinstance(o.value, float)


@snapshot_needed
def test_ensemble_sources_are_actually_independent_models():
    """Three separate NWP models, not the same one three times."""
    feed = SnapshotFeed()
    from src.feeds import districts
    got = feed.fetch_district(districts().iloc[0])
    fused = fuse_observations(got.observations)
    rain = fused.get("Rainfall")
    assert rain is not None
    assert rain.n_sources >= 2
    assert len(set(rain.sources)) == rain.n_sources


@snapshot_needed
def test_a_pass_produces_a_complete_record():
    from src.feeds import districts
    proc = StreamProcessor("snapshot")
    row = districts().iloc[0]
    rec = proc.step(row, explain=True)

    assert rec.district == row["district"]
    assert set(rec.hazards) == {"Flood", "Landslide", "Cyclone", "Heatwave"}
    assert all(0.0 <= p <= 1.0 for p in rec.hazards.values())
    assert rec.alert_level in ("Monitor", "Advisory", "Prepare", "Evacuate")
    assert rec.readings and rec.filter_step
    assert rec.headline_explanation
    assert rec.recommendation["checklist"]


@snapshot_needed
def test_the_explanation_never_leaks_a_node_name():
    """An officer must never see 'Rainfall__obs' or 'Hydrology__filter'."""
    from src.feeds import districts
    proc = StreamProcessor("snapshot")
    for _, row in districts().head(4).iterrows():
        rec = proc.step(row, explain=True)
        text = rec.headline_explanation + " ".join(rec.explanation.get("bullets", []))
        assert "__obs" not in text
        assert "__filter" not in text
        assert "Hydrology" not in text


@snapshot_needed
def test_polling_priority_is_non_negative_and_ranked():
    from src.feeds import districts
    proc = StreamProcessor("snapshot")
    rec = proc.step(districts().iloc[0], explain=False)
    gains = [r["expected_bits_gained"] for r in rec.polling_priority]
    assert all(g >= 0 for g in gains)
    assert rec.polling_priority


@snapshot_needed
def test_a_district_with_no_river_reach_still_gets_an_answer():
    """
    Chennai has no GloFAS river reach at its centroid. That is a real gap in the
    data, not a simulated outage, and the system has to cope with it.
    """
    from src.feeds import districts
    df = districts()
    row = df[df["district"] == "Chennai"].iloc[0]
    proc = StreamProcessor("snapshot")
    rec = proc.step(row, explain=True)
    assert not any(r["node"] == "RiverLevel" for r in rec.readings)
    assert 0.0 <= rec.hazards["Flood"] <= 1.0
    assert any("river reach" in e for e in rec.errors)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def test_corroborated_surprise_is_blamed_on_the_model_not_the_sensor():
    """
    The bug that real data taught us. Three independent forecast models agreeing on
    an unexpected value is evidence about the world, not a sensor fault - a broken
    gauge cannot make GFS, ECMWF and ICON agree. So the reading keeps full trust and
    the MODEL gets flagged.
    """
    proc = StreamProcessor("snapshot")
    fused = fuse_observations([
        _obs("WindSpeed", "wind_kmph", 190.0, "gfs_seamless"),
        _obs("WindSpeed", "wind_kmph", 191.0, "ecmwf_ifs025"),
        _obs("WindSpeed", "wind_kmph", 189.0, "icon_seamless"),
    ])
    before = list(fused["WindSpeed"].likelihood)
    evidence = proc.engine.install(fused, skip=HYDRO_NODES)
    events = proc._apply_surprise_gate(fused, evidence)

    assert events and events[0]["verdict"] == "model-surprise"
    assert fused["WindSpeed"].status == "model-surprise"
    assert fused["WindSpeed"].trust_weight == pytest.approx(
        fused["WindSpeed"].staleness_weight)
    assert fused["WindSpeed"].likelihood == pytest.approx(before, abs=1e-9)


def test_uncorroborated_surprise_is_discounted():
    proc = StreamProcessor("snapshot")
    fused = fuse_observations([
        _obs("WindSpeed", "wind_kmph", 190.0, "gfs_seamless"),
    ])
    evidence = proc.engine.install(fused, skip=HYDRO_NODES)
    events = proc._apply_surprise_gate(fused, evidence)

    assert events and events[0]["verdict"] == "suspect"
    assert fused["WindSpeed"].status == "suspect"
    assert fused["WindSpeed"].trust_weight < fused["WindSpeed"].staleness_weight
    # discounted evidence is closer to uniform than the raw reading was
    assert (max(fused["WindSpeed"].likelihood)
            < max(fused["WindSpeed"].raw_likelihood))


def test_a_plausible_reading_is_not_gated():
    proc = StreamProcessor("snapshot")
    fused = fuse_observations([
        _obs("Temperature", "temperature_c", 30.0, "gfs_seamless"),
        _obs("Temperature", "temperature_c", 30.4, "ecmwf_ifs025"),
    ])
    evidence = proc.engine.install(fused, skip=HYDRO_NODES)
    assert proc._apply_surprise_gate(fused, evidence) == []
    assert fused["Temperature"].status == "ok"


# ---------------------------------------------------------------------------
# hysteresis
# ---------------------------------------------------------------------------
def test_escalation_is_immediate():
    proc = StreamProcessor("snapshot")
    proc.alerts.clear()
    level, changed, info = proc._apply_hysteresis("X", "Evacuate", "t0")
    assert level == "Evacuate" and changed is True
    assert "immediately" in info["reason"]


def test_standing_down_needs_several_agreeing_frames():
    proc = StreamProcessor("snapshot")
    proc.alerts.clear()
    proc._apply_hysteresis("X", "Evacuate", "t0")

    for frame in range(DEESCALATE_FRAMES - 1):
        level, changed, info = proc._apply_hysteresis("X", "Monitor", f"t{frame}")
        assert level == "Evacuate", "must not drop on the first quiet frame"
        assert changed is False
    level, changed, _ = proc._apply_hysteresis("X", "Monitor", "tN")
    assert level == "Monitor" and changed is True


def test_an_interrupted_stand_down_restarts_the_count():
    proc = StreamProcessor("snapshot")
    proc.alerts.clear()
    proc._apply_hysteresis("X", "Prepare", "t0")
    proc._apply_hysteresis("X", "Monitor", "t1")
    proc._apply_hysteresis("X", "Prepare", "t2")        # back up again
    level, changed, info = proc._apply_hysteresis("X", "Monitor", "t3")
    assert level == "Prepare"
    assert info["pending_frames"] == 1


# ---------------------------------------------------------------------------
# soft vs naive, the comparison we quote
# ---------------------------------------------------------------------------
def test_hard_evidence_helper_picks_the_most_likely_bin():
    fused = fuse_observations([
        _obs("Rainfall", "rainfall_mm", 15.4, "gfs_seamless"),
        _obs("Rainfall", "rainfall_mm", 15.5, "ecmwf_ifs025"),
    ])
    hard = hard_evidence_from_fused(fused)
    assert hard["Rainfall"] in ("Low", "Moderate")
    assert hard["Rainfall"] == fused["Rainfall"].most_likely_state()
