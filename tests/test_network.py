"""
Tests for the network itself: is it a valid BN, do the CPTs mean what we think
they mean, and does the graph encode the independencies we claimed in the report?

Run:  python -m pytest tests -q
No network access, no API keys, a couple of seconds.
"""

import itertools

import numpy as np
import pytest

from src.config import STATES
from src.engine import RiskEngine
from src.networks import (EDGES, build_expert_network, describe_model,
                          noisy_or_cpd, ordered_logit_cpd)


@pytest.fixture(scope="module")
def model():
    return build_expert_network()


def test_model_is_valid(model):
    assert model.check_model()
    assert model.number_of_nodes() == len(STATES)


def test_every_node_has_a_cpd(model):
    for node in model.nodes():
        cpd = model.get_cpds(node)
        assert cpd is not None
        assert cpd.variable_card == len(STATES[node])


def test_all_cpds_are_normalised(model):
    for cpd in model.get_cpds():
        totals = np.asarray(cpd.get_values()).sum(axis=0)
        assert np.allclose(totals, 1.0), f"{cpd.variable} does not sum to 1"


def test_graph_is_acyclic(model):
    import networkx as nx
    assert nx.is_directed_acyclic_graph(nx.DiGraph(EDGES))


# ---------------------------------------------------------------------------
# CPT semantics. These are the tests that caught our parent-ordering bug: the
# table looked fine and the model validated, but the cells were transposed.
# ---------------------------------------------------------------------------
def test_flood_cpt_is_monotone_in_river_level(model):
    cpd = model.get_cpds("Flood")
    prev = -1.0
    for level in STATES["RiverLevel"]:
        p = cpd.get_value(Flood="Yes", RiverLevel=level,
                          SoilMoisture="Medium", Urbanisation="Low")
        assert p > prev, "flood risk must not fall as the river rises"
        prev = p


def test_dry_steep_slope_is_safe_but_wet_steep_slope_is_not(model):
    """The interaction term: slope alone must not trigger a landslide."""
    cpd = model.get_cpds("Landslide")
    dry = cpd.get_value(Landslide="Yes", Rainfall="Low",
                        SoilMoisture="Low", Slope="Steep")
    wet = cpd.get_value(Landslide="Yes", Rainfall="Heavy",
                        SoilMoisture="High", Slope="Steep")
    wet_flat = cpd.get_value(Landslide="Yes", Rainfall="Heavy",
                             SoilMoisture="High", Slope="Flat")
    assert dry < 0.05, f"a dry hill should not slide (got {dry:.3f})"
    assert wet > 0.85, f"a saturated steep slope should be dangerous (got {wet:.3f})"
    assert wet_flat < 0.20, "flat ground should stay low even when saturated"


def test_humidity_protects_against_heatwave(model):
    cpd = model.get_cpds("Heatwave")
    dry = cpd.get_value(Heatwave="Yes", Temperature="High", Humidity="Low")
    humid = cpd.get_value(Heatwave="Yes", Temperature="High", Humidity="High")
    assert dry > humid, "a negative weight on humidity should lower heat-wave risk"


def test_noisy_or_matches_the_closed_form():
    cpd = noisy_or_cpd("RoadBlocked", {"Flood": 0.75, "Landslide": 0.85,
                                       "Cyclone": 0.55}, leak=0.02)
    # both causes absent -> only the leak fires
    assert cpd.get_value(RoadBlocked="Yes", Flood="No", Landslide="No",
                         Cyclone="No") == pytest.approx(0.02, abs=1e-9)
    # one cause present
    expected = 1 - (1 - 0.02) * (1 - 0.85)
    assert cpd.get_value(RoadBlocked="Yes", Flood="No", Landslide="Yes",
                         Cyclone="No") == pytest.approx(expected, abs=1e-9)


def test_ordered_logit_columns_line_up_with_state_names():
    """
    pgmpy lays CPT columns out with the last parent varying fastest. If our
    _parent_grid ever disagrees, this test fails loudly.
    """
    cpd = ordered_logit_cpd("RiverLevel",
                            {"Rainfall": 3.0, "SoilMoisture": 2.8},
                            cutpoints=[1.2, 3.6])
    values = np.asarray(cpd.get_values())
    for i, (rain, soil) in enumerate(itertools.product(STATES["Rainfall"],
                                                       STATES["SoilMoisture"])):
        direct = cpd.get_value(RiverLevel="High", Rainfall=rain, SoilMoisture=soil)
        assert direct == pytest.approx(values[2, i])


def test_model_is_much_smaller_than_the_full_joint(model):
    info = describe_model(model)
    assert info["free_parameters"] < 300
    assert info["full_joint_size"] > 1_000_000
    assert info["compression"] > 1000


# ---------------------------------------------------------------------------
# d-separation: the claims we make on the "why a DAG" slide
# ---------------------------------------------------------------------------
def test_dseparation_claims(model):
    eng = RiskEngine(model)

    # rainfall reaches flood only through soil moisture and the river
    assert eng.is_dependent("Rainfall", "Flood", [])
    assert not eng.is_dependent("Rainfall", "Flood", ["RiverLevel", "SoilMoisture"])

    # flood and landslide share rainfall as a cause, so they are dependent
    assert eng.is_dependent("Flood", "Landslide", [])

    # ...and observing a common effect keeps them dependent (explaining away)
    assert eng.is_dependent("Flood", "Landslide", ["RoadBlocked"])

    # a heat wave has nothing to do with the river
    assert not eng.is_dependent("Heatwave", "RiverLevel", ["Rainfall", "Humidity"])
