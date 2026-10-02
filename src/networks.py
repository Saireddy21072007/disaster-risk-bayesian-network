"""
The Bayesian network itself: structure + the expert (prior) CPTs.

Why we did not type 81 numbers per table by hand
------------------------------------------------
Some of our nodes have three parents with three states each, i.e. 27 parent
configurations. Filling those by hand is both painful and impossible to defend
in a review ("where did 0.63 come from?"). So we used the two canonical CPT
families from the course instead:

  * ordered logit  - for ordinal children (SoilMoisture, RiverLevel, ...).
                     One weight per parent + cut-points. Monotone by
                     construction: more rain can never lower the flood risk.
  * noisy-OR       - for "any of these causes can independently trigger it"
                     children (RoadBlocked, RescueNeeded).

Every table is therefore generated from a handful of interpretable numbers that
we can argue about with a domain expert, and the full CPT is still a plain
TabularCPD that pgmpy does exact inference on. Section 6 of the report explains
how these priors are then refined from data (BDeu / Bayesian estimator).

Owner of this file: Rohit Vardhan M (structure + elicitation)
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from .config import STATES

# ----------------------------------------------------------------------------
# graph structure (the DAG)
# ----------------------------------------------------------------------------
EDGES: List[tuple] = [
    # context -> weather
    ("Season", "Rainfall"),
    ("Season", "Temperature"),
    ("Season", "SeaSurfaceTemp"),
    ("Rainfall", "Humidity"),
    # weather -> hidden physical state
    ("Rainfall", "SoilMoisture"),
    ("Humidity", "SoilMoisture"),
    ("Rainfall", "RiverLevel"),
    ("SoilMoisture", "RiverLevel"),
    # cyclone driver chain (SST warms the sea -> pressure falls -> wind picks up)
    ("SeaSurfaceTemp", "PressureDrop"),
    ("PressureDrop", "WindSpeed"),
    # hazards
    ("RiverLevel", "Flood"),
    ("SoilMoisture", "Flood"),
    ("Urbanisation", "Flood"),
    ("Rainfall", "Landslide"),
    ("SoilMoisture", "Landslide"),
    ("Slope", "Landslide"),
    ("PressureDrop", "Cyclone"),
    ("WindSpeed", "Cyclone"),
    ("SeaSurfaceTemp", "Cyclone"),
    ("Temperature", "Heatwave"),
    ("Humidity", "Heatwave"),
    # consequences
    ("Flood", "RoadBlocked"),
    ("Landslide", "RoadBlocked"),
    ("Cyclone", "RoadBlocked"),
    ("Flood", "RescueNeeded"),
    ("Landslide", "RescueNeeded"),
    ("Cyclone", "RescueNeeded"),
    ("RescueNeeded", "HospitalDemand"),
    ("Heatwave", "HospitalDemand"),
]


# ----------------------------------------------------------------------------
# small helpers for building CPTs
# ----------------------------------------------------------------------------
def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _severity(var: str, state_index: int) -> float:
    """Map a state index onto [0, 1]. State 0 is the mildest by convention."""
    card = len(STATES[var])
    if card == 1:
        return 0.0
    return state_index / (card - 1)


def _parent_grid(parents: Sequence[str]):
    """
    All parent configurations in the order pgmpy expects for TabularCPD.values.

    pgmpy lays the columns out with the LAST evidence variable changing fastest,
    which is exactly itertools.product over the parent state indices.
    """
    return itertools.product(*[range(len(STATES[p])) for p in parents])


def _cpd(var: str, parents: Sequence[str], columns: List[List[float]]) -> TabularCPD:
    """columns[i] = full distribution over `var` for the i-th parent config."""
    values = np.array(columns).T                       # (card, n_configs)
    values = values / values.sum(axis=0, keepdims=True)  # guard against rounding
    return TabularCPD(
        variable=var,
        variable_card=len(STATES[var]),
        values=values.tolist(),
        evidence=list(parents) or None,
        evidence_card=[len(STATES[p]) for p in parents] or None,
        state_names={v: STATES[v] for v in [var, *parents]},
    )


def ordered_logit_cpd(var: str, weights: Dict[str, float],
                      cutpoints: Sequence[float],
                      interactions: Dict[tuple, float] | None = None
                      ) -> TabularCPD:
    """
    Ordinal child. score = sum_p weight_p * severity(parent_p)
                           + sum_(a,b) weight_ab * severity_a * severity_b;
    P(Y <= k) = sigmoid(cut_k - score).

    A negative weight means the parent *protects* against the outcome, which is
    how we encode "humid air suppresses a heat wave".

    The interaction term exists because a purely additive logit gets landslides
    wrong: it lets a steep slope raise the risk on its own, and a dry hill does
    not slide. Slope only matters multiplied by water.
    """
    card = len(STATES[var])
    assert len(cutpoints) == card - 1, f"{var}: need {card - 1} cut-points"
    parents = list(weights)
    interactions = interactions or {}

    columns = []
    for config in _parent_grid(parents):
        sev = {p: _severity(p, s) for p, s in zip(parents, config)}
        score = sum(weights[p] * sev[p] for p in parents)
        for (a, b), w in interactions.items():
            score += w * sev[a] * sev[b]
        cum = [_sigmoid(c - score) for c in cutpoints] + [1.0]
        probs, prev = [], 0.0
        for c in cum:
            probs.append(max(c - prev, 1e-9))
            prev = c
        columns.append(probs)
    return _cpd(var, parents, columns)


def logit_cpd(var: str, weights: Dict[str, float], cutpoint: float,
              interactions: Dict[tuple, float] | None = None) -> TabularCPD:
    """Binary child (states ['No', 'Yes']) with a logistic link."""
    return ordered_logit_cpd(var, weights, [cutpoint], interactions)


def noisy_or_cpd(var: str, link: Dict[str, float], leak: float = 0.01) -> TabularCPD:
    """
    Classic noisy-OR. Each parent independently triggers the child with
    probability link[parent] (scaled by how severe that parent's state is), and
    `leak` covers causes we did not model.

        P(no | parents) = (1 - leak) * prod_p (1 - link_p * severity_p)
    """
    parents = list(link)
    columns = []
    for config in _parent_grid(parents):
        p_no = 1.0 - leak
        for p, s in zip(parents, config):
            p_no *= 1.0 - link[p] * _severity(p, s)
        columns.append([p_no, 1.0 - p_no])
    return _cpd(var, parents, columns)


def prior_cpd(var: str, probs: Sequence[float]) -> TabularCPD:
    return _cpd(var, [], [list(probs)])


def table_cpd(var: str, parent: str, rows: Dict[str, Sequence[float]]) -> TabularCPD:
    """Explicit single-parent table. Used where the parent is not really ordinal."""
    columns = [list(rows[state]) for state in STATES[parent]]
    return _cpd(var, [parent], columns)


# ----------------------------------------------------------------------------
# the expert CPTs
#
# Sources / reasoning for the numbers are in docs/cpt_justification.md - short
# version: rainfall and river-level effects come from the IMD/CWC bands, the
# landslide weights follow the "rain on saturated soil on a steep slope" rule of
# thumb used by GSI, and the consequence layer was elicited from the group after
# reading the 2018 Kerala flood post-event reports.
# ----------------------------------------------------------------------------
def expert_cpds() -> List[TabularCPD]:
    cpds = [
        # ---- roots ----
        prior_cpd("Season", [0.33, 0.30, 0.37]),          # Winter / Summer / Monsoon
        prior_cpd("Slope", [0.45, 0.35, 0.20]),           # terrain mix of our districts
        prior_cpd("Urbanisation", [0.65, 0.35]),

        # ---- weather ----
        table_cpd("Rainfall", "Season", {
            "Winter":  [0.75, 0.22, 0.03],
            "Summer":  [0.55, 0.35, 0.10],
            "Monsoon": [0.15, 0.45, 0.40],
        }),
        table_cpd("Temperature", "Season", {
            "Winter":  [0.55, 0.42, 0.03],
            "Summer":  [0.05, 0.50, 0.45],
            "Monsoon": [0.25, 0.68, 0.07],
        }),
        table_cpd("SeaSurfaceTemp", "Season", {
            "Winter":  [0.80, 0.20],
            "Summer":  [0.45, 0.55],
            "Monsoon": [0.35, 0.65],
        }),
        table_cpd("Humidity", "Rainfall", {
            "Low":      [0.45, 0.40, 0.15],
            "Moderate": [0.12, 0.48, 0.40],
            "Heavy":    [0.03, 0.27, 0.70],
        }),
        table_cpd("PressureDrop", "SeaSurfaceTemp", {
            "Normal": [0.92, 0.08],
            "Warm":   [0.70, 0.30],
        }),
        table_cpd("WindSpeed", "PressureDrop", {
            "No":  [0.42, 0.44, 0.12, 0.02],
            "Yes": [0.08, 0.26, 0.38, 0.28],
        }),

        # ---- hidden physical state ----
        ordered_logit_cpd("SoilMoisture",
                          {"Rainfall": 3.6, "Humidity": 1.4},
                          cutpoints=[0.9, 3.0]),
        ordered_logit_cpd("RiverLevel",
                          {"Rainfall": 3.0, "SoilMoisture": 2.8},
                          cutpoints=[1.2, 3.6]),

        # ---- hazards ----
        logit_cpd("Flood",
                  {"RiverLevel": 4.6, "SoilMoisture": 2.0, "Urbanisation": 1.1},
                  cutpoint=4.6),
        # slope x soil-moisture interaction: a steep slope is only dangerous once
        # the ground is wet, which a purely additive logit cannot express
        logit_cpd("Landslide",
                  {"Rainfall": 1.6, "SoilMoisture": 1.8, "Slope": 1.0},
                  cutpoint=5.8,
                  interactions={("Slope", "SoilMoisture"): 4.6}),
        logit_cpd("Cyclone",
                  {"PressureDrop": 2.6, "WindSpeed": 4.4, "SeaSurfaceTemp": 1.6},
                  cutpoint=5.6),
        # humidity has a NEGATIVE weight: dry air is what makes a heat wave
        logit_cpd("Heatwave",
                  {"Temperature": 6.0, "Humidity": -1.8},
                  cutpoint=4.4),

        # ---- consequences ----
        noisy_or_cpd("RoadBlocked",
                     {"Flood": 0.75, "Landslide": 0.85, "Cyclone": 0.55},
                     leak=0.02),
        noisy_or_cpd("RescueNeeded",
                     {"Flood": 0.70, "Landslide": 0.80, "Cyclone": 0.65},
                     leak=0.01),
        # a severe heat wave fills wards on its own - it does not need a rescue
        # operation first, which is what our earlier weight of 2.2 implied
        ordered_logit_cpd("HospitalDemand",
                          {"RescueNeeded": 3.4, "Heatwave": 3.8},
                          cutpoints=[1.0, 3.6]),
    ]
    return cpds


def build_expert_network() -> DiscreteBayesianNetwork:
    """The prior model - no data seen yet. This is what we demo in review 1."""
    model = DiscreteBayesianNetwork(EDGES)
    model.add_cpds(*expert_cpds())
    model.check_model()
    return model


def build_network_from_cpds(cpds: List[TabularCPD]) -> DiscreteBayesianNetwork:
    model = DiscreteBayesianNetwork(EDGES)
    model.add_cpds(*cpds)
    model.check_model()
    return model


# ----------------------------------------------------------------------------
# small utilities the rest of the project uses
# ----------------------------------------------------------------------------
def node_layers() -> Dict[str, List[str]]:
    """Grouping used for the graph drawing and for the slides."""
    return {
        "context":      ["Season", "Slope", "Urbanisation"],
        "sensors":      ["Rainfall", "Temperature", "Humidity",
                         "SeaSurfaceTemp", "PressureDrop", "WindSpeed"],
        "latent":       ["SoilMoisture", "RiverLevel"],
        "hazards":      ["Flood", "Landslide", "Cyclone", "Heatwave"],
        "consequences": ["RoadBlocked", "RescueNeeded", "HospitalDemand"],
    }


def describe_model(model: DiscreteBayesianNetwork) -> dict:
    """Numbers we quote on the 'model complexity' slide."""
    n_params = 0
    for cpd in model.get_cpds():
        card = cpd.variable_card
        parent_configs = int(np.prod([len(STATES[p]) for p in cpd.variables[1:]])) or 1
        n_params += (card - 1) * parent_configs
    full_joint = int(np.prod([len(STATES[v]) for v in model.nodes()]))
    return {
        "nodes": model.number_of_nodes(),
        "edges": model.number_of_edges(),
        "free_parameters": n_params,
        "full_joint_size": full_joint,
        "compression": round(full_joint / n_params, 1),
    }


if __name__ == "__main__":
    m = build_expert_network()
    print("model OK")
    for k, v in describe_model(m).items():
        print(f"  {k:>18}: {v}")
