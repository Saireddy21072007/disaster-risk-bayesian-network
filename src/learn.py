"""
Parameter learning (and a structure-learning sanity check).

Three ways of getting the numbers into the CPTs, all of which we compare:

  expert  - only the elicited priors from networks.py, no data at all.
  mle     - maximum likelihood: count(child, parents) / count(parents).
            Breaks on rare cells: 'WindSpeed = High' shows up in well under 1 %
            of the records, so some columns of the cyclone CPT are estimated
            from a handful of rows (or divide by zero).
  bayes   - Dirichlet posterior with the EXPERT CPT as the prior mean:
                alpha_ijk = ESS * P_expert(x_i = k | pa_i = j)
                P_hat     = (alpha_ijk + N_ijk) / (alpha_ij. + N_ij.)
            This is the conjugate update from the course notes. With ESS = 50 the
            prior is worth 50 imaginary observations per parent configuration, so
            well-populated cells follow the data and starved cells fall back on
            the expert. This is the model the dashboard actually serves.

We also run Hill-Climb structure search and compare the recovered DAG with our
hand-drawn one (Structural Hamming Distance) - not to replace our structure, but
to check the data really does carry the dependencies we assumed.

Owner: Sai Reddy A
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from .config import ARTIFACT_DIR, LEARNED_MODEL, RAW_EVENTS_CSV, SEED, STATES
from .discretize import discretize_frame
from .networks import EDGES, build_network_from_cpds, expert_cpds

EQUIVALENT_SAMPLE_SIZE = 50.0


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_discrete(test_size: float = 0.25, seed: int = SEED
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train_disc, test_disc, train_raw, test_raw) - raw kept for the
    sklearn baselines, which are allowed to see the un-binned numbers."""
    raw = pd.read_csv(RAW_EVENTS_CSV)
    rng = np.random.default_rng(seed)
    mask = rng.random(len(raw)) >= test_size
    train_raw, test_raw = raw[mask].reset_index(drop=True), raw[~mask].reset_index(drop=True)
    return (discretize_frame(train_raw), discretize_frame(test_raw),
            train_raw, test_raw)


# ----------------------------------------------------------------------------
# (de)serialisation - so the API does not have to re-learn on every start
# ----------------------------------------------------------------------------
def _align_to_expert(cpd: TabularCPD) -> TabularCPD:
    """
    pgmpy's estimators return the parents in alphabetical order, while our
    expert CPTs use the order written in networks.py. Comparing the two value
    arrays without fixing that would compare different cells - we lost an
    evening to this. reorder_parents() permutes the table properly.
    """
    expert_order = {c.variable: list(c.variables[1:]) for c in expert_cpds()}
    want = expert_order.get(cpd.variable, [])
    have = list(cpd.variables[1:])
    if want and have and want != have and sorted(want) == sorted(have):
        cpd = cpd.copy()
        cpd.reorder_parents(want)
    return cpd


def cpds_to_json(cpds: List[TabularCPD]) -> dict:
    out = {}
    for cpd in cpds:
        cpd = _align_to_expert(cpd)
        out[cpd.variable] = {
            "evidence": list(cpd.variables[1:]),
            "values": np.asarray(cpd.get_values()).tolist(),
        }
    return out


def cpds_from_json(blob: dict) -> List[TabularCPD]:
    cpds = []
    for var, spec in blob.items():
        parents = spec["evidence"]
        cpds.append(TabularCPD(
            variable=var,
            variable_card=len(STATES[var]),
            values=spec["values"],
            evidence=parents or None,
            evidence_card=[len(STATES[p]) for p in parents] or None,
            state_names={v: STATES[v] for v in [var, *parents]},
        ))
    return cpds


def save_model(cpds: List[TabularCPD], path=LEARNED_MODEL) -> None:
    path.write_text(json.dumps(cpds_to_json(cpds), indent=1), encoding="utf-8")


def load_model(path=LEARNED_MODEL) -> DiscreteBayesianNetwork:
    """Learned network if we have already fitted one, otherwise the expert prior."""
    if path.exists():
        return build_network_from_cpds(cpds_from_json(json.loads(path.read_text())))
    return build_network_from_cpds(expert_cpds())


# ----------------------------------------------------------------------------
# the three fitting routines
# ----------------------------------------------------------------------------
def _skeleton() -> DiscreteBayesianNetwork:
    return DiscreteBayesianNetwork(EDGES)


def fit_mle(train: pd.DataFrame) -> List[TabularCPD]:
    model = _skeleton()
    est = MaximumLikelihoodEstimator(model, train, state_names=STATES)
    cpds = []
    for node in model.nodes():
        cpd = est.estimate_cpd(node)
        # MLE leaves NaN where a parent configuration never occurred - fall back
        # to a uniform distribution there so the model is still valid.
        vals = np.asarray(cpd.get_values(), dtype=float)
        bad = ~np.isfinite(vals.sum(axis=0)) | (vals.sum(axis=0) == 0)
        if bad.any():
            vals[:, bad] = 1.0 / vals.shape[0]
            cpd.values = vals.reshape(cpd.cardinality)
        cpds.append(cpd)
    return cpds


def _expert_lookup() -> Dict[str, np.ndarray]:
    return {c.variable: np.asarray(c.get_values(), dtype=float) for c in expert_cpds()}


def _expert_pseudo_counts(node: str, parent_order: List[str], ess: float) -> np.ndarray:
    """
    alpha_ijk = ess * P_expert(node = k | parents = j), laid out in the parent
    order pgmpy is going to use internally (alphabetical), NOT the order we wrote
    the table in. Getting this wrong silently applies the prior to the wrong
    cells - it made a monsoon day look like a heat wave until we caught it.
    """
    cpd = {c.variable: c for c in expert_cpds()}[node].copy()
    if parent_order and list(cpd.variables[1:]) != parent_order:
        cpd.reorder_parents(parent_order)
    return np.asarray(cpd.get_values(), dtype=float) * ess


def fit_bayesian(train: pd.DataFrame,
                 ess: float = EQUIVALENT_SAMPLE_SIZE) -> List[TabularCPD]:
    """Dirichlet posterior whose prior mean is the elicited expert CPT."""
    model = _skeleton()
    est = BayesianEstimator(model, train, state_names=STATES)

    cpds = []
    for node in model.nodes():
        order = sorted(model.get_parents(node))
        pseudo = _expert_pseudo_counts(node, order, ess)
        cpd = est.estimate_cpd(node, prior_type="dirichlet", pseudo_counts=pseudo)
        # guard: if a future pgmpy stops sorting the parents, fail loudly here
        assert list(cpd.variables[1:]) == order, (
            f"{node}: expected parent order {order}, got {cpd.variables[1:]}")
        cpds.append(cpd)
    return cpds


def fit_bdeu(train: pd.DataFrame, ess: float = EQUIVALENT_SAMPLE_SIZE) -> List[TabularCPD]:
    """Uninformative BDeu prior - the 'we had no expert' control condition."""
    model = _skeleton()
    est = BayesianEstimator(model, train, state_names=STATES)
    return [est.estimate_cpd(n, prior_type="BDeu", equivalent_sample_size=ess)
            for n in model.nodes()]


# ----------------------------------------------------------------------------
# how far did the data move the expert away from their beliefs?
# ----------------------------------------------------------------------------
def cpt_shift(learned: List[TabularCPD]) -> pd.DataFrame:
    """Per-node mean |P_learned - P_expert| and mean KL(learned || expert)."""
    expert = _expert_lookup()
    rows = []
    for cpd in learned:
        cpd = _align_to_expert(cpd)
        a = np.asarray(cpd.get_values(), dtype=float)
        b = expert[cpd.variable]
        eps = 1e-12
        kl = (a * (np.log(a + eps) - np.log(b + eps))).sum(axis=0)
        rows.append({
            "node": cpd.variable,
            "parents": ", ".join(cpd.variables[1:]) or "-",
            "cells": a.shape[1],
            "mean_abs_diff": round(float(np.abs(a - b).mean()), 4),
            "max_abs_diff": round(float(np.abs(a - b).max()), 4),
            "mean_kl": round(float(kl.mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("mean_kl", ascending=False)


# ----------------------------------------------------------------------------
# structure learning check
# ----------------------------------------------------------------------------
def structure_check(train: pd.DataFrame, max_indegree: int = 3) -> dict:
    """
    Hill-climb search with the BIC score, started from an empty graph, then
    compared with our hand-built DAG. We only use this as evidence that the
    dependencies we assumed are visible in the data - the expert DAG stays.
    """
    from pgmpy.estimators import HillClimbSearch
    from pgmpy.metrics import SHD

    hc = HillClimbSearch(train)
    learned = hc.estimate(scoring_method="bic-d", max_indegree=max_indegree,
                          show_progress=False)

    expert_edges = set(EDGES)
    learned_edges = set(learned.edges())
    # an edge counts as "found" even if the search flipped its direction, since
    # a DAG is only identifiable up to its Markov equivalence class
    undirected = {frozenset(e) for e in learned_edges}
    recovered = sum(1 for e in expert_edges if frozenset(e) in undirected)

    try:
        learned_bn = DiscreteBayesianNetwork(list(learned_edges))
        learned_bn.add_nodes_from(STATES)          # SHD needs identical node sets
        shd = int(SHD().evaluate(_skeleton(), learned_bn))
    except Exception as exc:
        print(f"    (SHD unavailable: {exc})")
        shd = None

    return {
        "expert_edges": len(expert_edges),
        "learned_edges": len(learned_edges),
        "recovered_edges_ignoring_direction": recovered,
        "recovery_rate": round(recovered / len(expert_edges), 3),
        "extra_edges": len(undirected - {frozenset(e) for e in expert_edges}),
        "shd": shd,
    }


# ----------------------------------------------------------------------------
def main():
    train, test, _, _ = load_discrete()
    print(f"train = {len(train)} rows, test = {len(test)} rows\n")

    bayes = fit_bayesian(train)
    save_model(bayes)
    print(f"saved expert-prior Bayesian CPTs -> {LEARNED_MODEL}")

    # keep the other two around so the evaluation script can compare them
    save_model(fit_mle(train), ARTIFACT_DIR / "mle_cpts.json")
    save_model(fit_bdeu(train), ARTIFACT_DIR / "bdeu_cpts.json")

    shift = cpt_shift(bayes)
    shift.to_csv(ARTIFACT_DIR / "cpt_shift.csv", index=False)
    print("\nHow much the data moved each CPT away from our elicited prior:")
    print(shift.to_string(index=False))

    print("\nStructure-learning cross-check (Hill-Climb + BIC):")
    info = structure_check(train)
    for k, v in info.items():
        print(f"  {k:>34}: {v}")
    (ARTIFACT_DIR / "structure_check.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
