"""
The inference layer: one class that everything else (API, CLI, tests) talks to.

All the probability numbers the system ever shows come from exact inference
(Variable Elimination) over the discrete Bayesian network. Nothing here is a
heuristic score.

What the engine gives you for a single district reading:
  * P(hazard = Yes) for all four hazards, at once, from one shared piece of
    evidence - this is the "multi-hazard" claim in our synopsis.
  * P(consequences): road blocked, rescue needed, hospital demand.
  * the most probable explanation (MAP) over the latent nodes we cannot measure
    (soil moisture, river level when the gauge is down).
  * the posterior entropy, which we turn into a confidence label. A 60 % flood
    probability computed from two sensors is not the same claim as a 60 %
    computed from nine, and the dashboard should not pretend otherwise.

Owner: Sai Reddy A
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from pgmpy.inference import BeliefPropagation, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from .config import CONSEQUENCES, HAZARDS, OBSERVABLE, STATES
from .learn import load_model

LATENT = ["SoilMoisture", "RiverLevel"]


class InvalidEvidence(ValueError):
    pass


def validate_evidence(evidence: Dict[str, str],
                      allowed: Optional[Dict[str, List[str]]] = None
                      ) -> Dict[str, str]:
    """
    Reject anything that is not a real node / real state before it reaches pgmpy,
    because pgmpy's own error for a bad state name is cryptic.

    `allowed` defaults to the core state table. The soft-evidence engine passes
    its own, because its network has extra indicator leaves that config.STATES
    quite rightly knows nothing about.
    """
    allowed = allowed if allowed is not None else STATES
    clean = {}
    for node, state in evidence.items():
        if node not in allowed:
            raise InvalidEvidence(f"'{node}' is not a node in the network")
        if state not in allowed[node]:
            raise InvalidEvidence(
                f"'{state}' is not a state of {node}; allowed: {allowed[node]}")
        clean[node] = state
    return clean


def entropy(dist: Dict[str, float]) -> float:
    return -sum(p * math.log(p + 1e-12, 2) for p in dist.values())


class RiskEngine:
    """
    Thin wrapper around pgmpy inference. Deliberately the ONLY place in the
    project that imports pgmpy.inference, so swapping exact inference for an
    approximate sampler later is a one-file change.
    """

    def __init__(self, model: Optional[DiscreteBayesianNetwork] = None,
                 method: str = "ve"):
        self.model = model or load_model()
        self.method = method
        self._infer = (VariableElimination(self.model) if method == "ve"
                       else BeliefPropagation(self.model))
        if method == "bp":
            self._infer.calibrate()
        self._cache: Dict[tuple, Dict[str, Dict[str, float]]] = {}
        # state names taken from the model itself, so a subclass that adds nodes
        # does not have to touch config.STATES
        self._states: Dict[str, List[str]] = {
            n: list(self.model.get_cpds(n).state_names[n]) for n in self.model.nodes()
        }

    def _validate(self, evidence: Dict[str, str]) -> Dict[str, str]:
        return validate_evidence(evidence, self._states)

    # ------------------------------------------------------------------ core
    def posterior(self, targets: Iterable[str],
                  evidence: Optional[Dict[str, str]] = None
                  ) -> Dict[str, Dict[str, float]]:
        """P(target | evidence) for each target, as {target: {state: prob}}."""
        evidence = self._validate(dict(evidence or {}))
        targets = [t for t in targets if t not in evidence]

        key = (tuple(sorted(targets)), tuple(sorted(evidence.items())))
        if key in self._cache:
            return dict(self._cache[key])

        out: Dict[str, Dict[str, float]] = {}
        if targets:
            # one query per target keeps the factors small; the whole network is
            # tiny so this is still milliseconds
            for t in targets:
                factor = self._infer.query([t], evidence=evidence or None,
                                           show_progress=False)
                states = factor.state_names[t]
                out[t] = {s: float(p) for s, p in zip(states, factor.values)}

        # anything already observed has a degenerate posterior
        for node, state in evidence.items():
            out[node] = {s: (1.0 if s == state else 0.0)
                         for s in self._states[node]}

        self._cache[key] = out
        return dict(out)

    def probability(self, target: str, state: str,
                    evidence: Optional[Dict[str, str]] = None) -> float:
        return self.posterior([target], evidence)[target][state]

    # -------------------------------------------------------------- the bundle
    def hazard_probabilities(self, evidence: Dict[str, str]) -> Dict[str, float]:
        post = self.posterior(HAZARDS, evidence)
        return {h: post[h]["Yes"] for h in HAZARDS}

    def consequence_probabilities(self, evidence: Dict[str, str]) -> Dict[str, dict]:
        return self.posterior(CONSEQUENCES, evidence)

    def latent_map(self, evidence: Dict[str, str]) -> Dict[str, str]:
        """
        Most probable state of the physical quantities we did not observe.
        We use pgmpy's MAP query so this is a joint assignment, not four
        independent arg-maxes - it is the "most probable explanation" of the
        evidence, which is what we narrate in the explanation text.
        """
        unknown = [v for v in LATENT if v not in evidence]
        if not unknown:
            return {}
        result = self._infer.map_query(unknown, evidence=evidence or None,
                                       show_progress=False)
        return {k: str(v) for k, v in result.items()}

    def confidence(self, evidence: Dict[str, str], hazard: str) -> Dict[str, object]:
        """
        Two things decide how much we trust a number:
          - how peaked the posterior is (low entropy = the model is committed)
          - how much of the sensor set we actually received
        We report both, plus a word for the dashboard.
        """
        post = self.posterior([hazard], evidence)[hazard]
        h = entropy(post)                       # 0 (certain) .. 1 bit (coin flip)
        observed = sum(1 for v in OBSERVABLE if v in evidence)
        coverage = observed / len(OBSERVABLE)

        score = (1 - h) * 0.5 + coverage * 0.5
        label = "High" if score >= 0.66 else "Medium" if score >= 0.40 else "Low"
        return {
            "label": label,
            "score": round(score, 3),
            "posterior_entropy_bits": round(h, 3),
            "sensor_coverage": round(coverage, 3),
            "sensors_used": observed,
            "sensors_possible": len(OBSERVABLE),
        }

    def assess(self, evidence: Dict[str, str]) -> dict:
        """Everything the dashboard needs for one district, in one call."""
        evidence = self._validate(dict(evidence))
        hazards = self.hazard_probabilities(evidence)
        return {
            "evidence": evidence,
            "hazards": {h: round(p, 4) for h, p in hazards.items()},
            "consequences": {
                k: {s: round(p, 4) for s, p in v.items()}
                for k, v in self.consequence_probabilities(evidence).items()
            },
            "latent_map": self.latent_map(evidence),
            "confidence": {h: self.confidence(evidence, h) for h in HAZARDS},
        }

    # ------------------------------------------------- course-concept helpers
    def is_dependent(self, a: str, b: str,
                     observed: Optional[List[str]] = None) -> bool:
        """d-separation query. Used in tests and on the 'why a DAG' slide."""
        return bool(self.model.is_dconnected(a, b, observed=observed or []))

    def joint(self, variables: List[str],
              evidence: Optional[Dict[str, str]] = None):
        """Full joint factor over `variables` - handy for the CPT slides."""
        return self._infer.query(variables, evidence=evidence or None,
                                 show_progress=False)


# a module-level singleton, because building the junction tree twice per request
# was the first thing that made the API feel slow
_ENGINE: Optional[RiskEngine] = None


def get_engine() -> RiskEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RiskEngine()
    return _ENGINE


if __name__ == "__main__":
    eng = RiskEngine()
    demo = {"Season": "Monsoon", "Rainfall": "Heavy", "Humidity": "High",
            "Slope": "Steep", "Urbanisation": "Low"}
    import json
    print(json.dumps(eng.assess(demo), indent=2))
