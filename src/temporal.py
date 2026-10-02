"""
Recursive Bayesian filtering over the hydrology - the part that makes this a
real-time system rather than a system that happens to be called repeatedly.

Why a filter at all
-------------------
A flood is not caused by today's rain. It is caused by three days of rain landing
on ground that was already wet. Our static network cannot express that: give it
one snapshot and it has to pretend the catchment has no memory. We listed this as
a limitation at the last review, and processing a live stream is exactly the
situation that makes it unacceptable - the whole point of a stream is that
yesterday's frame informs today's.

So the two nodes that have memory, SoilMoisture and RiverLevel, are lifted into a
two-slice dynamic Bayesian network and tracked with the standard forward
recursion:

    predict   b'(s_t)  =  sum over s_{t-1} of  T(s_t | s_{t-1}, rain_t) b(s_{t-1})
    update    b(s_t)   =  normalise[ L(z_t | s_t) * b'(s_t) ]

`s` is the joint (SoilMoisture, RiverLevel), so nine states. `T` is built from the
same ordered-logit family we used for the static CPTs, which means the transition
is monotone by construction: more rain never dries the soil, and the river recedes
when the rain stops. `L` is the soft-evidence likelihood from measurement.py for
whatever direct hydrology readings arrived (GloFAS discharge, reanalysis soil
moisture) - and it is flat when nothing arrived, in which case the filter simply
coasts on the prediction, which is the correct behaviour for a dead gauge.

Rainfall is itself uncertain, so we do not condition on a single rainfall bin. We
marginalise the transition over the rainfall posterior:

    T_eff(s | s')  =  sum over r of  P(Rainfall = r | evidence_t) T(s | s', r)

Handing the result back to the static network
---------------------------------------------
The static network has its own opinion about the hydrology, formed from today's
rain through the Rainfall -> SoilMoisture -> RiverLevel chain. If we simply
multiplied the filtered belief into it we would count today's rainfall twice. So
we enter the filter's belief as virtual evidence with the network's own prediction
divided out:

    lambda(s) = b_filter(s) / P_network(s | evidence_t)

which makes the network's posterior over the hydrology exactly equal to the
filter's belief - replacing its memoryless guess rather than compounding with it.
`test_virtual_evidence_reproduces_filter_belief` asserts precisely that.

Owner: Sai Reddy A (recursion + virtual evidence) with Rohit Vardhan M (transition
elicitation)
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from .config import ARTIFACT_DIR, STATES
from .measurement import (HYDRO_NODES, FusedReading, SoftEvidenceEngine,
                          observation_node)

HYDRO_LEAF = "Hydrology__filter"
STATE_FILE = ARTIFACT_DIR / "filter_state.json"

SOIL = STATES["SoilMoisture"]
RIVER = STATES["RiverLevel"]
JOINT: List[Tuple[str, str]] = [(s, r) for s in SOIL for r in RIVER]
JOINT_INDEX: Dict[Tuple[str, str], int] = {p: i for i, p in enumerate(JOINT)}

# ---------------------------------------------------------------------------
# transition weights. Same ordered-logit family as networks.py: a score built
# from parent severities, then cut-points.
#
# soil:  persistence 3.2 (yesterday's soil is the strongest predictor - this IS
#        the memory), rainfall 3.0. Check: wet soil + no rain drains to
#        Low 0.17 / Medium 0.52 / High 0.31 in a day, dry soil + heavy rain
#        only reaches High 0.27 - it takes more than one day to saturate, which
#        is the accumulation effect we were missing.
# river: persistence 2.4 (recession is slower than the rain that caused it),
#        rainfall 2.6, soil 2.4 - a saturated catchment routes rain to the
#        channel instead of absorbing it.
SOIL_WEIGHTS = {"prev": 3.2, "rain": 3.0}
SOIL_CUTS = [1.6, 4.0]
RIVER_WEIGHTS = {"prev": 2.4, "rain": 2.6, "soil": 2.4}
RIVER_CUTS = [1.8, 4.4]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _severity(states: Sequence[str], state: str) -> float:
    idx = list(states).index(state)
    return idx / (len(states) - 1) if len(states) > 1 else 0.0


def _ordered(score: float, cuts: Sequence[float]) -> List[float]:
    cum = [_sigmoid(c - score) for c in cuts] + [1.0]
    out, prev = [], 0.0
    for c in cum:
        out.append(max(c - prev, 1e-12))
        prev = c
    return out


def soil_transition(prev_soil: str, rain: str, offset: float = 0.0) -> List[float]:
    score = (SOIL_WEIGHTS["prev"] * _severity(SOIL, prev_soil)
             + SOIL_WEIGHTS["rain"] * _severity(STATES["Rainfall"], rain))
    return _ordered(score, [c + offset for c in SOIL_CUTS])


def river_transition(prev_river: str, rain: str, soil: str,
                     offset: float = 0.0) -> List[float]:
    score = (RIVER_WEIGHTS["prev"] * _severity(RIVER, prev_river)
             + RIVER_WEIGHTS["rain"] * _severity(STATES["Rainfall"], rain)
             + RIVER_WEIGHTS["soil"] * _severity(SOIL, soil))
    return _ordered(score, [c + offset for c in RIVER_CUTS])


def transition_matrix(rain: str,
                      offsets: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """T[i, j] = P(joint state j at t | joint state i at t-1, rainfall = rain)."""
    soil_off, river_off = calibration_offsets() if offsets is None else offsets
    T = np.zeros((len(JOINT), len(JOINT)))
    for i, (ps, pr) in enumerate(JOINT):
        soil_next = soil_transition(ps, rain, soil_off)
        for si, s in enumerate(SOIL):
            river_next = river_transition(pr, rain, s, river_off)
            for ri, r in enumerate(RIVER):
                T[i, JOINT_INDEX[(s, r)]] = soil_next[si] * river_next[ri]
    return T


def expected_transition(rain_posterior: Dict[str, float],
                        offsets: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Transition marginalised over an uncertain rainfall band."""
    T = np.zeros((len(JOINT), len(JOINT)))
    total = sum(rain_posterior.values()) or 1.0
    for rain, p in rain_posterior.items():
        if p <= 1e-9:
            continue
        T += (p / total) * transition_matrix(rain, offsets)
    return T


# ---------------------------------------------------------------------------
# keeping the filter and the static network telling the same story
# ---------------------------------------------------------------------------
def stationary(T: np.ndarray, iters: int = 400) -> np.ndarray:
    """Long-run distribution of the chain, by power iteration."""
    b = np.full(T.shape[0], 1.0 / T.shape[0])
    for _ in range(iters):
        nxt = b @ T
        nxt = nxt / nxt.sum()
        if np.abs(nxt - b).max() < 1e-14:
            b = nxt
            break
        b = nxt
    return b


def static_hydrology_marginal() -> np.ndarray:
    """The static network's own belief about (SoilMoisture, RiverLevel) with no
    evidence - our reference for what 'a typical day' looks like."""
    from .engine import RiskEngine
    from .learn import load_model
    eng = RiskEngine(load_model())
    j = eng.joint(["SoilMoisture", "RiverLevel"], {})
    b = np.array([float(j.get_value(SoilMoisture=s, RiverLevel=r))
                  for s, r in JOINT])
    return b / b.sum()


def climatological_rain() -> Dict[str, float]:
    from .engine import RiskEngine
    from .learn import load_model
    return RiskEngine(load_model()).posterior(["Rainfall"], {})["Rainfall"]


def _stationary_kl(offsets: Tuple[float, float], rain: Dict[str, float],
                   target: np.ndarray) -> float:
    pi = stationary(expected_transition(rain, offsets=offsets))
    return float(np.sum(pi * (np.log(pi + 1e-15) - np.log(target + 1e-15))))


def fit_calibration_offsets(passes: int = 4) -> Dict[str, float]:
    """
    The transition model and the static CPTs are two separate elicitations of the
    same physical system, and left alone they disagree: run the chain forward under
    average rainfall and it settles somewhere wetter than the static network's
    marginal. A district with no readings at all then drifts to a flood probability
    of 0.31 against a base rate of 0.16, purely as an artefact of that mismatch -
    which is exactly what we saw the first time we ran the fleet on a snapshot with
    districts missing.

    So we fit TWO scalars: a shift of the soil cut-points and a shift of the river
    cut-points, chosen so the chain's stationary distribution under climatological
    rainfall matches the static marginal. (One shared scalar is not enough - it
    lines the river up and leaves the soil 0.26 out.) Two degrees of freedom,
    fitted against a reference the rest of the project already trusts, and asserted
    by a test. The dynamics - persistence, accumulation, recession - are untouched;
    only the resting point moves.

    Coordinate descent on a shrinking grid, because each evaluation needs a
    stationary solve and a 2-D sweep at this resolution would be wasteful.
    """
    target = static_hydrology_marginal()
    rain = climatological_rain()

    soil_off, river_off = 0.0, 0.0
    span, step = 3.0, 0.25
    for _ in range(passes):
        for axis in (0, 1):
            centre = soil_off if axis == 0 else river_off
            best_val, best_kl = centre, float("inf")
            grid = np.arange(centre - span, centre + span + 1e-9, step)
            for cand in grid:
                trial = ((float(cand), river_off) if axis == 0
                         else (soil_off, float(cand)))
                kl = _stationary_kl(trial, rain, target)
                if kl < best_kl:
                    best_kl, best_val = kl, float(cand)
            if axis == 0:
                soil_off = best_val
            else:
                river_off = best_val
        span, step = span / 3.0, step / 3.0

    return {"soil_offset": round(soil_off, 4),
            "river_offset": round(river_off, 4),
            "kl": _stationary_kl((soil_off, river_off), rain, target)}


_CAL_FILE = ARTIFACT_DIR / "transition_calibration.json"
_OFFSETS: Optional[Tuple[float, float]] = None


def calibration_offsets(refit: bool = False) -> Tuple[float, float]:
    """Cached (soil, river) cut-point shifts from fit_calibration_offsets."""
    global _OFFSETS
    if _OFFSETS is not None and not refit:
        return _OFFSETS
    if _CAL_FILE.exists() and not refit:
        try:
            blob = json.loads(_CAL_FILE.read_text(encoding="utf-8"))
            _OFFSETS = (float(blob["soil_offset"]), float(blob["river_offset"]))
            return _OFFSETS
        except (json.JSONDecodeError, KeyError, OSError, ValueError):
            pass

    _OFFSETS = (0.0, 0.0)          # so the fit itself does not recurse
    best = fit_calibration_offsets()
    _CAL_FILE.write_text(json.dumps({
        "soil_offset": best["soil_offset"],
        "river_offset": best["river_offset"],
        "stationary_kl_to_static_marginal": round(best["kl"], 6),
        "method": "shifts of the soil and river ordered-logit cut-points, chosen so "
                  "the chain's stationary distribution under climatological rainfall "
                  "matches the static network's hydrology marginal",
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=1), encoding="utf-8")
    _OFFSETS = (best["soil_offset"], best["river_offset"])
    return _OFFSETS


# ---------------------------------------------------------------------------
@dataclass
class FilterStep:
    """One frame of the recursion, kept for the dashboard and for the tests."""
    timestamp: str
    rain_posterior: Dict[str, float]
    predicted: List[float]
    likelihood: List[float]
    belief: List[float]
    used_readings: List[str] = field(default_factory=list)
    coasted: bool = False

    def marginals(self) -> Dict[str, Dict[str, float]]:
        b = np.asarray(self.belief).reshape(len(SOIL), len(RIVER))
        return {
            "SoilMoisture": {s: float(v) for s, v in zip(SOIL, b.sum(axis=1))},
            "RiverLevel": {r: float(v) for r, v in zip(RIVER, b.sum(axis=0))},
        }

    def to_dict(self) -> dict:
        m = self.marginals()
        return {
            "timestamp": self.timestamp,
            "coasted": self.coasted,
            "used_readings": self.used_readings,
            "rain_posterior": {k: round(v, 4) for k, v in self.rain_posterior.items()},
            "soil": {k: round(v, 4) for k, v in m["SoilMoisture"].items()},
            "river": {k: round(v, 4) for k, v in m["RiverLevel"].items()},
        }


class HydrologyFilter:
    """
    Forward filter over the joint (SoilMoisture, RiverLevel) for ONE district.

    Belief is persisted, so restarting the service does not throw away the
    catchment's history - which for a three-day accumulation matters.
    """

    def __init__(self, district: str, belief: Optional[Sequence[float]] = None):
        self.district = district
        if belief is None:
            belief = self._climatological_prior()
        self.belief = np.asarray(belief, dtype=float)
        self.belief /= self.belief.sum()
        self.history: List[FilterStep] = []

    @staticmethod
    def _climatological_prior() -> np.ndarray:
        """
        Where to start on a cold boot: the static network's own joint marginal for
        the hydrology with no evidence at all. Better than a uniform guess, and it
        means the first frame of a stream is not nonsense. Because the transition
        is calibrated to have this as its stationary distribution, a district that
        never reports anything stays here instead of drifting.
        """
        return static_hydrology_marginal()

    # ------------------------------------------------------------------ step
    def step(self, rain_posterior: Dict[str, float],
             hydro_readings: Optional[Dict[str, FusedReading]] = None,
             timestamp: Optional[str] = None) -> FilterStep:
        hydro_readings = hydro_readings or {}

        # ---- predict
        T = expected_transition(rain_posterior)
        predicted = self.belief @ T
        predicted = predicted / predicted.sum()

        # ---- update with whatever hydrology we actually measured
        lam = np.ones(len(JOINT))
        used = []
        for node in HYDRO_NODES:
            reading = hydro_readings.get(node)
            if reading is None:
                continue
            used.append(node)
            per_state = dict(zip(STATES[node], reading.likelihood))
            axis = 0 if node == "SoilMoisture" else 1
            for k, pair in enumerate(JOINT):
                lam[k] *= per_state[pair[axis]]

        posterior = predicted * lam
        if posterior.sum() <= 0:
            posterior = predicted.copy()
        posterior /= posterior.sum()

        self.belief = posterior
        step = FilterStep(
            timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
            rain_posterior=dict(rain_posterior),
            predicted=predicted.tolist(),
            likelihood=(lam / lam.sum()).tolist(),
            belief=posterior.tolist(),
            used_readings=used,
            coasted=not used,
        )
        self.history.append(step)
        self.history = self.history[-96:]          # four days of hourly frames
        return step

    # --------------------------------------------------------------- helpers
    def marginals(self) -> Dict[str, Dict[str, float]]:
        b = self.belief.reshape(len(SOIL), len(RIVER))
        return {
            "SoilMoisture": {s: float(v) for s, v in zip(SOIL, b.sum(axis=1))},
            "RiverLevel": {r: float(v) for r, v in zip(RIVER, b.sum(axis=0))},
        }

    def to_dict(self) -> dict:
        return {"district": self.district, "belief": self.belief.tolist(),
                "history": [h.to_dict() for h in self.history[-48:]]}


# ---------------------------------------------------------------------------
# handing the filtered belief to the static network
# ---------------------------------------------------------------------------
def _hydro_leaf_cpd(likelihood: Sequence[float]) -> TabularCPD:
    """
    P(Hydrology__filter = yes | SoilMoisture, RiverLevel) proportional to lambda.

    The parent order here must match pgmpy's column layout, which is
    itertools.product(SoilMoisture, RiverLevel) - the same order as JOINT.
    """
    lam = np.asarray(likelihood, dtype=float)
    lam = np.clip(lam, 1e-9, None)
    lam = lam / lam.max() * 0.98
    values = np.vstack([1.0 - lam, lam])
    return TabularCPD(
        variable=HYDRO_LEAF, variable_card=2, values=values.tolist(),
        evidence=["SoilMoisture", "RiverLevel"],
        evidence_card=[len(SOIL), len(RIVER)],
        state_names={HYDRO_LEAF: ["no", "yes"],
                     "SoilMoisture": SOIL, "RiverLevel": RIVER},
    )


def attach_hydrology_leaf(engine: SoftEvidenceEngine) -> None:
    """Add the joint indicator leaf, uninformative to begin with."""
    if HYDRO_LEAF in engine.model.nodes():
        return
    engine.model.add_edge("SoilMoisture", HYDRO_LEAF)
    engine.model.add_edge("RiverLevel", HYDRO_LEAF)
    engine.model.add_cpds(_hydro_leaf_cpd([1.0] * len(JOINT)))
    engine.model.check_model()
    engine._states[HYDRO_LEAF] = ["no", "yes"]
    engine._reset_inference()


def install_filtered_belief(engine: SoftEvidenceEngine, belief: Sequence[float],
                            evidence: Dict[str, str]) -> Dict[str, str]:
    """
    Enter the filter's belief as virtual evidence, dividing out the network's own
    prediction so today's rainfall is not counted twice.

    Returns the evidence dict extended with the leaf.
    """
    attach_hydrology_leaf(engine)

    # the network's belief about the hydrology from everything EXCEPT the filter
    without = {k: v for k, v in evidence.items() if k != HYDRO_LEAF}
    joint = engine.joint(["SoilMoisture", "RiverLevel"], without)
    prior = np.array([float(joint.get_value(SoilMoisture=s, RiverLevel=r))
                      for s, r in JOINT])
    prior = np.clip(prior, 1e-9, None)

    target = np.asarray(belief, dtype=float)
    target = target / target.sum()

    lam = target / prior                     # the correction, not the belief
    engine.model.add_cpds(_hydro_leaf_cpd(lam))
    engine.model.check_model()
    engine._reset_inference()

    # tell the engine how sharp the tracked hydrology is, so the confidence label
    # can credit the filter for it. A coasting filter is vague and should not be
    # allowed to look like two working gauges.
    card = len(target)
    h = -float(np.sum(target * np.log2(np.clip(target, 1e-12, None))))
    engine.hydro_sharpness = max(0.0, 1.0 - h / math.log2(card))

    out = dict(evidence)
    out[HYDRO_LEAF] = "yes"
    return out


# ---------------------------------------------------------------------------
# persistence for a fleet of districts
# ---------------------------------------------------------------------------
class FilterBank:
    """One filter per district, saved to disk between runs."""

    def __init__(self, path: Path = STATE_FILE):
        self.path = path
        self.filters: Dict[str, HydrologyFilter] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            blob = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for name, entry in blob.get("filters", {}).items():
            self.filters[name] = HydrologyFilter(name, entry.get("belief"))

    def get(self, district: str) -> HydrologyFilter:
        if district not in self.filters:
            self.filters[district] = HydrologyFilter(district)
        return self.filters[district]

    def save(self) -> Path:
        self.path.write_text(json.dumps({
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "filters": {k: v.to_dict() for k, v in self.filters.items()},
        }, indent=1), encoding="utf-8")
        return self.path

    def reset(self) -> None:
        self.filters.clear()
        if self.path.exists():
            self.path.unlink()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # three days of heavy rain, then it stops - watch the soil saturate and the
    # river rise and then recede. This is the accumulation the static model missed.
    f = HydrologyFilter("demo")
    print("start                     ", {k: round(v, 3)
                                          for k, v in f.marginals()["RiverLevel"].items()})
    heavy = {"Low": 0.02, "Moderate": 0.10, "Heavy": 0.88}
    dry = {"Low": 0.90, "Moderate": 0.08, "Heavy": 0.02}
    for day, rain in enumerate([heavy] * 3 + [dry] * 4, start=1):
        f.step(rain)
        m = f.marginals()
        label = "heavy rain" if rain is heavy else "dry"
        print(f"day {day} ({label:<10})  soil High={m['SoilMoisture']['High']:.3f}"
              f"   river High={m['RiverLevel']['High']:.3f}")
