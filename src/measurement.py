"""
The measurement layer: how a live number becomes evidence.

This is the second half of the answer to the zeroth-review feedback - "what
methods are you incorporating to process the real-time data feed?"

The naive pipeline is: take the number, look up which bin it falls in, enter that
bin as hard evidence. We do not do that, for three reasons that show up in the
real feed every single pass:

  * ECMWF said 15.0 mm and the IMD light/moderate boundary is 15.6 mm. Hard
    binning turns a 0.6 mm difference into a categorical claim.
  * GFS said 7.4 mm, ECMWF 15.0 mm and ICON 1.5 mm for the same district and
    hour. Hard binning has to pick one and throw the disagreement away.
  * The GloFAS discharge value was 17 hours old. Hard evidence treats a 17-hour-old
    reading exactly like one from 30 seconds ago.

So instead of hard evidence we compute VIRTUAL (soft) EVIDENCE: a likelihood
vector over the node's states, entered into the network through Pearl's indicator
construction. Three ideas do the work.

1. GAUSSIAN BIN LIKELIHOOD.
   A reading v with observation error sigma does not identify a bin, it induces a
   distribution over bins:

       lambda(state k) = Phi((e_k - v)/sigma) - Phi((e_{k-1} - v)/sigma)

   where e are the discretisation edges from config.BINS. A value sitting on an
   edge splits its mass across both neighbours, which is what it should do.

2. SIGMA COMES FROM THE ENSEMBLE, NOT FROM US.
   For every variable carried by all three forecast models we take the ensemble
   mean as the estimate and

       sigma_eff = sqrt(sigma_instrument^2 + s^2),   s = spread of the members

   which is Gaussian ensemble dressing, the standard post-processing move in
   numerical weather prediction. When GFS, ECMWF and ICON agree, sigma is small
   and the evidence is sharp. When they disagree, sigma grows, the likelihood
   flattens, the posterior widens and the confidence label drops - automatically,
   with no threshold anywhere. This is the part we are most pleased with: the
   reliability of the feed is measured from the feed itself, at that moment.

3. AGEING AND SUSPICION SHRINK EVIDENCE TOWARDS USELESS.
   A reading's likelihood is blended towards uniform with weight
   w = exp(-age / tau); at w = 0 the observation node is uninformative and
   entering it changes nothing. The Bayesian-surprise gate in stream.py shrinks w
   further for readings the model finds implausible.

Entering it into the network
----------------------------
For each node X that has an observation we attach a binary leaf X__obs with
P(X__obs = yes | X = s) = lambda(s), and we observe X__obs = yes. By the chain
rule this multiplies X's factor by lambda(s) - exactly virtual evidence, with no
change to the core network and no library support required. Setting lambda to a
one-hot vector recovers ordinary hard evidence, which is how we test it.

Owner: Sai Reddy A (virtual evidence + engine) and Jithin Reddy K (bin likelihood)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from .config import BINS, OBSERVABLE, STATES
from .discretize import COLUMN_TO_NODE
from .engine import RiskEngine, entropy
from .feeds import Observation
from .learn import load_model

OBS_SUFFIX = "__obs"

# Instrument / product error for one reading of each quantity, in the quantity's
# own units. These are elicited and we say so; the ensemble spread is what
# actually dominates sigma for the forecast variables.
#
#   rainfall     a 24-h accumulation from a model grid cell against a point gauge
#                is routinely out by a few mm
#   temperature  grid-cell vs screen-height station, ~1 C
#   humidity     ~5 % RH
#   wind         gusty and terrain-sensitive, ~4 kmph
#   pressure     the drop is a difference of two good measurements, ~0.5 hPa
#   river        GloFAS is a global reanalysis on a coarse reach; expressed as a
#                fraction of bankfull we allow 12 %
#   sst          satellite/analysis product, ~0.4 C
#   soil         a reanalysis estimate divided by an assumed porosity - by far our
#                least trustworthy input, so 0.12 on a 0-1 fraction
SIGMA_INSTRUMENT = {
    "rainfall_mm": 3.0,
    "temperature_c": 1.0,
    "humidity_pct": 5.0,
    "wind_kmph": 4.0,
    "pressure_drop_hpa": 0.5,
    "river_level_frac": 0.12,
    "sst_c": 0.4,
    "soil_moisture_frac": 0.12,
}

# How fast a reading stops being worth trusting, in minutes. A 24-hour rainfall
# accumulation is still meaningful hours later; a pressure drop is not.
STALENESS_TAU = {
    "Rainfall": 18 * 60,
    "Temperature": 6 * 60,
    "Humidity": 6 * 60,
    "WindSpeed": 3 * 60,
    "PressureDrop": 3 * 60,
    "RiverLevel": 30 * 60,
    "SeaSurfaceTemp": 48 * 60,
    "SoilMoisture": 12 * 60,
}
DEFAULT_TAU = 6 * 60


# ---------------------------------------------------------------------------
def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bin_likelihood(column: str, value: float, sigma: float) -> List[float]:
    """
    P(reading | node = each state), as a vector over that node's states.

    A Gaussian of width sigma centred on the reading, integrated over each bin.
    """
    edges, labels = BINS[column]
    sigma = max(float(sigma), 1e-6)
    cuts = [-math.inf, *edges, math.inf]
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        p_hi = 1.0 if hi == math.inf else _phi((hi - value) / sigma)
        p_lo = 0.0 if lo == -math.inf else _phi((lo - value) / sigma)
        out.append(max(p_hi - p_lo, 1e-12))
    assert len(out) == len(labels)
    return out


def soften(lam: Sequence[float], weight: float) -> List[float]:
    """
    Blend a likelihood towards uniform. weight = 1 keeps it, weight = 0 makes the
    observation carry no information at all.
    """
    lam = np.asarray(lam, dtype=float)
    lam = lam / lam.sum()
    uniform = np.full_like(lam, 1.0 / len(lam))
    w = min(max(float(weight), 0.0), 1.0)
    return (w * lam + (1 - w) * uniform).tolist()


def staleness_weight(node: str, age_minutes: float) -> float:
    tau = STALENESS_TAU.get(node, DEFAULT_TAU)
    return math.exp(-max(age_minutes, 0.0) / tau)


# ---------------------------------------------------------------------------
@dataclass
class FusedReading:
    """What the measurement layer produces for one node."""
    node: str
    column: str
    estimate: float
    likelihood: List[float]              # after staleness / suspicion softening
    raw_likelihood: List[float]          # before softening
    sigma_instrument: float
    ensemble_spread: float
    sigma_effective: float
    sources: List[str]
    n_sources: int
    age_minutes: float
    staleness_weight: float
    trust_weight: float                  # staleness x suspicion
    suspicion_weight: float = 1.0
    surprisal_bits: Optional[float] = None
    status: str = "ok"                   # ok | stale | suspect | disputed
    note: str = ""

    @property
    def sharpness(self) -> float:
        """1 = the evidence pins one state, 0 = it says nothing."""
        card = len(self.likelihood)
        h = entropy({str(i): p for i, p in enumerate(self.likelihood)})
        return max(0.0, 1.0 - h / math.log2(card)) if card > 1 else 1.0

    def most_likely_state(self) -> str:
        return STATES[self.node][int(np.argmax(self.likelihood))]

    def to_dict(self) -> dict:
        states = STATES[self.node]
        return {
            "node": self.node,
            "estimate": round(self.estimate, 3),
            "unit_column": self.column,
            "sources": self.sources,
            "n_sources": self.n_sources,
            "ensemble_spread": round(self.ensemble_spread, 3),
            "sigma_effective": round(self.sigma_effective, 3),
            "age_minutes": round(self.age_minutes, 1),
            "staleness_weight": round(self.staleness_weight, 3),
            "suspicion_weight": round(self.suspicion_weight, 3),
            "trust_weight": round(self.trust_weight, 3),
            "surprisal_bits": (None if self.surprisal_bits is None
                               else round(self.surprisal_bits, 2)),
            "status": self.status,
            "most_likely_state": self.most_likely_state(),
            "sharpness": round(self.sharpness, 3),
            "likelihood": {s: round(p, 4) for s, p in zip(states, self.likelihood)},
            "note": self.note,
        }


def fuse_observations(observations: Iterable[Observation],
                      now_age: Optional[Dict[str, float]] = None
                      ) -> Dict[str, FusedReading]:
    """
    Group the pass's observations by node and turn each group into one likelihood.

    Multiple sources for the same node are treated as an ensemble: the mean is the
    estimate and the spread inflates sigma (Gaussian ensemble dressing).
    """
    grouped: Dict[str, List[Observation]] = {}
    for o in observations:
        grouped.setdefault(o.node, []).append(o)

    fused: Dict[str, FusedReading] = {}
    for node, obs in grouped.items():
        column = obs[0].column
        if column not in BINS:
            continue
        values = [float(o.value) for o in obs]
        estimate = float(statistics.fmean(values))
        spread = float(statistics.stdev(values)) if len(values) > 1 else 0.0

        sigma_i = SIGMA_INSTRUMENT.get(column, 1.0)
        sigma_eff = math.sqrt(sigma_i ** 2 + spread ** 2)

        raw = bin_likelihood(column, estimate, sigma_eff)
        age = (now_age or {}).get(node)
        if age is None:
            age = float(statistics.fmean([o.age_minutes() for o in obs]))
        w = staleness_weight(node, age)

        status = "ok"
        if w < 0.55:
            status = "stale"
        if spread > 2.5 * sigma_i:
            status = "disputed"

        fused[node] = FusedReading(
            node=node, column=column, estimate=estimate,
            likelihood=soften(raw, w), raw_likelihood=soften(raw, 1.0),
            sigma_instrument=sigma_i, ensemble_spread=spread,
            sigma_effective=sigma_eff,
            sources=[o.source for o in obs], n_sources=len(obs),
            age_minutes=age, staleness_weight=w, trust_weight=w, status=status,
            note=obs[0].note,
        )
    return fused


# ---------------------------------------------------------------------------
# the augmented network
# ---------------------------------------------------------------------------
def observation_node(node: str) -> str:
    return f"{node}{OBS_SUFFIX}"


def base_node(obs_node: str) -> str:
    return obs_node[:-len(OBS_SUFFIX)] if obs_node.endswith(OBS_SUFFIX) else obs_node


SOFT_NODES = [n for n in COLUMN_TO_NODE.values() if n in STATES]

# The two nodes with memory. When the temporal filter in temporal.py is running it
# consumes their readings itself and supplies a filtered joint belief instead, so
# their per-node indicator leaves are left uninformative to avoid counting the same
# reading twice.
HYDRO_NODES = ["SoilMoisture", "RiverLevel"]


def build_augmented_network(core: Optional[DiscreteBayesianNetwork] = None
                            ) -> DiscreteBayesianNetwork:
    """
    Core network plus one binary indicator leaf per soft-evidence node.

    The leaves start uninformative (likelihood flat), so before any reading
    arrives the augmented network is distributionally identical to the core one -
    which is asserted in the tests.
    """
    core = core or load_model()
    model = DiscreteBayesianNetwork(list(core.edges()))
    model.add_nodes_from(core.nodes())
    model.add_cpds(*[cpd.copy() for cpd in core.get_cpds()])

    for node in SOFT_NODES:
        obs = observation_node(node)
        model.add_edge(node, obs)
        model.add_cpds(_indicator_cpd(node, [1.0] * len(STATES[node])))
    model.check_model()
    return model


def _indicator_cpd(node: str, likelihood: Sequence[float]) -> TabularCPD:
    """
    P(X__obs = yes | X = s) proportional to lambda(s).

    We rescale so the largest entry is 0.98 rather than 1.0: a likelihood of
    exactly 1 or 0 would make the leaf a hard constraint and can zero out a state
    permanently, which is precisely the brittleness we are trying to avoid.
    """
    lam = np.asarray(likelihood, dtype=float)
    lam = np.clip(lam, 1e-9, None)
    lam = lam / lam.max() * 0.98
    values = np.vstack([1.0 - lam, lam])          # rows: no, yes
    obs = observation_node(node)
    return TabularCPD(
        variable=obs, variable_card=2, values=values.tolist(),
        evidence=[node], evidence_card=[len(STATES[node])],
        state_names={obs: ["no", "yes"], node: STATES[node]},
    )


# ---------------------------------------------------------------------------
class SoftEvidenceEngine(RiskEngine):
    """
    A RiskEngine that reasons from soft evidence.

    Everything downstream - explain.py, decision.py - keeps working unchanged,
    because from their point of view this is still an engine with a `posterior`
    method and an evidence dict. The evidence dict just happens to name indicator
    leaves instead of the weather nodes themselves.
    """

    def __init__(self, core: Optional[DiscreteBayesianNetwork] = None):
        super().__init__(build_augmented_network(core), method="ve")
        self._installed: Dict[str, List[float]] = {}
        # set by temporal.install_filtered_belief when the filter is driving the
        # hydrology; 0 means "nothing tracked", 1 means "pinned down"
        self.hydro_sharpness: float = 0.0
        # the caller knows which channels this district could possibly report and
        # how much static context it supplied; without that we would count a
        # sea-surface sensor as missing for an inland district and mark its
        # confidence down for it
        self.expected_channels: Optional[int] = None
        self.context_credit: float = 0.0

    # ------------------------------------------------------------- evidence
    def install(self, fused: Dict[str, FusedReading],
                skip: Optional[Iterable[str]] = None) -> Dict[str, str]:
        """
        Point the indicator CPTs at these likelihoods and return the evidence dict
        to pass to queries. Nodes without a reading are reset to uninformative so
        state cannot leak between districts or between passes.

        `skip` names nodes whose readings are being consumed elsewhere - in
        practice the hydrology pair, which the temporal filter owns.
        """
        skip = set(skip or ())
        cpds = []
        for node in SOFT_NODES:
            reading = fused.get(node)
            lam = (reading.likelihood if (reading is not None and node not in skip)
                   else [1.0] * len(STATES[node]))
            self._installed[node] = list(lam)
            cpds.append(_indicator_cpd(node, lam))

        self.model.add_cpds(*cpds)       # add_cpds replaces by variable name
        self.model.check_model()
        self._reset_inference()
        return {observation_node(n): "yes" for n in fused
                if n in SOFT_NODES and n not in skip}

    def _reset_inference(self) -> None:
        from pgmpy.inference import VariableElimination
        self._infer = VariableElimination(self.model)
        self._cache.clear()

    # ---------------------------------------------------------- diagnostics
    def predictive(self, node: str, evidence: Dict[str, str]) -> Dict[str, float]:
        """P(node | evidence) with this node's own indicator left out - the
        model's belief *before* it sees this reading. Used by the surprise gate."""
        others = {k: v for k, v in evidence.items()
                  if base_node(k) != node}
        return self.posterior([node], others)[node]

    def surprisal_bits(self, node: str, likelihood: Sequence[float],
                       evidence: Dict[str, str]) -> float:
        """
        -log2 P(reading | everything else). High means the reading disagrees with
        what the rest of the network expects.
        """
        prior = self.predictive(node, evidence)
        lam = np.asarray(likelihood, dtype=float)
        lam = lam / lam.sum()
        p = sum(prior[s] * lam[i] for i, s in enumerate(STATES[node]))
        return -math.log2(max(p, 1e-12))

    def confidence(self, evidence: Dict[str, str], hazard: str) -> Dict[str, object]:
        """
        Same idea as the core engine, but coverage counts how much *usable*
        evidence arrived, not how many dicts keys exist. A stale or disputed
        reading contributes a fraction of a sensor.
        """
        post = self.posterior([hazard], evidence)[hazard]
        h = entropy(post)

        effective = 0.0
        for node, lam in self._installed.items():
            if node in HYDRO_NODES:
                continue          # credited below, via the filter
            if node not in OBSERVABLE:
                continue
            card = len(lam)
            if card > 1:
                arr = np.asarray(lam, dtype=float)
                arr = arr / arr.sum()
                hh = entropy({str(i): p for i, p in enumerate(arr)})
                effective += max(0.0, 1.0 - hh / math.log2(card))

        # the tracked hydrology stands in for the soil and river sensors, and is
        # worth as much as the filter is actually sure about it
        effective += 2.0 * self.hydro_sharpness
        # static context (season, terrain, land use) is known exactly
        effective += self.context_credit

        possible = self.expected_channels or (
            len([n for n in OBSERVABLE if n in SOFT_NODES
                 and n not in HYDRO_NODES]) + 2)
        coverage = min(effective / max(possible, 1), 1.0)

        score = (1 - h) * 0.5 + coverage * 0.5
        label = "High" if score >= 0.66 else "Medium" if score >= 0.40 else "Low"
        return {
            "label": label,
            "score": round(score, 3),
            "posterior_entropy_bits": round(h, 3),
            "sensor_coverage": round(coverage, 3),
            "effective_sensors": round(effective, 2),
            "sensors_possible": possible,
            "sensors_used": round(effective, 2),
        }


# ---------------------------------------------------------------------------
def hard_evidence_from_fused(fused: Dict[str, FusedReading]) -> Dict[str, str]:
    """
    What a naive pipeline would have done: take the most likely bin and assert it.
    Kept so evaluate_realtime.py can measure what the soft treatment buys us.
    """
    return {node: r.most_likely_state() for node, r in fused.items()
            if node in OBSERVABLE or node == "SoilMoisture"}


if __name__ == "__main__":
    from .feeds import SnapshotFeed, districts

    feed = SnapshotFeed()
    engine = SoftEvidenceEngine()
    for _, row in districts().iterrows():
        f = feed.fetch_district(row)
        if not f.observations:
            continue
        fused = fuse_observations(f.observations)
        evidence = engine.install(fused)
        print(f"\n=== {f.district} ===")
        for node, r in sorted(fused.items()):
            lam = " ".join(f"{s}={p:.2f}" for s, p in
                           zip(STATES[node], r.likelihood))
            print(f"  {node:<15} est {r.estimate:>7.2f}  spread {r.ensemble_spread:>5.2f}"
                  f"  sigma {r.sigma_effective:>5.2f}  [{lam}]  {r.status}")
        haz = engine.hazard_probabilities(evidence)
        print("  hazards:", {k: round(v, 3) for k, v in haz.items()})
        break
