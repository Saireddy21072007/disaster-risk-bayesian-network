"""
The real-time pipeline. One pass per district looks like this:

    fetch  ->  fuse  ->  surprise gate  ->  filter  ->  virtual evidence
           ->  assess  ->  decide  ->  hysteresis  ->  audit log

feeds.py does the fetch, measurement.py the fuse, temporal.py the filter. What is
left - and what lives here - are the three things that only exist because the data
is arriving continuously rather than sitting in a CSV:

1. A BAYESIAN SURPRISE GATE.
   A stuck gauge or a corrupted grid cell does not announce itself; it just sends a
   number. Before accepting a reading we ask the network how plausible it is given
   everything else we know:

       surprisal = -log2 P(reading | all other evidence)

   Above four bits - the reading had under a 6 % chance of being what it is - we do
   not discard it (discarding data on the model's say-so is how you miss the real
   emergency) but we shrink its trust weight, which flattens its likelihood. A
   suspect sensor is downgraded to a hint, and the event is logged. Note the
   self-consistency: the same posterior that the reading feeds is what judges it.

2. ALERT HYSTERESIS, DELIBERATELY ASYMMETRIC.
   Recomputing every fifteen minutes means the recommended action can oscillate
   across a decision boundary, and an evacuation order that flickers is worse than
   either state. So we escalate on the first frame that justifies it, and
   de-escalate only after three consecutive frames agree. The asymmetry is not
   arbitrary: it is the same utility table talking. A missed emergency costs about
   eight times a false alarm, so the cost of being slow to stand down is small and
   the cost of being slow to escalate is not.

3. MUTUAL-INFORMATION-DRIVEN POLLING.
   Every feed costs a request and every sensor has a repair queue. For each
   quantity we compute what a perfect reading would be worth right now,

       I(Hazard ; X | evidence) = H(Hazard | evidence) - E_x[ H(Hazard | evidence, X=x) ]

   and poll in that order. The district where the answer is already obvious gets
   left alone; the one sitting on a decision boundary gets the attention.

State - filter beliefs and alert levels - is persisted, because a stream processor
that forgets everything on restart is not tracking anything.

Owner: Sai Reddy A, with Sai Vandith on the gate and the scheduler
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .config import ARTIFACT_DIR, CONSEQUENCES, HAZARDS, STATES, risk_band
from .decision import ACTIONS, recommend
from .engine import entropy
from .explain import narrate
from .feeds import SOURCE_LABEL, districts, get_feed, save_snapshot
from .measurement import (HYDRO_NODES, SOFT_NODES, FusedReading,
                          SoftEvidenceEngine, fuse_observations,
                          hard_evidence_from_fused, observation_node, soften)
from .temporal import (HYDRO_LEAF, FilterBank, install_filtered_belief)

ALERT_STATE_FILE = ARTIFACT_DIR / "alert_state.json"
AUDIT_LOG = ARTIFACT_DIR / "alert_log.jsonl"

SURPRISE_THRESHOLD_BITS = 4.0
SUSPECT_TRUST = 0.35
DEESCALATE_FRAMES = 3

ACTION_RANK = {a: i for i, a in enumerate(ACTIONS)}


def current_season(when: Optional[datetime] = None,
                   coast_side: str = "none") -> str:
    """
    Season from the calendar, using the same coast-aware rule the training data
    uses - October to December is monsoon on the east coast, because that is when
    the retreating monsoon delivers most of its rain there. See observed.season_of
    for how the real data forced that correction on us.

    Training and inference MUST agree about this. If the CPTs were estimated with
    October labelled monsoon and the live path labelled it winter, every east-coast
    reading would be scored against the wrong column of the rainfall CPT.
    """
    d = when or datetime.now()
    from .observed import season_of
    return season_of(d.strftime("%Y-%m-%d"), coast_side)


def slope_state(slope_deg: float) -> str:
    from .discretize import bin_value
    return bin_value("slope_deg", slope_deg) or "Flat"


# ---------------------------------------------------------------------------
@dataclass
class AlertState:
    district: str
    level: str = "Monitor"
    pending: Optional[str] = None
    pending_frames: int = 0
    since: str = ""
    last_change: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StreamRecord:
    """Everything one pass produced for one district."""
    district: str
    state: str
    lat: float
    lon: float
    timestamp: str
    hazards: Dict[str, float]
    consequences: Dict[str, dict]
    dominant_hazard: str
    risk_band: str
    colour: str
    confidence: dict
    recommendation: dict
    raw_action: str
    alert_level: str
    alert_changed: bool
    hysteresis: dict
    readings: List[dict]
    filter_step: dict
    gate_events: List[dict]
    polling_priority: List[dict]
    errors: List[str]
    headline_explanation: str = ""
    explanation: dict = field(default_factory=dict)
    naive_comparison: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
class StreamProcessor:
    """
    Holds the engine, the filter bank and the alert state for a fleet of
    districts. `mode` is passed straight to feeds.get_feed:
    live (hit the APIs), snapshot (replay the saved pass), offline (cache only).
    """

    def __init__(self, mode: str = "live", engine: Optional[SoftEvidenceEngine] = None):
        self.mode = mode
        self.feed = get_feed(mode)
        # the live path uses the HYBRID model: weather layer estimated from real
        # observations, hazard layer still elicited. The all-simulated model stays
        # where it belongs, on the synthetic held-out evaluation.
        from .observed import load_hybrid_model
        self.engine = engine or SoftEvidenceEngine(load_hybrid_model())
        self.filters = FilterBank()
        self.alerts: Dict[str, AlertState] = {}
        self._load_alerts()
        self.records: Dict[str, StreamRecord] = {}
        # A pass REWRITES the indicator CPTs in the shared network, so two passes
        # running at once would read each other's evidence. FastAPI serves sync
        # endpoints from a thread pool, so this is a real risk, not a theoretical
        # one - every entry point that touches the engine takes this lock.
        self._lock = threading.RLock()

    # ----------------------------------------------------------- persistence
    def _load_alerts(self) -> None:
        if not ALERT_STATE_FILE.exists():
            return
        try:
            blob = json.loads(ALERT_STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for name, entry in blob.get("alerts", {}).items():
            self.alerts[name] = AlertState(**entry)

    def _save_alerts(self) -> None:
        ALERT_STATE_FILE.write_text(json.dumps({
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "alerts": {k: v.to_dict() for k, v in self.alerts.items()},
        }, indent=1), encoding="utf-8")

    def _audit(self, record: StreamRecord) -> None:
        """Append-only log, so every warning we ever issued can be reviewed after
        the event with the evidence that produced it."""
        with AUDIT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "timestamp": record.timestamp,
                "district": record.district,
                "alert_level": record.alert_level,
                "raw_action": record.raw_action,
                "hazards": record.hazards,
                "p_emergency": record.recommendation.get("p_emergency"),
                "confidence": record.confidence.get("label"),
                "readings": {r["node"]: {"estimate": r["estimate"],
                                         "status": r["status"],
                                         "sources": r["sources"]}
                             for r in record.readings},
                "errors": record.errors,
            }) + "\n")

    # ------------------------------------------------------------- the gate
    def _apply_surprise_gate(self, fused: Dict[str, FusedReading],
                             evidence: Dict[str, str]) -> List[dict]:
        """
        Score every weather reading against the rest of the network.

        A high surprisal has TWO possible explanations and our first version only
        considered one of them. It assumed the model was right and the sensor was
        broken, so it discounted the reading. The first time we pointed the pipeline
        at real data that went badly wrong: the simulated CPTs believed a monsoon day
        averaged tens of millimetres of rain, the actual reading for Coimbatore was
        1.8 mm from three independent forecast models in agreement, and the gate
        declared reality implausible and threw two thirds of it away.

        So we now separate the hypotheses using something the model cannot fake -
        whether the independent sources agree with each other:

          sources agree, model surprised   -> the MODEL is probably wrong. Keep the
                                              reading at full strength and raise a
                                              model-surprise flag for review.
          single source, model surprised   -> could be either. Discount, but log it.
          sources disagree, model surprised-> the feed is a mess. Discount.

        A sensor fault cannot make three independently-run numerical weather models
        agree, so agreement is genuine evidence that the reading is real.
        """
        events: List[dict] = []
        adjusted = False

        for node, reading in fused.items():
            if node not in SOFT_NODES or node in HYDRO_NODES:
                continue
            try:
                bits = self.engine.surprisal_bits(node, reading.raw_likelihood,
                                                  evidence)
            except Exception:                       # noqa: BLE001
                continue
            reading.surprisal_bits = bits
            if bits <= SURPRISE_THRESHOLD_BITS:
                continue

            corroborated = reading.n_sources >= 2 and reading.status != "disputed"
            if corroborated:
                # trust the data over ourselves, and say so out loud
                reading.status = "model-surprise"
                events.append({
                    "node": node,
                    "estimate": round(reading.estimate, 3),
                    "surprisal_bits": round(bits, 2),
                    "sources": reading.sources,
                    "verdict": "model-surprise",
                    "action": f"{reading.n_sources} independent sources agree, so "
                              "the reading is kept at full strength; the model's "
                              "prior for this variable looks wrong",
                })
            else:
                reading.suspicion_weight = SUSPECT_TRUST
                reading.trust_weight = reading.staleness_weight * SUSPECT_TRUST
                reading.likelihood = soften(reading.raw_likelihood,
                                            reading.trust_weight)
                reading.status = "suspect"
                adjusted = True
                events.append({
                    "node": node,
                    "estimate": round(reading.estimate, 3),
                    "surprisal_bits": round(bits, 2),
                    "sources": reading.sources,
                    "verdict": "suspect",
                    "action": "uncorroborated and implausible; trust reduced to "
                              f"{reading.trust_weight:.2f} and treated as a hint",
                })

        if adjusted:
            evidence.clear()
            evidence.update(self.engine.install(fused, skip=HYDRO_NODES))
        return events

    def _gate_hydrology(self, fused: Dict[str, FusedReading],
                        predicted_marginals: Dict[str, Dict[str, float]]
                        ) -> List[dict]:
        """
        Same idea for the hydrology, but judged against the FILTER's prediction
        rather than the static network - the filter is what has the memory, so it
        is the thing entitled to be surprised.
        """
        events = []
        for node in HYDRO_NODES:
            reading = fused.get(node)
            if reading is None:
                continue
            prior = predicted_marginals.get(node, {})
            if not prior:
                continue
            lam = np.asarray(reading.raw_likelihood, dtype=float)
            lam = lam / lam.sum()
            p = sum(prior.get(s, 0.0) * lam[i] for i, s in enumerate(STATES[node]))
            bits = -math.log2(max(p, 1e-12))
            reading.surprisal_bits = bits
            if bits > SURPRISE_THRESHOLD_BITS:
                reading.suspicion_weight = SUSPECT_TRUST
                reading.trust_weight = reading.staleness_weight * SUSPECT_TRUST
                reading.likelihood = soften(reading.raw_likelihood,
                                            reading.trust_weight)
                reading.status = "suspect"
                events.append({
                    "node": node,
                    "estimate": round(reading.estimate, 3),
                    "surprisal_bits": round(bits, 2),
                    "sources": reading.sources,
                    "action": "disagrees with the filter's prediction; "
                              "trust reduced",
                })
        return events

    # -------------------------------------------------------------- polling
    def polling_priority(self, evidence: Dict[str, str], hazard: str,
                         fused: Dict[str, FusedReading]) -> List[dict]:
        """
        I(hazard ; X | evidence) for each quantity: what a perfect reading of X
        would be worth right now, in bits.
        """
        base = entropy(self.engine.posterior([hazard], evidence)[hazard])
        rows = []
        for node in SOFT_NODES:
            try:
                marg = self.engine.posterior([node], evidence)[node]
                expected = 0.0
                for state, p in marg.items():
                    if p < 1e-6:
                        continue
                    post = self.engine.posterior([hazard],
                                                 {**evidence, node: state})[hazard]
                    expected += p * entropy(post)
            except Exception:                       # noqa: BLE001
                continue
            reading = fused.get(node)
            rows.append({
                "node": node,
                "expected_bits_gained": round(max(base - expected, 0.0), 4),
                "have_reading": reading is not None,
                "status": reading.status if reading else "missing",
                "sources": reading.sources if reading else [],
                "age_minutes": round(reading.age_minutes, 1) if reading else None,
            })
        # something we do not have at all outranks something we already trust
        rows.sort(key=lambda r: (r["expected_bits_gained"]
                                 * (1.6 if not r["have_reading"] else 1.0)),
                  reverse=True)
        return rows

    # ----------------------------------------------------------- hysteresis
    def _apply_hysteresis(self, district: str, proposed: str,
                          timestamp: str) -> tuple[str, bool, dict]:
        state = self.alerts.setdefault(district,
                                       AlertState(district=district,
                                                  since=timestamp,
                                                  last_change=timestamp))
        current_rank = ACTION_RANK.get(state.level, 0)
        proposed_rank = ACTION_RANK.get(proposed, 0)
        changed = False

        if proposed_rank > current_rank:
            # escalate at once - being slow here is the expensive mistake
            state.level = proposed
            state.pending, state.pending_frames = None, 0
            state.since = state.last_change = timestamp
            changed = True
            reason = "escalated immediately"
        elif proposed_rank < current_rank:
            if state.pending == proposed:
                state.pending_frames += 1
            else:
                state.pending, state.pending_frames = proposed, 1
            if state.pending_frames >= DEESCALATE_FRAMES:
                state.level = proposed
                state.pending, state.pending_frames = None, 0
                state.since = state.last_change = timestamp
                changed = True
                reason = f"stood down after {DEESCALATE_FRAMES} agreeing frames"
            else:
                reason = (f"holding {state.level}; {proposed} seen "
                          f"{state.pending_frames}/{DEESCALATE_FRAMES} times")
        else:
            state.pending, state.pending_frames = None, 0
            reason = "unchanged"

        return state.level, changed, {
            "held_level": state.level,
            "proposed": proposed,
            "pending": state.pending,
            "pending_frames": state.pending_frames,
            "frames_needed_to_stand_down": DEESCALATE_FRAMES,
            "reason": reason,
            "since": state.since,
        }

    # ------------------------------------------------------------- one pass
    def step(self, row, explain: bool = True) -> StreamRecord:
        with self._lock:
            return self._step_locked(row, explain)

    def _step_locked(self, row, explain: bool = True) -> StreamRecord:
        timestamp = datetime.now().isoformat(timespec="seconds")
        district = row["district"]

        feed = self.feed.fetch_district(row)
        errors = list(feed.errors)
        fused = fuse_observations(feed.observations)

        # ---- weather as soft evidence, then judged
        evidence = self.engine.install(fused, skip=HYDRO_NODES)

        # Static context is known exactly, so it goes in as ordinary hard evidence:
        # the season comes off the calendar, the terrain and land use off
        # districts.csv. We left this out of the first version of the pipeline and
        # the model was reasoning about Kerala in August without being told it was
        # the monsoon.
        context = {
            "Season": current_season(coast_side=str(row["coast_side"])),
            "Slope": slope_state(float(row["slope_deg"])),
            "Urbanisation": str(row["urbanisation"]),
        }
        evidence.update(context)
        self.engine.context_credit = float(len(context))
        # what this district could report at best: five weather channels, a sea
        # surface only if it has a coast, two hydrology channels, three context
        self.engine.expected_channels = 5 + (1 if str(row["coast_side"]) != "none"
                                             else 0) + 2 + len(context)

        gate_events = self._apply_surprise_gate(fused, evidence)

        # ---- the filter: predict, judge the hydrology, then update
        filt = self.filters.get(district)
        rain_posterior = self.engine.posterior(["Rainfall"], evidence)["Rainfall"]
        predicted = {
            "SoilMoisture": dict(zip(STATES["SoilMoisture"],
                                     np.asarray(filt.belief).reshape(3, 3).sum(axis=1))),
            "RiverLevel": dict(zip(STATES["RiverLevel"],
                                   np.asarray(filt.belief).reshape(3, 3).sum(axis=0))),
        }
        gate_events += self._gate_hydrology(fused, predicted)
        step = filt.step(rain_posterior,
                         {k: v for k, v in fused.items() if k in HYDRO_NODES},
                         timestamp=timestamp)

        # ---- hand the filtered hydrology to the static network
        evidence = install_filtered_belief(self.engine, filt.belief, evidence)

        # ---- assess and decide
        hazards = self.engine.hazard_probabilities(evidence)
        dominant = max(HAZARDS, key=lambda h: hazards[h])
        band, colour = risk_band(hazards[dominant])
        plan = recommend(self.engine, evidence, population=int(row["population"]))
        level, changed, hyst = self._apply_hysteresis(district, plan["action"],
                                                      timestamp)

        record = StreamRecord(
            district=district, state=row["state"],
            lat=float(row["lat"]), lon=float(row["lon"]),
            timestamp=timestamp,
            hazards={h: round(hazards[h], 4) for h in HAZARDS},
            consequences={k: {s: round(p, 4) for s, p in v.items()}
                          for k, v in self.engine.consequence_probabilities(evidence).items()},
            dominant_hazard=dominant, risk_band=band, colour=colour,
            confidence=self.engine.confidence(evidence, dominant),
            recommendation=plan, raw_action=plan["action"],
            alert_level=level, alert_changed=changed, hysteresis=hyst,
            readings=[fused[n].to_dict() for n in sorted(fused)],
            filter_step=step.to_dict(),
            gate_events=gate_events,
            polling_priority=self.polling_priority(evidence, dominant, fused)[:5],
            errors=errors,
        )

        if explain:
            ex = narrate(self.engine, evidence, dominant,
                         describe=self._describer(fused, filt))
            record.headline_explanation = ex["text"]
            record.explanation = ex
            record.naive_comparison = self._compare_with_naive(fused, hazards,
                                                              dominant)

        self.records[district] = record
        self._audit(record)
        return record

    # ------------------------------------------------------------- wording
    UNITS = {"Rainfall": "mm/24h", "Temperature": "C", "Humidity": "% RH",
             "WindSpeed": "kmph", "PressureDrop": "hPa/24h",
             "SeaSurfaceTemp": "C", "RiverLevel": "of bankfull",
             "SoilMoisture": "saturation"}

    def _describer(self, fused: Dict[str, FusedReading], filt):
        """
        Wording callback handed to explain.narrate: turns an indicator-leaf name
        into what the reading actually said, how many sources produced it and
        whether they agreed. The officer needs the number; the node name is our
        business, not theirs.
        """
        def describe(var: str) -> Optional[str]:
            if var == HYDRO_LEAF:
                m = filt.marginals()
                soil = max(m["SoilMoisture"], key=m["SoilMoisture"].get)
                river = max(m["RiverLevel"], key=m["RiverLevel"].get)
                return (f"the tracked catchment state (soil {soil.lower()}, "
                        f"river {river.lower()}, carried forward from previous "
                        f"readings)")
            if not var.endswith("__obs"):
                return None
            node = var[: -len("__obs")]
            r = fused.get(node)
            if r is None:
                return None
            unit = self.UNITS.get(node, "")
            if r.n_sources > 1:
                agree = ("agreeing" if r.status == "ok"
                         else "disagreeing" if r.status == "disputed"
                         else r.status)
                src = f"{r.n_sources} sources {agree}"
            else:
                label = SOURCE_LABEL.get(r.sources[0], r.sources[0])
                src = label if r.status == "ok" else f"{label}, {r.status}"
            return (f"{node.lower()} measured at {r.estimate:.1f} {unit} "
                    f"({src}) - most likely {r.most_likely_state().lower()}")

        return describe

    def _compare_with_naive(self, fused: Dict[str, FusedReading],
                            soft_hazards: Dict[str, float],
                            dominant: str) -> dict:
        """
        What a hard-binning pipeline would have concluded from the same feed. This
        is the number that shows the soft treatment is doing something, and it is
        cheap, so we compute it every pass.
        """
        from .engine import RiskEngine
        from .learn import load_model
        hard = hard_evidence_from_fused(fused)
        try:
            core = RiskEngine(load_model())
            hard_hazards = core.hazard_probabilities(hard)
        except Exception:                           # noqa: BLE001
            return {}
        return {
            "hard_evidence": hard,
            "hard_hazards": {h: round(v, 4) for h, v in hard_hazards.items()},
            "soft_hazards": {h: round(v, 4) for h, v in soft_hazards.items()},
            "dominant_gap": round(soft_hazards[dominant]
                                  - hard_hazards.get(dominant, 0.0), 4),
        }

    # ----------------------------------------------------------- whole fleet
    def step_all(self, only: Optional[Iterable[str]] = None,
                 explain: bool = False) -> List[StreamRecord]:
        df = districts()
        if only:
            wanted = {d.lower() for d in only}
            df = df[df["district"].str.lower().isin(wanted)]
        with self._lock:
            out = [self._step_locked(row, explain=explain)
                   for _, row in df.iterrows()]
            self.filters.save()
            self._save_alerts()
        return out

    def snapshot(self, only: Optional[Iterable[str]] = None) -> Path:
        """Save the raw feed so the demo can be replayed with the wifi off."""
        return save_snapshot(self.feed.fetch_all(only=only))


# ---------------------------------------------------------------------------
_PROCESSOR: Optional[StreamProcessor] = None


def get_processor(mode: Optional[str] = None) -> StreamProcessor:
    global _PROCESSOR
    if _PROCESSOR is None or (mode and mode != _PROCESSOR.mode):
        _PROCESSOR = StreamProcessor(mode or "live")
    return _PROCESSOR


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    mode = "snapshot" if "--snapshot" in sys.argv else "live"
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or None

    proc = StreamProcessor(mode)
    records = proc.step_all(only=names, explain=True)
    records.sort(key=lambda r: -max(r.hazards.values()))

    print(f"\nmode = {mode},  {len(records)} districts,  "
          f"{datetime.now():%Y-%m-%d %H:%M}")
    print("-" * 104)
    print(f"{'district':<15}{'top hazard':<12}{'P':>7}{'alert':>12}"
          f"{'conf':>8}{'src':>5}{'naive P':>9}  notes")
    print("-" * 104)
    for r in records:
        p = r.hazards[r.dominant_hazard]
        naive = r.naive_comparison.get("hard_hazards", {}).get(r.dominant_hazard)
        notes = []
        if r.gate_events:
            notes.append(f"{len(r.gate_events)} gated")
        disputed = [x["node"] for x in r.readings if x["status"] == "disputed"]
        if disputed:
            notes.append("disputed: " + ",".join(disputed))
        if r.errors:
            notes.append(r.errors[0][:34])
        print(f"{r.district:<15}{r.dominant_hazard:<12}{p:>7.3f}"
              f"{r.alert_level:>12}{r.confidence['label']:>8}"
              f"{len(r.readings):>5}"
              f"{(f'{naive:.3f}' if naive is not None else '-'):>9}  "
              f"{'; '.join(notes)}")

    top = records[0]
    print("\n" + "=" * 104)
    print(f"{top.district}: {top.headline_explanation}")
    print("\nWorth measuring next:")
    for row in top.polling_priority[:3]:
        print(f"  {row['node']:<16} {row['expected_bits_gained']:.3f} bits"
              f"   ({row['status']})")
