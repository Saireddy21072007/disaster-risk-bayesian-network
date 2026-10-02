"""
The explainable-AI layer.

A number on its own is useless to a tahsildar at 2 a.m. This module answers four
questions about every prediction, and all four answers come out of the same
Bayesian network - we did not bolt on a separate explainer like LIME/SHAP,
because in a BN the explanation IS a set of probability queries.

  1. WHY is it high?      evidence attribution by retraction.
                          For each observed variable we re-run inference with
                          that one reading removed and look at how far the
                          log-odds of the hazard move. That difference is exactly
                          the weight of evidence contributed by that reading
                          (Good's "weight of evidence", log-odds form), so it is
                          additive-ish and signed: a reading can also *lower* risk.

  2. WHAT is happening underneath?
                          the MAP assignment over the unobserved physical nodes,
                          narrated as a causal chain.

  3. WHAT IF the weather had been kinder?
                          counterfactual queries: force the top driver back to
                          its benign state and re-run.

  4. WHAT SHOULD WE MEASURE NEXT?
                          value of information. For every sensor we did NOT get,
                          compute the expected entropy drop
                              H(H | e) - sum_s P(s | e) H(H | e, s)
                          i.e. the mutual information I(Hazard ; Sensor | e).
                          The dashboard turns the winner into an instruction like
                          "read the Neyyar gauge before deciding".

Owner: Jithin Reddy K (attribution + text) and Sai Vandith (value of information)
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

from .config import BENIGN_STATE, OBSERVABLE, STATES, risk_band
from .engine import RiskEngine, entropy

# how we say each variable out loud
PHRASE = {
    "Rainfall":       {"Low": "light rainfall", "Moderate": "moderate rainfall",
                       "Heavy": "heavy rainfall (over 64.5 mm in 24 h)"},
    "RiverLevel":     {"Low": "the river well below warning level",
                       "Medium": "the river near its warning level",
                       "High": "the river at or past its danger level"},
    "SoilMoisture":   {"Low": "dry soil", "Medium": "partly wet soil",
                       "High": "already saturated soil"},
    "Humidity":       {"Low": "dry air", "Medium": "moderate humidity",
                       "High": "very humid air"},
    "Temperature":    {"Low": "cool temperatures", "Normal": "normal temperatures",
                       "High": "temperatures above 40 C"},
    "WindSpeed":      {"Calm": "light winds", "Breezy": "moderate winds",
                       "Strong": "strong winds (40-62 kmph)",
                       "Gale": "winds above cyclonic-storm strength (62 kmph)"},
    "SeaSurfaceTemp": {"Normal": "a normal sea surface", "Warm": "a sea surface above 28 C"},
    "PressureDrop":   {"No": "steady pressure", "Yes": "a sharp 24-hour pressure fall"},
    "Slope":          {"Flat": "flat terrain", "Moderate": "moderately sloped terrain",
                       "Steep": "steep hill slopes"},
    "Urbanisation":   {"Low": "mostly rural drainage", "High": "dense built-up drainage"},
    "Season":         {"Winter": "the winter season", "Summer": "the summer season",
                       "Monsoon": "the monsoon season"},
}

# how a variable is referred to inside a sentence
NICE_NAME = {
    "Rainfall": "the rainfall", "RiverLevel": "the river level",
    "SoilMoisture": "the soil moisture", "Humidity": "the humidity",
    "Temperature": "the temperature", "WindSpeed": "the wind speed",
    "SeaSurfaceTemp": "the sea surface temperature",
    "PressureDrop": "the pressure drop", "Slope": "the terrain",
    "Urbanisation": "the land use", "Season": "the season",
}

SENSOR_LABEL = {
    "Rainfall": "rain gauge", "RiverLevel": "river gauge",
    "Humidity": "humidity sensor", "Temperature": "thermometer",
    "WindSpeed": "anemometer", "SeaSurfaceTemp": "sea-surface temperature feed",
    "PressureDrop": "barometer", "Slope": "terrain record",
    "Urbanisation": "land-use record", "Season": "calendar",
}


def phrase(var: str, state: str) -> str:
    """
    Human wording for one piece of evidence.

    The real-time path enters evidence on indicator leaves rather than on the
    weather nodes themselves (see measurement.py), so `var` can be something like
    "Rainfall__obs" or "Hydrology__filter". Those must not reach a district
    officer's screen, so we translate them here rather than leaking node names.
    """
    if var == "Hydrology__filter":
        return "the tracked soil and river state"
    if var.endswith("__obs"):
        base = var[: -len("__obs")]
        nice = NICE_NAME.get(base, base.lower())
        return f"the live reading for {nice}"
    return PHRASE.get(var, {}).get(state, f"{var} = {state}")


def _log_odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# ----------------------------------------------------------------------------
# 1. attribution
# ----------------------------------------------------------------------------
def evidence_attribution(engine: RiskEngine, evidence: Dict[str, str],
                         hazard: str) -> List[dict]:
    """Signed weight of evidence for every reading we hold, biggest first."""
    p_full = engine.probability(hazard, "Yes", evidence)
    lo_full = _log_odds(p_full)

    rows = []
    for var, state in evidence.items():
        reduced = {k: v for k, v in evidence.items() if k != var}
        p_without = engine.probability(hazard, "Yes", reduced)
        woe = lo_full - _log_odds(p_without)      # in nats
        rows.append({
            "variable": var,
            "state": state,
            "phrase": phrase(var, state),
            "p_with": round(p_full, 4),
            "p_without": round(p_without, 4),
            "delta_prob": round(p_full - p_without, 4),
            "weight_of_evidence_bits": round(woe / math.log(2), 3),
            "odds_multiplier": round(math.exp(woe), 2),
            "direction": "raises" if woe > 0.05 else "lowers" if woe < -0.05 else "neutral",
        })
    rows.sort(key=lambda r: abs(r["weight_of_evidence_bits"]), reverse=True)
    return rows


# ----------------------------------------------------------------------------
# 2. hidden chain
# ----------------------------------------------------------------------------
def latent_chain(engine: RiskEngine, evidence: Dict[str, str]) -> List[str]:
    """Narrate the MAP state of what we could not measure."""
    m = engine.latent_map(evidence)
    return [phrase(var, state) for var, state in m.items()]


# ----------------------------------------------------------------------------
# 3. counterfactuals
# ----------------------------------------------------------------------------
def counterfactuals(engine: RiskEngine, evidence: Dict[str, str], hazard: str,
                    top: int = 2) -> List[dict]:
    """
    "What if that driver had been benign?" We only do this for readings that are
    not already benign, and only for the strongest few, otherwise the dashboard
    turns into a wall of hypotheticals.
    """
    p_full = engine.probability(hazard, "Yes", evidence)
    ranked = evidence_attribution(engine, evidence, hazard)

    out = []
    for row in ranked:
        var = row["variable"]
        benign = BENIGN_STATE.get(var)
        if benign is None or evidence.get(var) == benign:
            continue
        alt = dict(evidence)
        alt[var] = benign
        p_cf = engine.probability(hazard, "Yes", alt)
        name = NICE_NAME.get(var, var)
        out.append({
            "variable": var,
            "from_state": evidence[var],
            "to_state": benign,
            "p_actual": round(p_full, 4),
            "p_counterfactual": round(p_cf, 4),
            "sentence": (f"Had {name} been {benign.lower()} instead of "
                         f"{evidence[var].lower()}, the {hazard.lower()} probability "
                         f"would be {p_cf:.0%} rather than {p_full:.0%}."),
        })
        if len(out) >= top:
            break
    return out


# ----------------------------------------------------------------------------
# 4. value of information
# ----------------------------------------------------------------------------
def value_of_information(engine: RiskEngine, evidence: Dict[str, str],
                         hazard: str) -> List[dict]:
    """
    I(Hazard ; Sensor | evidence) in bits, for every sensor still missing.
    Tells the control room which single extra reading is worth chasing.
    """
    base_h = entropy(engine.posterior([hazard], evidence)[hazard])
    missing = [v for v in OBSERVABLE if v not in evidence]

    rows = []
    for var in missing:
        marg = engine.posterior([var], evidence)[var]
        expected = 0.0
        for state, p_state in marg.items():
            if p_state < 1e-6:
                continue
            post = engine.posterior([hazard], {**evidence, var: state})[hazard]
            expected += p_state * entropy(post)
        gain = base_h - expected
        rows.append({
            "variable": var,
            "sensor": SENSOR_LABEL.get(var, var),
            "expected_bits_gained": round(max(gain, 0.0), 4),
            "current_entropy_bits": round(base_h, 4),
        })
    rows.sort(key=lambda r: r["expected_bits_gained"], reverse=True)
    return rows


# ----------------------------------------------------------------------------
# putting it into English
# ----------------------------------------------------------------------------
def _bullet(row: dict) -> str:
    """
    One line for the dashboard. A 'neutral' reading is not noise - it usually
    means the variable is d-separated from the hazard by something we already
    observed (rainfall stops mattering once the river gauge is read), and saying
    so out loud is more honest than hiding the row.
    """
    if row["direction"] == "raises":
        return f"{row['phrase']} raises the risk (x{row['odds_multiplier']:.1f} on the odds)"
    if row["direction"] == "lowers":
        return f"{row['phrase']} lowers the risk (x{row['odds_multiplier']:.1f} on the odds)"
    return (f"{row['phrase']} adds nothing once the other readings are known "
            f"(its effect is already carried by them)")


def narrate(engine: RiskEngine, evidence: Dict[str, str], hazard: str,
            attribution: Optional[List[dict]] = None,
            describe: Optional[Callable[[str], Optional[str]]] = None) -> dict:
    """
    `describe` lets a caller supply better wording for a piece of evidence than
    this module can work out on its own. The real-time pipeline uses it to say
    "rainfall measured at 62.4 mm/24h (3 sources agree)" where we would otherwise
    only be able to say "the live reading for the rainfall" - it holds the actual
    numbers, we do not.
    """
    p = engine.probability(hazard, "Yes", evidence)
    band, colour = risk_band(p)
    prior = engine.probability(hazard, "Yes", {})
    attribution = attribution or evidence_attribution(engine, evidence, hazard)
    conf = engine.confidence(evidence, hazard)

    if describe is not None:
        for row in attribution:
            better = describe(row["variable"])
            if better:
                row["phrase"] = better

    raising = [r for r in attribution if r["direction"] == "raises"][:3]
    lowering = [r for r in attribution if r["direction"] == "lowers"][:2]

    lines = [f"{hazard} probability is {p:.0%} ({band} risk). "
             f"The long-run base rate for this hazard is {prior:.0%}."]

    if raising:
        bits = "; ".join(f"{r['phrase']} (x{r['odds_multiplier']:.1f} on the odds)"
                         for r in raising)
        lines.append(f"What pushed it up: {bits}.")
    if lowering:
        bits = "; ".join(r["phrase"] for r in lowering)
        lines.append(f"What is holding it down: {bits}.")

    chain = latent_chain(engine, evidence)
    if chain:
        lines.append("Our best reconstruction of what we cannot measure: "
                     + ", ".join(chain) + ".")

    cf = counterfactuals(engine, evidence, hazard, top=2)
    if cf:
        lines.append(cf[0]["sentence"])

    # how we word the coverage depends on whether the evidence was hard or soft:
    # the real-time path discounts readings for disagreement and age, so its count
    # is fractional and calling it "inputs available" would be misleading
    if "effective_sensors" in conf:
        coverage_text = (f"effective evidence {conf['effective_sensors']:.1f} of "
                         f"{conf['sensors_possible']} possible inputs, after "
                         f"discounting for source disagreement and age")
    else:
        coverage_text = (f"{conf['sensors_used']}/{conf['sensors_possible']} "
                         f"inputs available")

    voi = value_of_information(engine, evidence, hazard)
    if voi and voi[0]["expected_bits_gained"] > 0.02:
        best = voi[0]
        lines.append(f"Confidence is {conf['label'].lower()} ({coverage_text}). "
                     f"Getting a reading from the {best['sensor']} would remove the "
                     f"most uncertainty ({best['expected_bits_gained']:.2f} bits).")
    else:
        lines.append(f"Confidence is {conf['label'].lower()} ({coverage_text}).")

    return {
        "hazard": hazard,
        "probability": round(p, 4),
        "base_rate": round(prior, 4),
        "risk_band": band,
        "colour": colour,
        "confidence": conf,
        "text": " ".join(lines),
        "bullets": [_bullet(r) for r in attribution[:5]],
        "attribution": attribution,
        "counterfactuals": cf,
        "value_of_information": voi[:3],
        "latent_chain": chain,
    }


def explain_all(engine: RiskEngine, evidence: Dict[str, str]) -> Dict[str, dict]:
    """One explanation per hazard - what the /api/predict endpoint returns."""
    from .config import HAZARDS
    return {h: narrate(engine, evidence, h) for h in HAZARDS}


if __name__ == "__main__":
    eng = RiskEngine()
    ev = {"Season": "Monsoon", "Rainfall": "Heavy", "Humidity": "High",
          "RiverLevel": "High", "Slope": "Steep", "Urbanisation": "Low"}
    out = narrate(eng, ev, "Flood")
    print(out["text"])
    print()
    for b in out["bullets"]:
        print(" -", b)
