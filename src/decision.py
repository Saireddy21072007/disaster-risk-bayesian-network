"""
The decision layer - turning a posterior into an instruction.

Our synopsis said "if flood probability > 80 % then evacuate". We changed that
after the zeroth-review reading, and this file is the change:

    A fixed probability threshold is the wrong tool. Whether 40 % justifies an
    evacuation depends on how bad the event is, how much the evacuation costs and
    how many people are exposed - not on the number 0.8. So we made "Evacuate" a
    DECISION node in an influence diagram, attached a utility table, and let the
    system pick the action with maximum expected utility. The threshold then comes
    OUT of the model instead of being typed into it.

Three ideas from the course show up here:
  * chance node vs decision node vs utility node,
  * maximum expected utility as the decision rule,
  * expected value of perfect information (EVPI) - how much a perfect forecast
    would be worth, which is what justifies spending money on a river gauge.

Units: everything is in "cost units per 100 000 exposed people". They are
relative, not rupees. What matters for the argmax is the ratio between the rows,
and every number below is one we can defend and change in front of the class.

Owner: Sai Reddy A (utilities) and Rohit Vardhan M (checklists)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .config import HAZARDS
from .engine import RiskEngine

ACTIONS = ["Monitor", "Advisory", "Prepare", "Evacuate"]

# what it costs us to take the action, whether or not anything happens
ACTION_COST = {"Monitor": 0.0, "Advisory": 1.0, "Prepare": 6.0, "Evacuate": 25.0}

# residual harm if an emergency DOES happen and this is all we had done
HARM_IF_EMERGENCY = {"Monitor": 100.0, "Advisory": 80.0, "Prepare": 45.0,
                     "Evacuate": 12.0}

# cost of crying wolf: nothing happens and we had already acted. Includes the
# part we care about most - people ignoring the next warning.
FALSE_ALARM_COST = {"Monitor": 0.0, "Advisory": 1.0, "Prepare": 2.0,
                    "Evacuate": 8.0}

# fraction of a district's population that actually sits in the exposed zone.
# Crude planning figure, not a census number - flagged as such on the dashboard.
EXPOSURE_FRACTION = 0.12

CHECKLIST = {
    "Evacuate": [
        "Evacuate low-lying and hillside settlements to the listed shelter",
        "Close the affected highway / ghat stretch to traffic",
        "Open relief camps and move stock of food and drinking water",
        "Dispatch SDRF / NDRF teams with boats to the taluk headquarters",
        "Put the district hospital and nearby PHCs on emergency standby",
    ],
    "Prepare": [
        "Pre-position boats, ropes and rescue teams at the taluk office",
        "Get shelters and community halls ready but do not move people yet",
        "Warn the district hospital to keep beds and ambulances free",
        "Brief village officers and set up a 3-hourly reporting line",
    ],
    "Advisory": [
        "Issue a public advisory over local radio, SMS and WhatsApp groups",
        "Advise against night travel on ghat roads and low bridges",
        "Ask fishermen and quarry workers to stay off site",
    ],
    "Monitor": [
        "Continue automated 3-hourly monitoring, no field action needed",
        "Re-run the assessment when the next gauge reading arrives",
    ],
}

# hazard-specific lines appended to the checklist when that hazard dominates
HAZARD_SPECIFIC = {
    "Flood": ["Shut sluice gates as per the reservoir operation schedule",
              "Cut power to submerged feeders before water reaches switchgear"],
    "Landslide": ["Stop all traffic on the ghat road and post barriers",
                  "Move hillside labour camps and tea-estate quarters first"],
    "Cyclone": ["Broadcast the fishermen warning and recall boats to harbour",
                "Secure thatched roofs, hoardings and site cranes"],
    "Heatwave": ["Open cooling centres and keep ORS stocked at every PHC",
                 "Shift outdoor labour and school hours away from 12:00-16:00"],
}


# ----------------------------------------------------------------------------
# the influence diagram
# ----------------------------------------------------------------------------
def utility(action: str, emergency: bool) -> float:
    """U(action, state). Negative because every outcome here is a cost."""
    if emergency:
        return -(ACTION_COST[action] + HARM_IF_EMERGENCY[action])
    return -(ACTION_COST[action] + FALSE_ALARM_COST[action])


def expected_utility(action: str, p_emergency: float) -> float:
    return (p_emergency * utility(action, True)
            + (1 - p_emergency) * utility(action, False))


def best_action(p_emergency: float) -> str:
    return max(ACTIONS, key=lambda a: expected_utility(a, p_emergency))


def switch_points(resolution: int = 2001) -> Dict[str, tuple]:
    """
    Sweep P(emergency) from 0 to 1 and record the probability interval over which
    each action is optimal. These intervals ARE our alert thresholds, and they are
    a consequence of the utility table rather than an input to it. We print them
    on a slide next to the "why not just use 0.8" question.
    """
    ranges: Dict[str, list] = {}
    for i in range(resolution):
        p = i / (resolution - 1)
        a = best_action(p)
        if a not in ranges:
            ranges[a] = [p, p]
        else:
            ranges[a][1] = p
    return {a: (round(lo, 4), round(hi, 4)) for a, (lo, hi) in ranges.items()}


def evpi(p_emergency: float) -> float:
    """
    Expected value of perfect information: what we would pay for a crystal ball.
        EVPI = E_state[max_a U(a, state)] - max_a E_state[U(a, state)]
    Zero when the decision is already obvious, largest when we are on the fence -
    which is exactly when it is worth sending someone to read the gauge.
    """
    with_info = (p_emergency * max(utility(a, True) for a in ACTIONS)
                 + (1 - p_emergency) * max(utility(a, False) for a in ACTIONS))
    without_info = max(expected_utility(a, p_emergency) for a in ACTIONS)
    return with_info - without_info


# ----------------------------------------------------------------------------
# what the API returns
# ----------------------------------------------------------------------------
def p_emergency(engine: RiskEngine, evidence: Dict[str, str]) -> float:
    """
    The single state variable the decision hangs on:

        emergency  =  RescueNeeded = Yes   OR   HospitalDemand = High

    RescueNeeded already collects flood, landslide and cyclone into one "do
    people need help" answer, which is right - a district does not run three
    separate evacuations. But our first version used RescueNeeded alone, and then
    a 42 C heat wave came back as "Monitor", because heat waves reach people
    through hospitals, not through boats. So the emergency event is the UNION of
    the two, computed from the joint posterior rather than by adding
    probabilities (the two are far from independent - they share RescueNeeded).
    """
    factor = engine.joint(["RescueNeeded", "HospitalDemand"], evidence)
    p_quiet = 0.0
    for demand in ("Low", "Medium"):
        try:
            p_quiet += float(factor.get_value(RescueNeeded="No",
                                              HospitalDemand=demand))
        except Exception:
            # one of them was given as evidence, so it is not in the factor
            return engine.probability("RescueNeeded", "Yes", evidence)
    return 1.0 - p_quiet


# an "evacuation" is the wrong word for a heat wave - same decision, different
# vocabulary in front of the officer who has to act on it
ACTION_LABEL = {
    "Heatwave": {"Evacuate": "Mass relief operation", "Prepare": "Stage relief"},
}


def recommend(engine: RiskEngine, evidence: Dict[str, str],
              population: Optional[int] = None) -> dict:
    """Full recommendation for one district."""
    p_em = p_emergency(engine, evidence)
    hazards = engine.hazard_probabilities(evidence)
    dominant = max(HAZARDS, key=lambda h: hazards[h])

    table = [{
        "action": a,
        "expected_utility": round(expected_utility(a, p_em), 2),
        "cost_if_nothing_happens": round(-utility(a, False), 2),
        "cost_if_emergency": round(-utility(a, True), 2),
    } for a in ACTIONS]
    table.sort(key=lambda r: r["expected_utility"], reverse=True)

    action = table[0]["action"]
    runner_up = table[1]
    margin = table[0]["expected_utility"] - runner_up["expected_utility"]

    checklist = list(CHECKLIST[action])
    if action != "Monitor" and hazards[dominant] > 0.35:
        checklist += HAZARD_SPECIFIC.get(dominant, [])

    out = {
        "p_emergency": round(p_em, 4),
        "dominant_hazard": dominant,
        "dominant_hazard_probability": round(hazards[dominant], 4),
        "action": action,
        "action_label": ACTION_LABEL.get(dominant, {}).get(action, action),
        "runner_up": runner_up["action"],
        "decision_margin": round(margin, 2),
        "close_call": bool(margin < 3.0),
        "expected_utility_table": table,
        "thresholds": switch_points(),
        "evpi": round(evpi(p_em), 2),
        "checklist": checklist,
        "rationale": (
            f"P(rescue needed or hospital surge) = {p_em:.0%}. At that "
            f"probability the maximum-expected-utility action is '{action}' "
            f"(EU {table[0]['expected_utility']:.1f} versus "
            f"{runner_up['expected_utility']:.1f} for '{runner_up['action']}')."
        ),
    }

    if population:
        exposed = int(population * EXPOSURE_FRACTION)
        out["exposed_population"] = exposed
        out["expected_people_needing_help"] = int(exposed * p_em)
        out["evpi_scaled"] = round(out["evpi"] * exposed / 100_000, 2)

    if out["close_call"]:
        out["rationale"] += (" This is a close call - the two actions are within "
                             "3 cost units, so a human officer should sign off.")
    return out


if __name__ == "__main__":
    print("Optimal action as a function of P(emergency):")
    for a, (lo, hi) in switch_points().items():
        print(f"  {a:>9}: {lo:.3f} -> {hi:.3f}")
    print("\nEVPI at a few probabilities:")
    for p in (0.05, 0.2, 0.35, 0.5, 0.8):
        print(f"  P={p:.2f}  best={best_action(p):>9}  EVPI={evpi(p):6.2f}")
