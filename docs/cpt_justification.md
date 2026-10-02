# Where every number in the CPTs came from

Written so that if we are asked "why 4.6 and not 4.0" we have an answer that is
not "it looked right". Owner: Rohit Vardhan M, reviewed by all four of us.

We never typed a full CPT by hand. Instead every table is generated from a small
set of interpretable numbers using one of three canonical forms
(`src/networks.py`):

* **ordered logit** for ordinal children — one weight per parent, plus cut-points;
* **noisy-OR** for children where any cause can independently trigger the effect;
* **explicit small tables** where the parent is genuinely categorical.

Below, "severity" means the parent's state index scaled to `[0, 1]`, so `Low = 0`
and `Heavy = 1` for a three-state rainfall node.

---

## Discretisation cut-offs

These are not ours; they are the operational bands used in India, which is the
whole reason we chose them.

| Variable | Cut-offs | Source |
|---|---|---|
| Rainfall | 15.6 mm, 64.5 mm | IMD 24-hour rainfall classification (light / moderate / heavy) |
| Temperature | 30 °C, 40 °C | IMD heat-wave criterion for the plains is 40 °C |
| Wind speed | 20, 40, 62 kmph | 62 kmph is the IMD lower bound for a "cyclonic storm" |
| Sea surface temp | 28 °C | the conventional cyclogenesis threshold |
| Pressure drop | 4 hPa / 24 h | the fall associated with a developing depression |
| River level | 0.70, 0.95 of danger level | CWC publishes gauge readings against a danger level; warning level sits near 0.9 |
| Slope | 10°, 25° | GSI landslide susceptibility work treats ~25° as the steep class |

Four wind bins rather than three: with a single 30–62 kmph bucket, an ordinary
monsoon depression came out looking 60% like a cyclone, because the bin averaged
over everything up to storm strength.

---

## Root priors

| Node | Prior | Reasoning |
|---|---|---|
| Season | Winter 0.33, Summer 0.30, Monsoon 0.37 | the south-west monsoon plus the retreating monsoon cover roughly four and a half months of the year |
| Slope | Flat 0.45, Moderate 0.35, Steep 0.20 | the terrain mix of the 16 districts in `data/districts.csv` |
| Urbanisation | Low 0.65, High 0.35 | same source; "High" is our proxy for drainage that cannot cope |

---

## Weather layer

`Rainfall | Season`, `Temperature | Season`, `SeaSurfaceTemp | Season` and
`Humidity | Rainfall` are explicit 3×3 tables, because season is not an ordinal
quantity — summer is not "more" than winter. The numbers are our reading of
climatological normals for coastal south India: a dry winter, pre-monsoon showers
in summer, and 40% of monsoon days in the heavy band.

`PressureDrop | SeaSurfaceTemp` — 8% chance of a >4 hPa fall over a normal sea,
30% over a sea above 28 °C. `WindSpeed | PressureDrop` follows: without a
pressure fall, gale-force wind is a 2% event; with one, 28%.

Note the causal direction here: sea warms → pressure falls → wind picks up. We
deliberately did **not** draw `WindSpeed → PressureDrop`, even though wind is
easier to measure, because the explanation layer reads the arrows as causes.

---

## Latent physical state

`SoilMoisture` — ordered logit, weights Rainfall 3.6, Humidity 1.4, cut-points
0.9 and 3.0. Rainfall dominates; humidity matters because it controls how fast
the soil dries between spells. Check: heavy rain and humid air gives
P(saturated) = 0.88; light rain and dry air gives P(low) = 0.71.

`RiverLevel` — ordered logit, weights Rainfall 3.0, SoilMoisture 2.8, cut-points
1.2 and 3.6. The two weights are close on purpose: a catchment that is already
saturated sends nearly all new rain into the river, which is the mechanism behind
most sudden gauge rises. Check: heavy rain on saturated soil gives
P(above danger level) = 0.90; heavy rain on dry soil only 0.35.

---

## Hazards

`Flood` — logit, weights RiverLevel 4.6, SoilMoisture 2.0, Urbanisation 1.1,
cut-point 4.6. The gauge dominates by design: it is the variable a control room
actually trusts. Urbanisation is small but real — it is standing in for drainage
capacity. Check: river at danger level with saturated soil in a built-up district
gives 0.96; everything benign gives 0.01.

`Landslide` — logit, weights Rainfall 1.6, SoilMoisture 1.8, Slope 1.0, plus an
**interaction** term Slope × SoilMoisture of 4.6, cut-point 5.8.

> This is the one we got wrong first. Both the elicited table and the simulator's
> hazard trigger were purely additive in slope, so slope raised the risk on its
> own — end to end the system gave the Nilgiris a 23% landslide probability on a
> dry January day (it is 4% now). Slope is a *predisposing* factor and
> water is the *trigger*; the interaction term says so. After the change,
> heavy rain on saturated soil gives 0.08 on flat ground and 0.96 on a steep
> slope, and a dry steep slope gives 0.008. The elicitation error against the
> data dropped from 0.077 to 0.020 mean absolute difference per cell.

`Cyclone` — logit, weights PressureDrop 2.6, WindSpeed 4.4, SeaSurfaceTemp 1.6,
cut-point 5.6. Wind carries the most weight because it is the defining
observable, but it is not sufficient: gale-force wind with no pressure fall and a
normal sea gives 0.23, which is roughly right for a squall line rather than a
cyclone. That "not sufficient" behaviour is the explaining-away structure doing
its job.

`Heatwave` — logit, weights Temperature 6.0, **Humidity −1.8**, cut-point 4.4.
The negative weight is the interesting part: dry air is what makes a heat wave
dangerous in the plains, and it is also why coastal Kerala does not get one even
at 39 °C. This is the only protective weight in the model.

---

## Consequence layer

`RoadBlocked` and `RescueNeeded` are noisy-OR, which is the natural form for
"any of these three hazards can independently cause it".

| Cause | Road blocked | Rescue needed |
|---|---|---|
| Flood | 0.75 | 0.70 |
| Landslide | 0.85 | 0.80 |
| Cyclone | 0.55 | 0.65 |
| leak | 0.02 | 0.01 |

Landslide has the highest link probability for road blockage because a slip
blocks a ghat road completely, whereas a flood often leaves one lane. These were
elicited by the four of us after reading post-event reports from the 2018 Kerala
floods, and they are the numbers the data agreed with most closely (mean KL
0.002 after learning) — which we take as evidence the elicitation method works.

`HospitalDemand` — ordered logit, weights RescueNeeded 3.4, Heatwave 3.8,
cut-points 1.0 and 3.6. The heat-wave weight is deliberately larger than the
rescue weight: heat stroke cases arrive at the PHC without anyone being rescued
first. Our first version used 2.2 here, which meant a heat wave could never on
its own reach the top demand band — and that in turn made the decision layer
answer "Monitor" on a 42 °C day. See `tests/test_inference_and_pipeline.py::
test_heatwave_alone_can_trigger_a_response`.

---

## What happens to these numbers once we have data

They become the **prior mean** of a Dirichlet posterior, with an equivalent sample
size of 50 pseudo-observations per parent configuration:

```
alpha_ijk = 50 * P_expert(x_i = k | pa_i = j)
P_hat     = (alpha_ijk + N_ijk) / (alpha_ij. + N_ij.)
```

Well-populated cells end up following the data; starved cells (gale-force wind
appears in well under 1% of records) fall back on the elicitation. `artifacts/
cpt_shift.csv` lists, per node, how far the data moved us — which is also a
scorecard for how good our elicitation was.
