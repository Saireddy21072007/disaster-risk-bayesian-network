# Real-time sensor data: what we changed and what methods we used

This document answers the zeroth-review feedback directly:

> "It is mentioned real-time sensor data, but you are choosing the existing
> preprocessed data? Also, what methods are you incorporating to process the
> real-time data feed? It will be interesting to me. Proceed, all the best."

Both halves of that were fair. At the time our CPTs came from our own simulator and
the dashboard ran on a seeded scenario generator; there was no live feed anywhere in
the system, and "real-time sensor data" was a claim we had not earned. This is what
we did about it.

---

## Part 1 — the data is now genuinely live

Three public services, no API key, pulled per district on every pass
(`src/feeds.py`):

| Service | What it gives us | Used for |
|---|---|---|
| `api.open-meteo.com` | rainfall, temperature, humidity, wind, surface pressure | the weather layer, from **three independent models** |
| `flood-api.open-meteo.com` | GloFAS daily river discharge (m³/s) | `RiverLevel` |
| `marine-api.open-meteo.com` | sea surface temperature | `SeaSurfaceTemp` |

Three things there are modelling decisions rather than plumbing.

**We aggregate to the variable we actually model.** `Rainfall` is an IMD 24-hour
band, so we sum the trailing 24 hourly values rather than reading an
instantaneous rate. `Temperature` is a 24-hour maximum (the heat-wave criterion is
about the daily max), humidity a 24-hour mean, wind a 24-hour max, and
`PressureDrop` is `p(t−24h) − p(t)`. Reading the "current" field for any of these
would have silently modelled a different quantity.

**We take three forecast models, not one.** GFS, ECMWF-IFS and ICON are requested
separately for every variable that carries them. They disagree, and that
disagreement is information — see Part 3.

**The river threshold is calibrated, not invented.** GloFAS reports discharge;
our node is a fraction of the local danger level, which we do not have per
district. So for each district we pull six years of daily discharge and take the
**median annual maximum** — the 2-year return period, the standard bankfull proxy
in hydrology — and express today's discharge as a fraction of it. For Idukki that
is 220.5 m³/s, from six annual maxima. Cached for a month.

### And the CPTs are no longer ours either

`src/observed.py` pulls **35,072 real district-days** (16 districts, 2019-01-01 to
2024-12-31) from the archive versions of the same three services and re-estimates
every CPT whose node and whose parents are all directly observable:

| Node | Rows used | Mean shift from our elicitation |
|---|---|---|
| Season (prior) | 35,072 | 0.032 |
| Rainfall \| Season | 35,072 | **0.275** |
| Temperature \| Season | 35,072 | 0.150 |
| SeaSurfaceTemp \| Season | 6,921 | **0.351** |
| Humidity \| Rainfall | 35,072 | 0.262 |
| PressureDrop \| SeaSurfaceTemp | 6,921 | 0.186 |
| WindSpeed \| PressureDrop | 35,056 | 0.118 |

The estimator is the same one we already used — a Dirichlet posterior with the
elicited CPT as the prior mean, ESS 50 — just pointed at real observations.

**What we deliberately did not relearn**, and say so on the slide: `SoilMoisture`
and `RiverLevel` sit behind a latent parent, and `Flood`, `Landslide`, `Cyclone`,
`Heatwave` and the whole consequence layer need labelled events per district-day
that no public table provides. Those CPTs are still elicited and refined on
simulated data. Fitting them here with no labels would be inventing results.

So there are two models on purpose:

* `artifacts/learned_cpts.json` — every layer from the simulator. Used for the
  held-out **synthetic** evaluation, where the test labels come from the same
  generative process, so the weather layer has to match it.
* `artifacts/hybrid_cpts.json` — weather layer from real observations, hazard
  layer unchanged. Used by the **live** pipeline.

### The finding that made this worth doing

| P(Rainfall = … \| Season = Monsoon) | Low | Moderate | Heavy |
|---|---|---|---|
| our simulated model | 0.074 | 0.499 | **0.427** |
| 35,072 real district-days | 0.837 | 0.152 | **0.011** |

Our simulator drew monsoon rainfall from a gamma with a mean near 66 mm/day and
therefore believed **43% of monsoon days are "heavy" (>64.5 mm)**. The real figure
for these districts is **1.1%** — a 36× overestimate. Real mean daily rainfall is
5.70 mm overall and 8.16 mm in the monsoon.

That single number invalidated the weather layer of the model we presented at the
last review, and we would never have found it without connecting real data.

### A second correction the data forced on us

With one all-India season rule (June–September = monsoon), the real data came out
**wetter in "winter" (4.45 mm/day) than in "summer" (3.96 mm/day)** — not a climate
that exists. The cause is the north-east / retreating monsoon: the Tamil Nadu and
Andhra coasts get most of their rain in October–December. East-coast districts now
have October–December as their monsoon (`observed.season_of`), and the ordering
became Monsoon 8.16 > Winter 4.12 > Summer 3.78. `current_season` in the live path
uses the identical rule, because if training labelled October as monsoon and
inference labelled it winter, every east-coast reading would be scored against the
wrong column of the rainfall CPT.

### Real gaps, not simulated ones

Chennai has **no GloFAS river reach at its centroid** — all six years of discharge
come back null. That is a genuine hole in the data, and the network handles it the
way it handles any missing variable: it marginalises. Our earlier demo switched
three gauges off by hand to show graceful degradation; we no longer need to.

We also **removed** a soil-moisture feed we had already built. The forecast API's
`soil_moisture_9_to_27cm` and the archive's `soil_moisture_7_to_28cm_mean` are
different products on different scales — for the same Coimbatore cell one said
0.153 m³/m³ and the other 0.33 for comparable days. Training on one and inferring
from the other would have been a silent bug, and inventing a rescaling factor to
paper over it would have been worse. Soil moisture is now latent and the filter
infers it from rainfall history, which is also what a district with no soil probes
would have to do.

---

## Part 2 — recursive Bayesian filtering (`src/temporal.py`)

A flood is not caused by today's rain; it is caused by three days of rain landing on
ground that was already wet. A single snapshot cannot express that, and a stream is
precisely the setting where it becomes unacceptable — the point of a stream is that
yesterday's frame informs today's. We listed "the model is static" as a limitation
at the last review; this removes it.

`SoilMoisture` and `RiverLevel` are lifted into a two-slice dynamic Bayesian
network (9 joint states) and tracked with the standard forward recursion:

```
predict   b'(s_t)  =  Σ_{s_{t-1}}  T(s_t | s_{t-1}, rain_t) · b(s_{t-1})
update    b(s_t)   =  normalise[ L(z_t | s_t) · b'(s_t) ]
```

* `T` is built from the same ordered-logit family as our static CPTs, so it is
  monotone by construction: more rain never dries the soil, and the river recedes
  when the rain stops. Soil persistence carries weight 3.2 — that weight *is* the
  memory.
* Rainfall is itself uncertain, so we do not condition on one bin. We marginalise:
  `T_eff(s|s') = Σ_r P(Rainfall = r | evidence_t) · T(s|s',r)`.
* `L` is the soft-evidence likelihood from any direct hydrology reading. When
  nothing arrives it is flat and the filter **coasts on its prediction**, which is
  the correct behaviour for a dead gauge.

Measured behaviour (`tests/test_realtime.py`): three consecutive heavy-rain days
raise P(river high) monotonically and the third day is worth more than 1.8× the
first; four dry days cut it to under half its peak; and a catchment that was wet
four days ago is still wetter than a dry one after an identical dry day.

### Handing the result back without double counting

The static network has its own opinion about the hydrology, formed from today's rain
through `Rainfall → SoilMoisture → RiverLevel`. Multiplying the filtered belief into
it would count today's rainfall twice. So we enter the filter's belief as virtual
evidence with the network's own prediction divided out:

```
λ(s) = b_filter(s) / P_network(s | evidence_t)
```

which makes the network's posterior over the hydrology come out **equal** to the
filter's belief — replacing its memoryless guess rather than compounding with it.
`test_virtual_evidence_reproduces_the_filter_belief` asserts this to 1e-9; in
practice it holds to about 1e-16.

### Keeping the two elicitations consistent

The transition model and the static CPTs are two separate elicitations of the same
physical system, and left alone they disagreed: run the chain forward under average
rainfall and it settled somewhere wetter than the static marginal, so a district
with **no readings at all** drifted to a flood probability of 0.31 against a base
rate of 0.16 — a pure artefact. We caught it the first time we ran the fleet on a
snapshot with districts missing.

The fix is two fitted scalars: shifts of the soil and river cut-points chosen so the
chain's stationary distribution under climatological rainfall matches the static
marginal (soil +0.667, river +2.259; max joint discrepancy 0.038). One shared scalar
was not enough — it lined the river up and left the soil 0.26 out. The dynamics are
untouched; only the resting point moves. A test asserts a silent district stays
within 0.06 of the base rate over 30 frames.

---

## Part 3 — soft evidence, because a real reading is not a bin (`src/measurement.py`)

The naive pipeline takes the number, looks up its bin, and asserts that bin as hard
evidence. Every pass of the real feed shows why that is wrong:

* ECMWF said 15.0 mm; the IMD light/moderate boundary is 15.6 mm. Hard binning turns
  a 0.6 mm difference into a categorical claim.
* For the same district and hour, GFS said 7.4 mm, ECMWF 15.0 mm and ICON 1.5 mm.
  Hard binning has to pick one and discard the disagreement.
* The GloFAS reading was 17 hours old. Hard evidence treats it exactly like one from
  30 seconds ago.

So a reading becomes a **likelihood vector**, entered through Pearl's indicator
construction. Three mechanisms:

**1. Gaussian bin likelihood.** A reading `v` with observation error `σ` induces
`λ(k) = Φ((e_k − v)/σ) − Φ((e_{k−1} − v)/σ)` over the bins. A value on an edge splits
its mass. Live example: soil saturation 0.70 against bins at 0.40/0.70 came out
`Medium 0.48 / High 0.51` instead of asserting one of them.

**2. σ is measured from the ensemble, not chosen by us.**
`σ_eff = sqrt(σ_instrument² + s²)` where `s` is the spread of the three forecast
models — Gaussian ensemble dressing, the standard NWP post-processing move. When the
models agree the evidence is sharp; when they disagree the likelihood flattens, the
posterior widens and the confidence label drops, **with no threshold anywhere**.
This is the part we are most pleased with: the reliability of the feed is measured
from the feed itself, at that moment.

**3. Ageing shrinks evidence towards useless.** `λ` is blended towards uniform with
weight `exp(−age/τ)`, with τ per variable — 18 h for a rainfall accumulation, 3 h for
a pressure drop, 48 h for a sea surface temperature. At weight 0 the indicator leaf
is uninformative and entering it changes nothing.

The construction is exact, and two tests prove it: with flat likelihoods the
augmented network matches the core network to 1e-9, and with a one-hot likelihood it
reproduces ordinary hard conditioning to 1e-6.

---

## Part 4 — the stream-specific parts (`src/stream.py`)

### A Bayesian surprise gate — and the mistake we made building it

A stuck gauge does not announce itself, it just sends a number. So before accepting
a reading we ask the network how plausible it is:
`surprisal = −log₂ P(reading | all other evidence)`.

Our first version discounted anything above four bits. Pointed at real data, that
went badly wrong: the simulated CPTs believed monsoon days were far wetter than they
are, so the gate flagged **correct** readings as implausible — 1.8 mm of rain in
Coimbatore, agreed on by three independent models, was scored 5.16 bits surprising
and had two thirds of its evidence thrown away. The gate was assuming the model is
right and the sensor is wrong.

A high surprisal has two explanations, and we now separate them using something the
model cannot fake — whether the independent sources agree with each other:

| Situation | Verdict | What we do |
|---|---|---|
| sources agree, model surprised | **model-surprise** | keep the reading at full strength, flag the model for review |
| single source, model surprised | suspect | discount to 0.35 trust, log it |
| sources disagree, model surprised | suspect | discount; the feed is a mess |

A broken sensor cannot make GFS, ECMWF and ICON agree, so agreement is genuine
evidence that the reading is real. With the relearned weather layer the gate now
fires **zero** times on a normal pass, which is the right answer — the earlier
firing was the model being wrong, not the sensors.

### Asymmetric alert hysteresis

Recomputing every fifteen minutes lets the recommended action oscillate across a
decision boundary, and an evacuation order that flickers is worse than either state.
We escalate on the first frame that justifies it and stand down only after three
consecutive frames agree. The asymmetry is the utility table talking: a missed
emergency costs roughly eight times a false alarm, so being slow to stand down is
cheap and being slow to escalate is not. The dashboard shows the raw recommendation
next to the held level, so nothing is hidden.

### Mutual-information-driven polling

Every feed costs a request and every broken gauge sits in a repair queue. For each
quantity we compute what a perfect reading would be worth right now,
`I(Hazard ; X | evidence)`, and poll in that order. The district where the answer is
already obvious is left alone; the one on a decision boundary gets attention. On the
Idukki pass above, the river gauge was worth 0.13 bits and humidity 0.02 — so the
gauge gets chased first.

### Audit log

Every pass appends to `artifacts/alert_log.jsonl`: timestamp, district, alert level,
raw recommendation, hazard probabilities, and the readings with their status and
sources. We listed an audit trail as a deployment requirement last time; it is now
implemented, and `/api/live/audit` serves the tail of it.

---

## Running it

```bash
python -m src.feeds Idukki Chennai Madurai
```

```bash
python -m src.observed
```

```bash
python -m src.stream
```

```bash
python -m src.stream --snapshot
```

In the dashboard, the **Live feeds** toggle switches from the scenario generator to
the real pipeline; "replay last pull" reuses the saved snapshot so a demo works with
the wifi off, and "fetch now" hits the APIs.

`GET /api/live/provenance` returns, in machine-readable form, exactly which nodes
were relearned from real observations and which are still elicited.

## What is still not real

Honest list, unchanged in spirit from the last review but much shorter:

* **The hazard layer.** `Flood`, `Landslide`, `Cyclone`, `Heatwave` and the
  consequence nodes still come from elicitation plus simulated labels. Getting real
  labels means joining EM-DAT and state disaster reports to district-days, which is
  the next item on the weekly plan.
* **Soil moisture** is inferred, not measured, for the scale reason above.
* **The utility table** is still ours. Worth noting that against real August base
  rates it recommends "Prepare" for more districts than an operational service
  would; it was calibrated on simulated base rates and needs redoing against real
  event frequencies once we have them.
* **GloFAS is a global reanalysis**, not a CWC gauge. It has a real lag (our readings
  were around 17–20 hours old) and a coarse reach, which is exactly why the
  staleness decay exists.
