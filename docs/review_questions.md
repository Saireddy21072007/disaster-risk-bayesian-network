# Questions we expect, and how we answer them

We wrote this before the review by going round the table and each of us trying to
break the other three. Anyone in the group should be able to answer any of these,
not just the person who wrote the module.

---

## About the model

**Q. Write down the joint distribution your network represents.**

P(x₁ … x₁₈) = ∏ᵢ P(xᵢ | parents(xᵢ)). Concretely the first few factors are
P(Season) · P(Slope) · P(Urbanisation) · P(Rainfall | Season) ·
P(Temperature | Season) · … · P(HospitalDemand | RescueNeeded, Heatwave).

**Q. How many numbers would the full joint need, and how many do you store?**

13,436,928 entries in the full joint (product of all the cardinalities) against
164 free parameters in the CPTs. About 82,000× fewer. That compression is not
about disk space — it is why 9,000 training rows are enough to estimate anything
at all.

**Q. Which is your largest CPT?**

Landslide, given Rainfall (3) × SoilMoisture (3) × Slope (3) = 27 parent
configurations, 27 free parameters since the child is binary.

**Q. Give me a d-separation example from your own graph.**

Rainfall and Flood are dependent with no evidence. Observing RiverLevel and
SoilMoisture d-separates them — every path from Rainfall to Flood goes through
one of those two, so they block it. That is not a claim, it is asserted in
`tests/test_network.py::test_dseparation_claims` and the API reports it at
`/api/model`.

**Q. Where is there a collider in your network?**

RoadBlocked has Flood, Landslide and Cyclone as parents. Flood and Landslide are
dependent anyway because they share Rainfall as a cause, but conditioning on
RoadBlocked adds the explaining-away effect: learning the road is blocked and
that a landslide occurred lowers the probability that a flood also occurred.

**Q. Why exact inference and not sampling?**

The network is tiny. Variable elimination on 18 discrete variables with
cardinality 2–4 is milliseconds. Belief propagation is also wired in, and a test
asserts the two agree to 1e-6 — if they ever disagree, we are misusing one of
them.

---

## About the parameters

**Q. Where did your CPT numbers come from?**

Two stages. First, elicitation through canonical forms — ordered logit for ordinal
children, noisy-OR for "any cause triggers it" children — so each table comes from
a handful of interpretable weights rather than 27 hand-typed cells. Second, a
Dirichlet posterior with those elicited tables as the prior mean and an equivalent
sample size of 50. `docs/cpt_justification.md` justifies every weight.

**Q. Why is that Bayesian update legitimate?**

Dirichlet is conjugate to the multinomial, so the posterior is Dirichlet and its
mean is (pseudo-counts + counts) / (total pseudo-counts + total counts). With
ESS = 50 the prior is worth 50 imaginary observations per parent configuration:
cells with thousands of real rows follow the data, cells with four rows fall back
on the expert.

**Q. Isn't ESS = 50 arbitrary?**

Yes, it is a choice. It is one line in `src/learn.py` and sweeping it is on the
weekly plan for week 5. What we can say now is the direction of the effect: larger
ESS pulls the sparse cells further toward the elicitation, and the sparse cells
are precisely the extreme-weather ones where we have most reason to distrust four
noisy rows.

**Q. Did the data agree with your elicitation?**

Mostly. `artifacts/cpt_shift.csv` ranks the nodes by KL divergence between learned
and elicited. The noisy-OR consequence layer barely moved (mean KL 0.002) — that
elicitation was right. The nodes that moved most were Humidity | Rainfall and
RiverLevel | Rainfall, SoilMoisture, where our elicited curves were too gentle.

**Q. You ran structure learning. Why didn't you use the result?**

Hill-Climb with BIC recovers 86% of our edges, and adds shortcuts — mostly
Rainfall straight to the hazards, bypassing SoilMoisture. Those shortcuts improve
the score because rainfall is a good proxy, but they destroy the causal reading,
and the explanation layer depends on it: with a shortcut, "why is flood risk
high" can no longer answer "because the ground is already saturated". Also,
direction is not identifiable from observational data — only the Markov
equivalence class is — which is why we count undirected edge recovery.

---

## About the data

**Q. Is this real data?**

The weather layer is. We pull live from three public services (Open-Meteo weather,
GloFAS river discharge, marine SST) with no API key, and we re-estimated every CPT
whose node and parents are observable from **35,072 real district-days** across our
16 districts, 2019–2024.

The hazard layer is not, and we say so on the slide, in the README and in the module
docstring. `Flood`, `Landslide`, `Cyclone`, `Heatwave` and the consequence nodes
still come from elicitation plus simulated labels, because no public table gives a
verified event label per district-day. That join is weeks 1–3 of the plan.

**Q. So what exactly did the real data change?**

Two things, and both were embarrassing in a useful way.

First, our simulator believed **43% of monsoon days were "heavy" rainfall (>64.5 mm);
the real figure for these districts is 1.1%** — a 36× overestimate sitting inside the
model we presented at the zeroth review. Real mean daily rainfall is 5.70 mm, 8.16 mm
in the monsoon.

Second, our season calendar was wrong. With one all-India rule the data came out
wetter in "winter" (4.45 mm/day) than in "summer" (3.96 mm) — not a climate that
exists. The cause is the retreating monsoon: the Tamil Nadu and Andhra coasts get
their rain in October–December. East-coast districts now use that calendar, and the
live path uses the identical rule so the CPT columns line up.

Neither is a coding bug. Both are domain errors that only real observations could
expose.

**Q. Then isn't your synthetic evaluation circular?**

Two things prevent that. The simulator works on continuous physical quantities and
never touches the Bayesian network — the BN only ever sees a 2–4 bin discretised
view, which is the same handicap it faces on real gauge data. And the simulator's
*structure* matches our DAG but none of its *functional forms* do (gamma rainfall,
exponential saturation, logistic triggers, noisy-OR consequences), so the learned
CPTs are a genuine estimation result.

**Q. Why keep two models?**

Because each is valid for a different job, and mixing them would be wrong.
`learned_cpts.json` has every layer from the simulator and is used for the held-out
synthetic evaluation — the test labels come from that generative process, so the
weather layer has to match it. `hybrid_cpts.json` has the real weather layer and is
what the live pipeline serves. `/api/live/provenance` returns which nodes came from
which, in machine-readable form.

---

## About the real-time pipeline

**Q. You claimed "real-time sensor data" last time and had none. What is there now?**

Three public services pulled per district on every pass: Open-Meteo forecast
(rainfall, temperature, humidity, wind, surface pressure) from **three independent
numerical weather models** — GFS, ECMWF-IFS and ICON — plus GloFAS daily river
discharge and marine sea-surface temperature. No API key.

**Q. What is actually hard about ingesting an API?**

Three things that are modelling decisions, not plumbing.

We aggregate to the variable we model: `Rainfall` is an IMD 24-hour band so we sum
the trailing 24 hourly values; `Temperature` is a 24-hour max because the heat-wave
criterion is about the daily max; `PressureDrop` is p(t−24h) − p(t). Reading the
"current" field would have silently modelled a different quantity.

We calibrate the river threshold rather than invent it. GloFAS gives discharge in
m³/s; our node is a fraction of danger level, which we do not have per district. So
we take six years of daily discharge and use the **median annual maximum** — the
2-year return period, the standard bankfull proxy in hydrology. Idukki: 220.5 m³/s.

And we treat failure as an outcome. Chennai has **no GloFAS river reach at its
centroid** — six years of nulls. That is a real hole in the data, and the network
marginalises it, which is what it was already good at.

**Q. What methods do you use to process the feed?**

Four, all of them course material.

1. **Soft (virtual) evidence.** A reading becomes a likelihood over bins,
   `λ(k) = Φ((e_k − v)/σ) − Φ((e_{k−1} − v)/σ)`, entered via Pearl's indicator
   construction — a binary leaf per node with `P(leaf = yes | X = s) ∝ λ(s)`.
2. **σ measured from the ensemble.** `σ_eff = sqrt(σ_instrument² + s²)` with `s` the
   spread of the three forecast models — Gaussian ensemble dressing.
3. **Recursive Bayesian filtering** over the hydrology, a two-slice DBN with the
   standard predict/update recursion.
4. **A Bayesian surprise gate**, plus asymmetric alert hysteresis and
   mutual-information-driven polling.

**Q. Why not just take the bin the number falls in?**

Because on the live feed that throws away real information every pass. ECMWF reported
15.0 mm against a bin edge at 15.6 mm — hard binning turns a 0.6 mm difference into a
categorical claim. And for the same district and hour, GFS said 7.4, ECMWF 15.0 and
ICON 1.5 mm; hard binning has to pick one. A live soil reading of 0.70 against bins
at 0.40/0.70 came out Medium 0.48 / High 0.51 rather than asserting a bin.

**Q. How do you know the soft-evidence plumbing is correct?**

Two tests. With every likelihood flat, the augmented network reproduces the core
network to 1e-9 — so the leaves leak nothing. With a one-hot likelihood it reproduces
ordinary hard conditioning to 1e-6 — so the construction reduces to the thing it
generalises.

**Q. Where does the reliability number come from? Did you just pick it?**

For the forecast variables, no — and this is the part we like best. It is measured
from the ensemble spread at that moment. When the three models agree, σ is small and
the evidence is sharp; when they disagree, σ grows, the likelihood flattens, the
posterior widens and the confidence label drops, with no threshold anywhere. The
instrument floors (3 mm for rainfall, 1 °C for temperature, and so on) are elicited
and documented, but the spread usually dominates.

**Q. Why a filter? Isn't the BN enough?**

A flood is not caused by today's rain; it is caused by three days of rain landing on
already-wet ground. A single snapshot cannot represent that — we listed "the model is
static" as a limitation last time, and a stream is exactly where it becomes
unacceptable. `SoilMoisture` and `RiverLevel` are now tracked as a 9-state joint with
`predict: b'(s) = Σ T(s|s',rain) b(s')` and `update: b(s) ∝ L(z|s) b'(s)`. Measured:
three heavy days raise P(river high) monotonically, the third worth more than 1.8× the
first, and four dry days cut it below half its peak.

**Q. If both the filter and the network use rainfall, aren't you counting it twice?**

We would be, which is why we divide it out. The filter's belief is entered as virtual
evidence with the network's own prediction removed:
`λ(s) = b_filter(s) / P_network(s | evidence)`. That makes the network's posterior
over the hydrology come out *equal* to the filter's belief — replacing its memoryless
guess rather than compounding with it. A test asserts it; it holds to about 1e-16.

**Q. Your filter and your static CPTs are two different models of the same thing.
Do they agree?**

They did not, and it showed. Run the chain forward under average rainfall and it
settled wetter than the static marginal, so a district with **no readings at all**
drifted to a flood probability of 0.31 against a base rate of 0.16 — a pure artefact.
We caught it on the first fleet run over an incomplete snapshot. The fix is two fitted
scalars, shifts of the soil and river cut-points chosen so the chain's stationary
distribution matches the static marginal (+0.667 and +2.259; max joint discrepancy
0.038). One shared scalar was not enough. The dynamics are untouched — only the
resting point moves — and a test asserts a silent district stays within 0.06 of the
base rate over 30 frames.

**Q. Your gate decides a reading is implausible. What if the model is the one that
is wrong?**

That happened, and it is the best story we have. Version one discounted anything
above four bits of surprisal, which assumes the model is right and the sensor is
broken. Pointed at real data it declared reality implausible: 1.8 mm of rain in
Coimbatore, agreed on by three independent models, scored 5.16 bits and had two
thirds of its evidence discarded.

So we now separate the two hypotheses using something the model cannot fake — whether
the independent sources agree. Sources agree and the model is surprised → the *model*
is probably wrong, so keep the reading at full strength and flag the model. A single
source and a surprise → discount to 0.35 trust and log it. Sources disagree → the
feed is a mess, discount. A broken gauge cannot make GFS, ECMWF and ICON agree.

With the relearned weather layer the gate now fires zero times on a normal pass,
which is the right answer.

**Q. Why is the hysteresis asymmetric? Isn't that just a fudge?**

It is the utility table talking. A missed emergency costs roughly eight times a false
alarm, so being slow to stand down is cheap and being slow to escalate is not. We
escalate on the first frame that justifies it and stand down only after three
agreeing frames. The console shows the raw recommendation next to the held level, so
the smoothing is visible rather than hidden.

**Q. Does the soft treatment actually change the answers?**

Yes, and in both directions — which is the point: it is uncertainty being carried, not
a bias being added. We compute the comparison against hard binning on every pass and
plot it. What we cannot claim, without real hazard labels, is that soft evidence is
more *accurate*; only that it represents the reading honestly. We say that before
being asked.

**Q. Why did you remove the soil-moisture feed you had already built?**

Because the forecast API's `soil_moisture_9_to_27cm` and the archive's
`soil_moisture_7_to_28cm_mean` are different products on different scales — 0.153
against 0.33 for the same Coimbatore cell on comparable days. Training on one and
inferring from the other would have been a silent bug, and inventing a rescaling
factor to hide it would have been worse. Soil moisture is now latent and the filter
infers it from rainfall history, which is what a district with no soil probes would
have to do anyway.

**Q. What would change with real hazard labels?**

The CPT values in the hazard layer, and probably the utility calibration — against
real August base rates our table recommends "Prepare" for more districts than an
operational service would accept, because it was tuned on simulated base rates. The
two findings we lean on — the cost of discretisation and the robustness advantage
under missing sensors — are structural properties of the two model families, so we
expect those to survive.

---

## About the explanation

**Q. Why not LIME or SHAP?**

Because in a Bayesian network the explanation is already a query. We remove one
reading from the evidence set, re-run exact inference and report the shift in
log-odds — that is Good's weight of evidence, it is exact, and it is signed, so a
reading can also *lower* the risk. LIME and SHAP fit a second, approximate model
to explain the first one; we would be adding error, not removing it.

**Q. Your explanation says one reading "adds nothing". Is that a bug?**

No, it is d-separation showing up in the output. Once the river gauge is read,
rainfall carries no extra information about flood risk, because its whole effect
is mediated by the river level and soil moisture. We print that sentence rather
than hiding the row, because an officer needs to know the rainfall figure was
considered.

**Q. What is your confidence number?**

Half of it is 1 − posterior entropy in bits (how committed the model is), half is
sensor coverage (how many of the ten possible inputs actually arrived). The two
ingredients are standard; the 50/50 weighting is our design choice and we do not
pretend otherwise. The point is that a 60% flood probability from two sensors is
not the same claim as 60% from nine, and entropy alone cannot tell them apart.

**Q. How do you decide which sensor to ask for next?**

Mutual information: I(Hazard ; Sensor | evidence), computed as the current
posterior entropy minus the expected entropy after seeing that sensor, averaged
over the sensor's own predictive distribution. It is non-negative by construction
and we test that.

---

## About the decision layer

**Q. Your abstract said "evacuate above 80%". What happened?**

We changed our minds and that is on a slide. A fixed probability threshold ignores
what the response costs and how many people are exposed. So evacuation is a
decision node in a small influence diagram with a utility table, and the optimal
action is the one with maximum expected utility. The threshold then *comes out* of
the model — at P = 0.64 for our cost table — instead of being typed in. Change the
costs and it moves, which is the behaviour we want.

**Q. Where did the utility numbers come from?**

Us, and they are relative rather than rupees. What we defend is the method and the
ordering: evacuation is the most expensive action and also the one that leaves the
least residual harm; a false evacuation carries a real cost because people ignore
the next warning. Getting them signed off by a district officer is week 6 of the
plan.

**Q. What is EVPI and why do you compute it?**

Expected value of perfect information: the expected utility if we knew the outcome
in advance, minus the expected utility of the best action under our current
uncertainty. It is zero when the decision is obvious and largest when we are on
the fence. Scaled by exposed population it becomes a procurement ranking — which
river gauge to repair first.

**Q. Why does the decision hang on "rescue needed or hospital demand high"
rather than on the flood probability?**

Because a district runs one response, not one per hazard. RescueNeeded already
pools flood, landslide and cyclone. We had to add the hospital-demand term after
noticing that a 42 °C day came back as "Monitor" — heat waves reach people through
hospitals, not boats. It is a union of two events, so we compute it from the joint
posterior rather than by adding probabilities.

---

## Awkward questions

**Q. A plain logistic regression beats you. Why should anyone use this?**

With every sensor reporting, yes — by about 0.03 AUC, and that gap is the price of
discretisation. Three answers. First, at 60% of readings missing we are at 0.863
AUC and logistic regression is at 0.789, and missing readings are the normal case
for district gauge networks. Second, logistic regression cannot tell you why, and
an officer will not evacuate a taluk on an unexplained number. Third, the same
network answers questions the classifier was never trained for — P(soil saturated
| flood occurred), for instance.

**Q. What is genuinely new here? Flood prediction with a BN has been done.**

We are not claiming a new inference algorithm. What we have not seen combined in
one system is: multi-hazard with a shared consequence layer, explanation by
evidence retraction rather than a post-hoc surrogate, an explicit confidence
statement that separates model uncertainty from sensor coverage, value of
information driving a "measure this next" instruction, and a decision layer that
derives its own alert thresholds. The measured missing-sensor comparison against
imputation-based baselines is the piece we would write up.

**Q. Did you use AI tools?**

Yes, for scaffolding code and for drafting text, which the guidelines allow. The
modelling decisions are ours and we can defend each one — the landslide
interaction term, the four wind bins, the heat-wave weight in HospitalDemand and
the move away from a fixed threshold all came from looking at what the model said
and deciding it was wrong. The bugs listed in this document are ours too.

**Q. Show me something breaking.**

Ask for a dry January day in the Nilgiris (`calm_winter`, Nilgiris) and note the
landslide probability is now 4%. In our first version it was 22%, because the
elicited CPT was additive in slope. Then open
`tests/test_network.py::test_dry_steep_slope_is_safe_but_wet_steep_slope_is_not`,
which exists so it cannot come back.
