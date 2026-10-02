# Weekly plan to the final review

The project guidelines say a group aiming at a deployable product or a publication
should show a weekly plan. This is ours. Weeks 1–3 are the risky ones and we know
it — everything after them assumes the real-data join actually works.

| Week | Task | Owner | Done when |
|---|---|---|---|
| 1 | Pull NASA POWER daily weather (rainfall, T, RH, wind) for 6 districts, 2019–2024, and join with CWC gauge levels | Sai Vandith | one real CSV in `data/` with the same column names `simulate.py` produces, so nothing downstream changes |
| 2 | Label events: EM-DAT plus state disaster management reports and news archives for flood / landslide dates | Jithin Reddy | verified event dates for 3 monsoon seasons, with a written labelling rule |
| 3 | Refit on real data, re-run the whole evaluation | Sai Reddy | slides 13–15 regenerated from the real CSV; a written note on what changed |
| 4 | Dynamic BN over 3-day slices, so accumulated rainfall is in the model | Sai Reddy + Rohit | AUC comparison table: static vs dynamic, on the same test split |
| 5 | Sensitivity analysis: sweep the utility table and the Dirichlet ESS | Rohit | thresholds reported as ranges instead of point values |
| 6 | Take the checklists and the cost table to a district office for review | all four | written feedback from at least one officer, incorporated |
| 7 | Package it: Dockerfile, API docs, 5-minute demo video | Jithin + Sai Vandith | someone outside the group can run it from the README alone |
| 8 | Write-up as a short paper (target: a regional AI-for-disaster workshop) | all four | draft ready for internal review |

## Risks we have already thought about

**The real-data join may not produce enough positive events.** Six districts over
five monsoon seasons might only give us 20–40 labelled flood events. If that
happens, the Dirichlet prior becomes the main thing carrying the model, which is
fine — it is exactly the regime a Bayesian approach is for — but we will have to
present it as such rather than claiming a data-driven fit.

**Gauge data may not be downloadable in bulk.** CWC publishes readings but not
always in an archive-friendly form. Fallback: use reservoir inflow levels from
state irrigation departments, which are published daily.

**A dynamic BN multiplies the parameter count.** If week 4 blows up, we will keep
the static model and add engineered accumulation features (3-day rainfall total as
its own node) instead, which is a smaller change and still answers the same
criticism.

## What we would need to actually deploy this

Not part of the course project, but we should be able to answer it:

* a real-time feed (IMD's API or a state-run sensor network) instead of scenarios;
* per-district flood-plain and slope-zone population layers, to replace the flat
  12% exposure fraction;
* a utility table signed off by the state disaster management authority;
* an audit log — every warning issued, with the evidence and posterior that
  produced it, so decisions can be reviewed after the event.
