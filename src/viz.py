"""
Figures for the slide deck. Every chart here is generated from the artifacts
produced by learn.py / evaluate.py, so nothing on a slide is drawn by hand.

Run:  python -m src.viz
Output: artifacts/*.png

Owner: Rohit Vardhan M
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, HAZARDS, METRICS_JSON
from .decision import ACTIONS, expected_utility, switch_points
from .networks import EDGES, build_expert_network, describe_model, node_layers

PALETTE = {
    "context": "#8c8c8c", "sensors": "#2c7fb8", "latent": "#7b3294",
    "hazards": "#d7301f", "consequences": "#e08214",
}


# ----------------------------------------------------------------------------
SHORT_LABEL = {
    "SeaSurfaceTemp": "Sea\nSurface\nTemp",
    "PressureDrop": "Pressure\nDrop",
    "SoilMoisture": "Soil\nMoisture",
    "RiverLevel": "River\nLevel",
    "RoadBlocked": "Road\nBlocked",
    "RescueNeeded": "Rescue\nNeeded",
    "HospitalDemand": "Hospital\nDemand",
    "Urbanisation": "Urbani-\nsation",
    "Temperature": "Temper-\nature",
}


def draw_dag(path=None):
    """Layered drawing of the network - the single most important slide."""
    import networkx as nx

    layers = node_layers()
    order = ["context", "sensors", "latent", "hazards", "consequences"]
    pos, colour = {}, {}
    for col, group in enumerate(order):
        nodes = layers[group]
        for row, n in enumerate(nodes):
            pos[n] = (col * 2.6, -(row - (len(nodes) - 1) / 2) * 1.35)
            colour[n] = PALETTE[group]

    g = nx.DiGraph(EDGES)
    fig, ax = plt.subplots(figsize=(14, 7.6))
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, arrowsize=13,
                           edge_color="#9e9e9e", width=1.2,
                           connectionstyle="arc3,rad=0.06",
                           node_size=4200)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=4200,
                           node_color=[colour[n] for n in g.nodes()],
                           edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=7,
                            labels={n: SHORT_LABEL.get(n, n) for n in g.nodes()},
                            font_color="white", font_weight="bold")

    for col, group in enumerate(order):
        ax.text(col * 2.6, 4.6, group.upper(), ha="center", fontsize=10,
                color=PALETTE[group], fontweight="bold")

    info = describe_model(build_expert_network())
    ax.set_ylim(-4.6, 5.6)
    ax.set_title(f"Multi-hazard Bayesian network: {info['nodes']} nodes, "
                 f"{info['edges']} edges, {info['free_parameters']} free "
                 f"parameters", fontsize=13, pad=22)
    ax.axis("off")
    fig.tight_layout()
    out = path or ARTIFACT_DIR / "dag.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
def plot_missing_sensor(path=None):
    metrics = json.loads(METRICS_JSON.read_text())
    study = metrics["missing_sensor_study"]
    rates = list(study.keys())
    models = list(study[rates[0]].keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, hazard in zip(axes, ["Flood", "Landslide"]):
        for m in models:
            ys = [study[r][m][hazard] for r in rates]
            is_bn = m.startswith("BN")
            ax.plot(rates, ys, marker="o",
                    linewidth=2.8 if is_bn else 1.6,
                    linestyle="-" if is_bn else "--",
                    label=m)
        ax.set_title(f"{hazard}: ROC-AUC vs fraction of sensors missing")
        ax.set_xlabel("sensor readings missing at prediction time")
        ax.set_ylabel("ROC-AUC on held-out test set")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("The Bayesian network marginalises what it did not see; "
                 "the baselines have to impute it", fontsize=11)
    fig.tight_layout()
    out = path or ARTIFACT_DIR / "missing_sensors.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
def plot_calibration(path=None):
    metrics = json.loads(METRICS_JSON.read_text())
    rel = metrics["reliability"]

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    for h in HAZARDS:
        pts = rel.get(h, [])
        if not pts:
            continue
        ax.plot([p["predicted"] for p in pts], [p["observed"] for p in pts],
                marker="o", label=h)
    ax.set_xlabel("probability we forecast")
    ax.set_ylabel("fraction that actually happened")
    ax.set_title("Reliability of the shipped model\n(BN, expert-prior Dirichlet CPTs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = path or ARTIFACT_DIR / "calibration.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
def plot_model_comparison(path=None):
    df = pd.read_csv(ARTIFACT_DIR / "model_comparison.csv")
    pivot = df.pivot(index="model", columns="hazard", values="auc")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(pivot))
    width = 0.2
    for k, h in enumerate(HAZARDS):
        ax.bar(x + k * width, pivot[h], width, label=h)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(pivot.index, rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Held-out ROC-AUC. The BN sees only 2-3 bins per sensor; "
                 "the baselines see the raw numbers.")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = path or ARTIFACT_DIR / "model_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
def plot_decision_thresholds(path=None):
    ps = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for a in ACTIONS:
        ax.plot(ps, [expected_utility(a, p) for p in ps], linewidth=2, label=a)

    for a, (lo, hi) in switch_points().items():
        ax.axvline(lo, color="grey", linestyle=":", linewidth=1)
        ax.text(lo + 0.005, ax.get_ylim()[0] + 6, f"{a} from {lo:.2f}",
                rotation=90, fontsize=7.5, color="#444")

    ax.set_xlabel("P(emergency | evidence)   =   P(rescue needed OR hospital surge)")
    ax.set_ylabel("expected utility (cost units per 100k exposed)")
    ax.set_title("Alert thresholds are derived, not typed in:\n"
                 "the upper envelope of these four lines is the policy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = path or ARTIFACT_DIR / "decision_thresholds.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
def plot_cpt_shift(path=None):
    df = pd.read_csv(ARTIFACT_DIR / "cpt_shift.csv")
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.barh(df["node"][::-1], df["mean_kl"][::-1], color="#2c7fb8")
    ax.set_xlabel("mean KL( learned || elicited ) per parent configuration, nats")
    ax.set_title("Where the data disagreed with us\n"
                 "(small bars = our elicited prior was already right)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = path or ARTIFACT_DIR / "cpt_shift.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_console_panel(scenario: str = "monsoon_depression", path=None):
    """
    The district ranking exactly as the console computes it, rendered as a figure
    so the deck can show real model output next to the live screenshot.
    """
    from .decision import recommend
    from .discretize import evidence_from_reading
    from .engine import get_engine
    from .scenarios import SCENARIOS, build

    engine = get_engine()
    rows = []
    for r in build(scenario):
        ev = evidence_from_reading(r)
        hz = engine.hazard_probabilities(ev)
        plan = recommend(engine, ev, population=r["population"])
        top = max(hz, key=hz.get)
        rows.append((r["district"], top, hz[top],
                     plan.get("action_label", plan["action"]),
                     r.get("river_level_frac") is None))
    rows.sort(key=lambda t: t[2])

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    for i, (name, top, p, action, offline) in enumerate(rows):
        c = "#d7301f" if p >= 0.75 else "#e8912a" if p >= 0.40 else "#2f9e5e"
        ax.barh(i, p, color=c, alpha=0.35 if offline else 0.85,
                edgecolor=c, linewidth=2 if offline else 0)
        ax.text(-0.02, i, name, ha="right", va="center", fontsize=9)
        ax.text(p + 0.015, i, f"{top} {p:.0%}  →  {action}"
                + ("   (gauge offline)" if offline else ""),
                va="center", fontsize=8.5, color="#333")

    ax.set_xlim(0, 1.55)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("P(top hazard | today's readings)")
    ax.set_title(f"Console output - {SCENARIOS[scenario]}\n"
                 "faded bars with an outline = a sensor is not reporting",
                 fontsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = path or ARTIFACT_DIR / f"console_{scenario}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rainfall_prior_comparison(path=None):
    """
    The headline result of connecting real data: our simulated model thought 43 % of
    monsoon days were 'heavy'. Across 35,072 real district-days it is 1.1 % -
    a factor of about 40, computed from the two fitted CPT columns.
    """
    import json
    report_file = ARTIFACT_DIR / "weather_relearn_report.json"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    sim = report["monsoon_rainfall_simulated"]
    obs = report["monsoon_rainfall_observed"]
    bands = list(sim)

    x = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - 0.19, [sim[b] for b in bands], 0.38,
           label="our simulator (what we presented last time)", color="#c0392b")
    ax.bar(x + 0.19, [obs[b] for b in bands], 0.38,
           label=f"{report['rows_total']:,} real district-days", color="#1f77b4")
    for i, b in enumerate(bands):
        ax.text(i - 0.19, sim[b] + 0.015, f"{sim[b]:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.19, obs[b] + 0.015, f"{obs[b]:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Rainfall = {b}" for b in bands])
    ax.set_ylabel("P(Rainfall | Season = Monsoon)")
    ax.set_ylim(0, 1.0)
    # compute the ratio rather than hard-coding it: an earlier version said
    # 36x while the fitted CPT columns actually give about 40x
    ratio = sim["Heavy"] / max(obs["Heavy"], 1e-9)
    ax.set_title("The simulator believed monsoon days were "
                 f"{ratio:.0f}x more often 'heavy'" + chr(10) +
                 "than they actually are "
                 "(IMD heavy = over 64.5 mm in 24 h)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = path or ARTIFACT_DIR / "rainfall_prior_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_filter_trajectory(path=None):
    """Three heavy-rain days then four dry ones: accumulation and recession."""
    from .temporal import HydrologyFilter

    heavy = {"Low": 0.02, "Moderate": 0.10, "Heavy": 0.88}
    dry = {"Low": 0.92, "Moderate": 0.06, "Heavy": 0.02}
    schedule = [heavy] * 3 + [dry] * 5

    f = HydrologyFilter("figure")
    soil, river = [f.marginals()["SoilMoisture"]["High"]], \
                  [f.marginals()["RiverLevel"]["High"]]
    for rain in schedule:
        f.step(rain)
        soil.append(f.marginals()["SoilMoisture"]["High"])
        river.append(f.marginals()["RiverLevel"]["High"])

    days = np.arange(len(soil))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.axvspan(0.5, 3.5, color="#2c7fb8", alpha=0.12, label="heavy rain")
    ax.plot(days, soil, marker="o", linewidth=2.4, label="P(soil saturated)",
            color="#7b3294")
    ax.plot(days, river, marker="s", linewidth=2.4, label="P(river above danger)",
            color="#d7301f")
    ax.set_xlabel("day of the stream")
    ax.set_ylabel("probability")
    ax.set_title("Recursive filtering gives the catchment a memory:\n"
                 "one heavy day does not saturate it, three do, and then it recedes",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = path or ARTIFACT_DIR / "filter_trajectory.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_soft_vs_hard(path=None):
    """
    What the soft treatment changes on the real feed: per district, the dominant
    hazard probability from soft evidence against what hard binning would have said.
    """
    from .stream import StreamProcessor

    proc = StreamProcessor("snapshot")
    records = proc.step_all(explain=True)
    rows = []
    for r in records:
        nv = r.naive_comparison or {}
        hard = nv.get("hard_hazards", {}).get(r.dominant_hazard)
        if hard is None:
            continue
        rows.append((r.district, hard, r.hazards[r.dominant_hazard],
                     r.dominant_hazard))
    rows.sort(key=lambda t: t[2])

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    y = np.arange(len(rows))
    for i, (name, hard, soft, hazard) in enumerate(rows):
        ax.plot([hard, soft], [i, i], color="#9e9e9e", linewidth=1.4, zorder=1)
        ax.scatter(hard, i, s=52, color="#c0392b", zorder=2,
                   label="hard binning" if i == 0 else None)
        ax.scatter(soft, i, s=52, color="#1f77b4", zorder=2,
                   label="soft evidence (ours)" if i == 0 else None)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  ({h})" for n, _, _, h in rows], fontsize=8.5)
    ax.set_xlabel("P(dominant hazard) from the same live readings")
    ax.set_title("Same feed, two ways of entering it.\n"
                 "Hard binning asserts a bin; soft evidence keeps the uncertainty "
                 "the reading actually has.", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = path or ARTIFACT_DIR / "soft_vs_hard.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    for fn in (draw_dag, plot_model_comparison, plot_missing_sensor,
               plot_calibration, plot_decision_thresholds, plot_cpt_shift,
               plot_console_panel, plot_rainfall_prior_comparison,
               plot_filter_trajectory, plot_soft_vs_hard):
        try:
            print("wrote", fn())
        except FileNotFoundError as exc:
            print(f"skipped {fn.__name__}: {exc} (run src.learn and src.evaluate first)")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
def plot_missing_sensor_simple(path=None):
    """
    Plain-English version of plot_missing_sensor, for the simple deck.

    Same numbers, but no jargon on the chart itself: an audience that has to
    decode "ROC-AUC" and "marginalise" while also following the argument will
    follow neither.
    """
    metrics = json.loads(METRICS_JSON.read_text())
    study = metrics["missing_sensor_study"]
    rates = list(study.keys())

    nice = {
        "BN-Bayes (expert prior)": "Our system",
        "LogisticRegression": "Ordinary model A",
        "RandomForest": "Ordinary model B",
        "GradientBoosting": "Ordinary model C",
    }

    # explicit colours: letting matplotlib cycle gave "Our system" and one of the
    # ordinary models the same blue, which defeats the whole point of the chart
    colour = {"BN-Bayes (expert prior)": "#1f77b4",
              "LogisticRegression": "#e8912a",
              "RandomForest": "#2f9e5e",
              "GradientBoosting": "#c0392b"}

    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    for model in study[rates[0]]:
        ys = [study[r][model]["Flood"] for r in rates]
        ours = model.startswith("BN")
        ax.plot(rates, ys, marker="o",
                linewidth=3.6 if ours else 1.8,
                linestyle="-" if ours else "--",
                color=colour.get(model),
                label=nice.get(model, model))

    ax.set_xlabel("How many of the sensor readings were missing", fontsize=12)
    ax.set_ylabel("How good the prediction was\n(higher is better)", fontsize=12)
    ax.set_title("When sensors stop working, our system keeps working",
                 fontsize=14, pad=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = path or ARTIFACT_DIR / "missing_sensors_simple.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out
