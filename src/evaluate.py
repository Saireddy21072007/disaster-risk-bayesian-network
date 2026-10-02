"""
Held-out evaluation. This is the part that decides whether the project is any
good, so we tried to make the comparison hard on ourselves rather than easy.

Three questions:

  Q1  Does the Bayesian network predict as well as a black-box classifier?
      The BN is handicapped on purpose - it only ever sees each reading squeezed
      into 2 or 3 bins, while logistic regression and the tree ensembles get the
      raw floating-point numbers. If we come close anyway, the explanation layer
      is basically free.

  Q2  What happens when sensors go missing? This is the real question for Indian
      district data, where a river gauge is offline more often than not. A BN
      answers by marginalising out what it did not see. A sklearn model cannot,
      so it has to impute - and imputing invents data. We sweep the missing rate
      from 0 % to 60 % and watch both curves.

  Q3  Are the probabilities honest (calibrated)? A 70 % warning must be wrong
      about 30 % of the time, otherwise "70 %" is just a mood. We report the
      Brier score and a reliability curve.

Owner: whole group, run by Sai Vandith
"""

from __future__ import annotations

import json
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ARTIFACT_DIR, HAZARDS, METRICS_JSON, SEED
from .engine import RiskEngine
from .learn import (cpds_from_json, fit_bayesian, fit_bdeu, fit_mle,
                    load_discrete, save_model)
from .networks import build_network_from_cpds, expert_cpds

# what the BN is allowed to condition on. SoilMoisture is deliberately absent:
# nobody has a soil-moisture probe in every panchayat, so we treat it as latent
# and let the network infer it.
BN_INPUTS = ["Season", "Slope", "Urbanisation", "Rainfall", "Temperature",
             "Humidity", "SeaSurfaceTemp", "PressureDrop", "WindSpeed",
             "RiverLevel"]

# the sensors that can realistically drop out (terrain and calendar cannot)
DROPPABLE = ["Rainfall", "Temperature", "Humidity", "SeaSurfaceTemp",
             "PressureDrop", "WindSpeed", "RiverLevel"]

NUMERIC_FEATURES = ["rainfall_mm", "temperature_c", "humidity_pct", "sst_c",
                    "pressure_drop_hpa", "wind_kmph", "river_level_frac",
                    "slope_deg"]

MISSING_RATES = [0.0, 0.2, 0.4, 0.6]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    single_class = len(np.unique(y_true)) < 2
    return {
        "auc": None if single_class else round(float(roc_auc_score(y_true, p)), 4),
        "brier": round(float(brier_score_loss(y_true, p)), 4),
        "log_loss": round(float(log_loss(y_true, p, labels=[0, 1])), 4),
        "accuracy@0.5": round(float(((p >= 0.5).astype(int) == y_true).mean()), 4),
        "base_rate": round(float(y_true.mean()), 4),
    }


def _bn_predict(engine: RiskEngine, disc: pd.DataFrame,
                keep_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Multi-hazard prediction for a whole test frame.

    Two tricks make this fast enough to run in a demo: we ask for all four
    hazards in ONE variable-elimination query per row, and we memoise on the
    evidence tuple, because after discretisation thousands of rows collapse onto
    the same evidence pattern.
    """
    cache: Dict[tuple, Dict[str, float]] = {}
    preds = {h: np.zeros(len(disc)) for h in HAZARDS}

    cols = [c for c in BN_INPUTS if c in disc.columns]
    values = disc[cols].to_numpy(dtype=object)

    for i in range(len(disc)):
        evidence = {}
        for j, col in enumerate(cols):
            if keep_mask is not None and col in DROPPABLE:
                k = DROPPABLE.index(col)
                if not keep_mask[i, k]:
                    continue
            v = values[i, j]
            if isinstance(v, str):
                evidence[col] = v

        key = tuple(sorted(evidence.items()))
        if key not in cache:
            factor = engine.joint(HAZARDS, evidence)
            cache[key] = {
                h: float(factor.marginalize([x for x in HAZARDS if x != h],
                                            inplace=False)
                         .get_value(**{h: "Yes"}))
                for h in HAZARDS
            }
        for h in HAZARDS:
            preds[h][i] = cache[key][h]
    return preds


def _sk_models():
    return {
        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000)),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   random_state=SEED, n_jobs=-1)),
        "GradientBoosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(random_state=SEED)),
    }


def _design_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    X = raw[NUMERIC_FEATURES].copy()
    X["season_monsoon"] = (raw["season"] == "Monsoon").astype(int)
    X["season_summer"] = (raw["season"] == "Summer").astype(int)
    X["urban_high"] = (raw["urbanisation"] == "High").astype(int)
    return X


def _mask_numeric(X: pd.DataFrame, keep_mask: np.ndarray) -> pd.DataFrame:
    """Apply the same dropout to the baselines' features, then let the pipeline
    impute - which is the whole point of the comparison."""
    X = X.copy()
    col_for = {
        "Rainfall": "rainfall_mm", "Temperature": "temperature_c",
        "Humidity": "humidity_pct", "SeaSurfaceTemp": "sst_c",
        "PressureDrop": "pressure_drop_hpa", "WindSpeed": "wind_kmph",
        "RiverLevel": "river_level_frac",
    }
    for k, node in enumerate(DROPPABLE):
        col = col_for[node]
        X.loc[~keep_mask[:, k], col] = np.nan
    return X


# ----------------------------------------------------------------------------
# Q1 + Q3: accuracy and calibration with every sensor present
# ----------------------------------------------------------------------------
def compare_models(train_disc, test_disc, train_raw, test_raw) -> dict:
    variants = {
        "BN-expert (no data)": build_network_from_cpds(expert_cpds()),
        "BN-MLE": build_network_from_cpds(fit_mle(train_disc)),
        "BN-BDeu": build_network_from_cpds(fit_bdeu(train_disc)),
        "BN-Bayes (expert prior)": build_network_from_cpds(fit_bayesian(train_disc)),
    }

    results: Dict[str, dict] = {}
    for name, model in variants.items():
        engine = RiskEngine(model)
        preds = _bn_predict(engine, test_disc)
        results[name] = {h: _metrics(test_raw[h.lower()].to_numpy(), preds[h])
                         for h in HAZARDS}

    Xtr, Xte = _design_matrix(train_raw), _design_matrix(test_raw)
    for name, proto in _sk_models().items():
        per_hazard = {}
        for h in HAZARDS:
            from sklearn.base import clone
            clf = clone(proto)
            clf.fit(Xtr, train_raw[h.lower()])
            p = clf.predict_proba(Xte)[:, 1]
            per_hazard[h] = _metrics(test_raw[h.lower()].to_numpy(), p)
        results[name] = per_hazard
    return results


def reliability(y_true, p, bins: int = 10) -> List[dict]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if sel.sum() < 10:
            continue
        out.append({"bin_mid": round((lo + hi) / 2, 3),
                    "predicted": round(float(p[sel].mean()), 4),
                    "observed": round(float(y_true[sel].mean()), 4),
                    "n": int(sel.sum())})
    return out


# ----------------------------------------------------------------------------
# Q2: sensor dropout
# ----------------------------------------------------------------------------
def missing_sensor_study(train_disc, test_disc, train_raw, test_raw) -> dict:
    engine = RiskEngine(build_network_from_cpds(fit_bayesian(train_disc)))
    Xtr, Xte = _design_matrix(train_raw), _design_matrix(test_raw)

    from sklearn.base import clone
    fitted = {}
    for name, proto in _sk_models().items():
        fitted[name] = {}
        for h in HAZARDS:
            clf = clone(proto)
            clf.fit(Xtr, train_raw[h.lower()])
            fitted[name][h] = clf

    rng = np.random.default_rng(SEED)
    study: Dict[str, dict] = {}
    for rate in MISSING_RATES:
        keep = rng.random((len(test_raw), len(DROPPABLE))) >= rate
        row = {}

        bn_preds = _bn_predict(engine, test_disc, keep_mask=keep)
        row["BN-Bayes (expert prior)"] = {
            h: _metrics(test_raw[h.lower()].to_numpy(), bn_preds[h])["auc"]
            for h in HAZARDS}

        Xte_masked = _mask_numeric(Xte, keep)
        for name, per_h in fitted.items():
            row[name] = {
                h: _metrics(test_raw[h.lower()].to_numpy(),
                            per_h[h].predict_proba(Xte_masked)[:, 1])["auc"]
                for h in HAZARDS}
        study[f"{rate:.0%}"] = row
    return study


# ----------------------------------------------------------------------------
def main():
    train_disc, test_disc, train_raw, test_raw = load_discrete()
    print(f"train {len(train_disc)} / test {len(test_disc)} records\n")

    print("Q1+Q3  accuracy and calibration, all sensors present")
    table = compare_models(train_disc, test_disc, train_raw, test_raw)
    rows = []
    for model, per_h in table.items():
        for h, m in per_h.items():
            rows.append({"model": model, "hazard": h, **m})
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    pivot = df.pivot(index="model", columns="hazard", values="auc")
    pivot["mean AUC"] = pivot.mean(axis=1).round(4)
    print(pivot.sort_values("mean AUC", ascending=False).to_string())
    print()
    brier = df.pivot(index="model", columns="hazard", values="brier")
    brier["mean Brier"] = brier.mean(axis=1).round(4)
    print(brier.sort_values("mean Brier").to_string())

    print("\nQ2  AUC as sensors go missing (flood)")
    study = missing_sensor_study(train_disc, test_disc, train_raw, test_raw)
    miss_rows = []
    for rate, models in study.items():
        for model, per_h in models.items():
            miss_rows.append({"missing_rate": rate, "model": model, **per_h})
    mdf = pd.DataFrame(miss_rows)
    mdf.to_csv(ARTIFACT_DIR / "missing_sensor_study.csv", index=False)
    print(mdf.pivot(index="model", columns="missing_rate", values="Flood").to_string())

    # reliability curve for the model we ship
    engine = RiskEngine(build_network_from_cpds(fit_bayesian(train_disc)))
    preds = _bn_predict(engine, test_disc)
    rel = {h: reliability(test_raw[h.lower()].to_numpy(), preds[h]) for h in HAZARDS}

    METRICS_JSON.write_text(json.dumps({
        "n_train": len(train_disc), "n_test": len(test_disc),
        "comparison": table,
        "missing_sensor_study": study,
        "reliability": rel,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {METRICS_JSON}")


if __name__ == "__main__":
    main()
