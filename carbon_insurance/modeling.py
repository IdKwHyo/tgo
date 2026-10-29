from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .domain import Settings, digest, now, write_json
from .ingestion import records, save_table


def metrics(actual, prediction) -> dict:
    return {"mae": float(mean_absolute_error(actual, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(actual, prediction))), "n": len(actual)}


def annual_series(settings: Settings) -> tuple[list[dict], list[dict]]:
    rows = sorted([r for r in records(settings.data_dir, "aggregates") if r["kind"] == "standard_issuance"],
                  key=lambda r: r["fiscal_year_ce"])
    if not rows:
        return [], []
    # No publication-as-of guarantee for the latest reported fiscal year. Do not treat it as a full-year outcome.
    usable = rows[:-1]
    years = [r["fiscal_year_ce"] for r in usable]
    if years and years != list(range(min(years), max(years) + 1)):
        return [], rows
    return usable, rows[-1:]


def baseline_prediction(name: str, history: np.ndarray) -> float:
    if name in ("naive", "seasonal_naive_annual"):
        return float(history[-1])
    if name == "expanding_mean":
        return float(history.mean())
    if name == "expanding_median":
        return float(np.median(history))
    if name == "drift":
        return float(max(0, history[-1] + (history[-1] - history[0]) / (len(history) - 1)))
    raise ValueError(f"Unknown baseline {name}")


def evaluate_baselines(settings: Settings) -> dict:
    usable, excluded = annual_series(settings)
    if len(usable) < settings.baseline_min_observations:
        return {"name": "annual_baselines", "status": "INSUFFICIENT_DATA", "observations": len(usable),
                "excluded_latest_fiscal_year": excluded, "numerically_validated": False}
    values = np.array([r["tonnes"] for r in usable], dtype=float)
    prediction_rows, results = [], []
    for name in ("naive", "seasonal_naive_annual", "expanding_mean", "expanding_median", "drift"):
        actual, predicted = [], []
        for origin in range(6, len(values)):
            estimate = baseline_prediction(name, values[:origin])
            actual.append(values[origin])
            predicted.append(estimate)
            prediction_rows.append({"model": name, "fiscal_year_be": usable[origin]["fiscal_year_be"],
                                    "actual_tonnes": values[origin], "prediction_tonnes": estimate,
                                    "history_ends_fiscal_year_be": usable[origin - 1]["fiscal_year_be"],
                                    "source_id": usable[origin]["source_id"]})
        results.append({"model": name, **metrics(actual, predicted)})
    best = min(results, key=lambda r: (r["mae"], r["rmse"]))
    save_table(settings.data_dir, "baseline_backtests", prediction_rows)
    return {"name": "annual_baselines", "status": "EXPLORATORY_ONLY", "observations": len(values),
            "target": "Annual certified issuance for the mixed Category 13 population, NOT removal-only insurance losses",
            "validation": "Expanding-window one-step backtest; candidate ranking is diagnostic and is not an independent final test",
            "results": results, "diagnostic_best_model": best["model"],
            "excluded_latest_fiscal_year": excluded, "numerically_validated": False,
            "reasons": ["Very few backtest origins", "No independent test after comparing baseline candidates",
                        "Aggregate target is mixed activity and not an insurance exposure"],
            "annual_seasonality": "Seasonal naive with annual period 1 equals naive; no subannual seasonality inferred"}


def temporal_project_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.project_id.duplicated().any():
        raise ValueError("First-issuance training labels must have unique project groups")
    ordered = frame.sort_values(["approval_date", "project_id"]).reset_index(drop=True)
    # Keep equal approval dates together. No same-day labels on both sides of a split.
    dates = sorted(ordered.approval_date.unique())
    if len(dates) < 5:
        raise ValueError("Insufficient distinct approval dates for chronological validation")
    train_date = dates[max(1, int(len(dates) * 0.6))]
    test_date = dates[max(2, int(len(dates) * 0.8))]
    train = ordered[ordered.approval_date < train_date]
    validation = ordered[(ordered.approval_date >= train_date) & (ordered.approval_date < test_date)]
    test = ordered[ordered.approval_date >= test_date]
    if any(len(f) == 0 for f in (train, validation, test)):
        raise ValueError("Empty chronological project split")
    return train, validation, test


def paired_improvement_interval(actual, candidate, baseline) -> list[float] | None:
    improvement = np.abs(np.asarray(actual) - baseline) - np.abs(np.asarray(actual) - candidate)
    if len(improvement) < 20 or np.ptp(improvement) == 0:
        return None
    interval = bootstrap((improvement,), np.mean, confidence_level=0.95,
                         method="percentile", n_resamples=2000, rng=np.random.default_rng(19)).confidence_interval
    return [float(interval.low), float(interval.high)]


def train_catboost(settings: Settings, model_dir: Path) -> dict:
    candidates = records(settings.data_dir, "training_candidates")
    result = {"name": "catboost", "target": "First-cycle annualized certified tonnes conditional on issuance",
              "observations": len(candidates), "minimum_projects": settings.catboost_min_projects,
              "status": "INSUFFICIENT_DATA", "numerically_validated": False,
              "historical_features_verified": False, "selected_features": [],
              "reasons": ["No period-matched expected-delivery or insured-loss labels identified"],
              "forbidden_interpretations": ["Premium prediction", "Default frequency", "Reversal probability"]}
    if len(candidates) < settings.catboost_min_projects:
        result["reasons"].append("Not enough independent absorption-project issuance labels")
        return result
    frame = pd.DataFrame(candidates)
    try:
        train, validation, test = temporal_project_split(frame)
    except ValueError as exc:
        result["reasons"].append(str(exc))
        return result
    if len(train) < 20 or len(validation) < 5 or len(test) < settings.catboost_min_test_projects:
        result["reasons"].append("Not enough projects in chronological train/validation/test partitions")
        result["split_sizes"] = {"train": len(train), "validation": len(validation), "test": len(test)}
        return result
    from catboost import CatBoostRegressor, Pool
    features = ["expected_annual_tonnes", "monitoring_days", "methodology", "size", "program"]
    categorical = ["methodology", "size", "program"]
    for subset in (train, validation, test):
        subset.loc[:, "expected_annual_tonnes"] = subset.expected_annual_tonnes.fillna(-1)
    target = "target_annualized_certified_tonnes"
    model = CatBoostRegressor(iterations=250, depth=3, learning_rate=0.04, loss_function="RMSE",
                              random_seed=19, thread_count=1, verbose=False, allow_writing_files=False)
    model.fit(train[features], train[target], cat_features=categorical,
              eval_set=(validation[features], validation[target]), early_stopping_rounds=25)
    # Baseline choice uses validation, never final test labels.
    training_median = float(train[target].median())
    validation_baselines = {"training_median": np.full(len(validation), training_median),
                            "registered_annual_expectation": validation.expected_annual_tonnes.clip(lower=0).to_numpy()}
    best_baseline = min(validation_baselines, key=lambda n: metrics(validation[target], validation_baselines[n])["mae"])
    baseline_test = np.full(len(test), training_median) if best_baseline == "training_median" else test.expected_annual_tonnes.clip(lower=0).to_numpy()
    prediction = np.maximum(0, model.predict(test[features]))
    actual = test[target].to_numpy()
    candidate_metrics, baseline_metrics = metrics(actual, prediction), metrics(actual, baseline_test)
    interval = paired_improvement_interval(actual, prediction, baseline_test)
    validation_errors = np.abs(validation[target].to_numpy() - np.maximum(0, model.predict(validation[features])))
    # Split conformal-style residual interval is reported only with an adequate calibration partition.
    radius = float(np.quantile(validation_errors, min(1, np.ceil((len(validation_errors) + 1) * 0.8) / len(validation_errors)), method="higher")) if len(validation_errors) >= 20 else None
    coverage = float(np.mean(np.abs(actual - prediction) <= radius)) if radius is not None else None
    relative = (baseline_metrics["mae"] - candidate_metrics["mae"]) / max(baseline_metrics["mae"], 1e-12)
    numerical = (relative >= settings.minimum_relative_improvement and interval is not None and interval[0] > 0
                 and coverage is not None and coverage >= 0.75)
    historical = all(r["historical_features_verified"] for r in candidates)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "catboost.cbm"))
    write_json(model_dir / "catboost_card.json", {"features": features, "target": target,
               "training_project_ids": train.project_id.tolist(), "validation_project_ids": validation.project_id.tolist(),
               "test_project_ids": test.project_id.tolist(), "historical_features_verified": historical})
    test[["project_id", "approval_date", target, *features]].to_parquet(model_dir / "catboost_test_inputs.parquet", index=False)
    explanation = permutation_importance(model, test[features], actual, n_repeats=10,
                                         random_state=19, scoring="neg_mean_absolute_error")
    shap = model.get_feature_importance(Pool(test[features], cat_features=categorical), type="ShapValues")
    save_table(settings.data_dir, "catboost_explanations", [{"feature": name,
              "permutation_mae_increase": float(explanation.importances_mean[i]),
              "permutation_std": float(explanation.importances_std[i]),
              "mean_absolute_shap": float(np.abs(shap[:, i]).mean())} for i, name in enumerate(features)])
    save_table(settings.data_dir, "catboost_test_predictions", [{"project_id": str(pid), "actual": float(a),
              "prediction": float(p), "baseline": float(b)} for pid, a, p, b in zip(test.project_id, actual, prediction, baseline_test)])
    result.update(status="EVALUATED", selected_features=features, metrics=candidate_metrics,
                  baseline={"name": best_baseline, **baseline_metrics}, paired_mae_improvement_interval=interval,
                  interval_80_radius=radius, interval_test_coverage=coverage,
                  numerically_validated=numerical, historical_features_verified=historical,
                  model_path=str((model_dir / "catboost.cbm").relative_to(settings.data_dir)),
                  split_sizes={"train": len(train), "validation": len(validation), "test": len(test)},
                  feature_importance_limit="Predictive association, not causal importance or insurance pricing weight")
    if not historical:
        result["reasons"].append("Current registry fields lack historical point-in-time validation; model cannot be promoted")
    return result


def train_lstm(settings: Settings, model_dir: Path) -> dict:
    usable, excluded = annual_series(settings)
    result = {"name": "lstm", "status": "INSUFFICIENT_DATA", "observations": len(usable),
              "minimum_observations": settings.lstm_min_observations, "numerically_validated": False,
              "excluded_latest_fiscal_year": excluded,
              "target": "Mixed Category 13 annual issuance; outside removal-only insurance prediction scope",
              "scope_eligible": False}
    if len(usable) < settings.lstm_min_observations:
        result["reason"] = "Not enough genuinely temporal observations; no LSTM training or forecast executed"
        return result
    from .lstm import fit_temporal_candidate
    values = np.array([r["tonnes"] for r in usable], dtype=float)
    path = model_dir / "lstm.pt"
    fitted = fit_temporal_candidate(values, settings.lstm_lookback, path)
    origins = fitted["test_origins"]
    candidate = fitted["test_predictions"]
    actual = fitted["test_actual"]
    names = ("naive", "seasonal_naive_annual", "expanding_mean", "expanding_median", "drift")
    validation_origins = np.arange(fitted["train_end"], fitted["validation_end"])
    validation_errors = {name: metrics(values[validation_origins],
                         [baseline_prediction(name, values[:i]) for i in validation_origins])["mae"] for name in names}
    chosen = min(names, key=lambda name: validation_errors[name])
    baseline = np.array([baseline_prediction(chosen, values[:i]) for i in origins])
    matched_baselines = [{"name": name, **metrics(actual, [baseline_prediction(name, values[:i]) for i in origins])} for name in names]
    interval = paired_improvement_interval(actual, candidate, baseline)
    validation_residuals = np.abs(fitted["validation_actual"] - fitted["validation_predictions"])
    radius = float(np.quantile(validation_residuals, min(1, np.ceil((len(validation_residuals) + 1) * 0.8) / len(validation_residuals)), method="higher")) if len(validation_residuals) >= 20 else None
    coverage = float(np.mean(np.abs(actual - candidate) <= radius)) if radius is not None else None
    candidate_metrics, baseline_metrics = metrics(actual, candidate), metrics(actual, baseline)
    relative = (baseline_metrics["mae"] - candidate_metrics["mae"]) / max(baseline_metrics["mae"], 1e-12)
    numerical = relative >= settings.minimum_relative_improvement and interval is not None and interval[0] > 0 and coverage is not None and coverage >= 0.75
    save_table(settings.data_dir, "lstm_test_predictions", [{"fiscal_year_be": usable[int(i)]["fiscal_year_be"],
               "actual": float(a), "prediction": float(p), "selected_baseline": float(b)}
               for i, a, p, b in zip(origins, actual, candidate, baseline)])
    result.update(status="EVALUATED_RESEARCH_ONLY", metrics=candidate_metrics, baseline={"name": chosen, **baseline_metrics},
                  matched_test_baselines=matched_baselines, baseline_choice="Validation only, all baseline tests use identical LSTM origins",
                  paired_mae_improvement_interval=interval, epochs=fitted["epochs"],
                  interval_80_radius=radius, interval_test_coverage=coverage, numerically_validated=numerical,
                  model_path=str(path.relative_to(settings.data_dir)),
                  reason="This aggregate can be studied but cannot predict an absorption-only insured loss")
    return result


def train(settings: Settings) -> dict:
    ingestion = json.loads((settings.data_dir / "outputs/ingestion.json").read_text())
    version = "candidate_" + now().replace(":", "").replace(".", "_") + "_" + ingestion["data_version"][:10]
    model_dir = settings.data_dir / "models" / version
    model_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for name, factory in (("catboost", lambda: train_catboost(settings, model_dir)),
                          ("lstm", lambda: train_lstm(settings, model_dir)),
                          ("annual_baselines", lambda: evaluate_baselines(settings))):
        try:
            candidates.append(factory())
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            candidates.append({"name": name, "status": "FAILED_CANDIDATE", "numerically_validated": False,
                               "reason": f"{type(exc).__name__}: {str(exc)[:500]}"})
    evaluation = {"version": version, "data_version": ingestion["data_version"], "timestamp": now(),
                  "training_fingerprint": training_fingerprint(settings),
                  "candidates": candidates, "insurance_loss_model": "INSUFFICIENT_DATA",
                  "calibration_status": "No validated insured-loss frequency/severity model",
                  "feature_importance_status": "AVAILABLE" if any(c.get("status") == "EVALUATED" for c in candidates) else "NOT_ESTIMABLE_WITH_CURRENT_LABELS"}
    write_json(model_dir / "evaluation.json", evaluation)
    write_json(settings.data_dir / "outputs/model_evaluation.json", evaluation)
    return evaluation


def training_fingerprint(settings: Settings) -> str:
    implementation = {name: digest((Path(__file__).parent / name).read_bytes())
                      for name in ("features.py", "modeling.py", "lstm.py", "parsing.py")}
    configuration = {k: v for k, v in settings.model_dump(mode="json").items()
                     if k.startswith(("catboost_", "lstm_", "baseline_", "minimum_relative_"))}
    labels = sorted(records(settings.data_dir, "training_candidates"), key=lambda r: r["project_id"])
    return digest({"implementation": implementation, "configuration": configuration, "project_labels": labels})


def predict_catboost(model_path: Path, rows: pd.DataFrame) -> np.ndarray:
    from catboost import CatBoostRegressor
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return np.maximum(0, model.predict(rows[model.feature_names_]))
