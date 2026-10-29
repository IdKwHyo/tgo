from __future__ import annotations

import json

from .domain import Settings, now, write_json


def compare_champion(settings: Settings, champion: dict, candidate: dict) -> dict:
    """Require new common held-out projects, unseen in any earlier champion partition."""
    if champion.get("task") != candidate.get("target") or champion.get("name") != "catboost" or candidate.get("name") != "catboost":
        return {"replace": False, "reason": "No comparable supported champion/challenger task"}
    old_path = settings.data_dir / champion.get("model_path", "missing")
    new_path = settings.data_dir / candidate.get("model_path", "missing")
    paths = [old_path, new_path, old_path.parent / "catboost_card.json",
             new_path.parent / "catboost_card.json", new_path.parent / "catboost_test_inputs.parquet"]
    if not all(p.is_file() for p in paths):
        return {"replace": False, "reason": "Previous champion retained; paired common-test artifacts unavailable"}
    import pandas as pd
    from .modeling import metrics, paired_improvement_interval, predict_catboost
    old_card = json.loads(paths[2].read_text())
    new_card = json.loads(paths[3].read_text())
    if old_card["target"] != new_card["target"] or not old_card["historical_features_verified"] or not new_card["historical_features_verified"]:
        return {"replace": False, "reason": "Target mismatch or unverified historical features"}
    excluded = set(old_card["training_project_ids"] + old_card["validation_project_ids"] + old_card["test_project_ids"])
    test = pd.read_parquet(paths[4])
    fresh = test[~test.project_id.isin(excluded)]
    if len(fresh) < 20:
        return {"replace": False, "reason": "Fewer than 20 new common-test projects unseen in all previous champion partitions"}
    actual = fresh[new_card["target"]].to_numpy()
    old = predict_catboost(old_path, fresh)
    new = predict_catboost(new_path, fresh)
    old_metrics, new_metrics = metrics(actual, old), metrics(actual, new)
    interval = paired_improvement_interval(actual, new, old)
    relative = (old_metrics["mae"] - new_metrics["mae"]) / max(old_metrics["mae"], 1e-12)
    accepted = relative >= settings.minimum_relative_improvement and interval is not None and interval[0] > 0
    return {"replace": accepted, "reason": "Common-test improvement established" if accepted else "Common-test improvement not established",
            "previous_champion_metrics": old_metrics, "challenger_metrics": new_metrics,
            "paired_improvement_interval": interval, "project_ids": fresh.project_id.tolist()}


def promotion_decision(candidate: dict, qc: dict, data_quality_passed: bool,
                       minimum_improvement: float) -> tuple[bool, list[str]]:
    failures = []
    if not candidate.get("numerically_validated"):
        failures.append("Numerical out-of-sample and calibration gates failed or were not run")
    if not candidate.get("historical_features_verified"):
        failures.append("Historical point-in-time features not independently established")
    if not candidate.get("scope_eligible", candidate.get("name") == "catboost"):
        failures.append("Prediction target is outside the absorption-only project task")
    if not data_quality_passed:
        failures.append("Data quality or ingestion coverage gate failed")
    if qc.get("status") != "COMPLETED" or (qc.get("review") or {}).get("decision") != "ACCEPT":
        failures.append("No valid Gemini evidence-review acceptance")
    # Recheck metrics here so a flag or model/LLM assertion cannot override them.
    baseline, candidate_metric = candidate.get("baseline", {}), candidate.get("metrics", {})
    if not baseline.get("mae") or candidate_metric.get("mae") is None:
        failures.append("Missing independent candidate/baseline MAE")
    elif (baseline["mae"] - candidate_metric["mae"]) / baseline["mae"] < minimum_improvement:
        failures.append("Insufficient improvement over validation-selected baseline")
    interval = candidate.get("paired_mae_improvement_interval")
    if not interval or interval[0] <= 0:
        failures.append("Paired held-out improvement interval does not establish improvement")
    if candidate.get("interval_test_coverage") is None or candidate["interval_test_coverage"] < 0.75:
        failures.append("Insufficient independent prediction-interval coverage")
    return not failures, failures


def update_registry(settings: Settings, evaluation: dict, qc: dict, ingestion: dict) -> dict:
    path = settings.data_dir / "models/registry.json"
    previous = json.loads(path.read_text()) if path.exists() else {"champion": None, "history": []}
    quality = (ingestion["parsed_project_count"] == ingestion["discovered_project_count"]
               and len(ingestion["csv_validation"]) == 4
               and not ingestion["project_limit_applied"]
               and len(ingestion["listing_counts"]) == 4)
    decisions = []
    eligible = []
    for candidate in evaluation["candidates"]:
        accepted, failures = promotion_decision(candidate, qc, quality, settings.minimum_relative_improvement)
        decisions.append({"candidate": candidate["name"], "eligible_for_promotion": accepted,
                          "promoted": False, "reasons": failures})
        if accepted:
            eligible.append(candidate)
    champion = previous.get("champion")
    if eligible:
        # Targets must match before comparisons. Currently only the project-issuance task is eligible.
        best = min(eligible, key=lambda c: c["metrics"]["mae"])
        comparison = compare_champion(settings, champion, best) if champion else None
        if comparison is not None:
            next(d for d in decisions if d["candidate"] == best["name"])["champion_comparison"] = comparison
        if champion is None or comparison["replace"]:
            champion = {"version": evaluation["version"], "name": best["name"], "task": best["target"],
                        "data_version": evaluation["data_version"], "test_mae": best["metrics"]["mae"],
                        "model_path": best["model_path"], "approved_at": now()}
            next(d for d in decisions if d["candidate"] == best["name"])["promoted"] = True
        else:
            next(d for d in decisions if d["candidate"] == best["name"])["reasons"].append(
                comparison["reason"])
    result = {"state": "APPROVED_MODEL_AVAILABLE" if champion else "ABSTENTION",
              "champion": champion, "previous_champion": previous.get("champion"),
              "last_candidate_version": evaluation["version"], "data_version": evaluation["data_version"],
              "timestamp": now(), "decisions": decisions,
              "history": previous.get("history", []) + [{"version": evaluation["version"],
                           "data_version": evaluation["data_version"], "timestamp": now(), "decisions": decisions}]}
    write_json(path, result)
    write_json(settings.data_dir / "outputs/model_selection.json", result)
    return result
