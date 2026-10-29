from __future__ import annotations

import json

import pandas as pd

from .domain import Settings, write_json
from .ingestion import records
from .modeling import predict_catboost


def select_estimate(target: str, predictions: list[dict], observation: dict | None) -> dict:
    """Compare like targets only. Validated simple baselines can beat complex models."""
    qualified = [p for p in predictions if p.get("target") == target and p.get("numerically_validated")
                 and p.get("evidence_supported") and p.get("independent_test")
                 and p.get("value") is not None and p.get("test_mae") is not None]
    if qualified:
        best = min(qualified, key=lambda p: (p["test_mae"], p.get("complexity", 1)))
        return {"selection": "VALIDATED_BASELINE" if best.get("is_baseline") else "VALIDATED_PREDICTION", **best}
    if observation is not None and observation.get("value") is not None:
        return {"selection": "DIRECT_OBSERVATION", "target": target, **observation}
    return {"selection": "ABSTENTION", "target": target, "value": None}


def select_issuance_indicator(settings: Settings, project: dict, data_version: str) -> dict:
    target = "first_cycle_annualized_certified_tonnes"
    first = next((e for e in records(settings.data_dir, "event_features")
                  if e["project_id"] == project["project_id"] and e["cycle"] == 1 and e["usable_label"]), None)
    observed = {"value": first["annualized_certified_tonnes"], "unit": "tCO2e/year",
                "source_ids": [first["source_id"], first["certified_source_id"]],
                "interpretation": "Observed annualized first certified cycle; not expected loss or delivery shortfall"} if first else None
    registry_path = settings.data_dir / "models/registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"champion": None}
    champion = registry.get("champion")
    predictions, reasons = [], []
    if champion is None:
        reasons.append("No approved project-issuance model")
    elif champion["data_version"] != data_version:
        reasons.append("Approved model belongs to another source snapshot; revalidation required")
    elif champion["name"] != "catboost" or champion.get("task") != "First-cycle annualized certified tonnes conditional on issuance":
        reasons.append("Approved model does not implement the project-issuance target")
    elif not project.get("absorption_training_eligible"):
        reasons.append("Project methodology outside the supported absorption task")
    elif first is None:
        reasons.append("No observed first-cycle monitoring horizon; an underwriting horizon cannot be invented")
    else:
        model_path = settings.data_dir / champion["model_path"]
        card = json.loads((model_path.parent / "catboost_card.json").read_text())
        if project["project_id"] in card["training_project_ids"] + card["validation_project_ids"]:
            reasons.append("Project participated in fitting or model selection; no independent project inference claimed")
        elif not card["historical_features_verified"]:
            reasons.append("Historical features are not verified")
        else:
            inputs = {"expected_annual_tonnes": project["expected_annual_tonnes"] if project["expected_annual_tonnes"] is not None else -1,
                      "monitoring_days": first["period_days"], "methodology": project["methodology"] or "UNKNOWN",
                      "size": project["size"] or "UNKNOWN", "program": project["program"]}
            prediction = float(predict_catboost(model_path, pd.DataFrame([inputs]))[0])
            predictions.append({"target": target, "value": prediction, "unit": "tCO2e/year",
                                "model_version": champion["version"], "numerically_validated": True,
                                "evidence_supported": True, "independent_test": True,
                                "test_mae": champion["test_mae"], "complexity": 2,
                                "interpretation": "Retrospective held-out conditional issuance indicator; not an insured-loss prediction"})
    selected = select_estimate(target, predictions, observed)
    write_json(settings.data_dir / f"outputs/indicator_selection_{project['project_id']}.json",
               {"target": target, "registry_state": registry.get("state", "ABSTENTION"),
                "predictions": predictions, "observation": observed, "selected": selected, "inference_decline_reasons": reasons})
    return selected
