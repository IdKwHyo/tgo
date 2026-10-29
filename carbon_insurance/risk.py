from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from .domain import Settings, assessment_today, now, write_json
from .features import build_features, day_difference
from .ingestion import records
from .retrieval import EvidenceIndex
from .review import quality_review, review_findings
from .selection import select_issuance_indicator
from .store import SourceStore


def market_benchmark(settings: Settings) -> dict:
    rows = [r for r in records(settings.data_dir, "aggregates") if r["kind"] == "trading"]
    if not rows:
        return {"status": "NO_DEFENSIBLE_BENCHMARK", "selected_value": None}
    row = max(rows, key=lambda r: r["fiscal_year_be"])
    modified = row.get("resource_last_modified")
    source_age_days = None
    if modified:
        try:
            source_age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(modified).replace(tzinfo=timezone.utc)).days
        except ValueError:
            pass
    return {"status": "OBSERVED_CATEGORY_CONTEXT_ONLY", "selected_value": row["calculated_weighted_price"],
            "unit": "THB/tCO2e", "price_type": "Value-weighted trading price; NOT insurance premium",
            "scope": "FOR&AGR Category 13, mixed absorption and emission-reduction activities",
            "absorption_only_price_status": "NOT_SEPARATELY_IDENTIFIABLE",
            "project_specific_price": None, "fiscal_year_be": row["fiscal_year_be"],
            "fiscal_period_start": row["period_start"], "fiscal_period_end": row["period_end"],
            "snapshot_completeness": "Latest file may be a year-to-date snapshot; not assumed full-year",
            "observed_volume_tonnes": row["volume_tonnes"], "observed_value_baht": row["value_baht"],
            "reported_weighted_price": row["reported_weighted_price"],
            "observed_minimum": row["minimum_price_baht_per_tonne"],
            "observed_maximum": row["maximum_price_baht_per_tonne"],
            "range_ratio": row["maximum_price_baht_per_tonne"] / row["minimum_price_baht_per_tonne"] if row["minimum_price_baht_per_tonne"] > 0 else None,
            "range_interpretation": "Price range across trades; not a time-series volatility estimate or confidence interval",
            "source_id": row["source_id"], "source_url": row["source_url"],
            "source_row": row["source_row"], "resource_last_modified": modified,
            "source_timestamp_age_days": source_age_days,
            "freshness": "UNKNOWN" if source_age_days is None else "STALE_SOURCE_TIMESTAMP" if source_age_days > settings.stale_after_days else "WITHIN_CONFIGURED_SOURCE_AGE",
            "provenance": {"formula": "observed_value_baht / observed_volume_tonnes",
                           "input_source_ids": [row["source_id"]], "unit_interpretation": row["unit_warning"]}}


def assess(settings: Settings, project_id: str) -> dict:
    features = records(settings.data_dir, "project_features")
    if any(p.get("as_of_date") != assessment_today() for p in features):
        build_features(settings)
        features = records(settings.data_dir, "project_features")
    matches = [p for p in features if p["project_id"] == str(project_id)]
    if len(matches) != 1:
        raise ValueError(f"Project URL ID {project_id} is not in the ingested FOR&AGR project population")
    project = matches[0]
    ingestion = json.loads((settings.data_dir / "outputs/ingestion.json").read_text())
    evaluation = json.loads((settings.data_dir / "outputs/model_evaluation.json").read_text())
    if evaluation["data_version"] != ingestion["data_version"]:
        raise ValueError("Model evaluation is stale for current source data; train/evaluate again")
    market = market_benchmark(settings)
    evidence_index = EvidenceIndex(settings.data_dir)
    query = (f"{project['methodology']} {project['name']} "
             "non-permanence risk buffer monitoring verification issuance forestry "
             "ความเสี่ยง ความไม่ถาวร เครดิตสำรอง การทวนสอบ การรับรอง")
    evidence = evidence_index.search(query, project_id=str(project_id), program=project["program"],
                                     top_k=settings.retrieval_top_k)
    peer_values = [p["expected_annual_tonnes"] for p in features
                   if p["activity_class"] == project["activity_class"] and p["program"] == project["program"]
                   and p["expected_annual_tonnes"] is not None]
    percentile = float(np.mean(np.array(peer_values) <= project["expected_annual_tonnes"]) * 100) if peer_values and project["expected_annual_tonnes"] is not None else None
    premium = project["program"] == "Premium T-VER"
    eligible = project["activity_class"] == "ABSORPTION_METHOD_FAMILY"
    warnings = list(project["warnings"]) + [
        "No claims/exposure or coverage-specific loss dataset identified; actuarial expected loss and premium cannot be estimated",
        "Market benchmark pools Category 13 activities and is not a project-specific absorption price",
        "Project hazard probabilities, covered reversal severity and actual buffer deductions are not established",
        "Approval recency is not the date of a VVB verification report",
        "Missing or unextracted disclosures are missing evidence, not evidence of zero risk",
        "Historical feature vintages are unavailable; no trained model is approved from this snapshot",
    ]
    if not eligible:
        warnings.append("Methodology is reduction-only, mixed/unsegregated, or unresolved; removal-only insured tonnes not established")
    if market.get("freshness") == "STALE_SOURCE_TIMESTAMP":
        warnings.append("Official market resource timestamp exceeds configured freshness limit; fetching today does not refresh the observation date")
    if project["group"] == "2":
        warnings.append("Programme framework dates are not component-project crediting periods; components retained separately")
    components = {
        "permanence_and_reversal": {"status": "UNMEASURED", "score": None,
                                    "requirement_context": "Premium non-permanence/buffer rules require project-specific assessment" if premium else "Premium buffer rules not assumed applicable to Standard T-VER",
                                    "loss_probability": None, "buffer_rate": None},
        "issuance_performance": {"status": "DESCRIPTIVE_ONLY", "score": None,
                                "certified_total_tonnes": project["certified_total_tonnes"],
                                "usable_issuance_events": project["usable_issuance_events"],
                                "annualized_issuance_cv": project["issuance_variation_cv"],
                                "source_id": project["certified_total_source_id"],
                                "shortfall_status": "NOT_COMPARABLE_TO_CURRENT_ANNUAL_EXPECTATION"},
        "verification_and_methodology": {"status": "RECORD_AVAILABLE", "score": None,
                                        "methodology": project["methodology"], "method_family_class": project["activity_class"],
                                        "historical_version_compliance": "NOT_VERIFIED",
                                        "approval_recency_days": project["days_since_latest_issuance"],
                                        "latest_approval": project["latest_issuance_date"], "exact_verification_date": None},
        "operational_status": {"status": "OBSERVED", "score": None, "reported_status": project["status"],
                               "crediting_end": project["crediting_end"],
                               "crediting_period_ended": project["crediting_end"] < project["as_of_date"] if project["crediting_end"] else None},
        "market_value_exposure": {"status": "CATEGORY_CONTEXT_ONLY", "score": None,
                                  "benchmark_baht_per_tonne": market.get("selected_value"),
                                  "project_expected_tonnes_percentile": percentile,
                                  "percentile_peer_count": len(peer_values),
                                  "percentile_interpretation": "Relative registered project size, not a loss percentile or risk score",
                                  "sum_insured": None},
        "liquidity_and_valuation_uncertainty": {"status": "UNMEASURED_LIQUIDITY", "score": None,
                                               "observed_price_range_ratio": market.get("range_ratio"),
                                               "time_series_volatility": None, "bid_ask_spread": None},
        "documentation_quality": {"status": "INCOMPLETE_UNDERWRITING_EVIDENCE", "score": None,
                                   "registry_field_completeness": project["record_completeness_fraction"],
                                   "missing_registry_fields": project["missing_registry_fields"],
                                   "independently_verified_retrieved_chunks": len(evidence),
                                   "market_source_freshness": market.get("freshness"),
                                   "interpretation": "Documentation completeness is not physical risk or model confidence"},
        "model_uncertainty": {"status": "INSUFFICIENT_DATA", "score": None,
                              "confidence": None, "calibration": "NO_VALIDATED_INSURED_LOSS_MODEL"},
    }
    factors = ["methodology_and_version", "issuance_performance", "permanence_and_buffer",
               "operational_status", "market_valuation", "documentation_quality"]
    context = {"proposed_factors": factors, "project": project,
               "data_quality": {"ingestion_errors": ingestion["errors"], "warnings": warnings},
               "components": components, "market_context": market, "model_diagnostics": evaluation,
               "requested_decision_scope": "Evidence assessment only; no actuarial premium or calibrated risk score"}
    qc = quality_review(settings, context, evidence, numeric_gate_passed=False, tag=f"project_{project_id}")
    findings = review_findings(qc)
    warnings.extend(findings["warnings"])
    warnings.extend(f"Additional evidence requested by Gemini: {item}" for item in findings["additional_evidence"])
    indicator = select_issuance_indicator(settings, project, ingestion["data_version"])
    result = {"assessment_timestamp": now(), "as_of_date": project["as_of_date"],
              "data_version": ingestion["data_version"], "project_id": project["project_id"],
              "registration_number": project["registration_number"], "project_name": project["name"],
              "program": project["program"], "activity_class": project["activity_class"],
              "selected_output": {"selection": "DIRECT_OBSERVED_CONTEXT_WITH_ABSTENTION_ON_INSURANCE_PRICE",
                                  "carbon_credit_value": market, "absorption_only_project_valuation": None,
                                  "sum_insured_baht": None, "insurance_premium_baht": None,
                                  "pricing_status": "ACTUARIAL_PREMIUM_NOT_ESTIMABLE",
                                  "selected_operational_indicator": indicator},
              "dynamic_risk_rating": {"status": "NOT_RATABLE", "category": "NOT_RATABLE",
                                      "score": None, "model_confidence": None,
                                      "calibration": "No covered-loss or hazard calibration dataset",
                                      "components": components,
                                      "main_drivers": ["Unmeasured covered reversal/delivery risk", "Unpriced policy terms and insurer costs",
                                                       "Mixed-category valuation uncertainty", "Limited temporal/project labels"],
                                      "decline_reasons": warnings},
              "factor_evidence": project["classification_evidence"],
              "derivation_provenance": {
                  "record_completeness": {"formula": "present registry fields / 11", "source_ids": [project["source_id"]]},
                  "date_differences": {"formula": "as_of_date minus official registration/approval date", "source_ids": [project["source_id"]]},
                  "size_percentile": {"formula": "100 × count(peer expected annual tonnes <= project expected annual tonnes) / peer count",
                                      "source_ids": sorted({p["source_id"] for p in features if p["activity_class"] == project["activity_class"] and p["program"] == project["program"] and p["expected_annual_tonnes"] is not None})}},
              "gemini_quality_review_status": qc["status"],
              "factor_quality_review": findings,
              "warnings": warnings,
              "unpriced_components": ["Covered loss frequency", "Covered loss severity", "Policy exclusions, deductible and limit",
                                      "Buffer interaction and recoveries", "Expenses", "Risk/capital margin"],
              "audit_artifacts": {"retrieval": f"evidence_{project_id}.json", "quality_review": f"qc_project_{project_id}.json",
                                  "model_comparison": "model_evaluation.json", "source_inventory": "source_inventory.json",
                                  "numeric_lineage": "numeric_lineage.csv"}}
    write_json(settings.data_dir / f"outputs/evidence_{project_id}.json", evidence)
    write_json(settings.data_dir / f"outputs/assessment_{project_id}.json", result)
    return result
