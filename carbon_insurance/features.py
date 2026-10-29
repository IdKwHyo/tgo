from __future__ import annotations

import json
import re
from datetime import date

import numpy as np

from .domain import Settings, assessment_today, digest, now, write_json
from .ingestion import records, save_table

# These are activity classifications, not premium rates or calibrated risk weights.
# Each classification is enabled only when the family exists in the fetched official list.
ABSORPTION_FAMILIES = {
    "T-VER-S-METH-13-01", "T-VER-S-METH-13-03", "T-VER-S-METH-13-04",
    "T-VER-P-METH-13-01", "T-VER-P-METH-13-02", "T-VER-P-METH-13-04",
}
REDUCTION_FAMILIES = {"T-VER-S-METH-13-05", "T-VER-P-METH-13-08"}


def classify_activity(codes: list[str], method_inventory: list[dict]) -> dict:
    definitions = {m["code"]: m for m in method_inventory}
    supported = [definitions[c] for c in codes if c in definitions]
    families = [m.get("family_code", m["code"]) for m in supported]
    if not codes or len(supported) != len(codes):
        category = "UNRESOLVED_METHODOLOGY"
    elif all(c in ABSORPTION_FAMILIES for c in families):
        category = "ABSORPTION_METHOD_FAMILY"
    elif all(c in REDUCTION_FAMILIES for c in families):
        category = "REDUCTION_ONLY"
    else:
        category = "MIXED_OR_UNSEGREGATED"
    return {"activity_class": category,
            "absorption_training_eligible": category == "ABSORPTION_METHOD_FAMILY",
            "classification_evidence": [{"source_id": m["source_id"], "code": m["code"],
                                         "family_code": m.get("family_code", m["code"]),
                                         "title": m["title"], "locator": m["locator"]} for m in supported],
            "classification_limit": "Method family classification does not verify a particular vintage, historical version, additionality or all policy conditions."}


def day_difference(start: str | None, end: str | None, inclusive: bool = False) -> int | None:
    if not start or not end:
        return None
    return (date.fromisoformat(end) - date.fromisoformat(start)).days + int(inclusive)


def build_features(settings: Settings, *, as_of: str | None = None) -> dict:
    root = settings.data_dir
    timestamp = now()
    assessment_date = as_of or assessment_today()
    date.fromisoformat(assessment_date)
    projects = records(root, "projects")
    methods = records(root, "methodologies")
    all_events = records(root, "issuance_events")
    listings = {r["project_id"]: r for r in records(root, "listings")}
    events_by_project: dict[str, list] = {}
    errors = []
    valid_events = []
    seen = set()
    for event in all_events:
        key = (event["project_id"], event["cycle"])
        if key in seen:
            errors.append({"event": key, "error": "Duplicate project/cycle; excluded"})
            continue
        seen.add(key)
        days = day_difference(event["period_start"], event["period_end"], inclusive=True)
        reasons = []
        if days is None or days <= 0:
            reasons.append("Invalid monitoring period")
        if event["certified_tonnes"] is None:
            reasons.append("No matched certified-cycle quantity in official listing")
        if not event["approval_date"] or event["approval_date"] > assessment_date:
            reasons.append("Missing approval date or approval later than assessment")
        if event["period_end"] and event["approval_date"] and event["period_end"] > event["approval_date"]:
            reasons.append("Monitoring period ends after approval")
        event["period_days"] = days
        event["usable_label"] = not reasons
        event["label_warnings"] = reasons
        event["annualized_certified_tonnes"] = (event["certified_tonnes"] * 365.2425 / days) if not reasons else None
        event["quantity_semantics"] = "Certified quantity from listing; requested quantity retained separately"
        valid_events.append(event)
        events_by_project.setdefault(event["project_id"], []).append(event)
    features, candidates = [], []
    for project in projects:
        activity = classify_activity(project["methodology_codes"], methods)
        events = sorted(events_by_project.get(project["project_id"], []), key=lambda e: e["cycle"])
        usable = [e for e in events if e["usable_label"]]
        overlapping = any(a["period_end"] >= b["period_start"] for a, b in zip(usable, usable[1:]))
        latest_date = max((e["approval_date"] for e in usable), default=None)
        required = ["methodology", "registration_date", "crediting_start", "crediting_end",
                    "status", "size", "location", "developer", "owner", "validator", "expected_annual_tonnes"]
        present = sum(project.get(k) is not None and project.get(k) != "" for k in required)
        feature = {**project, **activity, "as_of_date": assessment_date,
                   "project_age_days": day_difference(project["registration_date"], assessment_date),
                   "crediting_period_days": day_difference(project["crediting_start"], project["crediting_end"], True),
                   "latest_issuance_date": latest_date,
                   "days_since_latest_issuance": day_difference(latest_date, assessment_date),
                   "usable_issuance_events": len(usable), "overlapping_monitoring_periods": overlapping,
                   "certified_total_tonnes": listings[project["project_id"]]["certified_total_tonnes"],
                   "certified_total_source_id": listings[project["project_id"]]["source_id"],
                   "record_completeness_fraction": present / len(required),
                   "missing_registry_fields": [k for k in required if project.get(k) is None or project.get(k) == ""],
                   "historical_features_verified": False,
                   "issuance_variation_cv": float(np.std([e["annualized_certified_tonnes"] for e in usable], ddof=1) /
                                                np.mean([e["annualized_certified_tonnes"] for e in usable])) if len(usable) >= 3 and not overlapping and np.mean([e["annualized_certified_tonnes"] for e in usable]) > 0 else None}
        if project["registration_date"] and project["registration_date"] > assessment_date:
            feature["absorption_training_eligible"] = False
            feature["warnings"] = project["warnings"] + ["Registration later than assessment date"]
        features.append(feature)
        # At most one first-issued label per project; non-issued projects are censored, not zero-loss rows.
        first = next((event for event in events if event["cycle"] == 1 and event["usable_label"]), None)
        if first and activity["absorption_training_eligible"] and project["group"] == "1" and not overlapping:
            if project["registration_date"] and project["registration_date"] <= first["approval_date"]:
                candidates.append({"project_id": project["project_id"], "approval_date": first["approval_date"],
                                   "methodology": project["methodology"] or "UNKNOWN", "size": project["size"] or "UNKNOWN",
                                   "program": project["program"], "expected_annual_tonnes": project["expected_annual_tonnes"],
                                   "monitoring_days": first["period_days"],
                                   "target_annualized_certified_tonnes": first["annualized_certified_tonnes"],
                                   "source_ids": [project["source_id"], first["source_id"], first["certified_source_id"]],
                                   "historical_features_verified": False,
                                   "label_scope": "Retrospective first-issuance amount conditional on issuance; not default, shortfall, reversal or insured loss"})
    catalog = [
        ("methodology_and_version", "OBSERVED", "Published project methodology; current snapshots are not historical feature vintages"),
        ("status", "OBSERVED", "Published operating status; not evidence that no reversal occurred"),
        ("expected_annual_tonnes", "OBSERVED", "Registered annual expectation; not verified tonnes"),
        ("certified_tonnes", "OBSERVED", "Listing amount matched by project and issuance cycle"),
        ("location_and_ownership", "OBSERVED", "Published text; no geocoded fire, flood or land-title probabilities inferred"),
        ("project_age_and_crediting_duration", "DERIVED", "Date differences from official fields"),
        ("verification_recency", "UNVERIFIED", "Approval recency is available; exact VVB verification date is not consistently extracted"),
        ("issuance_approval_recency", "DERIVED", "Days since latest approved certified cycle, not since the VVB report date"),
        ("annualized_issuance", "DERIVED", "Certified-cycle tonnes × 365.2425 / inclusive monitoring days; not an insurance loss"),
        ("issuance_variation", "DERIVED", "CV only with at least three nonoverlapping comparable monitoring intervals; still subject to area/version changes"),
        ("market_value_and_price_range", "OBSERVED", "Category 13 mixed-activity aggregate; not removal-only and not project-specific"),
        ("market_price_range_ratio", "DERIVED", "Maximum/minimum price; cross-sectional range, not temporal volatility or standard deviation"),
        ("liquidity", "UNVERIFIED", "Annual volume alone cannot estimate turnover, bid-ask spreads or exit time"),
        ("record_completeness_and_source_age", "DERIVED", "Documentation measurements; not physical risk"),
        ("non_permanence_and_buffer", "UNVERIFIED", "Official applicability guidance exists; actual project hazard and buffer deduction not consistently extracted"),
        ("matched_expected_vs_actual_shortfall", "UNVERIFIED", "Current annual expectations are not period-matched PDD delivery commitments"),
        ("insured_loss_frequency_and_severity", "UNVERIFIED", "No permitted claims/exposure or covered-loss dataset identified"),
    ]
    save_table(root, "project_features", features)
    save_table(root, "event_features", valid_events)
    save_table(root, "training_candidates", candidates)
    save_table(root, "factor_catalog", [{"factor": n, "measurement_class": c, "interpretation": why} for n, c, why in catalog])
    summary = {"timestamp": timestamp, "as_of_date": assessment_date, "projects": len(features),
               "activity_counts": {c: sum(p["activity_class"] == c for p in features) for c in sorted({p["activity_class"] for p in features})},
               "eligible_issuance_projects": len(candidates), "event_validation_errors": errors,
               "prohibited_targets": ["aggregate issuance / registrations", "current category price copied onto projects", "non-issued projects labeled zero loss"],
               "historical_snapshot_validation": "NOT_AVAILABLE: current registry snapshots cannot establish point-in-time features for past issuance"}
    write_json(root / "outputs/features.json", summary)
    return summary
