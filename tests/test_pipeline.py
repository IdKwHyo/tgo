"""Offline tests. Fixtures are real fetched TGO responses; no training observations are synthesized."""
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from carbon_insurance.domain import Settings, digest, require_tgo_url, write_json
from carbon_insurance.features import build_features, classify_activity
from carbon_insurance.ingestion import records, save_table
from carbon_insurance.modeling import evaluate_baselines, temporal_project_split, train_catboost, train_lstm
from carbon_insurance.parsing import (fiscal_period, listing_parameters, number, parse_csv,
                                      parse_listing, parse_method_history, parse_methods, parse_project, thai_date)
from carbon_insurance.registry import promotion_decision, update_registry
from carbon_insurance.review import quality_review, review_findings, validate_review
from carbon_insurance.selection import select_estimate
from carbon_insurance.store import SourceStore

FIXTURES = Path(__file__).parent / "fixtures"


def source(name="fixture"):
    return {"source_id": name, "url": "https://tver.tgo.or.th/fixture"}


def listings():
    return parse_listing((FIXTURES / "list_1_1.html").read_bytes(), source("listing"))


def test_official_market_csv_filter_units_and_reconciliation():
    rows, validation = parse_csv((FIXTURES / "trade.csv").read_bytes(), source(), "trading")
    assert validation["nonempty_rows"] == 8
    assert validation["eligible_rows"] == 1
    assert rows[0]["category"] == "FOR&AGR"
    assert rows[0]["volume_tonnes"] == 375933
    assert rows[0]["value_baht"] == 154586896
    assert rows[0]["calculated_weighted_price"] == pytest.approx(411.2086356)
    assert "Volume FY2026 (baht)" in rows[0]["original_values"]
    assert "header says baht" in rows[0]["unit_warning"]


def test_observed_zero_is_retained_and_fiscal_year_is_not_calendar_year():
    rows, _ = parse_csv((FIXTURES / "issuance.csv").read_bytes(), source(), "standard_issuance")
    assert len(rows) == 12
    assert rows[0]["tonnes"] == 0
    assert fiscal_period(2569)["period_start"] == "2025-10-01"
    assert fiscal_period(2569)["period_end"] == "2026-09-30"
    assert thai_date("11/03/2567") == "2024-03-11"
    assert thai_date("09/06/2607") == "2064-06-09"
    with pytest.raises(ValueError):
        thai_date("31/02/2569")
    with pytest.raises(ValueError):
        number("100 tonnes")


def test_premium_internal_category_id_is_discovered_not_assumed():
    values = listing_parameters((FIXTURES / "premium_list.html").read_bytes(), 2, 1)
    assert values["filter_category_id"] == "28"
    assert values["filter_program_id"] == "2"


def test_project_url_id_differs_from_registration_and_expected_is_not_issued():
    item = next(r for r in listings() if r["project_id"] == "449")
    p, events, children, links = parse_project((FIXTURES / "449.html").read_bytes(), source(), item)
    assert p.project_id == "449"
    assert p.registration_number == "393"
    assert p.expected_annual_tonnes == 259980
    assert events == []
    assert item["certified_total_tonnes"] == 0
    assert links


def test_programme_framework_is_not_component_crediting_period():
    # Listing adapter supplies actual values from the fetched programme record.
    item = {"project_id": "667", "group_id": 2, "name": "", "expected_annual_tonnes": 161,
            "certified_cycles": {}, "source_id": "listing"}
    p, events, children, _ = parse_project((FIXTURES / "667.html").read_bytes(), source(), item)
    assert p.crediting_start is None
    assert p.crediting_end is None
    assert p.expected_annual_tonnes == 161
    assert len(children) == 2
    assert p.raw_fields["registration"]["อายุกรอบแผนงาน"] == "20 ปี"


def test_requested_and_certified_cycle_amounts_keep_distinct_provenance():
    item = next(r for r in listings() if r["project_id"] == "530")
    _, events, _, _ = parse_project((FIXTURES / "issued_example.html").read_bytes(), source("detail"), item)
    assert events[0]["requested_tonnes"] == 16163
    assert events[0]["certified_tonnes"] == 16163
    assert events[0]["source_id"] == "detail"
    assert events[0]["certified_source_id"] == "listing"
    assert events[0]["approval_date"] == "2026-09-23"
    assert events[0]["period_start"] == "2024-04-05"


def test_methodology_classification_excludes_fertilizer_and_separates_mixed():
    methods = parse_methods((FIXTURES / "methods_1.html").read_bytes(), source(), 1)
    family = next(m for m in methods if m["code"] == "T-VER-S-METH-13-01")
    methods += parse_method_history((FIXTURES / "method_history_13_01.html").read_bytes(), source(), family)
    assert classify_activity(["T-VER-METH-FOR-01"], methods)["absorption_training_eligible"]
    assert classify_activity(["T-VER-S-METH-13-03"], methods)["absorption_training_eligible"]
    assert classify_activity(["T-VER-S-METH-13-05"], methods)["activity_class"] == "REDUCTION_ONLY"
    assert classify_activity(["T-VER-S-METH-13-06"], methods)["activity_class"] == "MIXED_OR_UNSEGREGATED"
    assert classify_activity(["T-VER-METH-FOR-99"], methods)["activity_class"] == "UNRESOLVED_METHODOLOGY"


@pytest.mark.parametrize("url", ["https://tgo.or.th.example.com/a", "http://tver.tgo.or.th/a",
                                "https://evil.example/a", "https://user:password@tver.tgo.or.th/a"])
def test_non_tgo_sources_and_embedded_credentials_rejected(url):
    with pytest.raises(ValueError):
        require_tgo_url(url)


def test_source_refresh_tracks_versions_and_offline_replay_detects_corruption(tmp_path, monkeypatch):
    raw = (FIXTURES / "trade.csv").read_bytes()
    client_class = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=raw, headers={"content-type": "text/csv"}))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_class(transport=transport, **kwargs))
    store = SourceStore(Settings(data_dir=tmp_path, retries=0))
    url = "https://opendata.tgo.or.th/fixture.csv"
    a, _ = store.fetch(url)
    b, _ = store.fetch(url)
    assert a["source_id"] == b["source_id"]
    with store.connect() as db:
        assert [r[0] for r in db.execute("SELECT changed FROM checks ORDER BY id")] == [1, 0]
    replay, saved = store.fetch(url, offline=True)
    assert replay["retrieved_at"] == a["retrieved_at"]
    assert saved == raw
    (tmp_path / a["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        store.fetch(url, offline=True)


def test_redirect_outside_tgo_is_rejected_before_following(tmp_path, monkeypatch):
    client_class = httpx.Client
    called = []
    def handler(request):
        called.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://external.example/file"})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs))
    store = SourceStore(Settings(data_dir=tmp_path, retries=0))
    with pytest.raises(RuntimeError, match="Only official"):
        store.fetch("https://opendata.tgo.or.th/test")
    assert called == ["https://opendata.tgo.or.th/test"]


def test_missing_gemini_credentials_never_produces_fake_review(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = quality_review(Settings(data_dir=tmp_path), {"proposed_factors": []}, [],
                            numeric_gate_passed=False, tag="test")
    assert result["status"] == "NOT_EXECUTED_MISSING_CREDENTIAL"
    assert result["review"] is None


def review_payload():
    # Stubbed provider response for validation tests only; never used as research evidence or output.
    return {"decision": "INVESTIGATE", "factors": [{"factor": "methodology", "support": "SUPPORTED",
            "insurance_relevance": "RELEVANT", "evidence_chunk_ids": ["actual-input-id"], "reason": "test"}],
            "statistical_credibility": "INSUFFICIENT", "material_model_disagreement": False,
            "eligibility_violation": False, "additional_evidence": [], "reason": "test"}


def test_llm_cannot_invent_citations_or_override_failed_numeric_gates():
    payload = review_payload()
    evidence = [{"chunk_id": "actual-input-id", "source_integrity_verified": True}]
    assert validate_review(payload, ["methodology"], evidence, False).decision == "INVESTIGATE"
    payload["factors"][0]["evidence_chunk_ids"] = ["invented"]
    with pytest.raises(ValueError, match="nonexistent"):
        validate_review(payload, ["methodology"], evidence, False)
    payload = review_payload()
    payload.update(decision="ACCEPT", statistical_credibility="SUPPORTED")
    with pytest.raises(ValueError, match="deterministic"):
        validate_review(payload, ["methodology"], evidence, False)


def test_forged_model_acceptance_does_not_promote_without_metrics():
    candidate = {"name": "catboost", "numerically_validated": True, "historical_features_verified": True}
    accepted, reasons = promotion_decision(candidate, {"status": "COMPLETED", "review": {"decision": "ACCEPT"}}, True, .05)
    assert not accepted
    assert any("MAE" in r for r in reasons)


def test_real_annual_series_abstains_lstm_and_backtests_without_future_leakage(tmp_path):
    (tmp_path / "processed").mkdir()
    (tmp_path / "outputs").mkdir()
    aggregates, _ = parse_csv((FIXTURES / "issuance.csv").read_bytes(), source(), "standard_issuance")
    save_table(tmp_path, "aggregates", aggregates)
    settings = Settings(data_dir=tmp_path)
    result = train_lstm(settings, tmp_path / "models")
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["observations"] == 11
    assert not (tmp_path / "models/lstm.pt").exists()
    baselines = evaluate_baselines(settings)
    assert baselines["status"] == "EXPLORATORY_ONLY"
    assert all(r["n"] == 5 for r in baselines["results"])
    assert baselines["excluded_latest_fiscal_year"][0]["fiscal_year_be"] == 2569
    backtests = records(tmp_path, "baseline_backtests")
    assert all(r["history_ends_fiscal_year_be"] < r["fiscal_year_be"] for r in backtests)


def test_catboost_has_a_real_minimum_data_gate(tmp_path):
    result = train_catboost(Settings(data_dir=tmp_path), tmp_path / "models")
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["selected_features"] == []
    assert not (tmp_path / "models").exists()


def test_lstm_architecture_accepts_sequences_without_manufacturing_training_data():
    import torch
    from carbon_insurance.lstm import LSTMForecaster
    model = LSTMForecaster()
    # Shape-only smoke check; no fit, target, loss history or research output is synthesized.
    with torch.no_grad():
        assert model(torch.zeros(2, 6, 1)).shape == (2,)


def test_later_valid_cycle_is_not_relabelled_as_first_cycle(tmp_path):
    SourceStore(Settings(data_dir=tmp_path))
    listing = next(r for r in listings() if r["project_id"] == "8")
    p, events, _, _ = parse_project((FIXTURES / "8.html").read_bytes(), source(), listing)
    methods = parse_methods((FIXTURES / "methods_1.html").read_bytes(), source(), 1)
    family = next(m for m in methods if m["code"] == "T-VER-S-METH-13-01")
    methods += parse_method_history((FIXTURES / "method_history_13_01.html").read_bytes(), source(), family)
    assert classify_activity(p.methodology_codes, methods)["absorption_training_eligible"]
    assert len(events) >= 2
    events[0]["certified_tonnes"] = None  # Simulate a missing official cycle match, not a new observation.
    save_table(tmp_path, "projects", [p.model_dump()])
    save_table(tmp_path, "listings", [listing])
    save_table(tmp_path, "methodologies", methods)
    save_table(tmp_path, "issuance_events", events)
    result = build_features(Settings(data_dir=tmp_path))
    assert result["eligible_issuance_projects"] == 0
    assert any(e["cycle"] == 2 and e["usable_label"] for e in records(tmp_path, "event_features"))


def test_gemini_rejection_and_missing_evidence_change_factor_findings():
    payload = review_payload()
    payload.update(decision="REJECT", eligibility_violation=True, additional_evidence=["Verified project risk report"])
    result = review_findings({"status": "COMPLETED", "review": payload})
    assert result["selected_factors"] == []
    assert result["additional_evidence"] == ["Verified project risk report"]
    assert any("eligibility" in w for w in result["warnings"])


def test_selection_compares_matching_targets_and_can_choose_simpler_baseline():
    # Selection-contract stubs, not training data or empirical model results.
    candidate = {"target": "issuance", "value": 1, "numerically_validated": True,
                 "evidence_supported": True, "independent_test": True, "test_mae": 2}
    baseline = {**candidate, "test_mae": 1, "is_baseline": True}
    assert select_estimate("issuance", [candidate, baseline], None)["selection"] == "VALIDATED_BASELINE"
    assert select_estimate("price", [candidate, baseline], None)["selection"] == "ABSTENTION"
    assert select_estimate("price", [candidate], {"value": 411})["selection"] == "DIRECT_OBSERVATION"


def test_registry_does_not_claim_promotion_when_previous_champion_is_kept(tmp_path):
    SourceStore(Settings(data_dir=tmp_path))
    prior = {"name": "catboost", "task": "issuance", "test_mae": 10, "version": "previous"}
    write_json(tmp_path / "models/registry.json", {"champion": prior, "history": []})
    candidate = {"name": "catboost", "target": "issuance", "numerically_validated": True,
                 "historical_features_verified": True, "metrics": {"mae": 1}, "baseline": {"mae": 2},
                 "paired_mae_improvement_interval": [.1, 1], "interval_test_coverage": .8,
                 "model_path": "not-loaded-by-selection.cbm"}
    evaluation = {"version": "current", "data_version": "current", "candidates": [candidate]}
    ingestion = {"parsed_project_count": 1, "discovered_project_count": 1, "csv_validation": [1, 2, 3, 4],
                 "listing_counts": [1, 2, 3, 4], "project_limit_applied": False}
    result = update_registry(Settings(data_dir=tmp_path), evaluation,
                             {"status": "COMPLETED", "review": {"decision": "ACCEPT"}}, ingestion)
    assert result["champion"] == prior
    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["eligible_for_promotion"]
    assert not result["decisions"][0]["promoted"]


def fixture_snapshot(tmp_path):
    """Small real-response snapshot; metadata is test-only and never used for model training."""
    settings = Settings(data_dir=tmp_path)
    SourceStore(settings)
    projects, sources, selected_listings = [], [], []
    for identifier, program in [("449", 1), ("718", 2)]:
        raw = (FIXTURES / f"{identifier}.html").read_bytes()
        path = tmp_path / f"raw/{identifier}.html"
        path.write_bytes(raw)
        url = f"https://tver.tgo.or.th/database/public/project/{program}/1/{identifier}?lang=th"
        metadata = {"source_id": digest(raw)[:24], "sha256": digest(raw), "url": url,
                    "title": f"TGO project {identifier}", "path": f"raw/{identifier}.html",
                    "retrieved_at": "2026-10-30T00:00:00+00:00", "role": "project", "project_id": identifier}
        listing = next(r for r in listings() if r["project_id"] == identifier) if identifier == "449" else {
            "project_id": identifier, "group_id": 1, "name": "", "expected_annual_tonnes": 4544,
            "certified_cycles": {}, "certified_total_tonnes": 0, "source_id": "listing"}
        project, _, _, _ = parse_project(raw, metadata, listing)
        projects.append(project.model_dump())
        sources.append(metadata)
        selected_listings.append(listing)
    methods = []
    for program in [1, 2]:
        methods.extend(parse_methods((FIXTURES / f"methods_{program}.html").read_bytes(), source(), program))
    save_table(tmp_path, "projects", projects)
    save_table(tmp_path, "listings", selected_listings)
    save_table(tmp_path, "methodologies", methods)
    save_table(tmp_path, "issuance_events", [])
    save_table(tmp_path, "knowledge_sources", sources)
    aggregates, _ = parse_csv((FIXTURES / "trade.csv").read_bytes(), source(), "trading")
    save_table(tmp_path, "aggregates", aggregates)
    write_json(tmp_path / "outputs/ingestion.json", {"data_version": "test-fixture", "errors": []})
    write_json(tmp_path / "outputs/model_evaluation.json", {"data_version": "test-fixture", "candidates": []})
    build_features(settings)
    return settings


def test_rag_rebuild_is_deterministic_and_other_project_citations_are_excluded(tmp_path):
    from carbon_insurance.retrieval import EvidenceIndex, build_index
    settings = fixture_snapshot(tmp_path)
    first = build_index(settings)
    sources = records(tmp_path, "knowledge_sources")
    save_table(tmp_path, "knowledge_sources", list(reversed(sources)))
    second = build_index(settings)
    assert first["index_version"] == second["index_version"]
    index = EvidenceIndex(tmp_path)
    passages = index.search("T-VER-S-METH-13-03 การกักเก็บคาร์บอน", project_id="449", program="Standard T-VER")
    assert passages
    assert all(p["project_id"] == "449" for p in passages)
    assert all(index.verify(p) for p in passages)
    assert not index.verify({**passages[0], "text": "tampered excerpt"})


def test_assessment_refreshes_cached_date_dependent_features(tmp_path, monkeypatch):
    from carbon_insurance.retrieval import build_index
    from carbon_insurance.risk import assess
    from carbon_insurance.domain import assessment_today
    settings = fixture_snapshot(tmp_path)
    build_index(settings)
    features = records(tmp_path, "project_features")
    for p in features:
        p["as_of_date"] = "2020-01-01"
        p["project_age_days"] = -1
    save_table(tmp_path, "project_features", features)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = assess(settings, "449")
    assert result["as_of_date"] == assessment_today()
    assert result["selected_output"]["insurance_premium_baht"] is None
    assert result["dynamic_risk_rating"]["status"] == "NOT_RATABLE"
    updated = next(p for p in records(tmp_path, "project_features") if p["project_id"] == "449")
    assert updated["project_age_days"] > 0


def test_model_failure_is_a_rejected_candidate_and_does_not_hide_other_results(tmp_path, monkeypatch):
    import carbon_insurance.modeling as modeling
    settings = fixture_snapshot(tmp_path)
    def unavailable_catboost(*args):
        raise RuntimeError("test-only provider/library failure")
    monkeypatch.setattr(modeling, "train_catboost", unavailable_catboost)
    result = modeling.train(settings)
    assert len(result["candidates"]) == 3
    assert result["candidates"][0]["status"] == "FAILED_CANDIDATE"
    assert result["candidates"][1]["status"] == "INSUFFICIENT_DATA"
    assert result["training_fingerprint"] == modeling.training_fingerprint(settings)
    different_settings = settings.model_copy(update={"catboost_min_projects": 41})
    assert modeling.training_fingerprint(settings) != modeling.training_fingerprint(different_settings)
