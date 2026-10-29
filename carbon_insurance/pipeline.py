from __future__ import annotations

import json

import numpy as np
from scipy.stats import wasserstein_distance

from .domain import Settings, now, write_json
from .features import build_features
from .ingestion import ingest, records
from .modeling import evaluate_baselines, train, training_fingerprint
from .registry import update_registry
from .retrieval import EvidenceIndex, build_index, implementation_fingerprint
from .review import quality_review
from .risk import assess, market_benchmark


def review_models(settings: Settings, evaluation: dict, ingestion: dict) -> dict:
    index = EvidenceIndex(settings.data_dir)
    evidence = index.search("T-VER forestry sequestration monitoring verification non-permanence buffer carbon risk ความเสี่ยง การทวนสอบ", top_k=settings.retrieval_top_k)
    numeric_gate = any(c.get("numerically_validated") and c.get("historical_features_verified")
                       and c.get("scope_eligible", c["name"] == "catboost") for c in evaluation["candidates"])
    qc = quality_review(settings,
                        {"proposed_factors": ["methodology", "issuance_labels", "temporal_validation", "non_permanence", "market_valuation"],
                         "model_diagnostics": evaluation, "data_quality": ingestion,
                         "requested_decision_scope": "Candidate-model promotion for absorption-only project issuance; never insured-loss calibration"},
                        evidence, numeric_gate_passed=numeric_gate, tag="models")
    write_json(settings.data_dir / "outputs/model_review_evidence.json", evidence)
    update_registry(settings, evaluation, qc, ingestion)
    return qc


def evaluate(settings: Settings) -> dict:
    root = settings.data_dir
    ingestion = json.loads((root / "outputs/ingestion.json").read_text())
    result = json.loads((root / "outputs/model_evaluation.json").read_text())
    if result["data_version"] != ingestion["data_version"]:
        raise ValueError("Evaluation belongs to another data snapshot; run train first")
    repeated_baselines = evaluate_baselines(settings)
    saved = next(c for c in result["candidates"] if c["name"] == "annual_baselines")
    if repeated_baselines != saved:
        raise ValueError("Baseline reproduction differs from saved evaluation")
    checked = {"timestamp": now(), "data_version": result["data_version"],
               "baseline_reproduction": "PASSED", "candidate_statuses": {c["name"]: c["status"] for c in result["candidates"]}}
    write_json(root / "outputs/evaluation_verification.json", checked)
    make_report(settings)
    return checked


def drift_summary(previous: list[dict], current: list[dict]) -> dict:
    if not previous:
        return {"status": "NO_PREVIOUS_FEATURE_SNAPSHOT"}
    a = [p["expected_annual_tonnes"] for p in previous if p.get("expected_annual_tonnes") is not None]
    b = [p["expected_annual_tonnes"] for p in current if p.get("expected_annual_tonnes") is not None]
    return {"status": "DESCRIPTIVE_COMPARISON", "previous_project_count": len(previous), "current_project_count": len(current),
            "expected_tonnes_wasserstein_distance": float(wasserstein_distance(a, b)) if a and b else None,
            "interpretation": "Distribution change, not a probability of physical risk; shared projects are not independent samples"}


def run_all(settings: Settings, *, offline: bool = False, force_train: bool = False) -> dict:
    root = settings.data_dir
    previous_features = records(root, "project_features")
    previous_evaluation_path = root / "outputs/model_evaluation.json"
    previous_evaluation = json.loads(previous_evaluation_path.read_text()) if previous_evaluation_path.exists() else None
    ingestion = ingest(settings, offline=offline)
    feature_summary = build_features(settings)
    index_path = root / "index/manifest.json"
    manifest = json.loads(index_path.read_text()) if index_path.exists() else {}
    if manifest.get("data_version") != ingestion["data_version"] or manifest.get("implementation_fingerprint") != implementation_fingerprint():
        build_index(settings)
        index_action = "INDEX_REBUILT"
    else:
        index_action = "UNCHANGED_INDEX_REUSED"
    data_changed = previous_evaluation is None or previous_evaluation["data_version"] != ingestion["data_version"]
    changed = data_changed or previous_evaluation.get("training_fingerprint") != training_fingerprint(settings)
    if changed or force_train:
        evaluation = train(settings)
        review_models(settings, evaluation, ingestion)
        training_action = "NEW_CANDIDATES_EVALUATED"
    else:
        evaluation = previous_evaluation
        training_action = "UNCHANGED_INPUTS_RETRAINING_SKIPPED"
    adaptive = {"timestamp": now(), "data_version": ingestion["data_version"], "source_data_changed": data_changed,
                "training_inputs_or_implementation_changed": changed, "index_action": index_action,
                "training_action": training_action, "forced": force_train,
                "drift": drift_summary(previous_features, records(root, "project_features"))}
    write_json(root / "outputs/adaptive_cycle.json", adaptive)
    assessments = []
    available = {p["project_id"] for p in records(root, "project_features")}
    for project_id in settings.sample_project_ids:
        if project_id in available:
            result = assess(settings, project_id)
            assessments.append({"project_id": project_id,
                                "pricing_status": result["selected_output"]["pricing_status"],
                                "rating": result["dynamic_risk_rating"]["category"]})
    make_report(settings)
    return {"data_version": ingestion["data_version"], "selected_market_context": market_benchmark(settings),
            "pricing_status": "ACTUARIAL_PREMIUM_NOT_ESTIMABLE", "risk_rating": "NOT_RATABLE",
            "sample_assessments": assessments, "adaptive_cycle": adaptive,
            "report": str(root / "outputs/MODEL_EVALUATION.md")}


def make_report(settings: Settings) -> str:
    root = settings.data_dir
    ingestion = json.loads((root / "outputs/ingestion.json").read_text())
    feature_summary = json.loads((root / "outputs/features.json").read_text())
    evaluation = json.loads((root / "outputs/model_evaluation.json").read_text())
    market = market_benchmark(settings)
    qc_path = root / "outputs/qc_models.json"
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {"status": "NOT_EXECUTED"}
    retrieval = json.loads((root / "outputs/retrieval_index.json").read_text())
    lines = ["# TGO carbon-absorption insurance research — executed results", "",
             f"Generated {now()}. Data version `{ingestion['data_version']}`.", "",
             "## Selected result", "",
             "No actuarially defensible insurance premium or calibrated physical-risk score can be estimated from this snapshot.",
             "The selected numerical result is only a directly observed, mixed Category 13 market context. It is not a removal-only project price, sum insured or premium.", "",
             f"Context benchmark: **{market.get('selected_value')} THB/tCO2e**. Source timestamp freshness: `{market.get('freshness')}`.", "",
             "## Actual ingestion", "", "| Dataset | Nonempty rows | FOR&AGR rows | Excluded rows |", "|---|---:|---:|---:|"]
    for dataset in ingestion["csv_validation"]:
        lines.append(f"| {dataset['kind']} | {dataset['nonempty_rows']} | {dataset['eligible_rows']} | {dataset['excluded_rows']} |")
    lines.extend(["", f"Discovered {ingestion['discovered_project_count']} project records; parsed {ingestion['parsed_project_count']}. "
                  f"Parsed {ingestion['issuance_events']} issuance events and {ingestion['subprojects']} programme components. "
                  f"Discovered {ingestion['document_links']} linked files; downloaded {ingestion['downloaded_project_pdfs']} project PDFs.", "",
                  "| Activity classification | Projects |", "|---|---:|"])
    for category, count in feature_summary["activity_counts"].items():
        lines.append(f"| {category} | {count} |")
    lines.extend(["", f"Usable first-issuance absorption-project candidates: {feature_summary['eligible_issuance_projects']}.", "",
                  "## Model evaluation", "", "| Candidate | Status | Observations | Reason / limit |", "|---|---|---:|---|"])
    for candidate in evaluation["candidates"]:
        reason = candidate.get("reason") or "; ".join(candidate.get("reasons", []))
        lines.append(f"| {candidate['name']} | {candidate['status']} | {candidate.get('observations', 0)} | {reason} |")
    baselines = next(c for c in evaluation["candidates"] if c["name"] == "annual_baselines")
    if baselines.get("results"):
        lines.extend(["", "Expanding-window diagnostics on reported mixed-category annual issuance (tonnes). "
                      "The newest reported fiscal year is excluded because its completeness is not established. "
                      "These are exploratory errors, with no independent final test after candidate comparison.", "",
                      "| Baseline | MAE (tonnes) | RMSE (tonnes) | Backtest origins |", "|---|---:|---:|---:|"])
        for row in baselines["results"]:
            lines.append(f"| {row['model']} | {row['mae']:.3f} | {row['rmse']:.3f} | {row['n']} |")
    lines.extend(["", "## Factors and underwriting", "",
                  "Observed factors: methodology/version, status, location/management, registered expected annual tonnes, "
                  "certified-cycle tonnes, crediting dates and Category 13 trading totals/range. Derived factors: age/duration, "
                  "approval recency, annualized issuance, descriptive issuance variation, documentation completeness, source timestamp age and relative project size.", "",
                  "There are no empirically validated insurance pricing weights or feature rankings. Predictive importance is not computed from invalid labels. "
                  "Unmeasured factors remain null: covered reversal/delivery frequency and severity, actual project buffer rate, liquidity, exact verification recency, matched expected-delivery shortfall, expenses and capital margin.", "",
                  "Dynamic assessments recompute reported operating status, dates, observed issuance, peer measurements and evidence gaps. "
                  "The rating remains `NOT_RATABLE` when loss/hazard calibration is absent; completeness and age are not substituted for physical risk.", "",
                  "## Evidence and Gemini", "", f"Gemini model-review execution: `{qc['status']}`.",
                  f"Indexed {retrieval['chunks']} chunks from {retrieval['sources_indexed']} official sources. "
                  f"Sources with insufficient/unusable extracted text: {len(retrieval.get('skipped', []))}. "
                  f"Sources with recorded PDF parser warnings: {len(retrieval.get('extraction_warnings', []))}; details are in `retrieval_index.json`.",
                  "RAG uses local character TF-IDF plus SVD embeddings and FAISS cosine search. Every supplied excerpt is checked against the immutable raw source. "
                  "Source integrity does not prove semantic support. Gemini cannot override failed deterministic gates or create a premium.", "",
                  "## Limitations", "",
                  "- Category 13 aggregates mix removal and emission reduction. No category average is copied onto project training rows.",
                  "- Zero issuance can mean an unissued/censored project, not delivery failure; such rows are not zero-loss labels.",
                  "- Requested issuance is distinct from certified issuance; cycle amounts are matched to the official listing.",
                  "- Current methodology/status/size values lack point-in-time historical snapshots; model promotion is blocked without independent verification.",
                  "- Programme framework dates and component crediting dates are distinct. Component quantities are not added to parent quantities.",
                  "- Premium buffer rules are not presumed to apply to Standard projects. No buffer percentage is invented.",
                  "- Source publication timestamps, fiscal-year labels and retrieval timestamps are distinct; a fresh download is not a fresh market observation.",
                  "- PDF coverage is bounded; OCR is not silently substituted. Missing/scanned text and fetch failures are disclosed.",
                  "- A versioned champion registry preserves abstention or the previous approved model; unchanged inputs do not trigger retraining.", "",
                  "## Audit artifacts", "",
                  "See `source_inventory.json`, `numeric_lineage.csv`, `factor_catalog.csv`, `model_evaluation.json`, "
                  "`model_selection.json`, `evidence_<project_id>.json`, `qc_project_<project_id>.json`, "
                  "`assessment_<project_id>.json` and `adaptive_cycle.json` in this directory.", "",
                  f"Ingestion exceptions: {len(ingestion['errors'])}. Full records are in `ingestion.json`."])
    report = "\n".join(lines) + "\n"
    (root / "outputs/MODEL_EVALUATION.md").write_text(report)
    return report
