# TGO carbon-absorption credit insurance research

A Python-only, evidence-constrained pipeline implementing the supplied handoff. It ingests real official TGO data, separates absorption activities from the wider FOR&AGR category, evaluates models only when their data gates pass, retrieves TGO evidence, and produces auditable project assessments. No frontend or deployment.

**The included run does not establish an actuarial insurance premium.** TGO trading prices value credits; they are not premiums. Missing claims, covered reversal losses, exposure histories, policy terms and insurer costs remain missing. See `data/outputs/MODEL_EVALUATION.md` for executed results and `docs/SPEC.md` for the original requirements.

## Install

Python 3.11+; tested here with Python 3.12. Use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
# CPU PyTorch keeps a Linux research installation smaller:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
```

On Apple Silicon, install the normal `torch` package rather than the Linux CPU wheel index. `pyproject.toml` declares project dependencies; `environment.lock.txt` records the actual execution environment (including runtime-provided packages), not a portable cross-platform lock.

## Run

All commands emit JSON to stdout and progress/errors to stderr. Global options go before the command.

```bash
python -m carbon_insurance --config config.yaml run-all
python -m carbon_insurance --config config.yaml ingest
python -m carbon_insurance --config config.yaml features
python -m carbon_insurance --config config.yaml build-index
python -m carbon_insurance --config config.yaml train
python -m carbon_insurance --config config.yaml evaluate
python -m carbon_insurance --config config.yaml assess --project-id 449
```

`project-id` is the integer at the end of the official project URL. Project URL ID **449** has registration number **393**; these identifiers are not interchangeable. Programme 667 and Premium project 718 are also sample assessments.

The GitHub repository contains source, offline test fixtures and selected executed reports in `reports/`. The downloadable project archive also contains the full `data/` snapshot. A fresh GitHub clone must run online ingestion once; offline replay requires that saved snapshot.

With the data snapshot present, run:

```bash
python -m carbon_insurance --config config.yaml run-all --offline
python -m pytest
```

Offline replay checks saved raw-file hashes and preserves source retrieval/publication dates. It does not claim that observations were refreshed. On unchanged inputs, `run-all` reuses the existing candidate evaluation. Use `--force-train` only for an intentional reproducibility experiment. This is a manually invoked adaptive cycle, not a background scheduler.

## Gemini

```bash
export GEMINI_API_KEY='your-key-in-your-local-shell'
export GEMINI_MODEL='a-current-model-supporting-structured-outputs'
python -m carbon_insurance --config config.yaml assess --project-id 449
```

Alternatively use `GOOGLE_API_KEY`. `.env.example` is a template; `.env` files are not automatically loaded. Never commit credentials. The configurable sample model is `gemini-3.8-flash`; verify availability in your account. The integration uses the `google-genai` SDK with a Pydantic response schema. Without a credential it records `NOT_EXECUTED_MISSING_CREDENTIAL`, without simulating approval. A live Gemini call was not performed in the included run. Public TGO passages and project metadata are the only evidence sent to it.

Gemini checks each proposed factor, insurance relevance, statistical credibility, model disagreement, eligibility and missing evidence. Returned factor names and chunk IDs are checked independently. Invalid citations, invalid schema and acceptance despite failed gates are rejected. Text inside documents is evidence, never an instruction to execute. A valid JSON response or matching citation alone does not prove that an assertion is true.

## What the pipeline does

- Discovers the four pinned TGO CSV resources through their actual CKAN `package_show` metadata. The official CSV URL remains a disclosed fallback if discovery fails.
- Parses the registry's real HTML form, then its observed `POST /database/Public/ProjectsResult` response. Standard Category 13 currently uses form value 13; Premium uses 28. It discovers these values from the forms instead of assuming a shared ID.
- Preserves immutable raw bytes, SHA-256 hashes, request identities, retrieval timestamps, response metadata and fetch-check history in SQLite. Changed sources produce new versions; failed fresh fetches do not silently use stale data.
- Filters FOR&AGR exactly and uses official methodology-family definitions and archived version tables to distinguish absorption, reduction-only, mixed/unsegregated and unresolved activities, including legacy forestry codes. A version-history entry does not prove project compliance with that methodology.
- Separates expected annual tonnes, requested issuance, certified-cycle quantities, programme framework dates and component crediting dates. It never divides aggregate annual issuance by aggregate registrations or labels an unissued project as a delivery failure.
- Downloads a configurable, bounded sample of project PDFs and the Category 13 methodology/guidance PDFs, retaining all discovered document links. Scanned or unusable PDF text is disclosed; no OCR content is invented.
- Builds local character TF-IDF/SVD embeddings and a FAISS cosine index. This is an explicit alternative local embedding system, not a Sentence Transformer. It needs no model download and handles Thai character sequences. Retrieval scores are similarity, not factual confidence. Every retrieved excerpt is rechecked against the saved original source; other projects and incompatible programmes are filtered out.

## Modeling and selection

**CatBoost:** implemented for first-cycle annualized certified tonnes among projects that actually issued credits. One label per project, chronological date-group splits, early stopping on validation only, a validation-selected median/registered-expectation baseline, MAE/RMSE, held-out permutation importance and CatBoost SHAP values when sample gates pass. This target is conditional on issuance and is not an insured-loss, reversal or shortfall model. Current registry features lack historical point-in-time provenance; promotion remains blocked unless that is independently established. The current ingestion deliberately never asserts that it has this historical evidence.

**LSTM:** implemented in PyTorch with chronological windows, train-only scaling, validation-only early stopping, saved checkpoints and inference. It compares naive, annual seasonal-naive, expanding mean/median and drift on exactly the same test origins, choosing its comparison baseline on validation only. At least 60 usable temporal observations are required. The currently identified annual series is much shorter. It returns `INSUFFICIENT_DATA` without fitting or forecasting. A future mixed Category 13 series would still be marked outside absorption-only insurance prediction scope.

**Baselines:** expanding-window naive, annual seasonal-naive (equivalent to naive at period 1), expanding mean, expanding median and drift. The latest reported fiscal-year total is excluded because its completeness is not established. Baseline errors are exploratory diagnostics; few annual origins and mixed activity do not support a project-insurance prediction.

**Champion/challenger:** all candidates retain diagnostics in separate artifacts. Promotion needs sufficient independent validation, paired error improvement, interval coverage, verified historical features, scope/data quality and Gemini acceptance. Replacing a previous champion additionally requires a paired comparison on at least 20 new held-out projects unseen in every previous champion partition. The prior model is preserved when those artifacts or evidence are absent; there is no cross-snapshot MAE shortcut. The current state is explicit abstention.

Primary output selection is target-aware: an issuance prediction cannot be promoted into a price. Where no validated absorption-price or insurance-loss estimate exists, the selected numeric context is the observed Category 13 weighted trading benchmark, clearly marked as mixed activity; project valuation, sum insured and premium remain null.

## Dynamic risk assessment

Assessments refresh date-dependent features in Thailand civil time, while audit timestamps retain UTC offsets. They recompute issuance observations, relative project size and documentation evidence. Components cover permanence, issuance performance, methodology/verification, operations, market exposure, liquidity, documentation and model uncertainty. Validated Gemini rejection/eligibility/support findings affect factor acceptance, warnings and evidence requests. A target-aware selector consults the model registry for supported operational indicators; an issuance quantity never becomes a price. They do not assign arbitrary weights or turn missing evidence into physical risk. Without covered-loss/hazard calibration the numerical score and model confidence remain null, with `NOT_RATABLE` and `ACTUARIAL_PREMIUM_NOT_ESTIMABLE` statuses.

The insurance framework is `Expected insured loss + Expenses + Risk/capital margin`. No term is numerically invented. Buffer applicability guidance is not an estimated buffer rate or insurer recovery, and Premium requirements are not applied to Standard projects by default.

## Outputs

`data/outputs/assessment_<id>.json` contains the selected result and rating. Model comparisons, failed candidates, full retrieved passages and QC results are separate files. `source_inventory.json`, `numeric_lineage.csv`, `dataset_metadata.csv`, `factor_catalog.csv` and the processed Parquet tables retain evidence and definitions. `data/lineage.sqlite` holds source versions and field facts. `MODEL_EVALUATION.md` summarizes actual row counts, backtests and limits.

Runtime paths in configuration are relative to your working directory. Models, index files and SQLite are local artifacts. Load only artifacts you trust: the index uses joblib, which is not a safe format for untrusted third-party files.

## Verification scope

Offline tests use actual saved TGO responses. Stubbed HTTP/Gemini responses only exercise validation; they are not training observations or research evidence. Tests cover category exclusion, units, date conversion, identifiers, programmes, requested/certified provenance, source integrity and redirect restrictions, missing credentials, citation checks, failed promotion, chronological backtests, minimum-data abstention and the LSTM input/output shape.

The code and included evidence snapshot are reproducible research scaffolding. Live inference, actuarial calibration, historical feature reconstruction, exhaustive PDF extraction and insurer-specific underwriting need additional real permitted evidence; they are not claimed as completed capabilities.
