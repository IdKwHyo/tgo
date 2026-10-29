# TGO carbon-absorption insurance research — executed results

Generated 2026-10-29T19:09:35.000585+00:00. Data version `df2a54129c689f5eae40dd9160e4182496209a00967fdd77a6b55fcbb5040fcc`.

## Selected result

No actuarially defensible insurance premium or calibrated physical-risk score can be estimated from this snapshot.
The selected numerical result is only a directly observed, mixed Category 13 market context. It is not a removal-only project price, sum insured or premium.

Context benchmark: **411.2086355813 THB/tCO2e**. Source timestamp freshness: `STALE_SOURCE_TIMESTAMP`.

## Actual ingestion

| Dataset | Nonempty rows | FOR&AGR rows | Excluded rows |
|---|---:|---:|---:|
| trading | 8 | 1 | 7 |
| premium_expected | 45 | 3 | 42 |
| standard_expected | 195 | 13 | 182 |
| standard_issuance | 180 | 12 | 168 |

Discovered 257 project records; parsed 257. Parsed 35 issuance events and 31 programme components. Discovered 1188 linked files; downloaded 12 project PDFs.

| Activity classification | Projects |
|---|---:|
| ABSORPTION_METHOD_FAMILY | 123 |
| MIXED_OR_UNSEGREGATED | 101 |
| REDUCTION_ONLY | 9 |
| UNRESOLVED_METHODOLOGY | 24 |

Usable first-issuance absorption-project candidates: 9.

## Model evaluation

| Candidate | Status | Observations | Reason / limit |
|---|---|---:|---|
| catboost | INSUFFICIENT_DATA | 9 | No period-matched expected-delivery or insured-loss labels identified; Not enough independent absorption-project issuance labels |
| lstm | INSUFFICIENT_DATA | 11 | Not enough genuinely temporal observations; no LSTM training or forecast executed |
| annual_baselines | EXPLORATORY_ONLY | 11 | Very few backtest origins; No independent test after comparing baseline candidates; Aggregate target is mixed activity and not an insurance exposure |

Expanding-window diagnostics on reported mixed-category annual issuance (tonnes). The newest reported fiscal year is excluded because its completeness is not established. These are exploratory errors, with no independent final test after candidate comparison.

| Baseline | MAE (tonnes) | RMSE (tonnes) | Backtest origins |
|---|---:|---:|---:|
| naive | 167684.000 | 235078.338 | 5 |
| seasonal_naive_annual | 167684.000 | 235078.338 | 5 |
| expanding_mean | 113096.501 | 196400.906 | 5 |
| expanding_median | 117596.400 | 202530.853 | 5 |
| drift | 174362.867 | 248357.488 | 5 |

## Factors and underwriting

Observed factors: methodology/version, status, location/management, registered expected annual tonnes, certified-cycle tonnes, crediting dates and Category 13 trading totals/range. Derived factors: age/duration, approval recency, annualized issuance, descriptive issuance variation, documentation completeness, source timestamp age and relative project size.

There are no empirically validated insurance pricing weights or feature rankings. Predictive importance is not computed from invalid labels. Unmeasured factors remain null: covered reversal/delivery frequency and severity, actual project buffer rate, liquidity, exact verification recency, matched expected-delivery shortfall, expenses and capital margin.

Dynamic assessments recompute reported operating status, dates, observed issuance, peer measurements and evidence gaps. The rating remains `NOT_RATABLE` when loss/hazard calibration is absent; completeness and age are not substituted for physical risk.

## Evidence and Gemini

Gemini model-review execution: `NOT_EXECUTED_MISSING_CREDENTIAL`.
Indexed 3385 chunks from 273 official sources. Sources with insufficient/unusable extracted text: 1. Sources with recorded PDF parser warnings: 3; details are in `retrieval_index.json`.
RAG uses local character TF-IDF plus SVD embeddings and FAISS cosine search. Every supplied excerpt is checked against the immutable raw source. Source integrity does not prove semantic support. Gemini cannot override failed deterministic gates or create a premium.

## Limitations

- Category 13 aggregates mix removal and emission reduction. No category average is copied onto project training rows.
- Zero issuance can mean an unissued/censored project, not delivery failure; such rows are not zero-loss labels.
- Requested issuance is distinct from certified issuance; cycle amounts are matched to the official listing.
- Current methodology/status/size values lack point-in-time historical snapshots; model promotion is blocked without independent verification.
- Programme framework dates and component crediting dates are distinct. Component quantities are not added to parent quantities.
- Premium buffer rules are not presumed to apply to Standard projects. No buffer percentage is invented.
- Source publication timestamps, fiscal-year labels and retrieval timestamps are distinct; a fresh download is not a fresh market observation.
- PDF coverage is bounded; OCR is not silently substituted. Missing/scanned text and fetch failures are disclosed.
- A versioned champion registry preserves abstention or the previous approved model; unchanged inputs do not trigger retraining.

## Audit artifacts

See `source_inventory.json`, `numeric_lineage.csv`, `factor_catalog.csv`, `model_evaluation.json`, `model_selection.json`, `evidence_<project_id>.json`, `qc_project_<project_id>.json`, `assessment_<project_id>.json` and `adaptive_cycle.json` in this directory.

Ingestion exceptions: 0. Full records are in `ingestion.json`.
