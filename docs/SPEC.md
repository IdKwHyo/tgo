# WORK HANDOFF — TGO Carbon Credit Insurance Pricing AI

## Objective

Build a production-oriented, Python-only machine-learning research pipeline that determines the most defensible pricing factors and dynamic risk ratings for **insurance covering Thai carbon-absorption credits**.

**No frontend, web app, dashboard, or deployment is required.**

Implement a complete, executable Python project, with real public data ingestion, CatBoost, LSTM, Gemini, retrieval-augmented generation (RAG), model evaluation, adaptive retraining, dynamic risk scoring, and auditable outputs.

Do not stop at an architecture proposal. Write the code, run tests, execute the pipeline with real available data, and report actual results and limitations.

### 1. Strict data scope

Use only publicly available, real TGO (Thailand Greenhouse Gas Management Organization) data relevant to the **FOR&AGR category (Category 13)**.

Distinguish actual carbon absorption/sequestration projects from other forestry/agricultural activities where the registry allows this. Do not assume every FOR&AGR project represents an identical removal activity.

Exclude unrelated project categories, including REF, EEP, EEB, SWM, and CCUS.

Do not use invented observations, synthetic insurance claims, fabricated loss histories, fictional training rows, or arbitrary premiums.

### 2. Verified initial TGO data sources

**Official FY2569 T-VER trading statistics:**

https://opendata.tgo.or.th/dataset/061c3709-1cdb-4af8-85f6-4c7cc6ce2b51/resource/1d9ad8ca-70c9-4f6e-bbd5-60eacb09d70e/download/cc_02_02.csv

Known FOR&AGR values:
- Volume: 375,933 tCO₂e
- Total value: ฿154,586,896
- Minimum price: ฿55/tCO₂e
- Maximum price: ฿3,000/tCO₂e
- Weighted average price: ฿411.2086356/tCO₂e

**Premium T-VER annual expected GHG reductions statistics:**

https://opendata.tgo.or.th/dataset/6c14f00b-31b4-49d4-b981-947c10bcab67/resource/dcb909e1-23c6-417e-9fe0-d6bd0491b116/download/pre_02_02.csv

**Standard T-VER annual registered expected reduction statistics:**

https://opendata.tgo.or.th/dataset/b9f275e5-6bbf-4c3b-907f-3a17c1540259/resource/bcd5564c-b149-4bb4-814b-55fec10a32b8/download/ra_02_02.csv

**Standard T-VER annual credit issuance statistics:**

https://opendata.tgo.or.th/dataset/90019e9a-228c-4374-bedb-418f761a7a1a/resource/2ee69181-7498-44f9-9ea6-18ae5292884d/download/ra_03_02.csv

**T-VER registered project database:**

https://tver.tgo.or.th/database/public/projects/1/1?lang=th

https://tver.tgo.or.th/database/public/project/1/1/449?lang=th

https://tver.tgo.or.th/database/public/project/1/2/667?lang=th

https://tver.tgo.or.th/database/public/project/2/1/718?lang=th

Investigate additional publicly available TGO sources for approved methodologies, project monitoring, verification, non-permanence risks, reserves, project status, and carbon-credit statistics.

Do not assume these are APIs. Inspect their actual document formats, download files where possible, and implement robust parsers for the official HTML/CSV structure.

Translate Buddhist Era years correctly, while preserving the original TGO fiscal-year definitions.

### 3. Required technology stack

- Python 3.11+
- pandas, NumPy, SciPy
- CatBoost
- PyTorch LSTM
- scikit-learn for baselines and evaluation
- google-genai SDK for Gemini
- Sentence Transformers or another suitable local embedding system
- FAISS or ChromaDB for document retrieval
- BeautifulSoup, requests/httpx for TGO ingestion
- SQLite for metadata and lineage
- Parquet for processed datasets
- Pydantic for schema validation
- pytest for automated tests

Use environment variables for credentials. Never hardcode API keys.

### 4. Data ingestion pipeline

Build a reusable pipeline capable of:

1. Discovering and downloading approved TGO datasets.
2. Filtering strictly for eligible FOR&AGR records.
3. Retrieving individual project records and linked public documentation.
4. Extracting project IDs, locations, methodologies, status, project size, expected tonnes, monitoring and verification information, and dates.
5. Keeping actual reported issuance separate from expected reductions.
6. Detecting duplicates, stale data, missing records, and incompatible definitions.
7. Saving raw immutable data with URL, retrieval timestamp, source hash, and original values.
8. Producing validated feature tables.
9. Detecting new or changed source data on subsequent runs.

Maintain source provenance for every numeric input.

Do not treat aggregate yearly issuance and aggregate yearly registrations as matched project cohorts without proof that they refer to comparable periods and populations.

### 5. Feature discovery

Investigate and quantify only factors that can be grounded in TGO data.

Candidate features include:

- Carbon-credit market value and trading volume
- Trading price dispersion
- Project methodology and version
- Project registration status
- Project size
- Crediting-period duration
- Project age
- Expected versus verified credit issuance, where genuinely comparable
- Monitoring and verification recency
- Project location
- Project ownership/management details where publicly supported
- Carbon-reversal and non-permanence indicators
- TGO buffer or reserve-credit requirements
- Historical issuance variation
- Available project-specific risk disclosures
- Data completeness and source staleness

Distinguish three classes of factors:

**Observed:** directly measured from official TGO data.

**Derived:** mathematically calculated from observed records.

**Unverified:** theoretically relevant but not measurable using the permitted datasets.

Do not assign unverified factors invented values.

### 6. CatBoost model

Use CatBoost for structured tabular data.

Candidate tasks:
- Predict verified credit issuance or issuance shortfall, if valid project-level labels are available.
- Estimate project-level operational/performance risk indicators.
- Analyze the relative predictive value of eligible pricing and risk factors.
- Predict actual traded credit prices only if transaction-level or sufficiently granular price labels exist.

Do not train a project-level pricing model by attaching the same category-average price to every project.

Use temporal or grouped validation, prevent project leakage, compare against simple baselines, and report MAE, RMSE and appropriate risk metrics.

Provide permutation/SHAP-based explanations where supported.

### 7. LSTM model

Use PyTorch LSTM for genuinely temporal series.

Potential targets:
- FOR&AGR issuance dynamics
- Credible project-level time sequences
- Historical market changes, if enough observations exist

**Important:** The currently identified annual TGO datasets contain only a small number of yearly observations. Do not force an LSTM to learn from an insufficient series.

Implement minimum-data checks, chronological splitting, and comparison with naive, seasonal-naive, and statistical baselines.

If insufficient observations exist, report `INSUFFICIENT_DATA` and disable the LSTM prediction. Keep the implemented architecture ready for future data.

Never label overfitting as successful adaptive learning.

### 8. Gemini + RAG decision and quality-control layer

Gemini is the **decision-support and evidence-review component**, not an unrestricted numeric price generator.

Create a searchable RAG knowledge base containing only permitted TGO methodology guidance, non-permanence requirements, relevant project documents, verification rules, and other official TGO material.

Each retrieved passage must retain:
- Source URL
- Document title
- Date/version, where available
- Text excerpt
- Chunk ID
- Retrieval score

Gemini receives:
- Validated input data
- CatBoost predictions and diagnostics
- LSTM predictions and diagnostics
- Baseline results
- Retrieved TGO evidence
- Data quality and uncertainty assessments

Gemini must evaluate:
1. Whether each proposed factor is supported by retrieved evidence.
2. Whether the factor is relevant to the insurance exposure.
3. Whether quantitative results are statistically credible.
4. Whether models disagree materially.
5. Whether the result violates TGO methodology or eligibility constraints.
6. Whether additional evidence is necessary.

Use structured JSON responses with Pydantic validation.

Gemini must not create source citations, claims frequencies, missing input data, or invented premiums.

Require independently validated retrieval evidence. An LLM assertion is not evidence by itself.

### 9. Dynamic risk-rating engine

Produce a dynamic, explainable rating rather than a static hardcoded score.

Risk dimensions should include, where evidenced:
- Carbon permanence/reversal
- Issuance or delivery performance
- Verification and methodology
- Project operational status
- Market value exposure
- Liquidity and valuation uncertainty
- Documentation quality
- Model uncertainty

Calibrate numeric factors using observed data where possible.

Keep uncertainty and missing evidence separate from measured physical risk.

If loss frequency or severity cannot be estimated, report an **evidence-based risk assessment**, not an actuarially validated expected loss.

Provide:
- Dynamic risk score, only if defensibly calibrated
- Rating category
- Component scores
- Model confidence
- Missing-evidence warnings
- Main factors driving the rating
- Timestamp and data version
- Reasons for any declined assessment

Use clear `NOT_RATABLE` and `INSUFFICIENT_DATA` statuses when necessary.

### 10. Insurance pricing methodology

Clearly distinguish:

**Carbon-credit value:** market value of a verified tonne.

**Sum insured:** the insurable financial interest and coverage limit, after applying policy terms.

**Insurance premium:** expected insured loss plus expenses, risk margin and other justified components.

Do not present the carbon-credit trading price as an insurance premium.

A general pricing framework is:

Gross Premium = Expected Insured Loss + Expenses + Risk/Capital Margin

Estimate expected insured loss only when actual credible claims, reversal, or other coverage-specific loss data support frequency and severity estimation.

If those data are unavailable, return:
- Observed TGO carbon-credit valuation benchmark
- Risk-factor assessment
- Evidence quality
- Potential underwriting concerns
- Unpriced risk components
- `ACTUARIAL_PREMIUM_NOT_ESTIMABLE`

Do not invent loss ratios, insurance rates, or insurer margins.

### 11. Adaptive learning

Implement adaptive learning through an auditable retraining cycle:

TGO Data Refresh → Validation → Feature Engineering → Model Training → Backtesting → RAG Review → Gemini Quality Review → Candidate Approval → Model Versioning

Use a champion/challenger framework.

Only promote a newly trained model if it satisfies defined out-of-sample performance, calibration, data-quality, and evidence requirements.

Gemini may recommend accepting, rejecting, or investigating a candidate, but it cannot override numerical validation failures.

Maintain a previous approved model, or an explicit abstention state if no model has yet qualified.

### 12. Best-output selection

The system must evaluate and select the most defensible available result.

Selection precedence:

1. Valid, independently tested prediction with supporting TGO evidence.
2. Simpler validated statistical baseline if superior to complex ML.
3. Directly observed TGO benchmark when predictive validation is insufficient.
4. Explicit abstention if no defensible estimate exists.

Never automatically favor CatBoost, LSTM, or Gemini merely because it is more sophisticated.

Report only the selected final estimate/rating as the primary output, but retain model comparison, failed candidates, and audit logs as separate artifacts.

### 13. Required deliverables

Create a complete Python project containing:

- `requirements.txt` or `pyproject.toml`
- `.env.example`
- `README.md`
- TGO ingestion scripts
- Cleaning and feature-engineering modules
- CatBoost training and inference
- PyTorch LSTM training and inference
- RAG indexing and retrieval
- Gemini decision and QC integration
- Dynamic risk-rating module
- Champion/challenger model selection
- CLI entrypoint
- Automated tests
- Sample configuration
- JSON/CSV outputs
- Model evaluation report
- Data lineage and source inventory

Provide a CLI supporting commands such as:

`python -m carbon_insurance ingest`

`python -m carbon_insurance build-index`

`python -m carbon_insurance train`

`python -m carbon_insurance evaluate`

`python -m carbon_insurance assess --project-id <TGO_PROJECT_ID>`

`python -m carbon_insurance run-all`

Make offline tests possible without Gemini credentials. Missing Gemini credentials must never trigger fabricated quality-control results.

### 14. Definition of success

Run the project against real available TGO data.

Demonstrate:
- Actual imported row counts and usable records
- Correct FOR&AGR filtering
- Source validation
- Actual selected features
- Working model training where data suffices
- Real backtesting results or an explicit insufficient-data finding
- Retrieved TGO evidence
- Gemini quality-control behavior if a credential is available
- A reproducible sample project assessment
- A transparent risk rating and insurance pricing feasibility decision

Explicitly disclose anything that could not be trained, validated, fetched, or executed.

Do not generate synthetic datasets to make the demo look complete.

**Final principle: the best available output is the most empirically defensible output, even if that means a simple baseline, a qualified risk rating, or a decision not to quote an insurance premium.**

Build the functioning Python codebase rather than an application mockup, conceptual diagram, or proposal.
