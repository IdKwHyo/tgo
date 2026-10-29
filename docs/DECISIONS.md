# Implementation decisions

The supplied SPEC is authoritative. This file records implementation choices and review standards.

- Keep official source bytes immutable. Numeric observations retain official values, units and locators; derived quantities identify formulas and input sources. Never substitute source retrieval time for observation time.
- Admit only official HTTPS TGO sources. Follow registry form behavior actually observed in public HTML; do not assume undocumented API schemas.
- Distinguish Category 13 membership, absorption-method-family classification and verified removal eligibility. Mixed or unresolved methodologies do not supply removal-only training labels.
- Missing, non-comparable and censored values are explicit. Never manufacture training observations, claims, insurance rates, premiums or physical-risk weights.
- One first-issued project label per project, chronological splits, and no same-day approval labels spanning partitions. Feature availability at prediction time must be established before promotion.
- Candidate metrics are diagnostics unless independent validation and calibration pass. Current-data sufficiency gates were configured before inspecting project-label counts: CatBoost 40 projects, LSTM 60 observations. These operational thresholds alone do not prove statistical credibility.
- Character TF-IDF plus truncated SVD is the selected local embedding system. FAISS scores are similarities, not probabilities. Mechanical excerpt verification is independent of Gemini; semantic correctness still needs evaluation.
- Gemini recommendations never override deterministic failures. Missing credentials record non-execution, with no replacement LLM or fabricated review.
- Default output is the selected result. Detailed diagnostics, rejected models, evidence passages and lineage remain separate audit artifacts.
- Tests exercise meaningful domain behavior using real saved TGO responses. Stubs test transport/validation only; they never become model observations or research results.
- Keep Python modules centered on ingestion, feature definitions, modeling, evidence retrieval, review and assessment. Do not add UI, deployment, external issue trackers, paid data or third-party carbon datasets.

The conservative current result can be abstention. The model architectures being implemented does not imply they were fitted or validated.
