# Review and verification

Build Router selected the Matt Pocock implementation/module-design workflow for this Python-only research project. The supplied handoff is `SPEC.md`; implementation choices are in `DECISIONS.md`.

## Standards review

A separate read-only review identified four correctness issues in the initial implementation. All were addressed:

- Cached assessment dates could produce stale recency values. Assessment now refreshes date-dependent features using Thailand civil time; UTC audit timestamps remain separate. Regression tested.
- A later usable issuance cycle could be mislabeled as the first. The first-issuance target now requires a usable official cycle 1. Regression tested with real project 8 responses.
- Promotion eligibility could be recorded as an actual promotion when a prior champion was retained. Those states are now separate; one final decision records the mutation. Regression tested.
- Concurrent ingestion order could change RAG builds. Sources/chunks are canonically sorted before vectorization and hashing. Reversed-source-order rebuild and citation-integrity regression tested.

## Spec review

A separate read-only requirements review found no fabricated premium or unsupported numerical physical-risk score. It identified three partial future paths, now addressed in code:

- Operational output selection now consults the champion registry, checks task/snapshot/input eligibility, and can select a supported issuance indicator. A target-aware selector prevents an issuance quantity from becoming a price and permits a superior validated baseline.
- Validated Gemini findings now affect factor acceptance, eligibility warnings, disagreement warnings and additional-evidence requests. They cannot override failed numeric gates or create prices.
- The LSTM implementation now compares all required baselines on its exact held-out origins, with baseline choice on validation only and explicit interval/error gates.

Further robustness work records PDF parser warnings, uses official archived methodology tables to resolve legacy forestry codes, fingerprints training implementations/configuration, and rejects failed candidates while retaining other results. Champion replacement requires a new paired held-out comparison rather than cross-snapshot MAE.

## Executed checks

The automated offline suite passed **26 tests**. A full CLI pipeline and subsequent reproducibility checks use the real saved TGO source snapshot. Exact data counts, baseline errors, source hashes, missing credentials and insufficient-data findings are recorded in `data/outputs/`.

No CatBoost or LSTM fit is claimed: the real data does not satisfy the configured gates. The LSTM architecture's shape test is not training or validation. No live Gemini call was made without a credential. Implemented future training/promotion paths are not represented as empirically validated merely because their code exists.
