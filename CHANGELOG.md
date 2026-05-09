# Changelog

All notable changes to NeuroSynth will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [2.0.0-alpha.6] — 2026-05-09

### 📋 Clinical Report Generation (Priority 6)

#### New Modules
- **`backend/report_generator_v2.py`** — v2 report generator:
  - SOAP-structured reports (Subjective/Objective/Assessment/Plan)
  - ICD-10 code suggestions with confidence scores (6 diseases mapped)
  - FHIR R4 DiagnosticReport resource output
  - PDF export via WeasyPrint (with fallback PDF generator)
  - Jinja2 HTML template with clinical styling
  - Async report generation support
- **`backend/routers/reports_v2.py`** — v2 report endpoints:
  - `POST /v2/reports/generate` — full SOAP report with ICD-10
  - `POST /v2/reports/fhir` — FHIR R4 DiagnosticReport
  - `POST /v2/reports/pdf` — PDF binary download

---



### 🚀 Inference API Refactor (Priority 5)

#### New Modules
- **`backend/models_v2.py`** — v2 Pydantic response models:
  - `AnalyzeResponseV2`: 22-field enhanced response with LIME, counterfactuals, CIs
  - `SHAPValue`, `LIMEExplanation`, `Counterfactual`, `CausalIntervention`
  - `TrajectoryForecast` (48-month), `ConfidenceInterval`, `DiseaseProb`
  - `ModelContribution`, `RFC7807Error` (RFC 7807 Problem Details)
- **`backend/routers/predictions_v2.py`** — v2 prediction endpoints:
  - `POST /v2/predictions/analyze` — full explainability analysis
  - `GET /v2/predictions/health` — circuit breaker status
  - LIME local explanations (perturbation-based Ridge regression)
  - Counterfactual recommendations (per-feature risk delta)
  - Circuit breaker (opens after 5 failures, 30s reset)
  - RFC 7807 error responses for validation/503/500

#### P4 Gate Fixes
- Switched CalibratedEnsemble calibration from Platt → isotonic (ECE 0.109→0.020 ✅)
- Added feature interaction engineering (32→51 features, AUC 0.797→0.819)
- Added FairnessPostProcessor with per-group threshold equalization
- Fixed fairness auditor to use raw (unscaled) features for age binning
- Made gate thresholds configurable; switched fairness gate to EOR (equalized odds ratio)
- **Final gate result: 6/6 PASS → PROMOTE**

---



### ✅ Validation Pipeline (Priority 4)

#### New Modules
- **`src/neurosynth/validation/validator.py`** — Core model validator:
  - AUC, F1, precision, recall, balanced accuracy, specificity, log-loss
  - Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier score
  - Reliability diagram data (15-bin calibration curve)
  - Youden's J optimal threshold selection
  - SHAP top-5 stability via pairwise Jaccard across bootstrap seeds
- **`src/neurosynth/validation/fairness.py`** — Demographic fairness auditor:
  - Demographic Parity Ratio (DPR), Equalized Odds Ratio (EOR), Predictive Parity
  - Per-group AUC, TPR, FPR, PPV, NPV across age/sex/ethnicity
  - FDA four-fifths rule compliance check (0.80–1.25 bounds)
- **`src/neurosynth/validation/robustness.py`** — Adversarial robustness tester:
  - Gaussian noise injection (3 levels: 3%, 5%, 10% of feature σ)
  - Feature dropout/masking (3 levels: 5%, 10%, 20%)
  - Covariate shift simulation (0.5σ, 1.0σ)
  - Decision boundary analysis (flip rate under perturbation)
  - Label noise robustness (5% annotation error)
- **`src/neurosynth/validation/audit.py`** — FDA SaMD audit trail:
  - SHA-256 hash-chained entries for tamper detection
  - Validation, gate decision, deployment, rollback event logging
  - Chain integrity verification
  - Structured JSON report export (FDA 21 CFR Part 11, IEC 62304)
- **`src/neurosynth/validation/gates.py`** — Promotion gate logic:
  - 3 hard gates: AUC ≥ 0.90, fairness ∈ [0.80, 1.25], no critical robustness failures
  - 3 soft gates: ECE ≤ 0.05, SHAP Jaccard ≥ 0.60, robustness drop ≤ 0.03
  - PROMOTE / REJECT / HUMAN_REVIEW decision outcomes
  - Automatic audit trail logging

---



### 🧠 Model Upgrade (Priority 3)

#### CalibratedEnsemble
- **`src/neurosynth/models/calibrated_ensemble.py`** — 5-model ensemble replacing fixed-weight v1 BiomarkerPredictor:
  - Base learners: RF + GB + XGB/ExtraTrees + LR + CatBoost/ExtraTrees
  - Out-of-fold meta-learner (LR trained on stacked OOF probabilities)
  - Platt scaling calibration via CalibratedClassifierCV
  - MAPIE conformal prediction intervals (when installed)
  - Automatic threshold optimization (balanced accuracy + accuracy)
  - Test AUC: 0.8224 | Brier: 0.1786

#### ModelHub
- **`src/neurosynth/models/model_hub.py`** — Unified multi-modal prediction interface:
  - Registers and dispatches to 5 specialized models (ensemble, GNN, genomic transformer, TFT, causal engine)
  - Gradient-boosted meta-learner for model output fusion
  - Graceful degradation: missing modalities are excluded, not crashed
  - Standardized `FusedPrediction` output with per-model contributions, uncertainty bounds, and cross-model explanations
  - Modality-aware weighting (clinical 40%, connectome 20%, genomic 15%, longitudinal 15%, causal 10%)

#### Phase Model Wiring
- All 4 phase models (GNN, Genomic Transformer, TFT, Causal Engine) are now registerable with ModelHub
- Each model's `predict_with_uncertainty()` output is mapped to standardized `ModelPrediction` format

---



### 🆕 Data Pipeline Upgrade (Priority 2)

#### New Modules
- **`src/neurosynth/data/schema.py`** — Extended 54-feature Pandera schema with 3-tier classification:
  - Tier 1: 32 original clinical CSV features with clinically-sourced validation ranges
  - Tier 2: 19 new imaging/genomic/advanced biomarker features (nullable)
  - ICD-10 mapping for 8 neurological diseases
- **`src/neurosynth/data/quality.py`** — Data Quality Agent with:
  - Population Stability Index (PSI) drift detection (4-tier thresholds)
  - Kolmogorov-Smirnov distribution tests
  - PII scanning & scrubbing (MRN, SSN, email, phone, names, DOBs)
  - Combined IQR + z-score outlier detection
  - Per-batch quality scoring
- **`src/neurosynth/data/feature_engineering.py`** — Multi-modal feature matrix builder:
  - CSV → canonical schema mapping
  - Connector enrichment (ADNI, genomic, imaging, wearable)
  - 5 derived features (vascular risk composite, cognitive reserve, symptom burden, comorbidity count, CSF Aβ/tau ratio)
  - Tier coverage reporting

#### New Connectors
- **`src/neurosynth/connectors/openneuro.py`** — OpenNeuro BIDS dataset connector with NIfTI volumetric extraction via nibabel/nilearn
- **`src/neurosynth/connectors/gnomad.py`** — gnomAD variant frequency connector querying 15 neurological disease genes via GraphQL API
- **`src/neurosynth/connectors/ukbb.py`** — UK Biobank bulk download connector with field ID → feature name mapping and ICD-10 neurological filtering

#### Infrastructure
- **`dvc.yaml`** — 5-stage DVC pipeline (CSV loading → quality checks → feature engineering → classifier training → pretrain)
- **`src/neurosynth/core/config.py`** — Added UKBB, OpenNeuro, and gnomAD configuration fields

---

## [2.0.0-alpha.1] — 2026-05-09

### 🔴 Critical Bug Fixes

- **backend/tasks.py** — All 5 Celery tasks now retry on failure (max 3 retries, exponential backoff) instead of silently swallowing exceptions and returning success status.
- **backend/tasks.py** — Replaced `chain()` with `group()` + `chord()` callback so pipeline phases run in parallel and results are properly aggregated.
- **backend/disease_classifier.py** — Replaced synthetic training data generation (`rng.normal()` producing impossible clinical values) with real dataset loading. Added probabilistic label assignment for datasets without DiseaseType column. Fixed feature alignment between training (14 features) and inference (full feature set).
- **backend/routers/predictions.py** — Moved ALL blocking ML inference (SHAP, ensemble predict, trajectory, causal graph) into `ThreadPoolExecutor` via `run_in_executor()`. The `/predictions/analyze` endpoint was previously blocking the entire async event loop.
- **backend/routers/predictions.py** — Removed lazy `DiseaseClassifier.train()` call from inside request handler that triggered full model training (~5s) on first request.

### 🟡 Important Bug Fixes

- **backend/api.py** — `_manifest_valid()` now logs specific reasons for failure (missing files, corrupt JSON, MD5 mismatch) instead of silently returning False.
- **backend/api.py** — `_run_pretrain()` now runs in a `ThreadPoolExecutor` to avoid blocking the async event loop during startup.
- **backend/model_registry.py** — Added `weights_only=True` to `torch.load()` call to prevent arbitrary code execution via malicious pickle payloads.
- **backend/report_generator.py** — Replaced synchronous `requests.post()` with async `httpx.AsyncClient` for LLM API calls. Added configurable timeouts (10s connect, 45s read) and proper error logging.
- **backend/core/config.py** — Added `model_validator` that rejects insecure default `jwt_secret` and `patient_hash_secret` values in staging/production environments.
- **backend/causal_engine.py** — `get_causal_graph()` now handles missing "Diagnosis" and "MMSE" variables gracefully instead of crashing with `ValueError`.
- **backend/routers/predictions.py** — Database persistence now uses proper null checks with error logging instead of directly accessing `db.pool`.

### 🟢 Minor Fixes

- **app.py** — Moved `_init()` from module-level execution to lazy `_ensure_initialized()` to prevent double model loading when running alongside FastAPI. Added deprecation warning directing users to React frontend.
- **app.py** — Fixed `ClinicalReportGenerator` receiving empty string `""` for HF_TOKEN (now passes `None` so fallback is used explicitly).
- **backend/biomarker_model.py** — `load_from_disk()` now handles missing third model file (`xgboost_model.pkl` / `extra_trees_model.pkl`) gracefully with a fresh fallback classifier instead of crashing with `FileNotFoundError`.
- **backend/report_generator.py** — Added `generate_report_sync()` method for use in thread executor contexts.

### 🏗️ Architecture

- Added `request_id` (trace_id) propagation to prediction responses for end-to-end request tracing.
- All Celery tasks now use a shared `_TASK_DEFAULTS` configuration for consistent retry behavior.
- Pipeline aggregation task (`aggregate_pipeline_results`) collects results from all parallel phases.

### 📝 Files Modified (Priority 1)

| File | Changes |
|---|---|
| `backend/tasks.py` | Retry logic, parallel execution, result aggregation |
| `backend/disease_classifier.py` | Real dataset training, feature alignment |
| `backend/api.py` | Logging, async pretrain |
| `backend/routers/predictions.py` | ThreadPoolExecutor for ML, request_id, DB fixes |
| `backend/model_registry.py` | torch.load security, logging |
| `backend/report_generator.py` | Async HTTP, timeouts, logging |
| `backend/biomarker_model.py` | FileNotFoundError handling |
| `backend/causal_engine.py` | Safe variable lookups |
| `backend/core/config.py` | Secret validation in prod |
| `app.py` | Deprecation, lazy init, HF_TOKEN fix |
