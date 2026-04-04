# NeuroSynth

<p align="center">
  Multi-modal neurological intelligence for prediction, causality, and clinical report generation.
</p>

<p align="center">
  Imaging + Genomics + Longitudinal Biomarkers + Causal Discovery + Clinical LLMs
</p>

---

## What This Is

NeuroSynth is a production-oriented platform that fuses neuroimaging, wearable streams, genomics, and temporal clinical data to:

1. Forecast patient deterioration trajectories
2. Surface likely causal pathways behind risk progression
3. Generate schema-constrained, clinically structured intervention reports

It is designed as an end-to-end system, not a single model repo.

## Why It Matters

- Multi-modal by design: one platform across imaging, genomics, biomarkers, and language
- Clinically actionable: uncertainty-aware predictions and explainable outputs
- Causality-aware: counterfactual and intervention simulation support
- Deployment-ready: CI/CD, infra templates, security scanning, release contracts

## Architecture At A Glance

### Phase 1. Data + Lakehouse
- ADNI, PPMI, MIMIC, and wearable connectors
- DICOM processing and normalized table pipelines
- Apache Iceberg domain tables
- Knowledge graph and timeseries feature processing

### Phase 2. Brain Connectome Modeling
- fMRI and structural feature graph construction
- Temporal graph batches and sequence datasets
- GNN training scaffolding with uncertainty hooks

### Phase 3. Genomic Intelligence
- Genomic preprocessing and annotation pathways
- Hierarchical variant transformer components
- Risk scoring and multi-head prediction outputs

### Phase 4. Temporal Forecasting
- Longitudinal biomarker dataset factory
- TFT wrapper, calibration, and validation modules
- Progression forecasting with interval outputs

### Phase 5. Causal Engine
- Differentiable causal discovery workflows
- Patient-level causal analysis and counterfactual simulation

### Phase 6. Clinical LLM Stack
- Corpus preparation and staged training utilities
- Retrieval and schema-constrained report generation

### Phase 7. MLOps + Deployment
- FastAPI orchestration and Kubeflow integration
- Helm and Terraform deployment templates
- CI pipeline with tests, release gating, and security scanning

## Repository Map

- src/neurosynth: platform source modules
- tests: unit and integration tests
- scripts: operational and release scripts
- artifacts: generated summaries and manifests
- helm/neurosynth: chart templates and values
- terraform: infra modules and production var templates
- .github/workflows: CI/CD workflows

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
docker compose up -d
pytest -q
```

## Training + Artifact Publication

Preview orchestration plan:

```bash
python scripts/train_orchestrator.py --dry-run
```

Run orchestration:

```bash
python scripts/train_orchestrator.py
```

Publish artifact manifest:

```bash
python scripts/publish_model_artifacts.py --model-dir artifacts
```

## Release Gate

Run pre-release checks:

```bash
python scripts/release_gate.py
```

Gate coverage includes:

- source compile checks
- placeholder leakage detection
- required runtime environment contract validation

## Production Configuration Contracts

- Environment template: .env.prod.example
- Terraform production values template: terraform/prod.tfvars.example

## CI/CD Expectations

The pipeline validates matrix builds on Python 3.11 and 3.12, executes tests, runs release gate checks, performs security scans, and only then allows staging/prod progression.

## Status

NeuroSynth is in production-hardening mode with full-stack scaffolding, active gate enforcement, and deployment pathways wired for staged promotion.
