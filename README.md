# NeuroSynth

NeuroSynth is a multi-modal neurological intelligence platform for forecasting deterioration, surfacing causal drivers, and generating clinically structured intervention reports.

It combines imaging, longitudinal biomarkers, genomics, causal discovery, and medical language generation into one production-oriented system.

## Why NeuroSynth

- Predicts neurological progression from heterogeneous patient signals.
- Connects prediction to causal mechanism, not only correlation.
- Produces explainable, uncertainty-aware outputs for clinical workflows.
- Ships with end-to-end infrastructure scaffolding across data, modeling, and deployment.

## Platform Modules

- Phase 1: Data Lakehouse and Ingestion
   - ADNI, PPMI, MIMIC, and wearable stream connectors
   - DICOM processing pipeline
   - Apache Iceberg tables for patient, longitudinal biomarker, imaging index, and genomics domains
   - Neo4j knowledge graph loader
   - TimescaleDB stream feature processing

- Phase 2: Brain Connectome GNN
   - Connectome construction from fMRI and structural data
   - Temporal graph datasets
   - GATv2 plus ODE-LSTM architecture
   - Evidential uncertainty and explainability utilities

- Phase 3: Genomic Transformer
   - QC and annotation wrappers for genomic preprocessing
   - DNABERT-based sequence context encoding
   - Hierarchical variant modeling and risk scoring

- Phase 4: Temporal Fusion Transformer
   - Longitudinal biomarker preprocessing and dataset factory
   - TFT wrapper, calibration, validation, and variable importance analysis

- Phase 5: Neural Causal Discovery
   - NOTEARS-inspired differentiable DAG learning
   - Patient-specific causal fine-tuning
   - Counterfactual intervention simulation and validation

- Phase 6: Clinical LLM
   - Corpus tooling, training stages, schema-constrained report generation
   - Retrieval pipeline and hallucination controls

- Phase 7: MLOps and Deployment
   - FastAPI orchestration layer
   - Kubeflow pipeline definitions
   - Helm chart and Terraform infrastructure modules
   - Monitoring, drift hooks, and CI deployment flow

## Repository Layout

- src/neurosynth: core platform packages
- tests: unit and integration-oriented tests
- helm/neurosynth: Kubernetes chart templates and values
- terraform: AWS infrastructure modules
- .github/workflows: CI and deployment workflows
- loadtest: Locust performance workload definitions

## Quick Start

1. Create and activate a Python 3.11 virtual environment.
2. Install project dependencies.
3. Start required local services with Docker Compose.
4. Run the test suite.

Example:

pip install -e '.[test]'
docker compose up -d
pytest -q

## Current State

This repository contains a complete multi-phase implementation scaffold with substantial functional code across all core subsystems. Production hardening tasks are tracked and being actively tightened module by module.

## Project Name

NeuroSynth
