# NeuroSynth v2.0

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-00a393)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-ee4c2c)
![React](https://img.shields.io/badge/React-18-61dafb)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-f7931e)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Mistral--7B-yellow)

NeuroSynth is a production-grade neurological AI platform for Alzheimer's risk intelligence. It combines ensemble machine learning, pseudo-longitudinal temporal modeling, neural causal discovery, and AI-generated structured clinical reports.

## What NeuroSynth Does

- Ingests 2,149-patient Alzheimer's tabular data with 34 columns
- Uses 32 predictive features after dropping identifier/administrative columns
- Predicts Alzheimer’s risk with a 4-model weighted ensemble
- Simulates 36-month progression trajectories using a PyTorch LSTM
- Learns directional biomarker relationships with NOTEARS-style causal discovery
- Generates 9-section clinical narratives using HuggingFace Inference (Mistral-7B)
- Serves a world-class React dashboard, FastAPI backend, and Gradio Spaces app

## Architecture

```text
alzheimers_disease_data.csv
        |
        v
backend/data_pipeline.py
        |
        +--> backend/biomarker_model.py (RF + GB + XGB/ET + LR)
        +--> backend/temporal_model.py (Pseudo-longitudinal LSTM)
        +--> backend/causal_engine.py (NOTEARS-MLP on 10 variables)
        +--> backend/report_generator.py (HF Mistral-7B + fallback)
        |
        v
backend/api.py (FastAPI)
        |
        +--> frontend/src/App.jsx (React + Recharts + Tailwind)
        +--> app.py (Gradio / Hugging Face Spaces)
```

## Dataset

- File expected in repository root: `alzheimers_disease_data.csv`
- Rows: 2,149
- Columns: 34
- Target: `Diagnosis` (0 = No AD, 1 = AD)
- Dropped if present: `PatientID`, `DoctorInCharge`
- Citation basis: Rabie El Kharoua (2024), Kaggle Alzheimer's disease data framing

## Model Stack

1. Biomarker Ensemble
- RandomForestClassifier (500 trees)
- GradientBoostingClassifier (300 estimators)
- XGBoost (if installed) or ExtraTrees fallback
- LogisticRegression
- Weighted probability fusion: `[0.35, 0.35, 0.20, 0.10]`

2. Temporal Progression
- PyTorch LSTM: `input=32`, `hidden=128`, `layers=3`, `dropout=0.4`
- Pseudo-longitudinal sequence construction from age and MMSE buckets
- 6-step trajectory forecast: months 6, 12, 18, 24, 30, 36

3. Causal Discovery
- NOTEARS-MLP over clinically important 10-variable subset
- Augmented Lagrangian optimization with acyclicity constraint
- Intervention simulator with risk delta interpretation

4. Clinical Reports
- Primary: HuggingFace Inference API (`mistralai/Mistral-7B-Instruct-v0.3`)
- Fallback: deterministic rule-based 9-section report

## Features

- FastAPI auto-training startup flow
- SHAP explanations (local + global summaries)
- Batch prediction endpoint (max 50 records)
- Causal graph + intervention simulation endpoint
- React SPA with 5 tabs: Dashboard, Patient Analysis, Causal Map, Research, About
- Gradio Spaces-ready interface with 4 functional tabs
- Dockerized deployment and docker-compose orchestration

## Quick Start (Local)

### 1) Backend

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r backend/requirements.txt
export HF_TOKEN=hf_your_token_here
python -m backend.api
```

### 2) Frontend

```bash
npm --prefix frontend install
npm --prefix frontend run dev -- --host 0.0.0.0 --port 5173
```

Open: `http://localhost:5173`

## Docker Start

```bash
docker-compose up --build
```

- API: `http://localhost:8000`
- Frontend static is served by FastAPI when built assets exist

## Hugging Face Spaces Deploy (Gradio)

- Entry file: `app.py`
- Set secret: `HF_TOKEN`
- Ensure dataset file is available in space storage: `alzheimers_disease_data.csv`

## API Endpoints

- `GET /health`
- `GET /dataset/stats`
- `GET /model/performance`
- `GET /model/feature_importance`
- `GET /model/shap_summary`
- `GET /causal/graph`
- `POST /predict`
- `POST /report`
- `POST /simulate`
- `POST /batch_predict`

## Screenshots

- Dashboard (placeholder)
- Patient Analysis (placeholder)
- Causal Map (placeholder)
- Clinical Report (placeholder)

## Environment

Use `.env` (not committed):

```bash
HF_TOKEN=hf_your_token_here
```

Example template: `.env.example`

## Disclaimer

NeuroSynth is a research and engineering platform. It is not a medical device and must not be used as a standalone diagnostic or treatment system. All outputs require qualified clinical oversight.
