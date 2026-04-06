# NeuroSynth

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-00a393)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-ee4c2c)
![React](https://img.shields.io/badge/React-18-61dafb)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference-yellow)

## What This Is

NeuroSynth is a production-grade neurological AI research platform that integrates:

- Biomarker-based dementia risk prediction
- Longitudinal progression forecasting
- Neural causal discovery over key biomarkers
- LLM-powered structured clinical report generation
- Interactive React dashboard for analysis and simulation

The platform trains directly from OASIS longitudinal data and exposes a unified API for inference, reporting, and causal intervention simulation.

## Architecture

```text
													+---------------------------+
													| oasis_longitudinal.csv    |
													+------------+--------------+
																			 |
																			 v
+--------------------+      +----------+-----------+      +----------------------+
| backend/data_      | ---> | backend/api.py       | ---> | React Frontend       |
| pipeline.py        |      | FastAPI orchestration|      | frontend/src/App.jsx |
+--------------------+      +----------+-----------+      +----------+-----------+
																			 |                             |
																			 v                             v
						 +------------------+  +-------------------+   +----------------------+
						 | Biomarker Ensemble| | Temporal LSTM      |   | Causal SVG Network   |
						 | RF + GB           | | Progression Model  |   | + Intervention Sim   |
						 +-------------------+ +--------------------+   +----------------------+
												 \             /         \
													\           /           \
													 v         v             v
										+--------------------------------------+
										| Clinical Report Generator (HF API)   |
										| mistralai/Mistral-7B-Instruct-v0.3   |
										+--------------------------------------+
```

## Feature List

- Prediction engine: ensemble RandomForest + GradientBoosting
- Temporal forecasting: PyTorch LSTM trajectory for 6 to 36 months
- Causal discovery: NOTEARS-style neural graph learning (pure PyTorch)
- Clinical reports: Hugging Face Inference API structured reports
- Frontend analytics: performance charts, feature importance, risk gauge, causal map
- Intervention simulation: estimated CDR risk change from modifiable biomarker shifts

## Quick Start

```bash
git clone https://github.com/DisturbedSage5840C/NeuroSynth.git
cd NeuroSynth

pip install -r backend/requirements.txt
python backend/api.py
```

In a second terminal for frontend:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Docker

```bash
docker-compose up --build
```

API will be available at http://localhost:8000

## Hugging Face Spaces Deployment

- Use `app.py` as the Space entry point
- Set `HF_TOKEN` in Space secrets
- Ensure `oasis_longitudinal.csv` and `models/` are available in runtime storage

## Dataset

NeuroSynth uses OASIS Longitudinal MRI and cognitive data.

- Source: https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers
- Expected local file: `oasis_longitudinal.csv` in repository root
- Target: `Group` mapped to Demented=1, Nondemented=0

## Model Architecture

### Biomarker Predictor

- RandomForestClassifier: `n_estimators=200`, `max_depth=10`, `class_weight=balanced`
- GradientBoostingClassifier: `n_estimators=100`
- Ensemble output: average of both probability outputs

### Temporal Progression Model

- PyTorch LSTM: `input_size=8`, `hidden_size=64`, `num_layers=2`, `dropout=0.3`
- Head: `Linear(64->32)->ReLU->Dropout->Linear(32->1)->Sigmoid`
- Output: 36-month projected risk trajectory at months `[6,12,18,24,30,36]`

### Causal Engine

- NOTEARS-inspired neural formulation
- Learnable adjacency matrix (`W_logits`) with acyclicity constraint
- Per-variable MLP structural equations
- Outputs directed edges, top causal drivers, and intervention simulation

## API Documentation

- `GET /health`
- `GET /model/performance`
- `GET /model/feature_importance`
- `GET /causal/graph`
- `POST /predict`
- `POST /report`
- `POST /simulate`
- `GET /dataset/stats`

## Environment Variables

Use `.env`:

```bash
HF_TOKEN=hf_your_token_here
```

A template is available in `.env.example`.

## Repository Layout

```text
backend/
	api.py
	data_pipeline.py
	biomarker_model.py
	temporal_model.py
	causal_engine.py
	report_generator.py
	requirements.txt
frontend/
	src/App.jsx
	package.json
app.py
Dockerfile
docker-compose.yml
models/
```

## Disclaimer

This project is a research and engineering tool only. It is not approved for clinical diagnosis, treatment decisions, or medical use. Any outputs must be validated by qualified professionals in regulated clinical workflows.
