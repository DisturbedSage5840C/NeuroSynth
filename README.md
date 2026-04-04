# NeuroSynth

Clinical-risk prediction, simplified to run fast and ship now.

## Overview

NeuroSynth now has a practical minimal core built on real OASIS longitudinal data.

- Task: binary prediction (`Demented` vs `Nondemented`)
- Model: `RandomForestClassifier`
- API: FastAPI (`/predict`)
- UI: local HTML form + Gradio deployment option

This keeps the repository name and structure as **NeuroSynth**, while making the core workflow runnable by one developer.

## Original vs Current

| Area | Original NeuroSynth | Current NeuroSynth Core |
|---|---|---|
| Runtime complexity | Enterprise-scale stack | Lightweight local stack |
| Deployment path | Infra-heavy cloud workflow | Local + free Hugging Face path |
| Modeling scope | Multi-module advanced architecture | Focused baseline classifier |
| Time to first result | High setup effort | Minutes |

## What Changed (Minimal Scope)

- Added `explore.py` for dataset inspection
- Added `train.py` for training and artifact export
- Added `main.py` for prediction API
- Added `index.html` for a quick browser UI
- Added `app.py` for Gradio deployment
- Added `requirements.txt` for minimal runtime dependencies
- Updated this `README.md`

Everything else in the repository remains intact.

## Data Contract

Place `oasis_longitudinal.csv` in the repository root.

Features used:

- `Age`
- `EDUC`
- `SES`
- `MMSE`
- `CDR`
- `eTIV`
- `nWBV`
- `ASF`

## Local Run (A to B to C)

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Explore and verify the dataset

```bash
python explore.py
```

3. Train model artifacts

```bash
python train.py
```

Outputs:

- `model.pkl`
- `label_encoder.pkl`

4. Start API

```bash
uvicorn main:app --reload
```

5. Test endpoints and UI

- Open `http://localhost:8000/docs`
- Open `index.html` in your browser

## Free Deployment (Hugging Face Spaces)

1. Create a new Space (Gradio)
2. Upload:
	- `app.py`
	- `model.pkl`
	- `label_encoder.pkl`
	- `requirements.txt`
3. Let Spaces auto-build and host the app

## Slow Path Back to Enterprise NeuroSynth

Phase 1: Baseline hardening

- Add stricter input validation and error handling
- Save training metrics per run
- Add reproducibility notes/model card

Phase 2: MLOps foundation

- Add experiment tracking
- Add containerized serving profile
- Add staging release checks

Phase 3: Advanced reintegration

- Reconnect temporal/genomic/connectome modules
- Re-enable full infra-backed deploy path
- Add production secrets, monitoring, and rollback gates

## Full Detailed Documentation

- HTML full document: `docs/project_a_to_z.html`
- PDF full document: `docs/project_a_to_z.pdf`
