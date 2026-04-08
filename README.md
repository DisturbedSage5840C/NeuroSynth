# NeuroSynth

NeuroSynth is a production clinical AI platform for neurological deterioration prediction.
It combines multimodal imaging, genomics, wearable biomarkers, and longitudinal clinical data through a multi-phase ML pipeline, exposed by a FastAPI backend and a React frontend.

## Repository Layout

- `backend/`: FastAPI API, auth/security, telemetry, rate limiting, async tasks, metrics.
- `frontend/`: Vite + React application.
  - Canonical UI source: `frontend/src/figma-system/`
  - Main entrypoint: `frontend/src/main.tsx`
- `src/neurosynth/`: core ML/data pipeline implementations.
- `scripts/`: orchestration and release checks.
- `tests/`: unit and integration tests.
- `docs/`: generated documentation artifacts (HTML/PDF).

## Frontend

### Install

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Default local URL:

- `http://localhost:5173`

### Production Build

```bash
npm run build
```

Build output is generated in `frontend/dist`.

## Backend

Python project constraints require Python 3.11/3.12.

### Install (recommended env)

```bash
.venv312/bin/python -m pip install -e '.[test]'
```

### Run API

```bash
.venv312/bin/python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

## Static Asset Serving

Backend serves static frontend from `static/` when present (Docker path), using:

- `backend/api.py` static mount logic

Vite frontend builds to `frontend/dist`.

## Testing

Run full test suite:

```bash
.venv312/bin/python -m pytest -q --tb=short
```

## Release Gate

Run release checks:

```bash
NEURO_JWKS_URL=https://example.local/jwks \
NEURO_ALLOWED_ORIGINS=http://localhost:5173 \
NEURO_REDIS_URL=redis://localhost:6379/0 \
.venv312/bin/python scripts/release_gate.py
```

## Docker

```bash
docker compose up --build
```

Note: Docker commands require the local Docker daemon to be running.
