# NeuroSynth

NeuroSynth is a production clinical AI platform for neurological deterioration prediction.
It combines multimodal imaging, genomics, wearable biomarkers, and longitudinal clinical data through a multi-phase ML pipeline, exposed by a FastAPI backend and a React frontend.

## Final Local Test URL

Use this single URL to test the integrated app:

- http://localhost:8000

## Repository Layout

- `backend/`: FastAPI API, auth/security, telemetry, rate limiting, async tasks, metrics.
- `frontend/`: Vite + React application.
  - Canonical UI source: `frontend/src/figma-system/`
  - Main entrypoint: `frontend/src/main.tsx`
- `src/neurosynth/`: core ML/data pipeline implementations.
- `scripts/`: orchestration and release checks.
- `tests/`: unit and integration tests.
- `docs/`: generated documentation artifacts (HTML/PDF).

## Quick Start (Integrated Local Stack)

This starts Redis + Celery + FastAPI, serves the frontend from backend static hosting, and gives one URL for testing.

1. Install Python dependencies:

```bash
.venv312/bin/python -m pip install -e '.[test]'
```

2. Install frontend dependencies and build:

```bash
cd frontend
npm install
npm run build
cd ..
```

3. Refresh static assets served by backend:

```bash
rm -rf static
mkdir -p static
cp -R frontend/dist/* static/
```

4. Start Redis (Terminal 1):

```bash
redis-server --port 6379
```

5. Start Celery worker (Terminal 2):

```bash
NEUROSYNTH_REDIS_URL=redis://localhost:6379/0 \
.venv312/bin/python -m celery -A backend.celery_app:celery_app worker -l info --concurrency=1
```

6. Start backend (Terminal 3):

```bash
NEUROSYNTH_REDIS_URL=redis://localhost:6379/0 \
.venv312/bin/python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

7. Open:

- http://localhost:8000

### Demo Login

- Email/username: `clinician@neurosynth.local`
- Password: `neurosynth`

## Frontend (Dev Mode)

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
NEUROSYNTH_REDIS_URL=redis://localhost:6379/0 \
.venv312/bin/python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

## Static Asset Serving

Backend serves frontend from `static/` when present, including SPA fallback routes.

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

## Troubleshooting

- If `http://localhost:8000` shows API JSON instead of UI, rebuild frontend and copy to `static/` again.
- If login or protected endpoints fail, make sure Redis and Celery are running.
- `/ready` may report `database: false` locally if Postgres is not started; the UI demo flow can still run with Redis/Celery.
