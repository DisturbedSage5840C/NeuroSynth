# NeuroSynth

NeuroSynth is a clinical AI platform for neurological risk analytics with:

- Synchronous disease/risk analysis APIs
- Multi-disease classification (Alzheimer's, Parkinson's, MS, Epilepsy, ALS, Huntington's)
- Longitudinal trajectory forecasting and SHAP explainability
- Causal graph outputs and generated clinical reports
- PostgreSQL persistence for patients and analysis history
- React clinical dashboard, report viewer, explorer timeline, and model performance page

## Single End-to-End Test Link

Use one URL for the complete integrated app:

- http://localhost:8000

## Deployment Targets

- Local full stack: FastAPI + React static bundle + PostgreSQL + Redis
- Vercel: frontend-only deployment (requires backend API URL env for live endpoints)

## System Architecture

```mermaid
flowchart LR
    A[React UI] -->|cookies + REST| B[FastAPI Backend]
    B --> C[Risk/Temporal/Causal Models]
    B --> D[Disease Classifier]
    B --> E[Clinical Report Generator]
    B --> F[(PostgreSQL)]
    B --> G[(Redis)]
    H[Celery Worker] --> G
    H --> C
```

## Clinical Data Flow

```mermaid
sequenceDiagram
    participant U as Clinician UI
    participant API as FastAPI
    participant ML as ML Stack
    participant DB as PostgreSQL

    U->>API: POST /auth/login
    API-->>U: session cookies
    U->>API: POST /predictions/analyze
    API->>ML: predictor + temporal + causal + disease classifier
    ML-->>API: probability, trajectory, SHAP, disease profile, report
    API->>DB: INSERT analyses
    API-->>U: analysis payload
    U->>API: GET /patients/{id}/analyses
    API->>DB: SELECT history
    API-->>U: timeline + comparison rows
```

## Feature Coverage Snapshot

| Capability | Status | Primary Endpoints/UI |
|---|---|---|
| Auth cookies + role access | Implemented | `/auth/login`, `/auth/refresh` |
| Synchronous risk analysis | Implemented | `/predictions/analyze` |
| Multi-disease classification | Implemented | analyze response `disease_classification` |
| Patient persistence | Implemented | `/patients/`, `/patients`, SQL schema |
| Analysis persistence/history | Implemented | `/patients/{patient_id}/analyses` |
| Report regeneration | Implemented | `/reports/generate`, Report tab actions |
| Model performance dashboard | Implemented | `/predictions/model/performance`, `/predictions/model/feature_importance`, `/performance` |

## Prerequisites

- Python venv at `.venv`
- Node/NPM for frontend build
- Redis running on `localhost:6379`
- PostgreSQL running on `localhost:5432`
- DB DSN (default): `postgresql://postgres:postgres@localhost:5432/neurosynth`

## One-Time Setup

### Option A: Docker Compose (recommended for local full stack)

1. Create `.env` from `.env.example` and keep values aligned.
2. Start dependencies:

```bash
docker compose up postgres redis -d
```

3. Start backend:

```bash
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

4. Build frontend and publish static assets used by backend:

```bash
cd frontend
npm install
npm run build
cd ..
rm -rf static
mkdir -p static
cp -R frontend/dist/* static/
```

### Option B: Manual local services

1. Install backend packages (including heavy ML runtime dependencies):

```bash
pip install -e '.[test]'
pip install torch requests
```

2. Build frontend and publish static assets for backend hosting:

```bash
cd frontend
npm install
npm run build
cd ..
rm -rf static
mkdir -p static
cp -R frontend/dist/* static/
```

3. Start PostgreSQL and apply schema:

```bash
brew services start postgresql@16
PAGER=cat /opt/homebrew/opt/postgresql@16/bin/psql -d postgres -c "ALTER ROLE postgres WITH LOGIN PASSWORD 'postgres';"
/opt/homebrew/opt/postgresql@16/bin/createdb -O postgres neurosynth || true
PGPASSWORD=postgres /opt/homebrew/opt/postgresql@16/bin/psql -h localhost -U postgres -d neurosynth -f backend/db_schema.sql
```

4. Start Redis:

```bash
redis-server --port 6379
```

5. Start backend API:

```bash
NEUROSYNTH_POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/neurosynth \
NEUROSYNTH_REDIS_URL=redis://localhost:6379/0 \
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

6. Optional: Start Celery worker:

```bash
NEUROSYNTH_REDIS_URL=redis://localhost:6379/0 \
python -m celery -A backend.celery_app:celery_app worker -l info --concurrency=1
```

## Vercel Frontend Deployment

Vercel project names must be lowercase, so use `neurosynth` as the project slug.

```bash
cd frontend
npx vercel project add neurosynth
npx vercel link --yes --scope disturbedsage5840cs-projects --project neurosynth
```

Set required frontend env vars in Vercel Project Settings (for example `VITE_API_BASE_URL`) so the deployed UI can call your backend.

The repository includes `frontend/vercel.json` with Vite output set to `dist`.

## Demo Credentials

- `clinician@neurosynth.local` / `neurosynth`
- `researcher@neurosynth.local` / `neurosynth`
- `admin@neurosynth.local` / `neurosynth`

## Verification Protocol (Science-Grade)

1. Login works and protected routes render.
2. Analyze endpoint returns model outputs with non-empty trajectory and SHAP values.
3. Disease card shows predicted disease + probability bars.
4. DB rows increment for both `patients` and `analyses`.
5. Data Explorer timeline shows patient history and SHAP comparison.
6. Performance page shows metrics, feature-importance bars, confusion matrix.
7. Report tab supports regenerate + print/PDF export.

## Useful Validation Commands

```bash
PGPASSWORD=postgres /opt/homebrew/opt/postgresql@16/bin/psql -h localhost -U postgres -d neurosynth -c "SELECT count(*) FROM patients; SELECT count(*) FROM analyses;"
```

```bash
python -m pytest -q tests/integration/test_pipeline_and_tasks_qa.py
```

## Troubleshooting

- If model startup logs `ml_models_load_failed`, install missing Python package from stack trace and restart API.
- If DB endpoints return fallback data, confirm `brew services list` shows `postgresql@16` started and re-run schema file.
- If UI is stale, rebuild frontend and re-copy `frontend/dist` to `static/`.
- If Docker build fails with daemon errors, start Docker Desktop first.
- If Vercel deployment returns an `Unexpected error` after upload, inspect build/deploy status in the Vercel dashboard and retry `npx vercel deploy --prod --yes`.
