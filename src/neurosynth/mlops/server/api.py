from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import asyncpg
import jwt
import redis
import structlog
from celery import Celery
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import jwk
from jose.utils import base64url_decode
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel


LOGGER = structlog.get_logger(__name__)
JWKS_URL = os.getenv("NEURO_JWKS_URL", "https://example.org/.well-known/jwks.json")
API_ALLOWED_ORIGINS = os.getenv("NEURO_ALLOWED_ORIGINS", "https://neurosynth.example.org").split(",")
TIMESCALE_DSN = os.getenv("NEURO_TIMESCALE_DSN", "postgresql://postgres:postgres@timescaledb:5432/neurosynth")

celery_app = Celery("neurosynth", broker=os.getenv("NEURO_REDIS_URL", "redis://localhost:6379/0"), backend=os.getenv("NEURO_REDIS_URL", "redis://localhost:6379/0"))


class PatientAnalysisRequest(BaseModel):
    patient_id: str
    analysis_config: dict[str, Any]


class AnalysisJobResponse(BaseModel):
    job_id: str
    status: str


app = FastAPI(title="NeuroSynth API", version="v1")
app.add_middleware(CORSMiddleware, allow_origins=API_ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

_rate = defaultdict(lambda: deque())
_pool: asyncpg.Pool | None = None


@app.on_event("startup")
async def startup() -> None:
    global _pool
    _pool = await asyncpg.create_pool(TIMESCALE_DSN, min_size=1, max_size=5)
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_request_log (
                ts TIMESTAMPTZ,
                endpoint TEXT,
                method TEXT,
                latency_ms DOUBLE PRECISION,
                status INT,
                api_key_hash TEXT
            );
            """
        )


async def _fetch_jwks() -> dict:
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(JWKS_URL)
        r.raise_for_status()
        return r.json()


async def verify_api_key(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1]

    jwks = await _fetch_jwks()
    headers = jwt.get_unverified_header(token)
    kid = headers.get("kid")
    key = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
    if key is None:
        raise HTTPException(status_code=401, detail="Unknown key id")

    message, encoded_sig = token.rsplit(".", 1)
    decoded_sig = base64url_decode(encoded_sig.encode("utf-8"))
    public_key = jwk.construct(key)
    if not public_key.verify(message.encode("utf-8"), decoded_sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = jwt.decode(token, options={"verify_signature": False})
    if payload.get("exp", 0) < int(time.time()):
        raise HTTPException(status_code=401, detail="Token expired")
    return payload


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
    dur = (time.time() - start) * 1000.0

    api_hash = str(hash(request.headers.get("Authorization", "")))
    LOGGER.info("http.request", endpoint=request.url.path, method=request.method, status=resp.status_code, latency_ms=dur)

    if _pool is not None:
        async with _pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO api_request_log(ts, endpoint, method, latency_ms, status, api_key_hash) VALUES($1,$2,$3,$4,$5,$6)",
                datetime.now(timezone.utc),
                request.url.path,
                request.method,
                float(dur),
                int(resp.status_code),
                api_hash,
            )

    return resp


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    auth = request.headers.get("Authorization", "")
    key = auth[:64]
    now = time.time()
    q = _rate[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= 100:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    q.append(now)
    return await call_next(request)


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=30)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"detail": "Request timeout"})


@app.post("/v1/analyze/patient")
async def analyze_patient(request: PatientAnalysisRequest, background_tasks: BackgroundTasks, api_key: dict = Depends(verify_api_key)) -> AnalysisJobResponse:
    _ = background_tasks
    if api_key.get("role") not in ["clinician", "admin", "researcher"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    # Stubbed patient existence/data availability check.
    if not request.patient_id:
        raise HTTPException(status_code=400, detail="Invalid patient_id")

    task = celery_app.send_task("analyze_patient", args=[request.patient_id, request.analysis_config])
    return AnalysisJobResponse(job_id=task.id, status="queued")


@app.get("/v1/analyze/status/{job_id}")
async def analyze_status(job_id: str, api_key: dict = Depends(verify_api_key)):
    _ = api_key
    result = celery_app.AsyncResult(job_id)
    return {"job_id": job_id, "status": result.status}


@app.get("/v1/analyze/result/{job_id}")
async def analyze_result(job_id: str, api_key: dict = Depends(verify_api_key)):
    _ = api_key
    result = celery_app.AsyncResult(job_id)
    if not result.ready():
        raise HTTPException(status_code=202, detail="Result not ready")
    return {"job_id": job_id, "result": result.result}


@app.post("/v1/simulate/intervention")
async def simulate_intervention(payload: dict, api_key: dict = Depends(verify_api_key)):
    if api_key.get("role") not in ["clinician", "admin"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"status": "accepted", "payload": payload}


@app.get("/v1/patient/{patient_id}/history")
async def patient_history(patient_id: str, api_key: dict = Depends(verify_api_key)):
    if api_key.get("role") == "researcher":
        raise HTTPException(status_code=403, detail="Researchers cannot access patient-level history")
    return {"patient_id": patient_id, "history": []}


@app.get("/health")
async def health():
    return {"status": "ok"}


