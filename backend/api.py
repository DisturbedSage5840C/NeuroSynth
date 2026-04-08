from __future__ import annotations

import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis
from slowapi.errors import RateLimitExceeded

from backend.celery_app import celery_app
from backend.core.config import get_settings
from backend.core.logging import configure_structlog, get_logger
from backend.core.metrics import CELERY_QUEUE_DEPTH, REQUEST_COUNT, REQUEST_LATENCY, render_metrics
from backend.core.rate_limit import limiter, rate_limit_exceeded_handler
from backend.core.security import ACCESS_COOKIE, Role, decode_token, hash_patient_id
from backend.db import get_db
from backend.deps import require_role
from backend.routers import admin, auth, biomarkers, causal, health, patients, pipelines, predictions, reports


async def _drain_celery_queue(timeout_seconds: int = 20) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout_seconds:
        try:
            inspect = celery_app.control.inspect(timeout=1.0)
            active = inspect.active() or {}
            reserved = inspect.reserved() or {}
        except Exception:
            # If broker is unavailable during shutdown, exit drain gracefully.
            return
        in_progress = sum(len(v) for v in active.values()) + sum(len(v) for v in reserved.values())
        if in_progress == 0:
            return
        await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_structlog()
    logger = get_logger("neurosynth.bootstrap")

    db = get_db()
    try:
        await db.connect()
        logger.info("database_connected")
    except Exception as exc:
        logger.warning("database_connect_failed", error=str(exc))

    try:
        app.state.redis = Redis.from_url(settings.redis_url, decode_responses=True)
        await app.state.redis.ping()
        logger.info("redis_connected")
    except Exception as exc:
        app.state.redis = None
        logger.warning("redis_connect_failed", error=str(exc))

    yield

    await _drain_celery_queue()
    redis_client = getattr(app.state, "redis", None)
    if redis_client is not None:
        await redis_client.close()
    await db.disconnect()


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production API for NeuroSynth healthcare AI workflows with async orchestration and role-based access.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.middleware("http")
async def auth_context_middleware(request: Request, call_next):
    token = request.cookies.get(ACCESS_COOKIE)
    request.state.user = None
    if token:
        try:
            payload = decode_token(token, expected_type="access")
            request.state.user = {
                "user_id": str(payload["sub"]),
                "role": str(payload["role"]),
            }
        except Exception:
            request.state.user = None
    return await call_next(request)


@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    logger = get_logger("neurosynth.request")
    trace_id = str(uuid4())
    started = time.perf_counter()

    patient_id = request.headers.get("x-patient-id")
    patient_id_hash = hash_patient_id(patient_id)
    response = await call_next(request)
    latency_s = time.perf_counter() - started
    user = getattr(request.state, "user", None) or {"role": "ANON"}
    path = request.url.path
    REQUEST_COUNT.labels(method=request.method, path=path, status=str(response.status_code)).inc()
    REQUEST_LATENCY.labels(method=request.method, path=path).observe(latency_s)

    logger.info(
        "request_completed",
        trace_id=trace_id,
        role=user.get("role"),
        patient_id=patient_id_hash,
        latency_ms=round(latency_s * 1000, 2),
        method=request.method,
        path=path,
        status_code=response.status_code,
    )
    response.headers["X-Trace-Id"] = trace_id
    return response


app.include_router(health.router)
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(predictions.router)
app.include_router(reports.router)
app.include_router(causal.router)
app.include_router(biomarkers.router)
app.include_router(admin.router)
app.include_router(pipelines.router)


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Admin-only metrics endpoint for scraping by Prometheus.",
)
async def metrics_root(_: object = Depends(require_role(Role.ADMIN))) -> Response:
    redis_client = getattr(app.state, "redis", None)
    if redis_client is not None:
        depth = int(await redis_client.llen("celery"))
        CELERY_QUEUE_DEPTH.labels(queue="celery").set(depth)
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


static_dir = Path("frontend/dist")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
