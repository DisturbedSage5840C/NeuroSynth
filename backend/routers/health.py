from fastapi import APIRouter, Depends, Request

from backend.db import Database
from backend.deps import get_database
from backend.models import HealthResponse, ReadyResponse

router = APIRouter(prefix="", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns process liveness for container orchestration health checks.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description="Checks readiness by verifying PostgreSQL and Redis connectivity.",
)
async def ready(request: Request, db: Database = Depends(get_database)) -> ReadyResponse:
    db_ok = False
    redis_ok = False
    redis_client = getattr(request.app.state, "redis", None)

    try:
        row = await db.fetchrow("SELECT 1 AS ok")
        db_ok = row is not None and row["ok"] == 1
    except Exception:
        db_ok = False

    try:
        redis_ok = bool(redis_client and await redis_client.ping())
    except Exception:
        redis_ok = False

    return ReadyResponse(status="ready" if db_ok and redis_ok else "degraded", database=db_ok, redis=redis_ok)
