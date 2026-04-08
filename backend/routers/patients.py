from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request

from backend.core.rate_limit import limiter, role_limit
from backend.core.security import Role
from backend.deps import get_current_user
from backend.models import PatientListResponse, PatientSummary, UserContext

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get(
    "",
    response_model=PatientListResponse,
    summary="List patients",
    description="Returns patient summaries accessible by clinicians, researchers, and admins.",
)
@limiter.limit(role_limit)
async def list_patients(request: Request, user: UserContext = Depends(get_current_user)) -> PatientListResponse:
    _ = request
    _ = user
    now = datetime.now(tz=UTC)
    items = [
        PatientSummary(patient_id="P-001", name="Patient P-001", updated_at=now),
        PatientSummary(patient_id="P-002", name="Patient P-002", updated_at=now),
    ]
    return PatientListResponse(items=items)


@router.get(
    "/{patient_id}",
    response_model=PatientSummary,
    summary="Get patient",
    description="Returns a single patient summary by ID.",
)
@limiter.limit(role_limit)
async def get_patient(patient_id: str, request: Request, user: UserContext = Depends(get_current_user)) -> PatientSummary:
    _ = request
    _ = user
    return PatientSummary(patient_id=patient_id, name=f"Patient {patient_id}", updated_at=datetime.now(tz=UTC))
