from fastapi import APIRouter, Depends, Request

from backend.core.rate_limit import limiter, role_limit
from backend.deps import get_current_user
from backend.models import ReportRequest, ReportResponse, UserContext
from backend.tasks import report_generation

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post(
    "/generate",
    response_model=ReportResponse,
    summary="Queue report generation",
    description="Queues the report generation phase and returns Celery task ID for polling.",
)
@limiter.limit(role_limit)
async def generate_report(payload: ReportRequest, request: Request, user: UserContext = Depends(get_current_user)) -> ReportResponse:
    _ = request
    _ = user
    task = report_generation.delay(payload.patient_id, payload.notes)
    return ReportResponse(task_id=task.id, patient_id=payload.patient_id, status="queued")
