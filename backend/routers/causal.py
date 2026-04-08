from fastapi import APIRouter, Depends, Request

from backend.core.rate_limit import limiter, role_limit
from backend.deps import get_current_user
from backend.models import CausalRequest, CausalResponse, UserContext
from backend.tasks import causal_analysis

router = APIRouter(prefix="/causal", tags=["causal"])


@router.post(
    "/analyze",
    response_model=CausalResponse,
    summary="Queue causal analysis",
    description="Queues causal analysis for patient features and intervention simulation.",
)
@limiter.limit(role_limit)
async def analyze_causal(payload: CausalRequest, request: Request, user: UserContext = Depends(get_current_user)) -> CausalResponse:
    _ = request
    _ = user
    task = causal_analysis.delay(payload.patient_id)
    return CausalResponse(task_id=task.id, patient_id=payload.patient_id, status="queued")
