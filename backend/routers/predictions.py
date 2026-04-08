from uuid import uuid4

import pandas as pd
import pandera as pa
from fastapi import APIRouter, Depends, Request
from pandera.typing import DataFrame, Series

from backend.core.rate_limit import limiter, role_limit
from backend.deps import get_current_user
from backend.models import FeatureVector, PredictionResponse, UserContext
from backend.tasks import enqueue_full_pipeline

router = APIRouter(prefix="/predictions", tags=["predictions"])


class PredictionInputSchema(pa.DataFrameModel):
    Age: Series[float] = pa.Field(ge=0, le=120)
    MMSE: Series[float] = pa.Field(ge=0, le=30)
    FunctionalAssessment: Series[float] = pa.Field(ge=0, le=10)
    ADL: Series[float] = pa.Field(ge=0, le=10)
    SleepQuality: Series[float] = pa.Field(ge=0, le=10)


@router.post(
    "/run",
    response_model=PredictionResponse,
    summary="Queue full prediction workflow",
    description="Validates model inputs with pandera and queues all ML phases in Celery.",
)
@limiter.limit(role_limit)
async def run_prediction(payload: FeatureVector, request: Request, user: UserContext = Depends(get_current_user)) -> PredictionResponse:
    _ = request
    _ = user
    frame = pd.DataFrame([payload.features])
    PredictionInputSchema.validate(frame)

    job_id = enqueue_full_pipeline(payload.patient_id)
    return PredictionResponse(
        job_id=job_id or uuid4().hex,
        patient_id=payload.patient_id,
        queued_phases=[
            "connectome_inference",
            "genomic_risk_score",
            "temporal_forecast",
            "causal_analysis",
            "report_generation",
        ],
    )
