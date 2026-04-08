from __future__ import annotations

import json
import time
from datetime import UTC, datetime

from celery import chain
from redis import Redis

from backend.celery_app import celery_app
from backend.core.config import get_settings
from backend.core.metrics import ML_INFERENCE_DURATION
from backend.core.security import hash_patient_id


def _publisher() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)


def _publish_progress(task_id: str, phase: str, progress: int, patient_id: str | None) -> None:
    payload = {
        "phase": phase,
        "task_id": task_id,
        "progress": progress,
        "patient_id_hash": hash_patient_id(patient_id),
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }
    redis_client = _publisher()
    redis_client.publish("biomarkers.progress", json.dumps(payload))


def _simulate_work(task_id: str, phase: str, patient_id: str | None) -> dict[str, str | int]:
    _publish_progress(task_id, phase, 0, patient_id)
    started = time.perf_counter()
    for progress in (20, 40, 60, 80, 100):
        time.sleep(0.15)
        _publish_progress(task_id, phase, progress, patient_id)
    duration = time.perf_counter() - started
    ML_INFERENCE_DURATION.labels(phase=phase).observe(duration)
    return {"phase": phase, "status": "completed", "duration_ms": int(duration * 1000)}


@celery_app.task(name="connectome_inference", bind=True)
def connectome_inference(self, patient_id: str) -> dict[str, str | int]:
    return _simulate_work(self.request.id, "connectome_inference", patient_id)


@celery_app.task(name="genomic_risk_score", bind=True)
def genomic_risk_score(self, patient_id: str) -> dict[str, str | int]:
    return _simulate_work(self.request.id, "genomic_risk_score", patient_id)


@celery_app.task(name="temporal_forecast", bind=True)
def temporal_forecast(self, patient_id: str) -> dict[str, str | int]:
    return _simulate_work(self.request.id, "temporal_forecast", patient_id)


@celery_app.task(name="causal_analysis", bind=True)
def causal_analysis(self, patient_id: str) -> dict[str, str | int]:
    return _simulate_work(self.request.id, "causal_analysis", patient_id)


@celery_app.task(name="report_generation", bind=True)
def report_generation(self, patient_id: str, notes: str | None = None) -> dict[str, str | int]:
    _ = notes
    return _simulate_work(self.request.id, "report_generation", patient_id)


def enqueue_full_pipeline(patient_id: str) -> str:
    workflow = chain(
        connectome_inference.si(patient_id),
        genomic_risk_score.si(patient_id),
        temporal_forecast.si(patient_id),
        causal_analysis.si(patient_id),
        report_generation.si(patient_id),
    )
    result = workflow.apply_async()
    return result.id
