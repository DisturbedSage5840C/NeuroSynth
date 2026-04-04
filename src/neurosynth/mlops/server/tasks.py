from __future__ import annotations

import os
import time

from celery import Celery

celery_app = Celery("neurosynth", broker=os.getenv("NEURO_REDIS_URL", "redis://localhost:6379/0"), backend=os.getenv("NEURO_REDIS_URL", "redis://localhost:6379/0"))


@celery_app.task(name="analyze_patient")
def analyze_patient(patient_id: str, analysis_config: dict):
    _ = analysis_config
    # Orchestration stub: replace with Kubeflow trigger or workflow submit.
    time.sleep(0.1)
    return {"patient_id": patient_id, "status": "completed", "report_path": f"s3://neurosynth/reports/{patient_id}.json"}
