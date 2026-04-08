from __future__ import annotations

from celery import Celery

from backend.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "neurosynth",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["backend.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    worker_prefetch_multiplier=1,
)
