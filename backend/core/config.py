from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="NEUROSYNTH_", extra="ignore")

    app_name: str = "NeuroSynth Clinical API"
    app_version: str = "3.0.0"
    app_env: Literal["dev", "staging", "prod"] = "dev"

    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 15
    refresh_token_days: int = 7
    auth_cookie_secure: bool = False

    patient_hash_secret: str = "change-me-patient-hmac"

    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/neurosynth"
    redis_url: str = "redis://localhost:6379/0"

    kubeflow_host: str = "http://localhost:8080"

    metrics_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
