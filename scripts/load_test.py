"""Load test script for NeuroSynth v2 API.

Usage:
    pip install locust
    locust -f scripts/load_test.py --headless -u 50 -r 5 --run-time 60s

Targets:
    - POST /predictions/analyze     (v1 inference)
    - POST /v2/predictions/analyze  (v2 inference + LIME + counterfactuals)
    - POST /v2/reports/generate     (SOAP report generation)
    - GET  /v2/predictions/health   (circuit breaker health)
    - GET  /predictions/model/performance (model metrics)
"""
from __future__ import annotations

import json
import os
import random

from locust import HttpUser, between, task

BASE_URL = os.getenv("NEUROSYNTH_API_URL", "http://localhost:8000")

# Realistic patient data ranges
FEATURES = [
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking",
    "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
    "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
    "Depression", "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP",
    "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
    "CholesterolTriglycerides", "MMSE", "FunctionalAssessment", "MemoryComplaints",
    "BehavioralProblems", "ADL", "Confusion", "Disorientation",
    "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness",
]

RANGES: dict[str, tuple[float, float]] = {
    "Age": (50, 90), "Gender": (0, 1), "Ethnicity": (0, 3),
    "EducationLevel": (5, 20), "BMI": (18, 40), "Smoking": (0, 1),
    "AlcoholConsumption": (0, 20), "PhysicalActivity": (0, 10),
    "DietQuality": (0, 10), "SleepQuality": (2, 10),
    "FamilyHistoryAlzheimers": (0, 1), "CardiovascularDisease": (0, 1),
    "Diabetes": (0, 1), "Depression": (0, 1), "HeadInjury": (0, 1),
    "Hypertension": (0, 1), "SystolicBP": (90, 180),
    "DiastolicBP": (60, 110), "CholesterolTotal": (150, 300),
    "CholesterolLDL": (50, 200), "CholesterolHDL": (30, 80),
    "CholesterolTriglycerides": (50, 400), "MMSE": (10, 30),
    "FunctionalAssessment": (1, 10), "MemoryComplaints": (0, 1),
    "BehavioralProblems": (0, 1), "ADL": (1, 10), "Confusion": (0, 1),
    "Disorientation": (0, 1), "PersonalityChanges": (0, 1),
    "DifficultyCompletingTasks": (0, 1), "Forgetfulness": (0, 1),
}


def random_patient() -> dict:
    """Generate a random patient payload."""
    patient: dict[str, float] = {}
    for f in FEATURES:
        lo, hi = RANGES.get(f, (0, 1))
        if hi <= 1:
            patient[f] = random.choice([0, 1])
        else:
            patient[f] = round(random.uniform(lo, hi), 2)
    return patient


class NeuroSynthUser(HttpUser):
    """Simulated clinical user interacting with NeuroSynth API."""

    host = BASE_URL
    wait_time = between(1, 3)

    def on_start(self) -> None:
        """Login to get auth token."""
        resp = self.client.post("/auth/login", json={
            "username": "loadtest",
            "password": "loadtest123",
        })
        if resp.status_code == 200:
            token = resp.json().get("access_token", "")
            self.client.headers.update({"Authorization": f"Bearer {token}"})

    @task(5)
    def v1_analyze(self) -> None:
        """v1 prediction endpoint — highest frequency."""
        self.client.post(
            "/predictions/analyze",
            json={"patient_id": f"LT-{random.randint(1000, 9999)}", "features": random_patient()},
            name="/predictions/analyze [v1]",
        )

    @task(3)
    def v2_analyze(self) -> None:
        """v2 prediction with LIME + counterfactuals."""
        self.client.post(
            "/v2/predictions/analyze",
            json={"patient_id": f"LT-{random.randint(1000, 9999)}", "features": random_patient()},
            name="/v2/predictions/analyze [v2]",
        )

    @task(1)
    def v2_report(self) -> None:
        """SOAP report generation — lowest frequency."""
        self.client.post(
            "/v2/reports/generate",
            json={"patient_id": f"LT-{random.randint(1000, 9999)}"},
            name="/v2/reports/generate",
        )

    @task(2)
    def health_check(self) -> None:
        """Circuit breaker health check."""
        self.client.get("/v2/predictions/health", name="/v2/predictions/health")

    @task(1)
    def model_performance(self) -> None:
        """Model performance metrics."""
        self.client.get("/predictions/model/performance", name="/predictions/model/performance")


class NeuroSynthStressUser(HttpUser):
    """High-throughput stress test user."""

    host = BASE_URL
    wait_time = between(0.1, 0.5)

    @task
    def rapid_v2_analyze(self) -> None:
        """Rapid-fire v2 predictions to test throughput."""
        self.client.post(
            "/v2/predictions/analyze",
            json={"patient_id": f"STRESS-{random.randint(1, 99999)}", "features": random_patient()},
            name="/v2/predictions/analyze [stress]",
        )
