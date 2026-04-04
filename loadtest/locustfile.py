from __future__ import annotations

import random

from locust import HttpUser, between, task


class NeuroSynthUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def analyze_patient(self):
        pid = f"p-{random.randint(1000,9999)}"
        self.client.post(
            "/v1/analyze/patient",
            json={"patient_id": pid, "analysis_config": {"full_pipeline": True}},
            headers={"Authorization": "Bearer test-token"},
        )
