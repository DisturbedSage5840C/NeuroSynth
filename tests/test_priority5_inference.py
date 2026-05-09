"""Priority 5 verification tests — Inference API Refactor."""
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import numpy as np

print("=" * 60)
print("NeuroSynth v2 — Priority 5 Inference API Verification")
print("=" * 60)

# 1. v2 Pydantic models
print("\n[1/5] v2 Pydantic Models (import + validation)...")
from backend.models_v2 import (
    AnalyzeResponseV2,
    CausalIntervention,
    ConfidenceInterval,
    Counterfactual,
    DiseaseProb,
    LIMEExplanation,
    ModelContribution,
    RFC7807Error,
    SHAPValue,
    TrajectoryForecast,
)

# Test model instantiation
response = AnalyzeResponseV2(
    patient_id="P-001",
    request_id="test-123",
    prediction=0,
    probability=0.35,
    risk_level="Low",
    confidence="Medium",
    shap_values=[SHAPValue(feature="MMSE", value=-0.15)],
    lime_explanation=[LIMEExplanation(feature="MMSE", weight=-0.12, direction="decreases_risk")],
    counterfactuals=[Counterfactual(feature="SleepQuality", current_value=4.0, target_value=7.0, risk_delta=-0.08)],
    trajectory_48mo=TrajectoryForecast(months=[6, 12, 18, 24, 30, 36, 42, 48], values=[0.3]*8),
    causal_interventions=[CausalIntervention(factor="PhysicalActivity", effect_size=0.05, direction="protective")],
    confidence_intervals=ConfidenceInterval(method="conformal", coverage=0.95, lower=0.28, upper=0.42),
)
assert response.probability == 0.35
assert len(response.shap_values) == 1
assert len(response.lime_explanation) == 1
assert len(response.counterfactuals) == 1
assert response.trajectory_48mo.months[-1] == 48
assert response.api_version == "v2"

# RFC 7807 error
error = RFC7807Error(
    type="https://neurosynth.dev/errors/validation",
    title="Validation Error",
    status=422,
    detail="MMSE must be between 0 and 30",
    instance="/v2/predictions/analyze",
    trace_id="test-trace",
)
assert error.status == 422
print(f"  AnalyzeResponseV2:  {len(response.model_fields)} fields")
print(f"  RFC7807Error:       {len(error.model_fields)} fields")
print("  PASSED")

# 2. LIME computation
print("\n[2/5] LIME Local Explanations...")
from backend.routers.predictions_v2 import _compute_lime

# Use a simple model for testing
from sklearn.ensemble import RandomForestClassifier
from backend.data_pipeline import DataPipeline

pipeline = DataPipeline()
X_train, X_test, y_train, y_test, feature_names, scaler, stats = pipeline.process()

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train.values, y_train.values)


def predict_proba_fn(X):
    return rf.predict_proba(X)[:, 1]


lime_results = _compute_lime(predict_proba_fn, X_test.values[0], feature_names)
print(f"  Top-5 LIME features:")
for r in lime_results[:5]:
    print(f"    {r['feature']:>25}: weight={r['weight']:+.4f} ({r['direction']})")
assert len(lime_results) > 0
assert all("feature" in r and "weight" in r for r in lime_results)
print("  PASSED")

# 3. Counterfactual generation
print("\n[3/5] Counterfactual Recommendations...")
from backend.routers.predictions_v2 import _generate_counterfactuals

current_prob = float(rf.predict_proba(X_test.values[:1])[:, 1][0])
counterfactuals = _generate_counterfactuals(
    predict_proba_fn, X_test.values[0], feature_names, current_prob,
)
print(f"  Current probability: {current_prob:.4f}")
print(f"  Counterfactuals found: {len(counterfactuals)}")
for cf in counterfactuals[:3]:
    print(f"    {cf['feature']:>25}: {cf['current_value']:.2f}→{cf['target_value']:.2f} (Δrisk={cf['risk_delta']:+.4f})")
assert all("feature" in cf and "risk_delta" in cf for cf in counterfactuals)
print("  PASSED")

# 4. Circuit Breaker
print("\n[4/5] Circuit Breaker...")
from backend.routers.predictions_v2 import _CircuitBreaker

breaker = _CircuitBreaker(threshold=3, reset_timeout=1.0)
assert not breaker.is_open

# Record failures up to threshold
breaker.record_failure()
breaker.record_failure()
assert not breaker.is_open
breaker.record_failure()
assert breaker.is_open

# Success resets
breaker.record_success()
assert not breaker.is_open
print("  Circuit breaker opens after 3 failures: OK")
print("  Circuit breaker resets on success:      OK")
print("  PASSED")

# 5. v2 Router registration
print("\n[5/5] v2 Router Registration...")
from backend.routers.predictions_v2 import router as v2_router

routes = [r.path for r in v2_router.routes]
print(f"  Router prefix: {v2_router.prefix}")
print(f"  Routes: {routes}")
assert "/analyze" in routes or any("/analyze" in r for r in routes)
assert any("health" in r for r in routes)
print("  PASSED")

print("\n" + "=" * 60)
print("ALL PRIORITY 5 VERIFICATION TESTS PASSED")
print("=" * 60)
