from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.biomarker_model import BiomarkerPredictor
from backend.causal_engine import NeuralCausalDiscovery
from backend.data_pipeline import DataPipeline
from backend.report_generator import ClinicalReportGenerator
from backend.temporal_model import TemporalProgressionModel


class PatientRequest(BaseModel):
    Age: float
    Gender: float
    Ethnicity: float
    EducationLevel: float
    BMI: float
    Smoking: float
    AlcoholConsumption: float
    PhysicalActivity: float
    DietQuality: float
    SleepQuality: float
    FamilyHistoryAlzheimers: float
    CardiovascularDisease: float
    Diabetes: float
    Depression: float
    HeadInjury: float
    Hypertension: float
    SystolicBP: float
    DiastolicBP: float
    CholesterolTotal: float
    CholesterolLDL: float
    CholesterolHDL: float
    CholesterolTriglycerides: float
    MMSE: float
    FunctionalAssessment: float
    MemoryComplaints: float
    BehavioralProblems: float
    ADL: float
    Confusion: float
    Disorientation: float
    PersonalityChanges: float
    DifficultyCompletingTasks: float
    Forgetfulness: float


class SimulationRequest(BaseModel):
    patient_data: dict[str, float]
    variable: str
    new_value: float = Field(ge=0.0, le=1.0)


app = FastAPI(title="NeuroSynth API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: dict[str, Any] = {
    "ready": False,
    "pipeline": None,
    "predictor": None,
    "temporal": None,
    "causal": None,
    "reporter": None,
    "feature_names": [],
    "dataset_stats": {},
    "X_test": None,
    "y_test": None,
    "X_test_raw": None,
    "causal_graph": None,
}


def _to_frame(patient: PatientRequest) -> pd.DataFrame:
    return pd.DataFrame([patient.model_dump()])


def _to_causal_patient_map(patient_dict: dict[str, float]) -> dict[str, float]:
    return {
        "Age": float(patient_dict.get("Age", 0.0)),
        "MMSE": float(patient_dict.get("MMSE", 0.0)),
        "FunctionalAssessment": float(patient_dict.get("FunctionalAssessment", 0.0)),
        "ADL": float(patient_dict.get("ADL", 0.0)),
        "MemoryComplaints": float(patient_dict.get("MemoryComplaints", 0.0)),
        "BehavioralProblems": float(patient_dict.get("BehavioralProblems", 0.0)),
        "Depression": float(patient_dict.get("Depression", 0.0)),
        "SleepQuality": float(patient_dict.get("SleepQuality", 0.0)),
        "PhysicalActivity": float(patient_dict.get("PhysicalActivity", 0.0)),
    }


@app.on_event("startup")
def startup_pipeline() -> None:
    print("[NeuroSynth] Startup: loading dataset and training models...")
    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test, feature_names, scaler, stats = pipeline.process()

    print("[NeuroSynth] Training ensemble biomarker model...")
    predictor = BiomarkerPredictor(feature_names=feature_names)
    predictor.train(X_train.values, y_train.values)

    print("[NeuroSynth] Training pseudo-temporal progression model...")
    temporal = TemporalProgressionModel(feature_names=feature_names)
    temporal.train_model(X_train.values, y_train.values)

    print("[NeuroSynth] Fitting neural causal discovery engine...")
    causal = NeuralCausalDiscovery()
    causal_vars = causal.variables
    causal_df = pipeline.df_processed[causal_vars].copy() if pipeline.df_processed is not None else pd.DataFrame()
    if not causal_df.empty:
        causal.fit(causal_df.values.astype(float), epochs=1000, outer_iters=15, inner_iters=100)

    reporter = ClinicalReportGenerator()

    STATE.update(
        {
            "ready": True,
            "pipeline": pipeline,
            "predictor": predictor,
            "temporal": temporal,
            "causal": causal,
            "reporter": reporter,
            "feature_names": feature_names,
            "dataset_stats": stats,
            "X_test": X_test.values,
            "y_test": y_test.values,
            "X_test_raw": pipeline.df_processed.loc[X_test.index, feature_names].values if pipeline.df_processed is not None else None,
            "causal_graph": causal.get_causal_graph(),
            "scaler": scaler,
        }
    )
    print("[NeuroSynth] Startup complete.")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "running", "ready": bool(STATE["ready"]) }


@app.get("/dataset/stats")
def dataset_stats() -> dict[str, Any]:
    return STATE["dataset_stats"]


@app.get("/model/performance")
def model_performance() -> dict[str, Any]:
    predictor: BiomarkerPredictor = STATE["predictor"]
    return predictor.evaluate(STATE["X_test"], STATE["y_test"])


@app.get("/model/feature_importance")
def model_feature_importance() -> dict[str, float]:
    predictor: BiomarkerPredictor = STATE["predictor"]
    return predictor.get_feature_importance()


@app.get("/model/shap_summary")
def model_shap_summary() -> dict[str, float]:
    predictor: BiomarkerPredictor = STATE["predictor"]
    X = STATE["X_test"]
    shap_vals = predictor.get_shap_values(X[: min(len(X), 300)])
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    pairs = sorted(zip(STATE["feature_names"], mean_abs.tolist()), key=lambda x: x[1], reverse=True)
    return {k: round(float(v), 6) for k, v in pairs}


@app.get("/causal/graph")
def causal_graph() -> dict[str, Any]:
    causal: NeuralCausalDiscovery = STATE["causal"]
    return causal.get_causal_graph()


@app.post("/predict")
def predict(patient: PatientRequest) -> dict[str, Any]:
    predictor: BiomarkerPredictor = STATE["predictor"]
    temporal: TemporalProgressionModel = STATE["temporal"]

    frame = _to_frame(patient)
    scaler = STATE["scaler"]
    scaled = scaler.transform(frame[STATE["feature_names"]])

    prediction = predictor.predict(scaled)
    traj = temporal.predict_trajectory(frame[STATE["feature_names"]].values[0], prediction["probability"])

    shap_vals = predictor.get_shap_values(scaled[:1])[0]
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
    shap_top = [
        {
            "feature": STATE["feature_names"][i],
            "value": round(float(shap_vals[i]), 4),
        }
        for i in top_idx
    ]

    return {
        **prediction,
        "trajectory": traj["trajectory"],
        "confidence_bands": traj["confidence_bands"],
        "shap_values": shap_top,
    }


@app.post("/report")
def report(patient: PatientRequest) -> dict[str, Any]:
    predictor: BiomarkerPredictor = STATE["predictor"]
    temporal: TemporalProgressionModel = STATE["temporal"]
    causal: NeuralCausalDiscovery = STATE["causal"]
    reporter: ClinicalReportGenerator = STATE["reporter"]

    frame = _to_frame(patient)
    scaled = STATE["scaler"].transform(frame[STATE["feature_names"]])

    prediction = predictor.predict(scaled)
    traj = temporal.predict_trajectory(frame[STATE["feature_names"]].values[0], prediction["probability"])

    shap_vals = predictor.get_shap_values(scaled[:1])[0]
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
    shap_top = [{"feature": STATE["feature_names"][i], "value": float(shap_vals[i])} for i in top_idx]

    return reporter.generate_report(
        patient_data=frame.iloc[0].to_dict(),
        prediction=prediction,
        trajectory=traj["trajectory"],
        causal_graph=causal.get_causal_graph(),
        shap_values=shap_top,
    )


@app.post("/simulate")
def simulate(req: SimulationRequest) -> dict[str, Any]:
    causal: NeuralCausalDiscovery = STATE["causal"]
    cpatient = _to_causal_patient_map(req.patient_data)
    return causal.simulate_intervention(req.variable, req.new_value, cpatient)


@app.post("/batch_predict")
def batch_predict(records: list[PatientRequest]) -> dict[str, Any]:
    if len(records) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 records per batch")

    results = []
    for rec in records:
        frame = _to_frame(rec)
        scaled = STATE["scaler"].transform(frame[STATE["feature_names"]])
        pred = STATE["predictor"].predict(scaled)
        results.append(pred)
    return {"count": len(results), "results": results}


static_dir = Path("frontend/dist")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
