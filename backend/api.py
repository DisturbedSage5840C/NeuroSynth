from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.biomarker_model import BiomarkerPredictor
from backend.causal_engine import NeuralCausalDiscovery
from backend.data_pipeline import DataPipeline
from backend.report_generator import ClinicalReportGenerator
from backend.temporal_model import TemporalProgressionModel


class PatientRequest(BaseModel):
    age: float
    educ: float
    ses: float
    mmse: float
    cdr: float
    etiv: float
    nwbv: float
    asf: float


class SimulationRequest(BaseModel):
    patient_data: Dict[str, float]
    intervention_variable: str
    new_value: float


app = FastAPI(title="NeuroSynth API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: Dict[str, Any] = {
    "models_loaded": False,
    "pipeline": None,
    "predictor": None,
    "temporal": None,
    "causal": None,
    "reporter": None,
    "X_test": None,
    "y_test": None,
    "dataset_size": 0,
}


def _patient_to_vector(patient: PatientRequest) -> np.ndarray:
    return np.array(
        [
            patient.age,
            patient.educ,
            patient.ses,
            patient.mmse,
            patient.cdr,
            patient.etiv,
            patient.nwbv,
            patient.asf,
        ],
        dtype=float,
    )


def _compute_top_risk_factors(feature_importance: Dict[str, float], patient_vector: np.ndarray) -> List[Dict[str, float]]:
    feature_names = ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
    factors = []
    for idx, name in enumerate(feature_names):
        importance = float(feature_importance.get(name, 0.0))
        value = float(patient_vector[idx])
        factors.append({"feature": name, "score": round(abs(value) * importance, 4)})
    factors.sort(key=lambda x: x["score"], reverse=True)
    return factors[:3]


@app.on_event("startup")
def startup_train_pipeline() -> None:
    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test, feature_names, scaler, patient_sequences = pipeline.process()

    predictor = BiomarkerPredictor(feature_names=feature_names)
    predictor.train(X_train, y_train)

    temporal = TemporalProgressionModel(fallback_predictor=lambda x: predictor.predict(x)["probability"])
    temporal.train_model(patient_sequences=patient_sequences, labels=pipeline.subject_labels, epochs=50, lr=0.001)

    causal = NeuralCausalDiscovery()
    X_scaled = np.vstack([X_train, X_test])
    causal.fit(X_scaled, epochs=500, lr=0.01, lambda1=0.01, lambda2=5.0)

    reporter = ClinicalReportGenerator()

    STATE.update(
        {
            "models_loaded": True,
            "pipeline": pipeline,
            "predictor": predictor,
            "temporal": temporal,
            "causal": causal,
            "reporter": reporter,
            "X_test": X_test,
            "y_test": y_test,
            "dataset_size": int(len(y_train) + len(y_test)),
            "feature_names": feature_names,
            "scaler": scaler,
        }
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "running",
        "models_loaded": bool(STATE["models_loaded"]),
        "dataset_size": int(STATE["dataset_size"]),
    }


@app.get("/model/performance")
def model_performance() -> Dict[str, Any]:
    predictor: BiomarkerPredictor = STATE["predictor"]
    return predictor.evaluate(STATE["X_test"], STATE["y_test"])


@app.get("/model/feature_importance")
def model_feature_importance() -> Dict[str, float]:
    predictor: BiomarkerPredictor = STATE["predictor"]
    return predictor.get_feature_importance()


@app.get("/causal/graph")
def causal_graph() -> Dict[str, Any]:
    causal: NeuralCausalDiscovery = STATE["causal"]
    return causal.get_causal_graph()


@app.post("/predict")
def predict(patient: PatientRequest) -> Dict[str, Any]:
    predictor: BiomarkerPredictor = STATE["predictor"]
    temporal: TemporalProgressionModel = STATE["temporal"]

    raw_vec = _patient_to_vector(patient)
    scaled_vec = STATE["scaler"].transform(raw_vec.reshape(1, -1))

    prediction = predictor.predict(scaled_vec)
    trajectory = temporal.predict_trajectory([scaled_vec.flatten().tolist()])
    feature_importance = predictor.get_feature_importance()
    top_risk_factors = _compute_top_risk_factors(feature_importance, raw_vec)

    return {
        "prediction": prediction["prediction"],
        "probability": prediction["probability"],
        "confidence": prediction["confidence"],
        "risk_level": prediction["risk_level"],
        "trajectory": trajectory,
        "feature_importance": feature_importance,
        "top_risk_factors": top_risk_factors,
    }


@app.post("/report")
def generate_report(patient: PatientRequest) -> Dict[str, Any]:
    predictor: BiomarkerPredictor = STATE["predictor"]
    temporal: TemporalProgressionModel = STATE["temporal"]
    causal: NeuralCausalDiscovery = STATE["causal"]
    reporter: ClinicalReportGenerator = STATE["reporter"]

    raw_vec = _patient_to_vector(patient)
    scaled_vec = STATE["scaler"].transform(raw_vec.reshape(1, -1))

    prediction = predictor.predict(scaled_vec)
    trajectory = temporal.predict_trajectory([scaled_vec.flatten().tolist()])
    graph = causal.get_causal_graph()

    patient_data = patient.model_dump()
    return reporter.generate_report(patient_data, prediction, trajectory, graph)


@app.post("/simulate")
def simulate_intervention(request: SimulationRequest) -> Dict[str, Any]:
    causal: NeuralCausalDiscovery = STATE["causal"]
    return causal.simulate_intervention(
        variable=request.intervention_variable,
        new_value=request.new_value,
        current_patient_data=request.patient_data,
    )


@app.get("/dataset/stats")
def dataset_stats() -> Dict[str, Any]:
    pipeline: DataPipeline = STATE["pipeline"]
    df = pipeline.df_clean.copy() if pipeline.df_clean is not None else pd.DataFrame()

    if df.empty:
        return {
            "n_patients": 0,
            "n_demented": 0,
            "n_nondemented": 0,
            "mean_age": 0.0,
            "feature_distributions": {},
        }

    feature_distributions: Dict[str, Dict[str, float]] = {}
    for col in pipeline.feature_columns:
        feature_distributions[col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
        }

    return {
        "n_patients": int(df["Subject ID"].nunique()),
        "n_demented": int((df["target"] == 1).sum()),
        "n_nondemented": int((df["target"] == 0).sum()),
        "mean_age": round(float(df["Age"].mean()), 4),
        "feature_distributions": feature_distributions,
    }


static_dir = Path("static")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=False)
