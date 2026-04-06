from __future__ import annotations

import os
from typing import Dict, List

import gradio as gr
import pandas as pd

from backend.biomarker_model import BiomarkerPredictor
from backend.causal_engine import NeuralCausalDiscovery
from backend.data_pipeline import DataPipeline
from backend.report_generator import ClinicalReportGenerator
from backend.temporal_model import TemporalProgressionModel


STATE: Dict[str, object] = {}


def _init_pipeline() -> None:
    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test, feature_names, scaler, patient_sequences = pipeline.process()

    predictor = BiomarkerPredictor(feature_names=feature_names)
    predictor.train(X_train, y_train)

    temporal = TemporalProgressionModel(fallback_predictor=lambda x: predictor.predict(x)["probability"])
    temporal.train_model(patient_sequences=patient_sequences, labels=pipeline.subject_labels, epochs=50, lr=0.001)

    causal = NeuralCausalDiscovery()
    causal.fit(pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)]).values, epochs=500, lr=0.01, lambda1=0.01, lambda2=5.0)

    reporter = ClinicalReportGenerator(os.getenv("HF_TOKEN"))

    STATE.update(
        {
            "pipeline": pipeline,
            "predictor": predictor,
            "temporal": temporal,
            "causal": causal,
            "reporter": reporter,
            "scaler": scaler,
            "feature_names": feature_names,
            "X_test": X_test,
            "y_test": y_test,
        }
    )


def _patient_payload(age, educ, ses, mmse, cdr, etiv, nwbv, asf) -> Dict[str, float]:
    return {
        "age": float(age),
        "educ": float(educ),
        "ses": float(ses),
        "mmse": float(mmse),
        "cdr": float(cdr),
        "etiv": float(etiv),
        "nwbv": float(nwbv),
        "asf": float(asf),
    }


def analyze_patient(age, educ, ses, mmse, cdr, etiv, nwbv, asf):
    payload = _patient_payload(age, educ, ses, mmse, cdr, etiv, nwbv, asf)
    raw_vec = [payload[k] for k in ["age", "educ", "ses", "mmse", "cdr", "etiv", "nwbv", "asf"]]
    scaled = STATE["scaler"].transform([raw_vec])

    prediction = STATE["predictor"].predict(scaled)
    trajectory = STATE["temporal"].predict_trajectory([scaled.flatten().tolist()])
    months = [6, 12, 18, 24, 30, 36]
    trajectory_df = pd.DataFrame({"month": months, "risk": trajectory})

    result = {
        "prediction": prediction["prediction"],
        "probability": prediction["probability"],
        "confidence": prediction["confidence"],
        "risk_level": prediction["risk_level"],
        "trajectory": trajectory,
        "feature_importance": STATE["predictor"].get_feature_importance(),
    }
    return result, trajectory_df


def generate_report(age, educ, ses, mmse, cdr, etiv, nwbv, asf):
    payload = _patient_payload(age, educ, ses, mmse, cdr, etiv, nwbv, asf)
    raw_vec = [payload[k] for k in ["age", "educ", "ses", "mmse", "cdr", "etiv", "nwbv", "asf"]]
    scaled = STATE["scaler"].transform([raw_vec])

    prediction = STATE["predictor"].predict(scaled)
    trajectory = STATE["temporal"].predict_trajectory([scaled.flatten().tolist()])
    graph = STATE["causal"].get_causal_graph()

    report = STATE["reporter"].generate_report(payload, prediction, trajectory, graph)
    md_sections = []
    for title, content in report["sections"].items():
        md_sections.append(f"### {title}\n{content}")
    return "\n\n".join(md_sections)


def causal_table_and_summary():
    graph = STATE["causal"].get_causal_graph()
    edges = graph.get("edges", [])
    edge_df = pd.DataFrame(edges) if edges else pd.DataFrame(columns=["from", "to", "strength"])

    summary = (
        "### Top causes of CDR\n"
        + "\n".join([f"- {x['variable']} ({x['strength']})" for x in graph.get("top_causes_of_CDR", [])])
        + "\n\n### Top causes of MMSE\n"
        + "\n".join([f"- {x['variable']} ({x['strength']})" for x in graph.get("top_causes_of_MMSE", [])])
        + "\n\n### Modifiable interventions\n"
        + "\n".join([f"- {x}" for x in graph.get("modifiable_interventions", [])])
    )
    return edge_df, summary


_init_pipeline()

with gr.Blocks(
    title="🧠 NeuroSynth — Neurological Deterioration Prediction Engine",
    theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="violet"),
    css="body { background: #0a0a0f; } .gradio-container { background: #0a0a0f !important; }",
) as demo:
    gr.Markdown("# 🧠 NeuroSynth — Neurological Deterioration Prediction Engine")
    gr.Markdown(
        "Use Patient Analysis for prediction and trajectory, Clinical Report for AI-generated assessment, "
        "and Causal Analysis for directional biomarker insights."
    )

    with gr.Tab("Patient Analysis"):
        with gr.Row():
            age = gr.Slider(50, 100, value=74, step=1, label="Age")
            educ = gr.Slider(0, 25, value=14, step=1, label="EDUC")
            ses = gr.Slider(1, 5, value=2, step=1, label="SES")
            mmse = gr.Slider(0, 30, value=24, step=1, label="MMSE")
        with gr.Row():
            cdr = gr.Slider(0, 3, value=0.5, step=0.5, label="CDR")
            etiv = gr.Slider(900, 2000, value=1450, step=1, label="eTIV")
            nwbv = gr.Slider(0.6, 0.9, value=0.72, step=0.001, label="nWBV")
            asf = gr.Slider(0.8, 1.8, value=1.2, step=0.001, label="ASF")

        analyze_btn = gr.Button("Analyze", variant="primary")
        analyze_json = gr.JSON(label="Prediction Results")
        traj_plot = gr.LinePlot(x="month", y="risk", title="36-Month Risk Trajectory")
        analyze_btn.click(
            analyze_patient,
            inputs=[age, educ, ses, mmse, cdr, etiv, nwbv, asf],
            outputs=[analyze_json, traj_plot],
        )

    with gr.Tab("Clinical Report"):
        with gr.Row():
            age_r = gr.Slider(50, 100, value=74, step=1, label="Age")
            educ_r = gr.Slider(0, 25, value=14, step=1, label="EDUC")
            ses_r = gr.Slider(1, 5, value=2, step=1, label="SES")
            mmse_r = gr.Slider(0, 30, value=24, step=1, label="MMSE")
        with gr.Row():
            cdr_r = gr.Slider(0, 3, value=0.5, step=0.5, label="CDR")
            etiv_r = gr.Slider(900, 2000, value=1450, step=1, label="eTIV")
            nwbv_r = gr.Slider(0.6, 0.9, value=0.72, step=0.001, label="nWBV")
            asf_r = gr.Slider(0.8, 1.8, value=1.2, step=0.001, label="ASF")

        report_btn = gr.Button("Generate Report", variant="primary")
        report_md = gr.Markdown()
        report_btn.click(
            generate_report,
            inputs=[age_r, educ_r, ses_r, mmse_r, cdr_r, etiv_r, nwbv_r, asf_r],
            outputs=report_md,
        )

    with gr.Tab("Causal Analysis"):
        causal_btn = gr.Button("Show Causal Graph", variant="primary")
        causal_df = gr.Dataframe(label="Causal Edges", headers=["from", "to", "strength"], interactive=False)
        causal_md = gr.Markdown()
        causal_btn.click(causal_table_and_summary, inputs=None, outputs=[causal_df, causal_md])


if __name__ == "__main__":
    demo.launch()
