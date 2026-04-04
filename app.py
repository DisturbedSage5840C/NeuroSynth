import gradio as gr
import joblib
import pandas as pd


model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")


def predict(age, educ, ses, mmse, cdr, etiv, nwbv, asf):
    features = pd.DataFrame(
        [[age, educ, ses, mmse, cdr, etiv, nwbv, asf]],
        columns=["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"],
    )
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()
    label = le.inverse_transform([prediction])[0]
    return f"Prediction: {label} | Confidence: {round(probability * 100, 2)}%"


gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Education (years)"),
        gr.Number(label="SES (1-5)"),
        gr.Number(label="MMSE Score"),
        gr.Number(label="CDR Score"),
        gr.Number(label="eTIV"),
        gr.Number(label="nWBV"),
        gr.Number(label="ASF"),
    ],
    outputs="text",
    title="NeuroSynth",
    description="Neurological risk prediction from patient biomarkers",
).launch()
