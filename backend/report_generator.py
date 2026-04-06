from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List

import requests


class ClinicalReportGenerator:
    model = "mistralai/Mistral-7B-Instruct-v0.3"
    endpoint = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

    def __init__(self, hf_token: str | None = None) -> None:
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")

    @staticmethod
    def _build_prompt(patient_data: dict, prediction: dict, trajectory: list, causal_graph: dict) -> str:
        return (
            "[INST] You are NeuroSynth's clinical AI. Generate a structured neurological assessment report.\n"
            f"Patient Data: {patient_data}\n"
            f"Risk Prediction: {prediction}\n"
            f"36-Month Trajectory: {trajectory}\n"
            f"Causal Analysis: {causal_graph}\n"
            "Generate a report with these exact sections:\n\n"
            "CLINICAL SUMMARY (2 sentences)\n"
            "RISK ASSESSMENT (current risk level and key drivers)\n"
            "PROGRESSION FORECAST (what the 36-month trajectory means clinically)\n"
            "CAUSAL PATHWAYS (which biomarkers are driving deterioration and why)\n"
            "INTERVENTION RECOMMENDATIONS (top 3 evidence-based, modifiable recommendations)\n"
            "MONITORING PROTOCOL (which biomarkers to track and how often)\n"
            "UNCERTAINTY FLAGS (what data would improve this assessment)\n"
            "Keep each section concise and clinically actionable. [/INST]"
        )

    @staticmethod
    def _parse_sections(text: str) -> Dict[str, str]:
        section_names = [
            "CLINICAL SUMMARY",
            "RISK ASSESSMENT",
            "PROGRESSION FORECAST",
            "CAUSAL PATHWAYS",
            "INTERVENTION RECOMMENDATIONS",
            "MONITORING PROTOCOL",
            "UNCERTAINTY FLAGS",
        ]

        lines = text.splitlines()
        sections: Dict[str, List[str]] = {name: [] for name in section_names}
        current = section_names[0]

        for raw_line in lines:
            line = raw_line.strip()
            matched = next((name for name in section_names if line.upper().startswith(name)), None)
            if matched:
                current = matched
                suffix = line[len(matched) :].lstrip(" :-")
                if suffix:
                    sections[current].append(suffix)
                continue
            sections[current].append(line)

        return {k: "\n".join(v).strip() for k, v in sections.items()}

    def _fallback_report(
        self,
        patient_data: dict,
        prediction: dict,
        trajectory: list,
        causal_graph: dict,
    ) -> Dict[str, object]:
        top_cdr = ", ".join([item["variable"] for item in causal_graph.get("top_causes_of_CDR", [])]) or "No strong drivers detected"
        top_modifiable = ", ".join(causal_graph.get("modifiable_interventions", [])) or "MMSE, SES, EDUC"

        sections = {
            "CLINICAL SUMMARY": (
                f"The current model estimates a {prediction.get('risk_level', 'Unknown')} neurological risk state. "
                f"The prediction confidence is {prediction.get('confidence', 'Unknown')} with probability {prediction.get('probability', 0)}."
            ),
            "RISK ASSESSMENT": (
                f"Risk level is {prediction.get('risk_level', 'Unknown')} based on integrated biomarker patterns. "
                f"Primary model drivers include: {top_cdr}."
            ),
            "PROGRESSION FORECAST": (
                f"Projected 36-month risk trajectory: {trajectory}. "
                "Upward movement indicates likely deterioration pressure, while stable or declining values suggest lower progression velocity."
            ),
            "CAUSAL PATHWAYS": (
                f"Causal graph highlights likely directional influences toward deterioration markers. "
                f"Most influential contributors for CDR in this run are: {top_cdr}."
            ),
            "INTERVENTION RECOMMENDATIONS": (
                f"1) Optimize modifiable biomarker pathways: {top_modifiable}. "
                "2) Increase follow-up cognitive assessments for trend confirmation. "
                "3) Pair risk monitoring with lifestyle and adherence interventions."
            ),
            "MONITORING PROTOCOL": (
                "Track MMSE and CDR monthly in high-risk cases; track SES-linked care factors quarterly; "
                "re-evaluate structural biomarkers every 6 to 12 months."
            ),
            "UNCERTAINTY FLAGS": (
                "Additional longitudinal visits, treatment adherence details, and richer multimodal data "
                "would reduce uncertainty and improve intervention confidence."
            ),
        }

        raw_text = "\n\n".join([f"{k}\n{v}" for k, v in sections.items()])
        return {
            "sections": sections,
            "raw_text": raw_text,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def generate_report(self, patient_data: dict, prediction: dict, trajectory: list, causal_graph: dict) -> Dict[str, object]:
        prompt = self._build_prompt(patient_data, prediction, trajectory, causal_graph)

        if not self.hf_token:
            return self._fallback_report(patient_data, prediction, trajectory, causal_graph)

        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 800,
                "temperature": 0.3,
                "return_full_text": False,
            },
        }

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list) and data and isinstance(data[0], dict):
                raw_text = data[0].get("generated_text", "")
            elif isinstance(data, dict) and "generated_text" in data:
                raw_text = data.get("generated_text", "")
            else:
                raw_text = str(data)

            sections = self._parse_sections(raw_text)
            return {
                "sections": sections,
                "raw_text": raw_text,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception:
            return self._fallback_report(patient_data, prediction, trajectory, causal_graph)
