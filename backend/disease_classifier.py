"""
Disease-type classifier. Takes patient features and returns the most likely
neurological disease type. Used to route to disease-specific risk models.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DISEASES = [
    "Alzheimer's Disease",
    "Parkinson's Disease",
    "Multiple Sclerosis",
    "Epilepsy",
    "ALS",
    "Huntington's Disease",
]


class DiseaseClassifier:
    def __init__(self, models_dir: str | Path = "models") -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.clf = RandomForestClassifier(n_estimators=200, random_state=42)
        self.le = LabelEncoder()
        self.feature_names: list[str] | None = None

    def generate_synthetic_training_data(self, n_per_class: int = 500):
        import pandas as pd

        all_rows: list[dict[str, float]] = []
        all_labels: list[str] = []
        rng = np.random.RandomState(42)

        # Alzheimer's profile
        for _ in range(n_per_class):
            row = {
                "Age": rng.normal(75, 8),
                "Gender": rng.randint(0, 2),
                "BMI": rng.normal(26, 4),
                "SystolicBP": rng.normal(138, 18),
                "DiastolicBP": rng.normal(84, 12),
                "CholesterolTotal": rng.normal(210, 35),
                "PhysicalActivity": rng.normal(3, 2),
                "SleepQuality": rng.normal(5, 2),
                "Depression": rng.binomial(1, 0.4),
                "MMSE": rng.normal(18, 6),
                "FunctionalAssessment": rng.normal(5, 2),
                "ADL": rng.normal(5, 2),
                "MemoryComplaints": rng.binomial(1, 0.85),
                "FamilyHistoryAlzheimers": rng.binomial(1, 0.35),
            }
            all_rows.append(row)
            all_labels.append("Alzheimer's Disease")

        # Parkinson's profile
        for _ in range(n_per_class):
            row = {
                "Age": rng.normal(68, 9),
                "Gender": rng.binomial(1, 0.6),
                "BMI": rng.normal(25, 4),
                "SystolicBP": rng.normal(130, 20),
                "DiastolicBP": rng.normal(80, 12),
                "CholesterolTotal": rng.normal(195, 30),
                "PhysicalActivity": rng.normal(3.5, 2),
                "SleepQuality": rng.normal(5.5, 2),
                "Depression": rng.binomial(1, 0.5),
                "MMSE": rng.normal(24, 4),
                "FunctionalAssessment": rng.normal(5.5, 2),
                "ADL": rng.normal(6, 2),
                "MemoryComplaints": rng.binomial(1, 0.35),
                "FamilyHistoryAlzheimers": rng.binomial(1, 0.1),
            }
            all_rows.append(row)
            all_labels.append("Parkinson's Disease")

        # MS profile
        for _ in range(n_per_class):
            row = {
                "Age": rng.normal(38, 10),
                "Gender": rng.binomial(1, 0.3),
                "BMI": rng.normal(25, 5),
                "SystolicBP": rng.normal(118, 14),
                "DiastolicBP": rng.normal(76, 10),
                "CholesterolTotal": rng.normal(185, 25),
                "PhysicalActivity": rng.normal(5, 2.5),
                "SleepQuality": rng.normal(6, 2),
                "Depression": rng.binomial(1, 0.45),
                "MMSE": rng.normal(27, 2),
                "FunctionalAssessment": rng.normal(6.5, 2),
                "ADL": rng.normal(7, 2),
                "MemoryComplaints": rng.binomial(1, 0.3),
                "FamilyHistoryAlzheimers": rng.binomial(1, 0.05),
            }
            all_rows.append(row)
            all_labels.append("Multiple Sclerosis")

        # Epilepsy profile
        for _ in range(n_per_class):
            row = {
                "Age": rng.normal(35, 18),
                "Gender": rng.randint(0, 2),
                "BMI": rng.normal(26, 5),
                "SystolicBP": rng.normal(122, 15),
                "DiastolicBP": rng.normal(78, 10),
                "CholesterolTotal": rng.normal(190, 30),
                "PhysicalActivity": rng.normal(5.5, 2),
                "SleepQuality": rng.normal(6.5, 1.5),
                "Depression": rng.binomial(1, 0.3),
                "MMSE": rng.normal(27, 2),
                "FunctionalAssessment": rng.normal(7, 1.5),
                "ADL": rng.normal(7.5, 1.5),
                "MemoryComplaints": rng.binomial(1, 0.25),
                "FamilyHistoryAlzheimers": rng.binomial(1, 0.08),
            }
            all_rows.append(row)
            all_labels.append("Epilepsy")

        # ALS profile
        for _ in range(n_per_class):
            row = {
                "Age": rng.normal(58, 10),
                "Gender": rng.binomial(1, 0.6),
                "BMI": rng.normal(24, 4),
                "SystolicBP": rng.normal(125, 16),
                "DiastolicBP": rng.normal(80, 10),
                "CholesterolTotal": rng.normal(188, 28),
                "PhysicalActivity": rng.normal(2.5, 2),
                "SleepQuality": rng.normal(5, 2),
                "Depression": rng.binomial(1, 0.3),
                "MMSE": rng.normal(27, 2),
                "FunctionalAssessment": rng.normal(4, 2.5),
                "ADL": rng.normal(4.5, 2.5),
                "MemoryComplaints": rng.binomial(1, 0.15),
                "FamilyHistoryAlzheimers": rng.binomial(1, 0.05),
            }
            all_rows.append(row)
            all_labels.append("ALS")

        # Huntington's profile
        for _ in range(n_per_class):
            row = {
                "Age": rng.normal(45, 12),
                "Gender": rng.randint(0, 2),
                "BMI": rng.normal(23, 4),
                "SystolicBP": rng.normal(120, 14),
                "DiastolicBP": rng.normal(78, 9),
                "CholesterolTotal": rng.normal(185, 25),
                "PhysicalActivity": rng.normal(3.5, 2),
                "SleepQuality": rng.normal(5.5, 2),
                "Depression": rng.binomial(1, 0.5),
                "MMSE": rng.normal(22, 5),
                "FunctionalAssessment": rng.normal(5, 2.5),
                "ADL": rng.normal(5.5, 2.5),
                "MemoryComplaints": rng.binomial(1, 0.6),
                "FamilyHistoryAlzheimers": rng.binomial(1, 0.05),
            }
            all_rows.append(row)
            all_labels.append("Huntington's Disease")

        df = pd.DataFrame(all_rows)
        df["Age"] = df["Age"].clip(20, 100)
        df["MMSE"] = df["MMSE"].clip(0, 30)
        df["FunctionalAssessment"] = df["FunctionalAssessment"].clip(0, 10)
        df["ADL"] = df["ADL"].clip(0, 10)
        df["PhysicalActivity"] = df["PhysicalActivity"].clip(0, 10)
        df["SleepQuality"] = df["SleepQuality"].clip(0, 10)

        return df, pd.Series(all_labels)

    def train(self) -> None:
        df, labels = self.generate_synthetic_training_data(n_per_class=800)
        y = self.le.fit_transform(labels)
        self.clf.fit(df.values, y)
        self.feature_names = list(df.columns)

        joblib.dump(self.clf, self.models_dir / "disease_clf.pkl")
        joblib.dump(self.le, self.models_dir / "disease_le.pkl")
        joblib.dump(self.feature_names, self.models_dir / "disease_features.pkl")

    def _lazy_load(self) -> None:
        if self.feature_names is not None:
            return
        self.feature_names = joblib.load(self.models_dir / "disease_features.pkl")
        self.clf = joblib.load(self.models_dir / "disease_clf.pkl")
        self.le = joblib.load(self.models_dir / "disease_le.pkl")

    def predict_disease(self, patient_features: dict) -> dict:
        import pandas as pd

        self._lazy_load()
        assert self.feature_names is not None

        row = {f: float(patient_features.get(f, 0.0)) for f in self.feature_names}
        df = pd.DataFrame([row])

        probs = self.clf.predict_proba(df.values)[0]
        pred_idx = int(np.argmax(probs))
        pred_disease = self.le.inverse_transform([pred_idx])[0]

        all_probs = {
            self.le.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(probs)
        }

        top_prob = float(probs[pred_idx])
        confidence = "High" if top_prob > 0.6 else "Medium" if top_prob > 0.4 else "Low"

        return {
            "predicted_disease": pred_disease,
            "disease_probabilities": all_probs,
            "confidence": confidence,
        }
