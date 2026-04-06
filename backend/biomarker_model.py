from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score


class BiomarkerPredictor:
    """Ensemble biomarker risk predictor using RF and GB models."""

    def __init__(self, feature_names: Iterable[str] | None = None, models_dir: str | Path = "models") -> None:
        self.feature_names = list(feature_names) if feature_names else [
            "Age",
            "EDUC",
            "SES",
            "MMSE",
            "CDR",
            "eTIV",
            "nWBV",
            "ASF",
        ]
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        )
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)
        joblib.dump(self.rf_model, self.models_dir / "rf_model.pkl")
        joblib.dump(self.gb_model, self.models_dir / "gb_model.pkl")

    def _ensemble_probability(self, X: np.ndarray) -> np.ndarray:
        rf_prob = self.rf_model.predict_proba(X)[:, 1]
        gb_prob = self.gb_model.predict_proba(X)[:, 1]
        return (rf_prob + gb_prob) / 2.0

    def predict(self, X: np.ndarray | List[float] | List[List[float]]) -> Dict[str, float | str]:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        probability = float(self._ensemble_probability(X_arr)[0])
        prediction = "Demented" if probability >= 0.5 else "Nondemented"

        if probability > 0.8:
            confidence = "High"
        elif probability > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"

        if probability > 0.85:
            risk_level = "Critical"
        elif probability > 0.7:
            risk_level = "High"
        elif probability > 0.5:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "confidence": confidence,
            "risk_level": risk_level,
        }

    def get_feature_importance(self) -> Dict[str, float]:
        rf_imp = self.rf_model.feature_importances_
        gb_imp = self.gb_model.feature_importances_
        mean_imp = (rf_imp + gb_imp) / 2.0

        sorted_pairs = sorted(
            zip(self.feature_names, mean_imp.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return {name: round(score, 6) for name, score in sorted_pairs}

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, object]:
        probs = self._ensemble_probability(X_test)
        y_pred = (probs >= 0.5).astype(int)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }
        return metrics
