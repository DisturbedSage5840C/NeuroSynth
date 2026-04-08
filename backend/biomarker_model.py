from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import shap
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


class BiomarkerPredictor:
    def __init__(self, feature_names: list[str], models_dir: str | Path = "models") -> None:
        self.feature_names = feature_names
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )
        if XGBClassifier is not None:
            self.third = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss",
            )
            self.third_name = "xgboost"
        else:
            self.third = ExtraTreesClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
            self.third_name = "extra_trees"

        self.lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        self.weights = np.array([0.35, 0.35, 0.20, 0.10], dtype=float)
        self.tree_explainer: shap.TreeExplainer | None = None

    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob >= 0.8:
            return "Critical"
        if prob >= 0.65:
            return "High"
        if prob >= 0.4:
            return "Moderate"
        return "Low"

    @staticmethod
    def _confidence(prob: float) -> str:
        margin = abs(prob - 0.5)
        if margin >= 0.3:
            return "High"
        if margin >= 0.15:
            return "Medium"
        return "Low"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.rf.fit(X_train, y_train)
        self.gb.fit(X_train, y_train)
        self.third.fit(X_train, y_train)
        self.lr.fit(X_train, y_train)

        self.tree_explainer = shap.TreeExplainer(self.rf)

        joblib.dump(self.rf, self.models_dir / "rf_model.pkl")
        joblib.dump(self.gb, self.models_dir / "gb_model.pkl")
        joblib.dump(self.third, self.models_dir / f"{self.third_name}_model.pkl")
        joblib.dump(self.lr, self.models_dir / "lr_model.pkl")

    def _ensemble_probs(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        rf_p = self.rf.predict_proba(X)[:, 1]
        gb_p = self.gb.predict_proba(X)[:, 1]
        third_p = self.third.predict_proba(X)[:, 1]
        lr_p = self.lr.predict_proba(X)[:, 1]

        stacked = np.vstack([rf_p, gb_p, third_p, lr_p])
        ensemble = np.average(stacked, axis=0, weights=self.weights)
        per_model = {
            "random_forest": rf_p,
            "gradient_boosting": gb_p,
            self.third_name: third_p,
            "logistic_regression": lr_p,
        }
        return ensemble, per_model

    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        if self.tree_explainer is None:
            self.tree_explainer = shap.TreeExplainer(self.rf)

        shap_values = self.tree_explainer.shap_values(X)
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                return np.asarray(shap_values[1])
            return np.asarray(shap_values[0])

        arr = np.asarray(shap_values)
        # Some SHAP versions return (n_samples, n_features, n_classes) for tree models.
        if arr.ndim == 3:
            if arr.shape[-1] == 2:
                return arr[:, :, 1]
            return arr.mean(axis=-1)
        return arr

    def predict(self, X: np.ndarray) -> dict[str, Any]:
        ensemble, per_model = self._ensemble_probs(X)
        prob = float(np.clip(ensemble[0], 0.0, 1.0))
        pred = int(prob >= 0.5)

        shap_vals = self.get_shap_values(X[:1])[0]
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
        top_risk_factors = [self.feature_names[i] for i in top_idx]

        return {
            "prediction": pred,
            "probability": round(prob, 4),
            "confidence": self._confidence(prob),
            "risk_level": self._risk_level(prob),
            "individual_model_probs": {
                k: round(float(v[0]), 4) for k, v in per_model.items()
            },
            "top_risk_factors": top_risk_factors,
        }

    def get_feature_importance(self) -> dict[str, float]:
        importances = []
        for model in [self.rf, self.gb, self.third]:
            if hasattr(model, "feature_importances_"):
                importances.append(np.asarray(model.feature_importances_, dtype=float))
        if not importances:
            return {name: 0.0 for name in self.feature_names}

        avg = np.mean(np.vstack(importances), axis=0)
        ranking = sorted(
            [(name, float(score)) for name, score in zip(self.feature_names, avg)],
            key=lambda x: x[1],
            reverse=True,
        )
        return {k: round(v, 6) for k, v in ranking}

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
        ensemble, _ = self._ensemble_probs(X_test)
        y_pred = (ensemble >= 0.5).astype(int)

        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
            "roc_auc": round(float(roc_auc_score(y_test, ensemble)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }
