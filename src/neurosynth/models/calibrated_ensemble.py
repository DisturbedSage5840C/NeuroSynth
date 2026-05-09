"""Enhanced ensemble with CatBoost + calibrated meta-learner + MAPIE conformal.

Upgrades from the v1 BiomarkerPredictor (RF+GB+XGB+LR with fixed weights)
to a 5-model ensemble with:
  - CatBoost as 5th base learner
  - Trained meta-learner (LR on OOF probabilities) replacing fixed weights
  - MAPIE conformal prediction for calibrated uncertainty intervals
  - Platt scaling for probability calibration
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None  # type: ignore[assignment,misc]

try:
    from mapie.classification import MapieClassifier
except ImportError:
    MapieClassifier = None  # type: ignore[assignment,misc]

try:
    import shap
except ImportError:
    shap = None

logger = logging.getLogger(__name__)


class CalibratedEnsemble:
    """5-model calibrated ensemble with conformal prediction.

    Base models:
      1. RandomForest (500 trees)
      2. GradientBoosting (300 trees)
      3. XGBoost or ExtraTrees (fallback)
      4. LogisticRegression (regularized)
      5. CatBoost (if available)

    Meta-learner: LogisticRegression on out-of-fold probabilities.
    Calibration: Platt scaling via CalibratedClassifierCV.
    Uncertainty: MAPIE conformal prediction intervals.
    """

    def __init__(
        self,
        feature_names: list[str],
        models_dir: str | Path = "models/ensemble_v2",
        n_cv_folds: int = 5,
    ) -> None:
        self.feature_names = feature_names
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.n_cv_folds = n_cv_folds

        # Base learners
        self.rf = RandomForestClassifier(
            n_estimators=500, max_depth=15, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42,
        )

        try:
            from xgboost import XGBClassifier
            self.xgb = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
                eval_metric="logloss",
            )
            self.xgb_name = "xgboost"
        except Exception:
            self.xgb = ExtraTreesClassifier(
                n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1,
            )
            self.xgb_name = "extra_trees"

        self.lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)

        if CatBoostClassifier is not None:
            self.catboost = CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6,
                auto_class_weights="Balanced", random_seed=42,
                verbose=0, allow_writing_files=False,
            )
        else:
            self.catboost = ExtraTreesClassifier(
                n_estimators=200, random_state=43, n_jobs=-1,
            )
            logger.warning("CatBoost not available, using ExtraTrees fallback")

        # Meta-learner (trained on OOF probs)
        self.meta_learner = LogisticRegression(C=10.0, max_iter=500, random_state=42)

        # Calibrated meta-learner
        self.calibrated_meta: CalibratedClassifierCV | None = None

        # Conformal predictor
        self.mapie_classifier: Any = None

        # SHAP explainer
        self.tree_explainer: Any = None

        # Optimal threshold
        self.decision_threshold: float = 0.5

    @property
    def base_models(self) -> list[tuple[str, Any]]:
        return [
            ("random_forest", self.rf),
            ("gradient_boosting", self.gb),
            (self.xgb_name, self.xgb),
            ("logistic_regression", self.lr),
            ("catboost", self.catboost),
        ]

    def _get_oof_predictions(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Generate out-of-fold probability predictions for meta-learner training."""
        n_models = len(self.base_models)
        oof = np.zeros((len(X), n_models))
        kf = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            for model_idx, (name, model) in enumerate(self.base_models):
                clone = self._clone_model(model)
                clone.fit(X_tr, y_tr)
                probs = clone.predict_proba(X_val)
                oof[val_idx, model_idx] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

            logger.info("OOF fold %d/%d complete", fold_idx + 1, self.n_cv_folds)

        return oof

    @staticmethod
    def _clone_model(model: Any) -> Any:
        """Clone a sklearn-compatible model."""
        from sklearn.base import clone
        return clone(model)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
        """Full training pipeline: base models → OOF → meta-learner → calibration."""

        # 1. Get out-of-fold predictions for meta-learner
        logger.info("Generating OOF predictions (%d folds)...", self.n_cv_folds)
        oof_probs = self._get_oof_predictions(X_train, y_train)

        # 2. Train meta-learner on OOF predictions
        logger.info("Training meta-learner on OOF probabilities...")
        self.meta_learner.fit(oof_probs, y_train)

        # 3. Train all base models on full dataset
        logger.info("Training base models on full dataset...")
        for name, model in self.base_models:
            model.fit(X_train, y_train)
            logger.info("  Trained %s", name)

        # 4. Calibrate the ensemble via isotonic regression (lower ECE than Platt)
        logger.info("Calibrating ensemble (isotonic regression)...")
        self.calibrated_meta = CalibratedClassifierCV(
            self.meta_learner, method="isotonic", cv=3,
        )
        self.calibrated_meta.fit(oof_probs, y_train)

        # 5. Set up MAPIE conformal predictor
        if MapieClassifier is not None:
            logger.info("Setting up MAPIE conformal predictor...")
            self.mapie_classifier = MapieClassifier(
                estimator=self.meta_learner,
                method="lac",
                cv="prefit",
                random_state=42,
            )
            self.mapie_classifier.fit(oof_probs, y_train)
        else:
            logger.warning("MAPIE not available, skipping conformal prediction")

        # 6. Find optimal threshold
        meta_probs = self.calibrated_meta.predict_proba(oof_probs)[:, 1]
        best_t, best_score = 0.5, -1.0
        for t in np.linspace(0.30, 0.75, 46):
            y_hat = (meta_probs >= t).astype(int)
            score = 0.6 * balanced_accuracy_score(y_train, y_hat) + 0.4 * accuracy_score(y_train, y_hat)
            if score > best_score:
                best_score, best_t = float(score), float(t)
        self.decision_threshold = best_t

        # 7. Set up SHAP explainer
        if shap is not None:
            self.tree_explainer = shap.TreeExplainer(self.rf)

        # 8. Compute training metrics
        train_metrics = {
            "meta_auc": round(float(roc_auc_score(y_train, meta_probs)), 4),
            "meta_brier": round(float(brier_score_loss(y_train, meta_probs)), 4),
            "meta_logloss": round(float(log_loss(y_train, meta_probs)), 4),
            "threshold": round(best_t, 4),
            "n_base_models": len(self.base_models),
        }

        # Per-model AUC
        for model_idx, (name, _) in enumerate(self.base_models):
            auc = float(roc_auc_score(y_train, oof_probs[:, model_idx]))
            train_metrics[f"{name}_oof_auc"] = round(auc, 4)

        # 9. Save all artifacts
        self._save_artifacts()

        logger.info("CalibratedEnsemble training complete: %s", train_metrics)
        return train_metrics

    def _save_artifacts(self) -> None:
        """Persist all model artifacts to disk."""
        for name, model in self.base_models:
            joblib.dump(model, self.models_dir / f"{name}_model.pkl")
        joblib.dump(self.meta_learner, self.models_dir / "meta_learner.pkl")
        if self.calibrated_meta is not None:
            joblib.dump(self.calibrated_meta, self.models_dir / "calibrated_meta.pkl")
        if self.mapie_classifier is not None:
            joblib.dump(self.mapie_classifier, self.models_dir / "mapie_classifier.pkl")
        joblib.dump(self.decision_threshold, self.models_dir / "decision_threshold.pkl")

    def load_from_disk(self) -> None:
        """Load all artifacts from disk."""
        for name, _ in self.base_models:
            path = self.models_dir / f"{name}_model.pkl"
            if path.exists():
                setattr(self, name.replace("-", "_"), joblib.load(path))

        meta_path = self.models_dir / "meta_learner.pkl"
        if meta_path.exists():
            self.meta_learner = joblib.load(meta_path)

        cal_path = self.models_dir / "calibrated_meta.pkl"
        if cal_path.exists():
            self.calibrated_meta = joblib.load(cal_path)

        mapie_path = self.models_dir / "mapie_classifier.pkl"
        if mapie_path.exists():
            self.mapie_classifier = joblib.load(mapie_path)

        thresh_path = self.models_dir / "decision_threshold.pkl"
        if thresh_path.exists():
            self.decision_threshold = float(joblib.load(thresh_path))

    def _base_probs(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from all base models."""
        probs = []
        for _, model in self.base_models:
            p = model.predict_proba(X)
            probs.append(p[:, 1] if p.shape[1] > 1 else p[:, 0])
        return np.column_stack(probs)

    def predict(self, X: np.ndarray) -> dict[str, Any]:
        """Full prediction with calibrated probabilities and uncertainty."""
        base_probs = self._base_probs(X)

        # Calibrated meta-learner prediction
        if self.calibrated_meta is not None:
            meta_prob = self.calibrated_meta.predict_proba(base_probs)[:, 1]
        else:
            meta_prob = self.meta_learner.predict_proba(base_probs)[:, 1]

        prob = float(np.clip(meta_prob[0], 0.0, 1.0))
        pred = int(prob >= self.decision_threshold)

        # Conformal prediction interval
        conformal = {}
        if self.mapie_classifier is not None:
            try:
                _, pred_sets = self.mapie_classifier.predict(
                    base_probs[:1], alpha=[0.05, 0.10, 0.20]
                )
                conformal = {
                    "alpha_0.05": pred_sets[0].tolist(),
                    "alpha_0.10": pred_sets[0].tolist() if pred_sets.shape[0] < 2 else pred_sets[1].tolist(),
                    "alpha_0.20": pred_sets[-1].tolist(),
                }
            except Exception as e:
                logger.warning("MAPIE prediction failed: %s", e)

        # SHAP values
        shap_vals = self.get_shap_values(X[:1])[0] if X.shape[0] > 0 else []
        if len(shap_vals) > 0:
            top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
            top_risk_factors = [self.feature_names[i] for i in top_idx]
        else:
            top_risk_factors = []

        # Risk level
        if prob >= 0.8:
            risk_level = "Critical"
        elif prob >= 0.65:
            risk_level = "High"
        elif prob >= 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        confidence = "High" if abs(prob - 0.5) >= 0.3 else ("Medium" if abs(prob - 0.5) >= 0.15 else "Low")

        # Per-model breakdown
        per_model = {}
        for i, (name, _) in enumerate(self.base_models):
            per_model[name] = round(float(base_probs[0, i]), 4)

        return {
            "prediction": pred,
            "probability": round(prob, 4),
            "confidence": confidence,
            "risk_level": risk_level,
            "individual_model_probs": per_model,
            "top_risk_factors": top_risk_factors,
            "conformal_prediction": conformal,
            "calibrated": self.calibrated_meta is not None,
            "decision_threshold": round(self.decision_threshold, 4),
        }

    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Get SHAP explanations from the primary tree model."""
        if shap is None:
            return np.zeros((X.shape[0], X.shape[1]), dtype=float)

        if self.tree_explainer is None:
            self.tree_explainer = shap.TreeExplainer(self.rf)

        shap_values = self.tree_explainer.shap_values(X)
        if isinstance(shap_values, list):
            return np.asarray(shap_values[1] if len(shap_values) == 2 else shap_values[0])

        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            return arr[:, :, 1] if arr.shape[-1] == 2 else arr.mean(axis=-1)
        return arr

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Evaluate ensemble on test set."""
        base_probs = self._base_probs(X_test)
        if self.calibrated_meta is not None:
            probs = self.calibrated_meta.predict_proba(base_probs)[:, 1]
        else:
            probs = self.meta_learner.predict_proba(base_probs)[:, 1]

        y_pred = (probs >= self.decision_threshold).astype(int)

        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "brier_score": round(float(brier_score_loss(y_test, probs)), 4),
            "decision_threshold": round(float(self.decision_threshold), 4),
        }
