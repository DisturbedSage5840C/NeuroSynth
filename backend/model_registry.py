from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json

import joblib
import numpy as np
import torch


class ModelRegistry:
    def __init__(self, models_dir: str | Path = "models") -> None:
        self.models_dir = Path(models_dir)

    def _load_feature_names(self, scaler: Any) -> list[str]:
        names = getattr(scaler, "feature_names_in_", None)
        if names is None:
            return []
        return [str(n) for n in names]

    def load_all(self) -> SimpleNamespace:
        from backend.biomarker_model import BiomarkerPredictor, MultiDiseasePredictor
        from backend.causal_engine import NeuralCausalDiscovery
        from backend.disease_classifier import DISEASES, DiseaseClassifier
        from backend.temporal_model import TemporalProgressionModel

        scaler = joblib.load(self.models_dir / "scaler.pkl")
        feature_names = self._load_feature_names(scaler)

        predictor = BiomarkerPredictor(feature_names)
        predictor.rf = joblib.load(self.models_dir / "rf_model.pkl")
        predictor.gb = joblib.load(self.models_dir / "gb_model.pkl")

        third_model = self.models_dir / "xgboost_model.pkl"
        if third_model.exists():
            predictor.third = joblib.load(third_model)
            predictor.third_name = "xgboost"
        else:
            predictor.third = joblib.load(self.models_dir / "extra_trees_model.pkl")
            predictor.third_name = "extra_trees"

        lr_model = self.models_dir / "lr_model.pkl"
        if lr_model.exists():
            predictor.lr = joblib.load(lr_model)

        temporal = TemporalProgressionModel(feature_names)
        lstm_state = torch.load(self.models_dir / "lstm_model.pt", map_location="cpu")
        temporal.model.load_state_dict(lstm_state)

        variables = None
        vars_file = self.models_dir / "causal_vars.json"
        if vars_file.exists():
            variables = json.loads(vars_file.read_text(encoding="utf-8"))

        causal_model = NeuralCausalDiscovery(variables=variables)
        causal_path = self.models_dir / "causal_graph.npy"
        if causal_path.exists():
            causal_model.latest_W = np.load(causal_path)

        disease_clf = DiseaseClassifier(models_dir=self.models_dir)
        disease_clf._lazy_load()

        multi_predictor = MultiDiseasePredictor(
            feature_names=feature_names,
            diseases=DISEASES,
            models_dir=self.models_dir / "multi",
        )
        try:
            multi_predictor.load_from_disk()
        except Exception:
            multi_predictor = None

        manifest = {}
        manifest_path = self.models_dir / "model_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        return SimpleNamespace(
            scaler=scaler,
            predictor=predictor,
            temporal=temporal,
            causal=causal_model,
            disease_classifier=disease_clf,
            multi_predictor=multi_predictor,
            feature_names=feature_names,
            manifest=manifest,
            dataset_stats=manifest.get("dataset_stats", {}),
        )
