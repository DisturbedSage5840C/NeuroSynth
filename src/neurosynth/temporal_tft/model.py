from __future__ import annotations

import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


class NeuroTFT:
    def __init__(self, model: TemporalFusionTransformer) -> None:
        self.model = model
        self.quantiles = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

    @classmethod
    def from_dataset(cls, training_dataset):
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=4.2e-4,
            hidden_size=192,
            attention_head_size=6,
            dropout=0.18,
            hidden_continuous_size=80,
            output_size=7,
            loss=QuantileLoss(quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]),
            log_interval=10,
            log_val_interval=5,
            reduce_on_plateau_patience=8,
            monotone_constaints={"delta_hippocampus": -1, "nfl_plasma": 1},
        )
        return cls(tft)

    def _enforce_progressive(self, median: np.ndarray) -> np.ndarray:
        return np.maximum.accumulate(median, axis=-1)

    def predict_with_uncertainty(self, patient_df: pd.DataFrame) -> dict:
        raw, x = self.model.predict(patient_df, mode="raw", return_x=True)
        pred = raw["prediction"].detach().cpu().numpy()
        # shape expected: [B, decoder_length, quantiles]
        q_map = {q: i for i, q in enumerate(self.quantiles)}

        median = pred[0, :, q_map[0.5]]
        p10 = pred[0, :, q_map[0.1]]
        p90 = pred[0, :, q_map[0.9]]
        p05 = pred[0, :, q_map[0.05]]
        p95 = pred[0, :, q_map[0.95]]

        median = self._enforce_progressive(median)

        var_imp = self.model.interpret_output(raw, reduction="mean").get("encoder_variables", None)
        if var_imp is None:
            variable_importances = pd.DataFrame(columns=["variable", "importance"])
        else:
            variable_importances = pd.DataFrame({"variable": list(var_imp.keys()), "importance": [float(np.mean(v)) for v in var_imp.values()]})

        att = raw.get("attention")
        enc_att = att[0].detach().cpu().numpy() if att is not None else np.zeros((8, 6), dtype=np.float32)

        thr = 60.0
        above = np.where(median > thr)[0]
        months_to_threshold = float((above[0] + 1) * 6) if len(above) > 0 else float("inf")

        slope = (median[-1] - median[0]) / max(len(median) - 1, 1)
        if slope < 1.0:
            rate = "slow"
        elif slope < 3.0:
            rate = "moderate"
        else:
            rate = "rapid"

        return {
            "median_forecast": median,
            "prediction_interval_80": np.stack([p10, p90], axis=-1),
            "prediction_interval_90": np.stack([p05, p95], axis=-1),
            "variable_importances": variable_importances,
            "encoder_attention": enc_att,
            "months_to_threshold": months_to_threshold,
            "progression_rate_category": rate,
        }
