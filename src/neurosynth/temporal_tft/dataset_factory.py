from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from torch.utils.data import WeightedRandomSampler


@dataclass
class DatasetFactory:
    min_encoder_length: int = 4
    max_encoder_length: int = 8
    min_prediction_length: int = 1
    max_prediction_length: int = 6

    def _base_dataset(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="dci",
            group_ids=["patient_id"],
            min_encoder_length=self.min_encoder_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=self.min_prediction_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["sex", "apoe_e4_cat", "disease_subtype", "cohort", "site_region"],
            static_reals=["age_at_enrollment", "education_years", "prs_ad_normalized", "prs_pd_normalized", "apoe_e4_count"],
            time_varying_known_categoricals=["medication_class", "season"],
            time_varying_known_reals=["age_at_visit", "visit_number", "months_since_diagnosis", "total_drug_burden_score"],
            time_varying_unknown_reals=[
                "csf_abeta42", "csf_ptau181", "csf_total_tau", "csf_ratio", "hippocampal_volume", "entorhinal_volume",
                "fusiform_volume", "ventricle_volume", "whole_brain_volume", "atrophy_asymmetry", "cdrsb", "mmse", "moca",
                "adas13", "nfl_plasma", "alpha_syn_csf", "gait_speed", "tremor_index", "bradykinesia_score", "step_count_daily",
                "sleep_efficiency", "delta_hippocampus", "delta_nfl", "delta_cdrsb", "accel_hippocampus", "accel_nfl",
            ],
            target_normalizer=GroupNormalizer(groups=["patient_id", "disease_subtype"], transformation="softplus"),
            categorical_encoders={
                "sex": NaNLabelEncoder(add_nan=True),
                "apoe_e4_cat": NaNLabelEncoder(add_nan=True),
                "disease_subtype": NaNLabelEncoder(add_nan=True),
                "medication_class": NaNLabelEncoder(add_nan=True),
            },
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

    def create_datasets(self, df: pd.DataFrame, train_cutoff_date: str, val_cutoff_date: str):
        d = df.copy()
        d["visit_date"] = pd.to_datetime(d["visit_date"])
        train_cutoff = pd.to_datetime(train_cutoff_date)
        val_cutoff = pd.to_datetime(val_cutoff_date)

        train_df = d[d["visit_date"] <= train_cutoff].copy()
        val_df = d[(d["visit_date"] > train_cutoff) & (d["visit_date"] <= val_cutoff)].copy()
        test_df = d[d["visit_date"] > val_cutoff].copy()

        training = self._base_dataset(train_df)
        validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
        test = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
        return training, validation, test

    def create_weighted_sampler(self, dataset: TimeSeriesDataSet) -> WeightedRandomSampler:
        target = dataset.data["target"].detach().cpu().numpy().reshape(-1)
        bins = pd.qcut(target, q=4, labels=False, duplicates="drop")
        freq = pd.Series(bins).value_counts().to_dict()
        weights = np.array([1.0 / freq.get(int(b), 1.0) for b in bins], dtype=np.float64)
        return WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
