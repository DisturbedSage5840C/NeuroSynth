from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPipeline:
    """Load and preprocess OASIS longitudinal data for NeuroSynth."""

    feature_columns = ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
    target_column = "Group"

    def __init__(self, csv_path: str | None = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else self._resolve_csv_path()
        self.models_dir = self._resolve_models_dir()
        self.df_clean: pd.DataFrame | None = None
        self.subject_labels: Dict[str, int] = {}
        self.scaler: StandardScaler | None = None

    def _resolve_csv_path(self) -> Path:
        candidates = [
            Path.cwd() / "oasis_longitudinal.csv",
            Path(__file__).resolve().parent.parent / "oasis_longitudinal.csv",
            Path("/app/oasis_longitudinal.csv"),
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError("oasis_longitudinal.csv not found in expected locations")

    def _resolve_models_dir(self) -> Path:
        candidates = [Path.cwd() / "models", Path(__file__).resolve().parent.parent / "models", Path("/app/models")]
        for path in candidates:
            if path.exists() or path.parent.exists():
                path.mkdir(parents=True, exist_ok=True)
                return path
        fallback = Path.cwd() / "models"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[df[self.target_column] != "Converted"]
        df["SES"] = df["SES"].fillna(df["SES"].median())
        df["MMSE"] = df["MMSE"].fillna(df["MMSE"].median())
        df["nWBV"] = df["nWBV"].fillna(df["nWBV"].mean())
        df = df.dropna(subset=self.feature_columns + [self.target_column, "Subject ID", "Visit"])
        df["target"] = df[self.target_column].map({"Demented": 1, "Nondemented": 0}).astype(int)
        return df

    def _build_patient_sequences(self, df: pd.DataFrame) -> Dict[str, List[List[float]]]:
        sequences: Dict[str, List[List[float]]] = {}
        for subject_id, subject_df in df.groupby("Subject ID"):
            sorted_df = subject_df.sort_values("Visit")
            vectors = sorted_df[self.feature_columns].astype(float).values.tolist()
            sequences[str(subject_id)] = vectors
            self.subject_labels[str(subject_id)] = int(sorted_df["target"].iloc[-1])
        return sequences

    def process(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler, Dict[str, List[List[float]]]]:
        raw_df = pd.read_csv(self.csv_path)
        df = self._clean_dataframe(raw_df)
        self.df_clean = df

        X = df[self.feature_columns].astype(float).values
        y = df["target"].values

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        self.scaler = scaler

        patient_sequences = self._build_patient_sequences(df)

        scaler_path = self.models_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)

        return X_train, X_test, y_train, y_test, self.feature_columns, scaler, patient_sequences
