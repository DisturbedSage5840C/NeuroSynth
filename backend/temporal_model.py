from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
from torch import nn


class TemporalLSTM(nn.Module):
    def __init__(self, input_size: int = 8, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed_out, _ = self.lstm(x)
        idx = (lengths - 1).clamp(min=0)
        last_hidden = packed_out[torch.arange(x.shape[0]), idx]
        return self.head(last_hidden)


class TemporalProgressionModel:
    def __init__(self, models_dir: str | Path = "models", fallback_predictor: Callable[[np.ndarray], float] | None = None) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model = TemporalLSTM()
        self.fallback_predictor = fallback_predictor
        self.progression_slope = 0.03
        self.use_fallback = False

    def _pad_sequences(self, sequences: List[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        max_len = int(lengths.max().item())
        n_features = sequences[0].shape[1]
        padded = torch.zeros((len(sequences), max_len, n_features), dtype=torch.float32)
        for i, seq in enumerate(sequences):
            padded[i, : len(seq), :] = torch.tensor(seq, dtype=torch.float32)
        return padded, lengths

    def train_model(
        self,
        patient_sequences: Dict[str, List[List[float]]],
        labels: Dict[str, int],
        epochs: int = 50,
        lr: float = 0.001,
    ) -> None:
        usable_subjects = [sid for sid, seq in patient_sequences.items() if len(seq) > 0 and sid in labels]
        if len(usable_subjects) < 2:
            self.use_fallback = True
            return

        sequences = [np.asarray(patient_sequences[sid], dtype=float) for sid in usable_subjects]
        y = np.asarray([labels[sid] for sid in usable_subjects], dtype=np.float32).reshape(-1, 1)

        padded, lengths = self._pad_sequences(sequences)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.model(padded, lengths)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()

        trends = []
        for sid in usable_subjects:
            seq = np.asarray(patient_sequences[sid], dtype=float)
            if len(seq) > 1:
                cdr_first = float(seq[0, 4])
                cdr_last = float(seq[-1, 4])
                trends.append((cdr_last - cdr_first) / max(1, len(seq) - 1))
        if trends:
            self.progression_slope = float(np.clip(np.mean(trends) * 0.15 + 0.02, 0.005, 0.09))

        torch.save(self.model.state_dict(), self.models_dir / "lstm_model.pt")

    def _fallback_probability(self, visit_sequence: List[List[float]] | np.ndarray) -> float:
        arr = np.asarray(visit_sequence, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        latest = arr[-1]

        if self.fallback_predictor is not None:
            try:
                return float(np.clip(self.fallback_predictor(latest), 0.0, 1.0))
            except Exception:
                pass

        cdr = float(latest[4])
        mmse = float(latest[3])
        heuristic = (cdr / 3.0) * 0.65 + (1.0 - min(mmse, 30.0) / 30.0) * 0.35
        return float(np.clip(heuristic, 0.0, 1.0))

    def predict_trajectory(self, visit_sequence: List[List[float]] | np.ndarray) -> List[float]:
        months = [6, 12, 18, 24, 30, 36]

        if self.use_fallback:
            base = self._fallback_probability(visit_sequence)
            return [round(float(np.clip(base + i * 0.04, 0.0, 1.0)), 4) for i, _ in enumerate(months)]

        arr = np.asarray(visit_sequence, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            lengths = torch.tensor([arr.shape[0]], dtype=torch.long)
            base_prob = float(self.model(x, lengths).item())

        trajectory = []
        for idx, _ in enumerate(months, start=1):
            projected = np.clip(base_prob + self.progression_slope * idx, 0.0, 1.0)
            trajectory.append(round(float(projected), 4))
        return trajectory
