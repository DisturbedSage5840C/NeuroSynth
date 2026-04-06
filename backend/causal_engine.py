from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn


class _VariableMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralCausalDiscovery(nn.Module):
    variables = ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]

    def __init__(self, models_dir: str | Path = "models") -> None:
        super().__init__()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.W_logits = nn.Parameter(torch.zeros(8, 8))
        self.mlps = nn.ModuleList([_VariableMLP() for _ in range(8)])
        self._latest_W: np.ndarray | None = None

    def get_adjacency(self) -> torch.Tensor:
        W = torch.sigmoid(self.W_logits)
        W = W * (1 - torch.eye(8, device=W.device))
        return W

    def acyclicity_constraint(self, W: torch.Tensor) -> torch.Tensor:
        return torch.trace(torch.matrix_exp(W * W)) - 8

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        W = self.get_adjacency()
        preds = []
        for j in range(8):
            weighted_input = X * W[:, j]
            preds.append(self.mlps[j](weighted_input))
        X_hat = torch.cat(preds, dim=1)
        return X_hat, W

    def fit(
        self,
        X_data: np.ndarray,
        epochs: int = 500,
        lr: float = 0.01,
        lambda1: float = 0.01,
        lambda2: float = 5.0,
    ) -> None:
        X = torch.tensor(X_data, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        outer_iters = 10
        inner_iters = max(1, epochs // outer_iters)
        alpha = torch.tensor(0.0)

        for _ in range(outer_iters):
            for _ in range(inner_iters):
                optimizer.zero_grad()
                X_hat, W = self.forward(X)
                recon_loss = ((X_hat - X) ** 2).mean()
                sparsity = lambda1 * torch.sum(torch.abs(W))
                h = self.acyclicity_constraint(W)
                loss = recon_loss + sparsity + alpha * h + 0.5 * lambda2 * (h ** 2)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                h_val = self.acyclicity_constraint(self.get_adjacency())
                alpha = alpha + lambda2 * h_val

        W_np = self.get_adjacency().detach().cpu().numpy()
        self._latest_W = W_np
        np.save(self.models_dir / "causal_graph.npy", W_np)

    def get_causal_graph(self) -> Dict[str, object]:
        if self._latest_W is None:
            saved = self.models_dir / "causal_graph.npy"
            if saved.exists():
                self._latest_W = np.load(saved)
            else:
                self._latest_W = np.zeros((8, 8), dtype=float)

        W = self._latest_W
        edges = []
        for i, src in enumerate(self.variables):
            for j, dst in enumerate(self.variables):
                strength = float(W[i, j])
                if i != j and strength > 0.3:
                    edges.append({"from": src, "to": dst, "strength": round(strength, 4)})

        cdr_idx = self.variables.index("CDR")
        mmse_idx = self.variables.index("MMSE")

        def _top_causes(target_idx: int) -> List[Dict[str, float]]:
            causes = []
            for i, name in enumerate(self.variables):
                if i == target_idx:
                    continue
                causes.append({"variable": name, "strength": float(W[i, target_idx])})
            causes.sort(key=lambda x: x["strength"], reverse=True)
            return [{"variable": c["variable"], "strength": round(c["strength"], 4)} for c in causes[:3]]

        top_cdr = _top_causes(cdr_idx)
        top_mmse = _top_causes(mmse_idx)

        modifiable_set = {"MMSE", "SES", "EDUC"}
        modifiable_interventions = [
            item["variable"] for item in top_cdr if item["variable"] in modifiable_set and item["strength"] > 0
        ]

        return {
            "edges": sorted(edges, key=lambda x: x["strength"], reverse=True),
            "top_causes_of_CDR": top_cdr,
            "top_causes_of_MMSE": top_mmse,
            "modifiable_interventions": modifiable_interventions,
        }

    def simulate_intervention(
        self,
        variable: str,
        new_value: float,
        current_patient_data: Dict[str, float],
    ) -> Dict[str, object]:
        graph = self.get_causal_graph()
        W = self._latest_W if self._latest_W is not None else np.zeros((8, 8))

        var_to_payload = {
            "Age": "age",
            "EDUC": "educ",
            "SES": "ses",
            "MMSE": "mmse",
            "CDR": "cdr",
            "eTIV": "etiv",
            "nWBV": "nwbv",
            "ASF": "asf",
        }

        x = np.array([float(current_patient_data[var_to_payload[v]]) for v in self.variables], dtype=float)
        cdr_idx = self.variables.index("CDR")

        incoming = W[:, cdr_idx]
        original_score = (x[cdr_idx] / 3.0) + float(np.dot(incoming, x) / (len(x) * 2.0))
        original_risk = float(np.clip(1.0 / (1.0 + np.exp(-original_score)), 0.0, 1.0))

        if variable not in self.variables:
            raise ValueError(f"Unknown intervention variable: {variable}")

        x_intervened = x.copy()
        x_intervened[self.variables.index(variable)] = float(new_value)
        intervened_score = (x_intervened[cdr_idx] / 3.0) + float(np.dot(incoming, x_intervened) / (len(x_intervened) * 2.0))
        intervened_risk = float(np.clip(1.0 / (1.0 + np.exp(-intervened_score)), 0.0, 1.0))

        improvement = original_risk - intervened_risk
        direction = "improves" if improvement > 0 else "worsens"
        interpretation = (
            f"Adjusting {variable} to {new_value:.2f} {direction} estimated CDR-linked risk by {abs(improvement):.3f}. "
            f"This estimate is derived from learned causal strengths and should be interpreted as directional evidence."
        )

        return {
            "original_CDR_risk": round(original_risk, 4),
            "intervened_CDR_risk": round(intervened_risk, 4),
            "estimated_improvement": round(improvement, 4),
            "interpretation": interpretation,
        }
