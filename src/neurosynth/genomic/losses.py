from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMultiTaskLoss(nn.Module):
    def __init__(self, w_clinical: float = 1.0, w_prs: float = 0.3, w_apoe: float = 0.5, w_dirichlet_kl: float = 0.1) -> None:
        super().__init__()
        self.w_clinical = w_clinical
        self.w_prs = w_prs
        self.w_apoe = w_apoe
        self.w_dirichlet_kl = w_dirichlet_kl

    def _dirichlet_kl(self, alpha: torch.Tensor) -> torch.Tensor:
        k = alpha.shape[-1]
        prior = torch.ones_like(alpha)
        sum_alpha = alpha.sum(dim=-1, keepdim=True)
        sum_prior = prior.sum(dim=-1, keepdim=True)
        term1 = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
        term2 = torch.lgamma(prior).sum(dim=-1, keepdim=True) - torch.lgamma(sum_prior)
        term3 = ((alpha - prior) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=-1, keepdim=True)
        _ = k
        return (term1 + term2 + term3).mean()

    def forward(self, outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        clinical = F.cross_entropy(outputs["diagnosis_logits"], labels["diagnosis_class"])
        prs = F.mse_loss(outputs["prs_pred"], labels["prs"])
        apoe = F.cross_entropy(outputs["apoe_logits"], labels["apoe_count"])
        kl = self._dirichlet_kl(outputs["dirichlet_alpha"])

        total = self.w_clinical * clinical + self.w_prs * prs + self.w_apoe * apoe + self.w_dirichlet_kl * kl
        return total, {
            "clinical_loss": float(clinical.detach()),
            "prs_loss": float(prs.detach()),
            "apoe_loss": float(apoe.detach()),
            "dirichlet_kl": float(kl.detach()),
        }
