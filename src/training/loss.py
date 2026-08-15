from __future__ import annotations

import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    def __init__(
        self,
        lambda_quantile: float = 0.3,
        lambda_dir: float = 0.5,
        lambda_vol: float = 0.1,
        use_dir_loss: bool = True,
    ):
        super().__init__()
        self.lambda_quantile = lambda_quantile
        self.lambda_dir = lambda_dir if use_dir_loss else 0.0
        self.lambda_vol = lambda_vol

    def forward(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        y = batch["y"]
        point = out["point"]
        point_loss = torch.mean((point - y) ** 2)
        q10 = torch.maximum(0.1 * (y - out["q10"]), (0.1 - 1.0) * (y - out["q10"]))
        q90 = torch.maximum(0.9 * (y - out["q90"]), (0.9 - 1.0) * (y - out["q90"]))
        quantile_loss = torch.mean(q10 + q90)
        direction = torch.sign(y - batch["p_last"])
        dir_loss = torch.mean(torch.clamp(-out["dir_logit"] * direction, min=0.0))
        pred_vol = torch.std(point, unbiased=False)
        real_vol = torch.std(y, unbiased=False)
        vol_loss = (pred_vol - real_vol) ** 2
        total = (
            point_loss
            + self.lambda_quantile * quantile_loss
            + self.lambda_dir * dir_loss
            + self.lambda_vol * vol_loss
        )
        return {
            "loss": total,
            "point": point_loss.detach(),
            "quantile": quantile_loss.detach(),
            "dir": dir_loss.detach(),
            "vol": vol_loss.detach(),
        }
