from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.msdat import count_params, format_param_line
from .loss import CompositeLoss
from .metrics import evaluate_arrays


class Trainer:
    def __init__(self, model, cfg: dict, device: str | None = None):
        self.model = model
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(model, "to"):
            self.model.to(self.device)
        loss_cfg = cfg.get("loss", {})
        self.criterion = CompositeLoss(
            lambda_quantile=loss_cfg.get("lambda_quantile", 0.3),
            lambda_dir=loss_cfg.get("lambda_dir", 0.5),
            lambda_vol=loss_cfg.get("lambda_vol", 0.1),
            use_dir_loss=loss_cfg.get("use_dir_loss", True),
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=cfg.get("lr_factor", 0.5),
            patience=cfg.get("lr_patience", 5),
        )

    def param_line(self) -> str:
        return format_param_line(count_params(self.model))

    def _step(self, batch: dict, train: bool = True) -> dict[str, float]:
        batch_t = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        out = self.model(batch_t["x"], scales=batch_t.get("scales"), vol=batch_t.get("vol"))
        losses = self.criterion(out, batch_t)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.get("grad_clip", 1.0))
            self.optimizer.step()
        return {k: float(v.detach().cpu()) for k, v in losses.items()}

    @torch.no_grad()
    def predict_loader(self, loader: DataLoader) -> dict[str, np.ndarray]:
        self.model.eval()
        store = {k: [] for k in ("y_true", "y_pred", "q10", "q90", "p_last", "y_true_raw", "p_last_raw", "dir_logit")}
        ids, dates, symbols = [], [], []
        for batch in loader:
            batch_t = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = self.model(batch_t["x"], scales=batch_t.get("scales"), vol=batch_t.get("vol"))
            store["y_true"].append(batch_t["y"].cpu().numpy())
            store["y_pred"].append(out["point"].cpu().numpy())
            store["q10"].append(out["q10"].cpu().numpy())
            store["q90"].append(out["q90"].cpu().numpy())
            store["p_last"].append(batch_t["p_last"].cpu().numpy())
            store["y_true_raw"].append(batch_t["y_raw"].cpu().numpy())
            store["p_last_raw"].append(batch_t["p_last_raw"].cpu().numpy())
            store["dir_logit"].append(out["dir_logit"].cpu().numpy())
            ids.extend(batch["id"])
            dates.extend(batch["date"])
            symbols.extend(batch["symbol"])
        arrays = {k: np.concatenate(v, axis=0) for k, v in store.items()}
        arrays["id"] = np.array(ids)
        arrays["date"] = np.array(dates)
        arrays["symbol"] = np.array(symbols)
        return arrays

    def evaluate_loader(self, loader: DataLoader) -> dict[str, float]:
        pred = self.predict_loader(loader)
        scale = np.divide(
            pred["y_true_raw"] - pred["p_last_raw"] + 1e-8,
            pred["y_true"] - pred["p_last"] + 1e-8,
        )
        y_pred_raw = pred["p_last_raw"] + (pred["y_pred"] - pred["p_last"]) * np.abs(scale)
        return evaluate_arrays(
            pred["y_true"], pred["y_pred"], pred["p_last"],
            pred["y_true_raw"], y_pred_raw, pred["q10"], pred["q90"],
        )

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg,
            "param_stats": count_params(self.model),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
