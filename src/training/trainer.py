import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.training.loss import CompositeLoss
from src.training.metrics import compute_metrics


class Trainer:
    def __init__(self, model, cfg, device=None):
        self.model = model
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, factor=cfg["lr_decay"], patience=cfg["lr_patience"]
        )
        self.criterion = CompositeLoss(cfg["lambda_quantile"], cfg["lambda_direction"], cfg["lambda_vol"])
        self.best_val = float("inf")
        self.patience_counter = 0
        self.recent_preds = []

    def _forward(self, batch):
        x = batch["x"].to(self.device)
        vol = batch["vol"].to(self.device)
        close = batch.get("close_window")
        if close is not None:
            close = close.to(self.device)
        kwargs = {"vol": vol}
        if hasattr(self.model, "forward") and "close_window" in self.model.forward.__code__.co_varnames:
            kwargs["close_window"] = close
        return self.model(x, **kwargs)

    def train_epoch(self, loader):
        self.model.train()
        total = 0.0
        for batch in loader:
            self.optimizer.zero_grad()
            out = self._forward(batch)
            loss = self.criterion(out, batch, self.recent_preds)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            self.recent_preds.append(out[0].detach().mean())
            if len(self.recent_preds) > 20:
                self.recent_preds.pop(0)
        return total / max(len(loader), 1)

    @torch.no_grad()
    def evaluate(self, loader, return_preds=False):
        self.model.eval()
        preds, targets, raw_t, raw_p, p_t_list = [], [], [], [], []
        total = 0.0
        for batch in loader:
            out = self._forward(batch)
            loss = self.criterion(out, batch)
            total += loss.item()
            preds.append(out[0].cpu().numpy())
            targets.append(batch["y"].numpy())
            raw_t.append(batch["y_raw"].numpy())
            raw_p.append(out[0].cpu().numpy())
            p_t_list.append(batch["p_t"].numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        metrics = compute_metrics(targets, preds, np.concatenate(raw_t), np.concatenate(raw_p), np.concatenate(p_t_list))
        metrics["loss"] = total / max(len(loader), 1)
        if return_preds:
            return metrics, preds, targets, np.concatenate(raw_t), np.concatenate(p_t_list)
        return metrics

    def fit(self, train_loader, val_loader, save_path=None):
        history = []
        for epoch in range(self.cfg["max_epochs"]):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step(val_metrics["loss"])
            history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
            if val_metrics["loss"] < self.best_val:
                self.best_val = val_metrics["loss"]
                self.patience_counter = 0
                if save_path:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg["early_stop_patience"]:
                    break
        return history

    def benchmark_inference(self, loader, n_warmup=10):
        self.model.eval()
        latencies = []
        count = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                vol = batch["vol"].to(self.device)
                if count < n_warmup:
                    if hasattr(self.model, "forward"):
                        kw = {"vol": vol}
                        if "close_window" in self.model.forward.__code__.co_varnames:
                            kw["close_window"] = batch.get("close_window", x[..., 3]).to(self.device)
                        self.model(x, **kw)
                    count += 1
                    continue
                t0 = time.perf_counter()
                kw = {"vol": vol}
                if "close_window" in self.model.forward.__code__.co_varnames:
                    kw["close_window"] = batch.get("close_window", x[..., 3]).to(self.device)
                self.model(x, **kw)
                latencies.append((time.perf_counter() - t0) / x.size(0) * 1000)
        return float(np.mean(latencies)) if latencies else 0.0

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
