#!/usr/bin/env python
"""Train a configured model on a cached feature split."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_default, load_yaml
from src.data.dataset import WindowDataset, collate_windows
from src.models import build_model
from src.training.metrics import summarize
from src.training.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_ablation(cfg: dict, name: str | None) -> dict:
    extra = {}
    if not name:
        return extra
    variants = load_yaml("configs/ablation.yaml")["variants"]
    if name not in variants:
        raise SystemExit(f"unknown ablation {name}; choose from {sorted(variants)}")
    extra = dict(variants[name])
    if "n_scales" in extra:
        cfg["n_scales"] = extra["n_scales"]
    cfg.setdefault("loss", {})
    if "use_dir_loss" in extra:
        cfg["loss"]["use_dir_loss"] = extra["use_dir_loss"]
    return extra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MSDAT")
    parser.add_argument("--dataset", default="SPX")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--n-scales", type=int, default=None)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_default()
    if args.horizon is not None:
        cfg["horizon"] = args.horizon
    if args.lookback is not None:
        cfg["lookback"] = args.lookback
    if args.n_scales is not None:
        cfg["n_scales"] = args.n_scales
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    extra = apply_ablation(cfg, args.ablation)
    if args.n_scales is not None:
        extra["n_scales"] = args.n_scales

    set_seed(args.seed)
    tag = args.model.lower().replace("-", "")
    if args.ablation:
        tag = f"ablation_{args.dataset.lower()}_{args.ablation}"
    else:
        tag = f"{tag}_{args.dataset.lower()}_s{args.seed}"
    if args.horizon and args.horizon != 1:
        tag = f"{tag}_h{args.horizon}"
    if args.lookback and args.lookback != cfg.get("lookback", args.lookback):
        tag = f"{tag}_L{args.lookback}"
    ckpt_dir = ROOT / "checkpoints" / tag

    train_ds = WindowDataset(ROOT / "data" / "features" / args.dataset / "train.npz")
    val_ds = WindowDataset(ROOT / "data" / "features" / args.dataset / "val.npz")

    if args.model.upper() in {"ARIMA", "ARIMA(1,1,1)"}:
        from src.models.baselines.arima import ARIMABaseline
        from src.training.metrics import evaluate_arrays

        model = ARIMABaseline()
        pred = model.predict_batch(val_ds.x)
        metrics = evaluate_arrays(val_ds.y, pred, val_ds.p_last, val_ds.y_raw, pred, None, None)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with (ckpt_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(summarize(metrics))
        return

    model = build_model(args.model, cfg, extra)
    trainer = Trainer(model, cfg)
    print(trainer.param_line())

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_windows)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_windows)

    best = 1e9
    bad = 0
    history = []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        losses = []
        for batch in train_loader:
            losses.append(trainer._step(batch, train=True)["loss"])
        val = trainer.evaluate_loader(val_loader)
        trainer.scheduler.step(val["rmse"])
        row = {"epoch": epoch, "train_loss": float(sum(losses) / len(losses)), **val}
        history.append(row)
        print(f"epoch {epoch:03d} train_loss={row['train_loss']:.4f} {summarize(val)}")
        extra_ckpt = {
            "epoch": epoch,
            "best_val_rmse": min(best, val["rmse"]),
            "dataset": args.dataset,
            "model_name": args.model,
            "extra": extra,
        }
        if val["rmse"] < best:
            best = val["rmse"]
            bad = 0
            trainer.save(ckpt_dir / "best.pt", extra=extra_ckpt)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print("early stop")
                trainer.save(ckpt_dir / "last.pt", extra=extra_ckpt)
                break
        trainer.save(ckpt_dir / "last.pt", extra=extra_ckpt)

    metrics = {"best_val_rmse": best, "history": history}
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
