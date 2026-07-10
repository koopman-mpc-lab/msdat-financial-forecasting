import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data import load_splits, make_dataloader
from src.models.factory import build_model
from src.training.trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", required=True, choices=["spx", "csi300", "nasdaq20", "btc_usd"])
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.horizon:
        cfg["horizon"] = args.horizon
    set_seed(args.seed)
    train_ds, val_ds, test_ds, _ = load_splits(args.dataset, cfg)
    train_loader = make_dataloader(train_ds, cfg["batch_size"], shuffle=True)
    val_loader = make_dataloader(val_ds, cfg["batch_size"])
    test_loader = make_dataloader(test_ds, cfg["batch_size"])
    out_dir = Path(cfg["output_dir"]) / args.dataset / args.model / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "best.pt"
    if args.model == "arima":
        from src.models.baselines.arima import fit_arima_batch
        from src.training.metrics import compute_metrics
        close_windows = [test_ds[i]["close_window"].numpy() for i in range(len(test_ds))]
        preds = fit_arima_batch(close_windows)
        targets = np.array([test_ds[i]["y"].item() for i in range(len(test_ds))])
        raw_t = np.array([test_ds[i]["y_raw"].item() for i in range(len(test_ds))])
        p_t = np.array([test_ds[i]["p_t"].item() for i in range(len(test_ds))])
        metrics = compute_metrics(targets, preds, raw_t, preds, p_t)
    else:
        ablation = cfg.pop("ablation", {})
        cfg.update(ablation)
        model = build_model(args.model, cfg)
        trainer = Trainer(model, cfg)
        trainer.fit(train_loader, val_loader, save_path)
        model.load_state_dict(torch.load(save_path, map_location=trainer.device))
        metrics, _, _, _, _ = trainer.evaluate(test_loader, return_preds=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
