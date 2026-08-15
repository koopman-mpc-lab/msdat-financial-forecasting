#!/usr/bin/env python
"""Evaluate a saved checkpoint on a cached feature split."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_default
from src.data.dataset import WindowDataset, collate_windows
from src.models import build_model, count_params, format_param_line
from src.training.metrics import evaluate_arrays, summarize


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("eval")


def invert_scale(y_norm, p_last, y_raw, p_last_raw, y_pred):
    scale = np.divide(y_raw - p_last_raw + 1e-8, y_norm - p_last + 1e-8)
    return p_last_raw + (y_pred - p_last) * np.abs(scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    log = setup_logger()
    cfg = load_default()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = payload.get("model_name", "MSDAT")
    dataset = args.dataset or payload.get("dataset", "SPX")
    extra = payload.get("extra", {})
    model = build_model(model_name, payload.get("cfg", cfg), extra)
    model.load_state_dict(payload["model"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    stats = payload.get("param_stats") or count_params(model)
    log.info("device=%s dataset=%s split=%s", device, dataset, args.split)
    log.info("ckpt=%s", ckpt_path)
    log.info("%s", format_param_line(stats))

    data_path = ROOT / "data" / "features" / dataset / f"{args.split}.npz"
    dataset_obj = WindowDataset(data_path)
    loader = DataLoader(
        dataset_obj,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_windows,
    )
    n = len(dataset_obj)
    n_batches = len(loader)
    log.info("n_windows=%d n_batches=%d", n, n_batches)

    store = {k: [] for k in ("y_true", "y_pred", "q10", "q90", "p_last", "y_true_raw", "p_last_raw")}
    ids, dates, symbols = [], [], []
    t_start = time.perf_counter()
    done = 0
    for i, batch in enumerate(loader, start=1):
        batch_t = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch_t["x"], scales=batch_t.get("scales"), vol=batch_t.get("vol"))
        store["y_true"].append(batch_t["y"].cpu().numpy())
        store["y_pred"].append(out["point"].cpu().numpy())
        store["q10"].append(out["q10"].cpu().numpy())
        store["q90"].append(out["q90"].cpu().numpy())
        store["p_last"].append(batch_t["p_last"].cpu().numpy())
        store["y_true_raw"].append(batch_t["y_raw"].cpu().numpy())
        store["p_last_raw"].append(batch_t["p_last_raw"].cpu().numpy())
        ids.extend(batch["id"])
        dates.extend(batch["date"])
        symbols.extend(batch["symbol"])
        done += len(batch["id"])
        log.info("batch %d/%d windows=%d/%d", i, n_batches, done, n)

    arrays = {k: np.concatenate(v, axis=0) for k, v in store.items()}
    y_pred_raw = invert_scale(
        arrays["y_true"], arrays["p_last"], arrays["y_true_raw"],
        arrays["p_last_raw"], arrays["y_pred"],
    )
    metrics = evaluate_arrays(
        arrays["y_true"], arrays["y_pred"], arrays["p_last"],
        arrays["y_true_raw"], y_pred_raw, arrays["q10"], arrays["q90"],
    )
    log.info("done in %.1fs", time.perf_counter() - t_start)
    log.info("%s", summarize(metrics))

    out_path = Path(args.out) if args.out else ROOT / "results" / "predictions" / f"{dataset}_{model_name}_{args.split}_eval.csv"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "id": ids,
        "date": dates,
        "symbol": symbols,
        "y_true": arrays["y_true"],
        "y_pred": arrays["y_pred"],
        "q10": arrays["q10"],
        "q90": arrays["q90"],
        "y_true_raw": arrays["y_true_raw"],
        "y_pred_raw": y_pred_raw,
        "p_last": arrays["p_last"],
        "p_last_raw": arrays["p_last_raw"],
    }).to_csv(out_path, index=False)
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
