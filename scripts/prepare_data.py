#!/usr/bin/env python
"""Build lookback-window caches from local OHLCV CSVs.

Expected CSV columns: date,open,high,low,close,volume[,turnover].
Place files under data/raw/<dataset>/<symbol>.csv and run this script.

Min-max statistics are estimated from the training partition only.
CEEMDAN is applied independently to each lookback window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import load_datasets, load_default
from src.data.features import build_features
from src.decomposition.ceemdan import window_ceemdan


def _liquidity(df: pd.DataFrame, name: str | None) -> np.ndarray | None:
    if name and name in df.columns:
        return df[name].to_numpy(dtype=np.float64)
    return None


def windows_from_frame(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    n_train: int,
    n_val: int,
    n_test: int,
    liquidity_col: str | None,
    decomp_cfg: dict,
) -> dict[str, dict[str, np.ndarray]]:
    ohlcv = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    feats = build_features(ohlcv, _liquidity(df, liquidity_col))
    close = ohlcv[:, 3]
    n_days = len(df)
    if n_train + n_val + n_test > n_days:
        raise SystemExit(f"split {n_train, n_val, n_test} exceeds {n_days} rows")

    train_feats = feats[:n_train]
    lo = train_feats.min(axis=0)
    hi = train_feats.max(axis=0)
    span = np.clip(hi - lo, 1e-8, None)
    feats_n = (feats - lo) / span
    close_n = feats_n[:, 3]

    ensemble = int(decomp_cfg.get("ensemble", 50))
    tau_high = float(decomp_cfg.get("tau_high", 5))
    tau_mid = float(decomp_cfg.get("tau_mid", 20))
    refresh_every = int(decomp_cfg.get("refresh_every", 5))
    n_scales = int(decomp_cfg.get("n_scales", 3))

    bounds = {
        "train": n_train,
        "val": n_train + n_val,
        "test": n_train + n_val + n_test,
    }
    out: dict[str, dict[str, list]] = {
        name: {k: [] for k in ("x", "y", "y_raw", "p_last", "p_last_raw", "vol", "scales", "dates")}
        for name in bounds
    }
    cached_scales = None
    cached_t = None
    last_t = n_train + n_val + n_test - horizon
    for t in range(lookback - 1, last_t):
        target = t + horizon
        if target < bounds["train"]:
            split = "train"
        elif target < bounds["val"]:
            split = "val"
        else:
            split = "test"
        if cached_scales is None or cached_t is None or (t - cached_t) >= refresh_every:
            dec = window_ceemdan(
                close[t - lookback + 1:t + 1],
                ensemble=ensemble,
                tau_high=tau_high,
                tau_mid=tau_mid,
                n_scales=n_scales,
                seed=t,
            )
            raw_scales = dec["scales"]
            cmin, cmax = close[t - lookback + 1:t + 1].min(), close[t - lookback + 1:t + 1].max()
            cached_scales = (raw_scales - cmin) / max(cmax - cmin, 1e-8)
            cached_t = t
        ret = np.diff(np.log(np.clip(close[max(0, t - 4):t + 1], 1e-8, None)))
        bucket = out[split]
        bucket["x"].append(feats_n[t - lookback + 1:t + 1])
        bucket["y"].append(close_n[t + horizon])
        bucket["y_raw"].append(close[t + horizon])
        bucket["p_last"].append(close_n[t])
        bucket["p_last_raw"].append(close[t])
        bucket["vol"].append(float(np.std(ret)) if len(ret) else 0.0)
        bucket["scales"].append(cached_scales)
        bucket["dates"].append(str(df["date"].iloc[t]))

    packed = {}
    for split, bucket in out.items():
        if not bucket["x"]:
            raise SystemExit(f"no {split} windows; check lookback/horizon versus the split")
        packed[split] = {
            "x": np.stack(bucket["x"]).astype(np.float32),
            "y": np.asarray(bucket["y"], dtype=np.float32),
            "y_raw": np.asarray(bucket["y_raw"], dtype=np.float32),
            "p_last": np.asarray(bucket["p_last"], dtype=np.float32),
            "p_last_raw": np.asarray(bucket["p_last_raw"], dtype=np.float32),
            "vol": np.asarray(bucket["vol"], dtype=np.float32),
            "scales": np.stack(bucket["scales"]).astype(np.float32),
            "dates": np.asarray(bucket["dates"]),
        }
    return packed


def _concat(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = frames[0].keys()
    return {k: np.concatenate([frame[k] for frame in frames], axis=0) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()

    cfg = load_default()
    meta = load_datasets()[args.dataset]
    raw_dir = ROOT / args.raw_dir / args.dataset
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSVs under {raw_dir}")

    n_train, n_val, n_test = meta["split"]
    decomp_cfg = dict(cfg.get("decomposition", {}))
    decomp_cfg["n_scales"] = cfg.get("n_scales", 3)
    out_dir = ROOT / "data" / "features" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    per_split: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "val": [], "test": []}
    for path in files:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        packed = windows_from_frame(
            df,
            cfg["lookback"],
            cfg["horizon"],
            n_train,
            n_val,
            n_test,
            "turnover",
            decomp_cfg,
        )
        for split, payload in packed.items():
            n = len(payload["y"])
            payload["symbols"] = np.array([path.stem] * n)
            payload["ids"] = np.array([f"{path.stem}_{i:05d}" for i in range(n)])
            per_split[split].append(payload)

    for split, frames in per_split.items():
        payload = _concat(frames)
        dest = out_dir / f"{split}.npz"
        np.savez_compressed(dest, **payload)
        print("wrote", dest, len(payload["y"]))


if __name__ == "__main__":
    main()
