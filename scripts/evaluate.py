#!/usr/bin/env python
"""Recompute metrics from a prediction CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.training.metrics import evaluate_arrays, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.pred)
    metrics = evaluate_arrays(
        df["y_true"].to_numpy(),
        df["y_pred"].to_numpy(),
        df["p_last"].to_numpy(),
        df["y_true_raw"].to_numpy() if "y_true_raw" in df else None,
        df["y_pred_raw"].to_numpy() if "y_pred_raw" in df else None,
        df["q10"].to_numpy() if "q10" in df else None,
        df["q90"].to_numpy() if "q90" in df else None,
    )
    print(Path(args.pred).name, summarize(metrics))


if __name__ == "__main__":
    main()
