#!/usr/bin/env python
"""Diebold-Mariano test on two prediction CSVs, with optional Holm adjustment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.analysis.dm_test import diebold_mariano, holm_adjust


def _errors(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return df["y_true"].to_numpy() - df["y_pred"].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-a", required=True, help="MSDAT or reference prediction CSV")
    parser.add_argument("--pred-b", action="append", required=True, help="Baseline prediction CSV; repeatable")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    err_a = _errors(Path(args.pred_a))
    rows = []
    for pred_b in args.pred_b:
        err_b = _errors(Path(pred_b))
        n = min(len(err_a), len(err_b))
        result = diebold_mariano(err_a[:n], err_b[:n])
        result["pred_a"] = args.pred_a
        result["pred_b"] = pred_b
        rows.append(result)

    adjusted = holm_adjust([row["p_value"] for row in rows])
    for row, p_adj in zip(rows, adjusted):
        row["p_holm"] = p_adj
        print(f"{Path(row['pred_b']).name}: DM={row['dm_stat']:.3f} p={row['p_value']:.4f} p_holm={p_adj:.4f}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
