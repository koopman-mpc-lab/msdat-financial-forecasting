#!/usr/bin/env python
"""Report parameter counts and per-sample latency for the configured models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.analysis.cost import measure_latency
from src.config import load_default
from src.models import build_model, count_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SPX")
    parser.add_argument("--models", nargs="+", default=["MSDAT", "iTransformer", "PatchTST", "Informer", "Autoformer"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_default()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {
        "x": torch.randn(8, cfg["lookback"], cfg["n_features"], device=device),
        "scales": torch.randn(8, cfg["n_scales"], cfg["lookback"], device=device),
        "vol": torch.rand(8, device=device),
    }
    rows = []
    for name in args.models:
        model = build_model(name, cfg).to(device)
        stats = count_params(model)
        ms = measure_latency(model, batch, device=device, repeats=args.repeats)
        row = {"model": name, "dataset": args.dataset, "device": device, "latency_ms": ms, **stats}
        rows.append(row)
        print(f"{name:14s} params={stats['trainable']:,}  latency={ms:.3f} ms/sample")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
