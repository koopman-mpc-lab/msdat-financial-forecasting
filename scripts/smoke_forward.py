#!/usr/bin/env python
"""Forward-pass smoke check for MSDAT and the neural baselines."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.config import load_default
from src.models import build_model, count_params, format_param_line


def main() -> None:
    cfg = load_default()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, cfg["lookback"], cfg["n_features"], device=device)
    scales = torch.randn(4, cfg["n_scales"], cfg["lookback"], device=device)
    vol = torch.rand(4, device=device)
    names = [
        "MSDAT", "GRU", "LSTM", "CNN-LSTM", "N-HiTS",
        "iTransformer", "PatchTST", "TimesNet", "Informer", "Autoformer",
    ]
    print(f"device={device} lookback={cfg['lookback']} C={cfg['n_features']}")
    for name in names:
        model = build_model(name, cfg)
        model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(x, scales=scales if name == "MSDAT" else None, vol=vol)
        stats = count_params(model)
        print(f"{name:14s} {format_param_line(stats)}  point={tuple(out['point'].shape)}")
    print("smoke_forward: ok")


if __name__ == "__main__":
    main()
