#!/usr/bin/env python
"""Compare aggregated channel-attention ranks with a LightGBM-SHAP ranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.analysis.explainability import spearman
from src.config import load_default
from src.data.dataset import WindowDataset, collate_windows
from src.data.features import FEATURE_NAMES
from src.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", default="SPX")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_default()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(payload.get("model_name", "MSDAT"), payload.get("cfg", cfg), payload.get("extra", {}))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    data = WindowDataset(ROOT / "data" / "features" / args.dataset / f"{args.split}.npz")
    loader = DataLoader(data, batch_size=32, shuffle=False, collate_fn=collate_windows)
    attns = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"], scales=batch.get("scales"), vol=batch.get("vol"))
            attns.append(out["channel_attn"].mean(dim=1).cpu().numpy())
    attn_w = np.concatenate(attns, axis=0).mean(axis=0)

    try:
        import lightgbm as lgb
        import shap
    except ImportError as exc:
        raise SystemExit("lightgbm and shap are required for this script") from exc

    y = data.y
    booster = lgb.LGBMRegressor(n_estimators=500, max_depth=6, verbosity=-1)
    booster.fit(data.x.mean(axis=1), y)
    explainer = shap.TreeExplainer(booster)
    shap_w = np.abs(explainer.shap_values(data.x.mean(axis=1))).mean(axis=0)
    rho = spearman(attn_w, shap_w)
    ranking = {
        "dataset": args.dataset,
        "spearman": rho,
        "attention": {name: float(w) for name, w in zip(FEATURE_NAMES, attn_w)},
        "shap": {name: float(w) for name, w in zip(FEATURE_NAMES, shap_w)},
    }
    print(f"Spearman(attention, SHAP) = {rho:.3f}")
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(ranking, handle, indent=2)


if __name__ == "__main__":
    main()
