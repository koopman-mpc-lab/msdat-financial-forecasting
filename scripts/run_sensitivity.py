#!/usr/bin/env python
"""Sweep lookback length and the number of frequency scales."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SPX")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_yaml("configs/sensitivity.yaml")
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    for lookback in cfg["lookbacks"]:
        cmd = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--model", "MSDAT", "--dataset", args.dataset,
            "--seed", str(seed), "--lookback", str(lookback),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    for n_scales in cfg["n_scales"]:
        cmd = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--model", "MSDAT", "--dataset", args.dataset,
            "--seed", str(seed), "--n-scales", str(n_scales),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
