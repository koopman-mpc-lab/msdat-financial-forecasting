#!/usr/bin/env python
"""Train the multi-horizon grid from configs/horizon.yaml."""

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
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_yaml("configs/horizon.yaml")
    datasets = [args.dataset] if args.dataset else cfg["datasets"]
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    for dataset in datasets:
        for model in cfg["models"]:
            for horizon in cfg["horizons"]:
                for seed in seeds:
                    cmd = [
                        sys.executable, str(ROOT / "scripts" / "train.py"),
                        "--model", model, "--dataset", dataset,
                        "--seed", str(seed), "--horizon", str(horizon),
                    ]
                    print(" ".join(cmd))
                    subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
