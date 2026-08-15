#!/usr/bin/env python
"""Train selected models on one or more datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["SPX"])
    parser.add_argument("--models", nargs="+", default=["MSDAT"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                cmd = [
                    sys.executable, str(ROOT / "scripts" / "train.py"),
                    "--dataset", dataset, "--model", model, "--seed", str(seed),
                ]
                print(" ".join(cmd))
                subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
