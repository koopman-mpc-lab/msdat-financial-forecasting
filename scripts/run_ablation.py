#!/usr/bin/env python
"""Run the component-ablation grid on one dataset."""

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", default=None)
    args = parser.parse_args()
    variants = args.variants or list(load_yaml("configs/ablation.yaml")["variants"])
    for name in variants:
        cmd = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--model", "MSDAT", "--dataset", args.dataset,
            "--seed", str(args.seed), "--ablation", name,
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
