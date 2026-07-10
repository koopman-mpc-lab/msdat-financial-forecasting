import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.config import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", default="spx")
    p.add_argument("--models", nargs="+", default=["msdat", "itransformer"])
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    results = {}
    for h in args.horizons:
        results[h] = {}
        for model in args.models:
            cmd = [sys.executable, str(root / "scripts" / "train.py"),
                   "--dataset", args.dataset, "--model", model,
                   "--seed", str(args.seed), "--horizon", str(h)]
            if args.config:
                cmd += ["--config", args.config]
            subprocess.run(cmd, check=True, cwd=str(root))
            mpath = Path(cfg["output_dir"]) / args.dataset / model / f"seed_{args.seed}" / "metrics.json"
            with open(mpath, encoding="utf-8") as f:
                results[h][model] = json.load(f)
    out = Path(cfg["output_dir"]) / f"multi_horizon_{args.dataset}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
