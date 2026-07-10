import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.config import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    datasets = args.datasets or cfg["datasets"]
    models = args.models or cfg["models"]
    seeds = args.seeds or list(range(cfg["num_seeds"]))
    root = Path(__file__).resolve().parents[1]
    results = {}
    for ds in datasets:
        results[ds] = {}
        for model in models:
            metrics_list = []
            for seed in seeds:
                cmd = [sys.executable, str(root / "scripts" / "train.py"),
                       "--dataset", ds, "--model", model, "--seed", str(seed)]
                if args.config:
                    cmd += ["--config", args.config]
                subprocess.run(cmd, check=True, cwd=str(root))
                mpath = root / cfg["output_dir"].replace(str(root) + "\\", "").replace(str(root) + "/", "")
                mpath = Path(cfg["output_dir"]) / ds / model / f"seed_{seed}" / "metrics.json"
                with open(mpath, encoding="utf-8") as f:
                    metrics_list.append(json.load(f))
            avg = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}
            results[ds][model] = avg
    out = Path(cfg["output_dir"]) / "overall_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
