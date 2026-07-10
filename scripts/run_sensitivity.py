import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.config import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", default="spx")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    results = {"lookback": {}, "num_scales": {}}
    for L in [20, 40, 60, 80, 100]:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["lookback"] = L
        cfg_path = Path(cfg["output_dir"]) / f"sens_L{L}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(run_cfg, f)
        cmd = [sys.executable, str(root / "scripts" / "train.py"),
               "--config", str(cfg_path), "--dataset", args.dataset,
               "--model", "msdat", "--seed", str(args.seed)]
        subprocess.run(cmd, check=True, cwd=str(root))
        mpath = Path(cfg["output_dir"]) / args.dataset / "msdat" / f"seed_{args.seed}" / "metrics.json"
        with open(mpath, encoding="utf-8") as f:
            results["lookback"][L] = json.load(f)
    for K in [2, 3, 4, 5]:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["num_scales"] = K
        cfg_path = Path(cfg["output_dir"]) / f"sens_K{K}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(run_cfg, f)
        cmd = [sys.executable, str(root / "scripts" / "train.py"),
               "--config", str(cfg_path), "--dataset", args.dataset,
               "--model", "msdat", "--seed", str(args.seed)]
        subprocess.run(cmd, check=True, cwd=str(root))
        mpath = Path(cfg["output_dir"]) / args.dataset / "msdat" / f"seed_{args.seed}" / "metrics.json"
        with open(mpath, encoding="utf-8") as f:
            results["num_scales"][K] = json.load(f)
    out = Path(cfg["output_dir"]) / f"sensitivity_{args.dataset}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
