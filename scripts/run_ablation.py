import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from src.config import load_config


ABLATIONS = [
    {"name": "w/o decomposition", "ablation": {"use_decomposition": False}},
    {"name": "w/o temporal attention (GRU)", "ablation": {"use_temporal_attention": False, "temporal_backend": "gru"}},
    {"name": "w/o channel attention", "ablation": {"use_channel_attention": False}},
    {"name": "w/o cross-scale gating", "ablation": {"use_gating": False}},
    {"name": "w/o directional loss", "ablation": {}, "lambda_direction": 0.0},
    {"name": "Full MSDAT", "ablation": {}},
]


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
    results = {}
    for ab in ABLATIONS:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["ablation"] = ab.get("ablation", {})
        if "lambda_direction" in ab:
            run_cfg["lambda_direction"] = ab["lambda_direction"]
        cfg_path = Path(cfg["output_dir"]) / "ablation_cfg.json"
        import yaml
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(run_cfg, f)
        tag = ab["name"].replace(" ", "_").replace("/", "_")
        cmd = [sys.executable, str(root / "scripts" / "train.py"),
               "--config", str(cfg_path), "--dataset", args.dataset,
               "--model", "msdat", "--seed", str(args.seed)]
        subprocess.run(cmd, check=True, cwd=str(root))
        mpath = Path(cfg["output_dir"]) / args.dataset / "msdat" / f"seed_{args.seed}" / "metrics.json"
        with open(mpath, encoding="utf-8") as f:
            results[ab["name"]] = json.load(f)
    out = Path(cfg["output_dir"]) / f"ablation_{args.dataset}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
