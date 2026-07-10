import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis.dm_test import diebold_mariano_test, holm_adjust
from src.config import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--baseline-model", required=True)
    p.add_argument("--msdat-preds", required=True)
    p.add_argument("--baseline-preds", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    msdat = np.load(args.msdat_preds)
    base = np.load(args.baseline_preds)
    e1 = msdat["targets"] - msdat["preds"]
    e2 = base["targets"] - base["preds"]
    dm, p = diebold_mariano_test(e1, e2)
    out = {"dm_stat": dm, "p_value": p, "baseline": args.baseline_model, "dataset": args.dataset}
    out_path = Path(cfg["output_dir"]) / f"dm_{args.dataset}_{args.baseline_model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
