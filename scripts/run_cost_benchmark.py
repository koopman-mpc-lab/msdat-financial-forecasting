import argparse
import json
from pathlib import Path

from src.analysis.cost import benchmark_model
from src.config import load_config
from src.data import load_splits, make_dataloader
from src.models.factory import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", default="spx")
    p.add_argument("--models", nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    models = args.models or [m for m in cfg["models"] if m != "arima"]
    train_ds, _, _, _ = load_splits(args.dataset, cfg)
    loader = make_dataloader(train_ds, cfg["batch_size"], shuffle=True)
    results = {}
    for name in models:
        model = build_model(name, cfg)
        results[name] = benchmark_model(model, loader, cfg)
    out = Path(cfg["output_dir"]) / f"cost_{args.dataset}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
