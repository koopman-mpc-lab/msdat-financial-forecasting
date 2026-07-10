import argparse
import json
from pathlib import Path

import torch

from src.config import load_config
from src.data import load_splits, make_dataloader
from src.models.factory import build_model
from src.training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--save-preds", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    _, _, test_ds, _ = load_splits(args.dataset, cfg)
    test_loader = make_dataloader(test_ds, cfg["batch_size"])
    model = build_model(args.model, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    trainer = Trainer(model, cfg, device)
    metrics, preds, targets, raw_t, p_t = trainer.evaluate(test_loader, return_preds=True)
    print(json.dumps(metrics, indent=2))
    if args.save_preds:
        out = Path(args.checkpoint).parent / "predictions.npz"
        import numpy as np
        np.savez(out, preds=preds, targets=targets, y_raw=raw_t, p_t=p_t)


if __name__ == "__main__":
    main()
