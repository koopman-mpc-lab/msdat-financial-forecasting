import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.analysis.explainability import attention_shap_correlation
from src.config import FEATURE_COLS, load_config
from src.data import load_splits, make_dataloader
from src.models.factory import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    _, _, test_ds, _ = load_splits(args.dataset, cfg)
    loader = make_dataloader(test_ds, cfg["batch_size"])
    model = build_model("msdat", cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()
    attn_weights = []
    X_list, y_list = [], []
    for batch in loader:
        x = batch["x"].to(device)
        vol = batch["vol"].to(device)
        close = batch["close_window"].to(device)
        _, _, _, _, aw, _ = model(x, vol=vol, close_window=close, return_aux=True)
        attn_weights.extend([a.cpu().numpy() for a in aw])
        X_list.append(batch["x"][:, -1].numpy())
        y_list.append(batch["y"].numpy())
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    rho, pval, attn_rank, shap_vals = attention_shap_correlation(attn_weights, X, y, FEATURE_COLS)
    out = {"spearman_rho": rho, "p_value": pval,
           "attention": attn_rank.tolist(), "shap": shap_vals.tolist()}
    out_path = Path(cfg["output_dir"]) / f"explainability_{args.dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
