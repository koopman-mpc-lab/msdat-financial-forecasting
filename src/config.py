from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "ma5", "ma10", "ma20", "rsi14", "macd_hist",
    "bb_width", "liquidity",
]

DATASET_FILES = {
    "spx": "spx.csv",
    "csi300": "csi300.csv",
    "nasdaq20": "nasdaq20.csv",
    "btc_usd": "btc_usd.csv",
}


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["root"] = str(ROOT)
    cfg["data_dir"] = str(ROOT / cfg["data_dir"])
    cfg["output_dir"] = str(ROOT / cfg["output_dir"])
    return cfg
