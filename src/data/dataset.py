from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import resolve


class WindowDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        payload = np.load(resolve(npz_path), allow_pickle=True)
        self.x = payload["x"].astype(np.float32)
        self.y = payload["y"].astype(np.float32)
        self.y_raw = payload["y_raw"].astype(np.float32)
        self.p_last = payload["p_last"].astype(np.float32)
        self.p_last_raw = payload["p_last_raw"].astype(np.float32)
        self.vol = payload["vol"].astype(np.float32)
        self.scales = payload["scales"].astype(np.float32)
        self.ids = payload["ids"]
        self.dates = payload["dates"]
        self.symbols = payload["symbols"] if "symbols" in payload.files else np.array(["."] * len(self.y))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {
            "x": torch.from_numpy(self.x[idx]),
            "y": torch.tensor(self.y[idx]),
            "y_raw": torch.tensor(self.y_raw[idx]),
            "p_last": torch.tensor(self.p_last[idx]),
            "p_last_raw": torch.tensor(self.p_last_raw[idx]),
            "vol": torch.tensor(self.vol[idx]),
            "scales": torch.from_numpy(self.scales[idx]),
            "id": str(self.ids[idx]),
            "date": str(self.dates[idx]),
            "symbol": str(self.symbols[idx]),
        }


def load_split(dataset: str, split: str) -> WindowDataset:
    return WindowDataset(Path("data") / "features" / dataset / f"{split}.npz")


def collate_windows(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "y_raw": torch.stack([item["y_raw"] for item in batch], dim=0),
        "p_last": torch.stack([item["p_last"] for item in batch], dim=0),
        "p_last_raw": torch.stack([item["p_last_raw"] for item in batch], dim=0),
        "vol": torch.stack([item["vol"] for item in batch], dim=0),
        "scales": torch.stack([item["scales"] for item in batch], dim=0),
        "id": [item["id"] for item in batch],
        "date": [item["date"] for item in batch],
        "symbol": [item["symbol"] for item in batch],
    }
