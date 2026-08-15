from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_default() -> dict[str, Any]:
    return load_yaml("configs/default.yaml")


def load_datasets() -> dict[str, Any]:
    return load_yaml("configs/datasets.yaml")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
