from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def measure_latency(model, batch: dict, device: str = "cpu", repeats: int = 20) -> float:
    model.eval()
    x = batch["x"].to(device)
    scales = batch.get("scales")
    vol = batch.get("vol")
    if scales is not None:
        scales = scales.to(device)
    if vol is not None:
        vol = vol.to(device)
    for _ in range(5):
        model(x, scales=scales, vol=vol)
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(x, scales=scales, vol=vol)
    elapsed = time.perf_counter() - t0
    return float(1000.0 * elapsed / (repeats * x.size(0)))
