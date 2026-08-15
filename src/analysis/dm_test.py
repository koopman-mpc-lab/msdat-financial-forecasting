from __future__ import annotations

import numpy as np


def _newey_west(values: np.ndarray, lags: int | None = None) -> float:
    n = len(values)
    if lags is None:
        lags = int(np.floor(n ** (1 / 3)))
    centered = values - values.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    var = gamma0
    for lag in range(1, lags + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (lags + 1)
        var += 2.0 * weight * gamma
    return var


def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray) -> dict[str, float]:
    """DM test on squared-error loss; positive statistic favours model a (MSDAT)."""
    d = err_b ** 2 - err_a ** 2
    mean = float(d.mean())
    var = _newey_west(d)
    stat = mean / np.sqrt(max(var, 1e-18) / len(d))
    from math import erfc, sqrt
    p = float(erfc(abs(stat) / sqrt(2.0)))
    return {"dm_stat": float(stat), "p_value": p}


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted
