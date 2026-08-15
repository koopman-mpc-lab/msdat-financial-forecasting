from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true_raw), 1e-8, None)
    return float(np.mean(np.abs(y_true_raw - y_pred_raw) / denom) * 100.0)


def r2_score(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> float:
    ss_res = np.sum((y_true_raw - y_pred_raw) ** 2)
    ss_tot = np.sum((y_true_raw - np.mean(y_true_raw)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, p_last: np.ndarray) -> float:
    true_dir = np.sign(y_true - p_last)
    pred_dir = np.sign(y_pred - p_last)
    true_dir[true_dir == 0] = 1
    pred_dir[pred_dir == 0] = 1
    return float(np.mean(true_dir == pred_dir) * 100.0)


def quantile_coverage(y_true: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> float:
    return float(np.mean((y_true >= q10) & (y_true <= q90)))


def evaluate_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_last: np.ndarray,
    y_true_raw: np.ndarray | None = None,
    y_pred_raw: np.ndarray | None = None,
    q10: np.ndarray | None = None,
    q90: np.ndarray | None = None,
) -> dict[str, float]:
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "da": directional_accuracy(y_true, y_pred, p_last),
    }
    if y_true_raw is not None and y_pred_raw is not None:
        out["mape"] = mape(y_true_raw, y_pred_raw)
        out["r2"] = r2_score(y_true_raw, y_pred_raw)
    if q10 is not None and q90 is not None:
        out["coverage80"] = quantile_coverage(y_true, q10, q90)
    return out


def summarize(metrics: dict[str, float]) -> str:
    parts = [f"RMSE={metrics['rmse']:.4f}", f"MAE={metrics['mae']:.4f}", f"DA={metrics['da']:.1f}%"]
    if "mape" in metrics:
        parts.append(f"MAPE={metrics['mape']:.2f}%")
    if "r2" in metrics:
        parts.append(f"R2={metrics['r2']:.4f}")
    if "coverage80" in metrics:
        parts.append(f"cov80={metrics['coverage80']:.3f}")
    return " ".join(parts)
