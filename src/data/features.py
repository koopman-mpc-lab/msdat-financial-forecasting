from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "open", "high", "low", "close", "volume",
    "ma5", "ma10", "ma20", "rsi14", "macd_hist",
    "bb_width", "liquidity",
]


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def build_features(ohlcv: np.ndarray, liquidity: np.ndarray | None = None) -> np.ndarray:
    """Build the twelve-channel daily feature matrix from OHLCV.

    ohlcv: (T, 5) columns [open, high, low, close, volume]
    """
    ohlcv = np.asarray(ohlcv, dtype=np.float64)
    close = ohlcv[:, 3]
    t = len(close)
    ma5 = _rolling_mean(close, 5)
    ma10 = _rolling_mean(close, 10)
    ma20 = _rolling_mean(close, 20)
    rsi = _rsi(close, 14)
    macd = _ema(close, 12) - _ema(close, 26)
    macd_hist = macd - _ema(macd, 9)
    std20 = _rolling_std(close, 20)
    bb_width = np.divide(4.0 * std20, ma20, out=np.zeros(t), where=ma20 > 0)
    if liquidity is None:
        ret = np.zeros(t)
        ret[1:] = np.diff(np.log(np.clip(close, 1e-8, None)))
        liquidity = _rolling_std(ret, 5)
    feats = np.stack(
        [
            ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], close, ohlcv[:, 4],
            ma5, ma10, ma20, rsi, macd_hist, bb_width, np.asarray(liquidity, dtype=np.float64),
        ],
        axis=1,
    )
    return feats


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    csum = np.cumsum(values)
    for i in range(len(values)):
        left = max(0, i - window + 1)
        n = i - left + 1
        prev = csum[left - 1] if left else 0.0
        out[i] = (csum[i] - prev) / n
    return out


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    for i in range(len(values)):
        left = max(0, i - window + 1)
        out[i] = float(np.std(values[left:i + 1]))
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, 0.0, None)
    loss = np.clip(-delta, 0.0, None)
    avg_gain = _rolling_mean(gain, period)
    avg_loss = _rolling_mean(loss, period)
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss > 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)
