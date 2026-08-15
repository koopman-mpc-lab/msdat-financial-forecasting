from __future__ import annotations

import numpy as np


class ARIMABaseline:
    """Rolling-window ARIMA(1,1,1) used as the statistical reference."""

    order = (1, 1, 1)

    def fit_predict_window(self, close: np.ndarray) -> float:
        y = np.asarray(close, dtype=np.float64)
        try:
            from statsmodels.tsa.arima.model import ARIMA
            fitted = ARIMA(y, order=self.order).fit()
            return float(fitted.forecast(1)[0])
        except Exception:
            if len(y) < 3:
                return float(y[-1])
            dy = np.diff(y)
            return float(y[-1] + dy[-1])

    def predict_batch(self, windows: np.ndarray, close_index: int = 3) -> np.ndarray:
        return np.asarray(
            [self.fit_predict_window(windows[i, :, close_index]) for i in range(len(windows))],
            dtype=np.float32,
        )
