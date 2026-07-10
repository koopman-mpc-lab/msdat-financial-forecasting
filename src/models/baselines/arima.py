import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def fit_arima_batch(close_windows, order=(1, 1, 1)):
    preds = []
    for w in close_windows:
        try:
            model = ARIMA(w, order=order)
            res = model.fit()
            preds.append(float(res.forecast(1)[0]))
        except Exception:
            preds.append(float(w[-1]))
    return np.array(preds, dtype=np.float32)
