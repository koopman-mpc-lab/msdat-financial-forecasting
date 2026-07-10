import numpy as np
from sklearn.metrics import r2_score


def compute_metrics(y_true_norm, y_pred_norm, y_true_raw, y_pred_raw, p_t):
    y_true_norm = np.asarray(y_true_norm)
    y_pred_norm = np.asarray(y_pred_norm)
    y_true_raw = np.asarray(y_true_raw)
    y_pred_raw = np.asarray(y_pred_raw)
    p_t = np.asarray(p_t)
    rmse = float(np.sqrt(np.mean((y_true_norm - y_pred_norm) ** 2)))
    mae = float(np.mean(np.abs(y_true_norm - y_pred_norm)))
    mape = float(np.mean(np.abs((y_true_raw - y_pred_raw) / (y_true_raw + 1e-8))) * 100)
    r2 = float(r2_score(y_true_raw, y_pred_raw))
    da = float(np.mean(np.sign(y_pred_raw - p_t) == np.sign(y_true_raw - p_t)) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "da": da}
