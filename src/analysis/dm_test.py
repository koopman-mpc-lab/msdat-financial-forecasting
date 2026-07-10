import numpy as np
from scipy import stats


def _nw_variance(d, max_lag=None):
    n = len(d)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    d = d - d.mean()
    gamma0 = np.dot(d, d) / n
    var = gamma0
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)
        gamma = np.dot(d[lag:], d[:-lag]) / n
        var += 2 * w * gamma
    return var / n


def diebold_mariano_test(errors1, errors2):
    d = np.asarray(errors1) ** 2 - np.asarray(errors2) ** 2
    n = len(d)
    var_d = _nw_variance(d)
    if var_d <= 0:
        return 0.0, 1.0
    dm = d.mean() / np.sqrt(var_d)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def holm_adjust(p_values, alpha=0.05):
    p = np.asarray(p_values)
    order = np.argsort(p)
    adjusted = np.ones_like(p)
    m = len(p)
    for i, idx in enumerate(order):
        adjusted[idx] = min(1.0, p[idx] * (m - i))
    for i in range(1, m):
        idx = order[i]
        prev = order[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev])
    return adjusted.tolist(), [a < alpha for a in adjusted]
