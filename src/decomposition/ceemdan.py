import numpy as np

try:
    from PyEMD import CEEMDAN
except ImportError:
    CEEMDAN = None


def _mean_period(imf):
    zero_cross = np.where(np.diff(np.signbit(imf)))[0]
    if len(zero_cross) < 2:
        return float(len(imf))
    return float(np.mean(np.diff(zero_cross)) * 2)


def group_imfs(imfs, residual, tau1=5, tau2=20, num_scales=3):
    if num_scales == 2:
        high = np.zeros_like(residual)
        low = residual.copy()
        for imf in imfs:
            period = _mean_period(imf)
            if period < tau2:
                high += imf
            else:
                low += imf
        return [high, low]
    high, mid, low = np.zeros_like(residual), np.zeros_like(residual), residual.copy()
    for imf in imfs:
        period = _mean_period(imf)
        if period < tau1:
            high += imf
        elif period < tau2:
            mid += imf
        else:
            low += imf
    return [high, mid, low]


def decompose_window(price_window, ensemble=50, tau1=5, tau2=20, num_scales=3):
    signal = np.asarray(price_window, dtype=np.float64)
    if CEEMDAN is None:
        n = len(signal)
        t = np.arange(n)
        trend = np.polyval(np.polyfit(t, signal, 2), t)
        detrended = signal - trend
        high = detrended - np.convolve(detrended, np.ones(5) / 5, mode="same")
        mid = np.convolve(detrended, np.ones(5) / 5, mode="same") - np.convolve(detrended, np.ones(15) / 15, mode="same")
        low = trend
        if num_scales == 2:
            return [high, low + mid]
        return [high, mid, low]
    ce = CEEMDAN(trials=ensemble)
    imfs = ce(signal)
    if imfs.ndim == 1:
        imfs = imfs.reshape(1, -1)
    residual = signal - imfs.sum(axis=0)
    return group_imfs(imfs, residual, tau1, tau2, num_scales)


class DecompositionCache:
    def __init__(self, refresh=5, **decomp_kwargs):
        self.refresh = refresh
        self.decomp_kwargs = decomp_kwargs
        self._cache = {}
        self._step = 0

    def get(self, key, price_window):
        self._step += 1
        if key not in self._cache or self._step % self.refresh == 0:
            self._cache[key] = decompose_window(price_window, **self.decomp_kwargs)
        return self._cache[key]

    def clear(self):
        self._cache.clear()
        self._step = 0
