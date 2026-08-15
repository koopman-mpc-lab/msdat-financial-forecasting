from __future__ import annotations

import numpy as np


def mean_period(values: np.ndarray) -> float:
    crossings = np.flatnonzero(np.diff(np.signbit(values)))
    if len(crossings) < 2:
        return float(len(values))
    return float(2.0 * np.mean(np.diff(crossings)))


def group_imfs(
    imfs: np.ndarray,
    residual: np.ndarray,
    tau_high: float = 5.0,
    tau_mid: float = 20.0,
    n_scales: int = 3,
) -> np.ndarray:
    """Group IMFs by mean instantaneous period into n_scales reconstructions."""
    length = residual.shape[0]
    bands = np.zeros((n_scales, length), dtype=np.float64)
    if n_scales == 1:
        bands[0] = np.sum(imfs, axis=0) + residual
        return bands

    edges = np.linspace(0.0, 1.0, n_scales + 1)
    # Default 3-scale rule uses the manuscript thresholds; extra scales
    # split the same period axis into equal log-period bins.
    if n_scales == 3:
        cuts = [tau_high, tau_mid]
    else:
        periods = [mean_period(imf) for imf in imfs] + [float(length)]
        lo, hi = min(periods), max(periods)
        cuts = [lo + (hi - lo) * e for e in edges[1:-1]]

    for imf in imfs:
        period = mean_period(imf)
        assigned = n_scales - 1
        for i, cut in enumerate(cuts):
            if period < cut:
                assigned = i
                break
        bands[assigned] += imf
    bands[-1] += residual
    return bands


def _sift(signal: np.ndarray, max_imfs: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight successive-envelope sifting used when PyEMD is absent."""
    residual = signal.astype(np.float64).copy()
    imfs = []
    t = np.arange(len(signal), dtype=np.float64)
    for _ in range(max_imfs):
        proto = residual.copy()
        for _ in range(8):
            peaks = np.r_[0, np.flatnonzero((proto[1:-1] > proto[:-2]) & (proto[1:-1] >= proto[2:])) + 1, len(proto) - 1]
            troughs = np.r_[0, np.flatnonzero((proto[1:-1] < proto[:-2]) & (proto[1:-1] <= proto[2:])) + 1, len(proto) - 1]
            if len(peaks) < 3 or len(troughs) < 3:
                break
            upper = np.interp(t, peaks, proto[peaks])
            lower = np.interp(t, troughs, proto[troughs])
            proto = proto - 0.5 * (upper + lower)
        if np.allclose(proto, 0.0, atol=1e-8):
            break
        imfs.append(proto)
        residual = residual - proto
        if np.std(residual) < 0.05 * (np.std(signal) + 1e-8):
            break
    if not imfs:
        imfs.append(residual.copy())
        residual = np.zeros_like(residual)
    return np.stack(imfs, axis=0), residual


def window_ceemdan(
    close: np.ndarray,
    ensemble: int = 50,
    tau_high: float = 5.0,
    tau_mid: float = 20.0,
    n_scales: int = 3,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    close = np.asarray(close, dtype=np.float64)
    try:
        from PyEMD import CEEMDAN
        decomposer = CEEMDAN(trials=int(ensemble), parallel=False)
        decomposer.noise_seed(int(seed))
        imfs = decomposer(close)
        if imfs.ndim == 1:
            imfs = imfs[None, :]
        residual = imfs[-1]
        imfs = imfs[:-1] if imfs.shape[0] > 1 else imfs
    except Exception:
        imfs, residual = _sift(close)

    scales = group_imfs(imfs, residual, tau_high, tau_mid, n_scales)
    return {"imfs": imfs, "residual": residual, "scales": scales}
