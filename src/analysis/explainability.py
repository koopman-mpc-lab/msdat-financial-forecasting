from __future__ import annotations

import numpy as np


def rank_features(weights: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(weights), kind="stable")
    ranks = np.empty(len(weights), dtype=int)
    ranks[order] = np.arange(1, len(weights) + 1)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = rank_features(a)
    rb = rank_features(b)
    ra = ra.astype(float)
    rb = rb.astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    return float(np.dot(ra, rb) / max(denom, 1e-12))
