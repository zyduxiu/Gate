from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def squared_l2_to_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    diff = matrix - query.astype(np.float32, copy=False)
    return np.einsum("ij,ij->i", diff, diff)


def recall_at_k(found: Sequence[int], truth: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    truth_k = set(int(x) for x in truth[:k])
    if not truth_k:
        return 0.0
    found_k = set(int(x) for x in found[:k])
    return len(found_k & truth_k) / float(len(truth_k))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def gini(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(np.maximum(arr, 0.0))
    n = arr.size
    weighted = np.sum((np.arange(1, n + 1) * arr))
    return float((2.0 * weighted) / (n * np.sum(arr)) - (n + 1.0) / n)


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def geometric_mean(values: Sequence[float]) -> float:
    vals = [max(float(v), 1e-12) for v in values]
    if not vals:
        return 0.0
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))
