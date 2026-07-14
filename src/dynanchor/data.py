from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StreamingScenario:
    initial: np.ndarray
    queries_before: np.ndarray
    insert_batches: list[np.ndarray]
    delete_ids: list[int]
    queries_after: np.ndarray


def make_gaussian_mixture(
    n: int,
    dim: int,
    centers: np.ndarray,
    std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, centers.shape[0], size=n)
    vectors = centers[labels] + rng.normal(scale=std, size=(n, dim))
    return vectors.astype(np.float32), labels.astype(np.int64)


def make_streaming_drift_scenario(
    n_initial: int = 900,
    n_insert: int = 240,
    n_delete: int = 120,
    n_queries: int = 240,
    dim: int = 16,
    batches: int = 3,
    seed: int = 7,
) -> StreamingScenario:
    """Create a compact clustered workload with base-data distribution drift."""

    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(4, dim)).astype(np.float32)
    initial, labels = make_gaussian_mixture(n_initial, dim, centers, std=0.55, seed=seed + 1)

    q_labels = rng.integers(0, centers.shape[0], size=n_queries)
    queries_before = centers[q_labels] + rng.normal(scale=0.65, size=(n_queries, dim))

    drift_center = centers.mean(axis=0)
    drift_center[: min(4, dim)] += 8.0
    drift = drift_center + rng.normal(scale=0.42, size=(n_insert, dim))

    insert_batches = [
        chunk.astype(np.float32)
        for chunk in np.array_split(drift.astype(np.float32), max(1, int(batches)))
    ]

    delete_pool = np.flatnonzero(labels == int(labels[0]))
    if delete_pool.size < n_delete:
        delete_pool = np.arange(n_initial)
    delete_ids = rng.choice(delete_pool, size=min(n_delete, delete_pool.size), replace=False)

    after_old = centers[rng.integers(0, centers.shape[0], size=n_queries // 2)]
    after_new = np.repeat(drift_center.reshape(1, -1), n_queries - (n_queries // 2), axis=0)
    queries_after = np.vstack(
        [
            after_old + rng.normal(scale=0.65, size=after_old.shape),
            after_new + rng.normal(scale=0.45, size=after_new.shape),
        ]
    )
    rng.shuffle(queries_after, axis=0)

    return StreamingScenario(
        initial=initial.astype(np.float32),
        queries_before=queries_before.astype(np.float32),
        insert_batches=insert_batches,
        delete_ids=[int(x) for x in delete_ids],
        queries_after=queries_after.astype(np.float32),
    )
