from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_fvecs(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors, dtype="<f4")
    dim = np.int32(vectors.shape[1])
    with path.open("wb") as handle:
        for row in vectors:
            handle.write(dim.tobytes())
            handle.write(row.tobytes())


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4")
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    raw_i32 = raw.view("<i4")
    dim = int(raw_i32[0])
    if dim <= 0:
        raise ValueError(f"invalid fvecs dimension {dim} in {path}")
    row_width = dim + 1
    if raw.size % row_width != 0:
        raise ValueError(f"{path} does not look like fvecs; size is not divisible by {row_width}")
    rows_i32 = raw_i32.reshape(-1, row_width)
    if not np.all(rows_i32[:, 0] == dim):
        raise ValueError(f"{path} has inconsistent fvecs row dimensions")
    rows = raw.reshape(-1, row_width)
    return rows[:, 1:].astype(np.float32, copy=True)


def read_fvecs_slice(path: Path, limit: int | None = None, offset: int = 0) -> np.ndarray:
    """Read a row slice from an fvecs file without loading the whole file."""

    if limit is not None and limit <= 0:
        return np.empty((0, 0), dtype=np.float32)
    if offset < 0:
        raise ValueError("offset must be non-negative")
    with path.open("rb") as handle:
        dim_raw = handle.read(4)
        if len(dim_raw) != 4:
            return np.empty((0, 0), dtype=np.float32)
        dim = int(np.frombuffer(dim_raw, dtype="<i4")[0])
        if dim <= 0:
            raise ValueError(f"invalid fvecs dimension {dim} in {path}")
        row_bytes = 4 + dim * 4
        size = path.stat().st_size
        if size % row_bytes != 0:
            raise ValueError(f"{path} does not look like fixed-width fvecs")
        rows = size // row_bytes
        if offset >= rows:
            return np.empty((0, dim), dtype=np.float32)
        count = rows - offset if limit is None else min(int(limit), rows - offset)
        handle.seek(offset * row_bytes)
        raw = np.fromfile(handle, dtype="<f4", count=count * (dim + 1))
    if raw.size != count * (dim + 1):
        raise ValueError(f"truncated fvecs slice in {path}")
    raw_i32 = raw.view("<i4").reshape(count, dim + 1)
    if not np.all(raw_i32[:, 0] == dim):
        raise ValueError(f"{path} has inconsistent fvecs row dimensions")
    return raw.reshape(count, dim + 1)[:, 1:].astype(np.float32, copy=True)


def write_u32_list(path: Path, values: list[int] | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype="<u4")
    header = np.asarray([arr.size], dtype="<u4")
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(arr.tobytes())


def write_ivecs(path: Path, ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(ids, dtype="<i4")
    dim = np.int32(ids.shape[1])
    with path.open("wb") as handle:
        for row in ids:
            handle.write(dim.tobytes())
            handle.write(row.tobytes())


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i4")
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"invalid ivecs dimension {dim} in {path}")
    row_width = dim + 1
    if raw.size % row_width != 0:
        raise ValueError(f"{path} does not look like ivecs; size is not divisible by {row_width}")
    rows = raw.reshape(-1, row_width)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"{path} has inconsistent ivecs row dimensions")
    return rows[:, 1:].astype(np.int32, copy=True)


def write_diskann_fbin(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors, dtype="<f4")
    header = np.asarray([vectors.shape[0], vectors.shape[1]], dtype="<u4")
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(vectors.tobytes())


def write_diskann_gt(path: Path, ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(ids, dtype="<u4")
    header = np.asarray([ids.shape[0], ids.shape[1]], dtype="<u4")
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(ids.tobytes())


def write_nsg_graph(path: Path, ids: np.ndarray) -> None:
    write_ivecs(path, ids)


def exact_knn(
    base: np.ndarray,
    queries: np.ndarray,
    k: int,
    batch_size: int = 256,
    exclude_self: bool = False,
) -> np.ndarray:
    base = np.asarray(base, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if k <= 0:
        raise ValueError("k must be positive")
    if base.ndim != 2 or queries.ndim != 2:
        raise ValueError("base and queries must be 2D arrays")
    if base.shape[1] != queries.shape[1]:
        raise ValueError("base and query dimensions differ")
    max_k = base.shape[0] - (1 if exclude_self else 0)
    if k > max_k:
        raise ValueError(f"k={k} is larger than available neighbors={max_k}")

    result = np.empty((queries.shape[0], k), dtype=np.int32)
    base_norm = np.sum(base * base, axis=1)
    for start in range(0, queries.shape[0], batch_size):
        stop = min(start + batch_size, queries.shape[0])
        q = queries[start:stop]
        dists = np.sum(q * q, axis=1, keepdims=True) + base_norm.reshape(1, -1) - 2.0 * q @ base.T
        if exclude_self and base.shape[0] == queries.shape[0]:
            rows = np.arange(stop - start)
            dists[rows, start + rows] = np.inf
        part = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        row = np.arange(part.shape[0])[:, None]
        ordered = part[row, np.argsort(dists[row, part], axis=1)]
        result[start:stop] = ordered.astype(np.int32)
    return result


def generate_clustered_vectors(
    n_base: int,
    n_query: int,
    dim: int,
    clusters: int,
    drift_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(clusters, dim)).astype(np.float32)
    labels = rng.integers(0, clusters, size=n_base)
    base = centers[labels] + rng.normal(scale=0.55, size=(n_base, dim))

    drift_n = int(round(n_base * drift_fraction))
    if drift_n > 0:
        drift_center = centers.mean(axis=0)
        drift_center[: min(8, dim)] += 8.0
        base[-drift_n:] = drift_center + rng.normal(scale=0.42, size=(drift_n, dim))

    old_q = n_query // 2
    query_labels = rng.integers(0, clusters, size=old_q)
    old_queries = centers[query_labels] + rng.normal(scale=0.65, size=(old_q, dim))
    drift_center = centers.mean(axis=0)
    drift_center[: min(8, dim)] += 8.0
    new_queries = drift_center + rng.normal(scale=0.45, size=(n_query - old_q, dim))
    queries = np.vstack([old_queries, new_queries])
    rng.shuffle(queries, axis=0)
    return base.astype(np.float32), queries.astype(np.float32)


def prepare_external_dataset(
    out_dir: Path,
    n_base: int = 10_000,
    n_query: int = 1_000,
    dim: int = 64,
    clusters: int = 8,
    drift_fraction: float = 0.20,
    gt_k: int = 100,
    nsg_graph_k: int = 100,
    seed: int = 17,
    batch_size: int = 128,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base, queries = generate_clustered_vectors(
        n_base=n_base,
        n_query=n_query,
        dim=dim,
        clusters=clusters,
        drift_fraction=drift_fraction,
        seed=seed,
    )
    gt = exact_knn(base, queries, k=gt_k, batch_size=batch_size)
    nsg_graph = exact_knn(base, base, k=nsg_graph_k, batch_size=batch_size, exclude_self=True)

    paths = {
        "base_fvecs": out_dir / "base.fvecs",
        "query_fvecs": out_dir / "query.fvecs",
        "truth_ivecs": out_dir / "truth.ivecs",
        "nsg_knn_graph": out_dir / "base_exact_knn.graph",
        "base_fbin": out_dir / "base.fbin",
        "query_fbin": out_dir / "query.fbin",
        "truth_bin": out_dir / "truth.bin",
    }
    write_fvecs(paths["base_fvecs"], base)
    write_fvecs(paths["query_fvecs"], queries)
    write_ivecs(paths["truth_ivecs"], gt)
    write_nsg_graph(paths["nsg_knn_graph"], nsg_graph)
    write_diskann_fbin(paths["base_fbin"], base)
    write_diskann_fbin(paths["query_fbin"], queries)
    write_diskann_gt(paths["truth_bin"], gt)

    manifest = {
        "n_base": n_base,
        "n_query": n_query,
        "dim": dim,
        "clusters": clusters,
        "drift_fraction": drift_fraction,
        "gt_k": gt_k,
        "nsg_graph_k": nsg_graph_k,
        "seed": seed,
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
    }
    manifest_path = out_dir / "manifest.json"
    manifest["manifest"] = str(manifest_path.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_external_dataset_from_arrays(
    out_dir: Path,
    base: np.ndarray,
    queries: np.ndarray,
    gt_k: int = 100,
    nsg_graph_k: int = 100,
    batch_size: int = 128,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = np.asarray(base, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    gt = exact_knn(base, queries, k=gt_k, batch_size=batch_size)
    nsg_graph = exact_knn(base, base, k=nsg_graph_k, batch_size=batch_size, exclude_self=True)

    paths = {
        "base_fvecs": out_dir / "base.fvecs",
        "query_fvecs": out_dir / "query.fvecs",
        "truth_ivecs": out_dir / "truth.ivecs",
        "nsg_knn_graph": out_dir / "base_exact_knn.graph",
        "base_fbin": out_dir / "base.fbin",
        "query_fbin": out_dir / "query.fbin",
        "truth_bin": out_dir / "truth.bin",
    }
    write_fvecs(paths["base_fvecs"], base)
    write_fvecs(paths["query_fvecs"], queries)
    write_ivecs(paths["truth_ivecs"], gt)
    write_nsg_graph(paths["nsg_knn_graph"], nsg_graph)
    write_diskann_fbin(paths["base_fbin"], base)
    write_diskann_fbin(paths["query_fbin"], queries)
    write_diskann_gt(paths["truth_bin"], gt)

    manifest = {
        "n_base": int(base.shape[0]),
        "n_query": int(queries.shape[0]),
        "dim": int(base.shape[1]),
        "gt_k": int(gt_k),
        "nsg_graph_k": int(nsg_graph_k),
        "metadata": metadata or {},
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
    }
    manifest_path = out_dir / "manifest.json"
    manifest["manifest"] = str(manifest_path.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recall_from_ivecs(result_path: Path, truth_path: Path, k: int) -> dict[str, float | int]:
    result = read_ivecs(result_path)
    truth = read_ivecs(truth_path)
    if result.shape[0] != truth.shape[0]:
        raise ValueError(f"query count mismatch: result={result.shape[0]} truth={truth.shape[0]}")
    if k > result.shape[1] or k > truth.shape[1]:
        raise ValueError(f"k={k} exceeds result/truth width: {result.shape[1]}/{truth.shape[1]}")
    hits = 0
    for found, exact in zip(result[:, :k], truth[:, :k]):
        hits += len(set(int(x) for x in found) & set(int(x) for x in exact))
    recall = hits / float(result.shape[0] * k) if result.shape[0] else 0.0
    return {
        "queries": int(result.shape[0]),
        "k": int(k),
        "hits": int(hits),
        "recall": float(recall),
    }


def read_sptag_txt_results(path: Path, k: int) -> np.ndarray:
    rows: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        _, _, payload = line.partition(":")
        ids: list[int] = []
        for item in payload.split("|"):
            if not item:
                continue
            _, _, node_id = item.partition("@")
            if node_id:
                ids.append(int(node_id))
            if len(ids) >= k:
                break
        rows.append(ids)
    if not rows:
        return np.empty((0, 0), dtype=np.int32)
    width = max(len(row) for row in rows)
    result = np.full((len(rows), width), -1, dtype=np.int32)
    for row_idx, row in enumerate(rows):
        result[row_idx, : len(row)] = row
    return result


def recall_from_sptag_txt(result_path: Path, truth_path: Path, k: int) -> dict[str, float | int]:
    result = read_sptag_txt_results(result_path, k)
    truth = read_ivecs(truth_path)
    if result.shape[0] != truth.shape[0]:
        raise ValueError(f"query count mismatch: result={result.shape[0]} truth={truth.shape[0]}")
    if k > result.shape[1] or k > truth.shape[1]:
        raise ValueError(f"k={k} exceeds result/truth width: {result.shape[1]}/{truth.shape[1]}")
    hits = 0
    for found, exact in zip(result[:, :k], truth[:, :k]):
        hits += len(set(int(x) for x in found if x >= 0) & set(int(x) for x in exact))
    recall = hits / float(result.shape[0] * k) if result.shape[0] else 0.0
    return {
        "queries": int(result.shape[0]),
        "k": int(k),
        "hits": int(hits),
        "recall": float(recall),
    }
