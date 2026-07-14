from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.external import CommandResult, run_command, run_external_binary, to_wsl_path
from dynanchor.external import run_nsg_build
from dynanchor.io_formats import (
    read_fvecs_slice,
    write_diskann_fbin,
    write_diskann_gt,
    write_fvecs,
    write_ivecs,
    write_u32_list,
)


SIFT_URLS = [
    "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
    "https://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
]


ANN_BENCHMARK_URLS = {
    "fashion-mnist-784-euclidean": "https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
    "mnist-784-euclidean": "https://ann-benchmarks.com/mnist-784-euclidean.hdf5",
    "glove-100-angular": "https://ann-benchmarks.com/glove-100-angular.hdf5",
    "nytimes-256-angular": "https://ann-benchmarks.com/nytimes-256-angular.hdf5",
    "gist-960-euclidean": "https://ann-benchmarks.com/gist-960-euclidean.hdf5",
    "deep-image-96-angular": "https://ann-benchmarks.com/deep-image-96-angular.hdf5",
}


def save_command(path: Path, result: CommandResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "command": result.command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def safe_extract_tar_gz(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (out_dir / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe tar member path: {member.name}")
        tar.extractall(out_dir)


def find_sift_root(raw_dir: Path) -> Path:
    candidates = [
        raw_dir,
        raw_dir / "sift",
        raw_dir / "sift1m",
    ]
    for candidate in candidates:
        if (candidate / "sift_base.fvecs").exists() and (candidate / "sift_query.fvecs").exists():
            return candidate
    raise FileNotFoundError(f"cannot find sift_base.fvecs under {raw_dir}")


def curl_executable() -> str:
    return "curl.exe" if os.name == "nt" else "curl"


def download_sift(raw_dir: Path, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "sift.tar.gz"
    if not archive.exists() or force:
        last: subprocess.CompletedProcess[str] | None = None
        for url in SIFT_URLS:
            command = [
                curl_executable(),
                "-L",
                "--retry",
                "5",
                "--connect-timeout",
                "30",
                "-C",
                "-",
                "-o",
                str(archive),
                url,
            ]
            if url.startswith("https://"):
                command.insert(1, "--insecure")
            print("$ " + " ".join(command))
            last = subprocess.run(command, text=True, encoding="utf-8", errors="replace", check=False)
            if last.returncode == 0 and archive.exists() and archive.stat().st_size > 100_000_000:
                break
        else:
            raise RuntimeError(f"failed to download SIFT1M; last return code={last.returncode if last else 'n/a'}")

    sift_root = raw_dir / "sift"
    if force or not (sift_root / "sift_base.fvecs").exists():
        print(f"Extracting {archive} -> {raw_dir}")
        safe_extract_tar_gz(archive, raw_dir)
    return find_sift_root(raw_dir)


def download_hdf5(dataset: str, raw_dir: Path, force: bool = False) -> Path:
    if dataset not in ANN_BENCHMARK_URLS:
        known = ", ".join(sorted(ANN_BENCHMARK_URLS))
        raise ValueError(f"unknown ANN-Benchmarks dataset {dataset!r}; known: {known}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{dataset}.hdf5"
    if out.exists() and not force:
        return out
    url = ANN_BENCHMARK_URLS[dataset]
    command = [
        curl_executable(),
        "-L",
        "--retry",
        "5",
        "--connect-timeout",
        "30",
        "-C",
        "-",
        "-o",
        str(out),
        url,
    ]
    print("$ " + " ".join(command))
    result = subprocess.run(command, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0 or not out.exists() or out.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"failed to download {dataset} from {url}")
    return out


def read_hdf5_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required for ANN-Benchmarks HDF5 datasets; run `python -m pip install h5py`") from error
    with h5py.File(path, "r") as handle:
        if "train" not in handle or "test" not in handle:
            raise KeyError(f"{path} must contain train and test datasets")
        train = np.asarray(handle["train"], dtype=np.float32)
        test = np.asarray(handle["test"], dtype=np.float32)
    return train, test


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def squared_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return (
        np.sum(points * points, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1).reshape(1, -1)
        - 2.0 * points @ centers.T
    )


def exact_knn_labels(
    base: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    k: int,
    batch_size: int,
) -> np.ndarray:
    base = np.asarray(base, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint32)
    queries = np.asarray(queries, dtype=np.float32)
    if k > base.shape[0]:
        raise ValueError(f"k={k} exceeds base size={base.shape[0]}")
    result = np.empty((queries.shape[0], k), dtype=np.uint32)
    base_norm = np.sum(base * base, axis=1)
    for start in range(0, queries.shape[0], batch_size):
        stop = min(start + batch_size, queries.shape[0])
        q = queries[start:stop]
        dists = np.sum(q * q, axis=1, keepdims=True) + base_norm.reshape(1, -1) - 2.0 * q @ base.T
        part = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        row = np.arange(part.shape[0])[:, None]
        ordered = part[row, np.argsort(dists[row, part], axis=1)]
        result[start:stop] = labels[ordered]
    return result


def projection_score(vectors: np.ndarray) -> np.ndarray:
    width = min(16, vectors.shape[1])
    return np.mean(vectors[:, :width], axis=1)


def choose_dynamic_split(
    base_pool: np.ndarray,
    n_initial: int,
    n_insert: int,
    n_delete: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_initial + n_insert > base_pool.shape[0]:
        raise ValueError("pool is too small for requested initial+insert split")
    rng = np.random.default_rng(seed)
    scores = projection_score(base_pool)
    order = np.argsort(scores)
    insert_source = order[-n_insert:]
    initial_candidates = order[: base_pool.shape[0] - n_insert]
    if initial_candidates.size < n_initial:
        raise ValueError("not enough non-insert candidates for initial set")
    initial_source = rng.choice(initial_candidates, size=n_initial, replace=False)
    initial_scores = scores[initial_source]
    delete_order = np.argsort(initial_scores)[: min(n_delete, n_initial)]
    delete_ids = delete_order.astype(np.uint32)
    return initial_source.astype(np.int64), insert_source.astype(np.int64), delete_ids


def choose_queries(query_pool: np.ndarray, n_query: int, seed: int) -> np.ndarray:
    if n_query > query_pool.shape[0]:
        raise ValueError("not enough SIFT queries")
    rng = np.random.default_rng(seed)
    scores = projection_score(query_pool)
    order = np.argsort(scores)
    high_count = n_query // 2
    high = order[-high_count:] if high_count else np.empty(0, dtype=np.int64)
    rest = np.asarray([idx for idx in range(query_pool.shape[0]) if idx not in set(high)], dtype=np.int64)
    random_count = n_query - high.size
    random_part = rng.choice(rest, size=random_count, replace=False) if random_count else np.empty(0, dtype=np.int64)
    selected = np.concatenate([high, random_part])
    rng.shuffle(selected)
    return query_pool[selected].astype(np.float32, copy=True)


def estimate_density(vectors: np.ndarray, ref: np.ndarray, density_k: int) -> np.ndarray:
    distances = squared_distances(vectors, ref)
    k = min(max(1, density_k), ref.shape[0])
    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    return 1.0 / (np.mean(np.sqrt(np.maximum(nearest, 0.0)), axis=1) + 1e-6)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def select_density_labels(
    vectors: np.ndarray,
    labels: np.ndarray,
    count: int,
    seed: int,
    candidate_count: int,
    density_sample: int,
    density_k: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    vectors = np.asarray(vectors, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint32)
    candidate_count = min(candidate_count, vectors.shape[0])
    candidate_pos = rng.choice(vectors.shape[0], size=candidate_count, replace=False)
    candidates = vectors[candidate_pos]
    candidate_labels = labels[candidate_pos]

    ref_count = min(density_sample, vectors.shape[0])
    ref_pos = rng.choice(vectors.shape[0], size=ref_count, replace=False)
    density = normalize(estimate_density(candidates, vectors[ref_pos], density_k=density_k))

    selected_positions: list[int] = [int(np.argmax(density))]
    min_dist = squared_distances(candidates, candidates[selected_positions]).reshape(-1)
    while len(selected_positions) < min(count, candidate_count):
        coverage = normalize(min_dist)
        score = 0.70 * coverage + 0.30 * density
        score[np.asarray(selected_positions, dtype=np.int64)] = -1.0
        next_pos = int(np.argmax(score))
        selected_positions.append(next_pos)
        min_dist = np.minimum(min_dist, squared_distances(candidates, candidates[[next_pos]]).reshape(-1))
    return [int(candidate_labels[pos]) for pos in selected_positions]


def medoid_label(vectors: np.ndarray, labels: np.ndarray) -> int:
    center = np.mean(vectors, axis=0, dtype=np.float64).astype(np.float32)
    distances = np.sum((vectors - center.reshape(1, -1)) ** 2, axis=1)
    return int(labels[int(np.argmin(distances))])


def write_anchor_file(path: Path, anchors: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(x)) for x in anchors) + "\n", encoding="utf-8")


def remap_anchors_to_compact(anchors: list[int], dynamic_labels: np.ndarray) -> list[int]:
    label_to_compact = {int(label): int(pos) for pos, label in enumerate(dynamic_labels)}
    remapped = []
    seen = set()
    for anchor in anchors:
        compact = label_to_compact.get(int(anchor))
        if compact is None or compact in seen:
            continue
        seen.add(compact)
        remapped.append(compact)
    return remapped


def prepare_sift_dynamic(args: argparse.Namespace) -> dict[str, Any]:
    sift_root = find_sift_root(args.raw_dir)
    base_path = sift_root / "sift_base.fvecs"
    query_path = sift_root / "sift_query.fvecs"
    pool_size = max(args.pool_size, args.n_initial + args.n_insert)
    print(f"Reading SIFT base pool: {pool_size} rows")
    base_pool = read_fvecs_slice(base_path, limit=pool_size)
    query_pool = read_fvecs_slice(query_path, limit=max(10_000, args.n_query))

    initial_idx, insert_idx, delete_ids = choose_dynamic_split(
        base_pool=base_pool,
        n_initial=args.n_initial,
        n_insert=args.n_insert,
        n_delete=args.n_delete,
        seed=args.seed,
    )
    initial = base_pool[initial_idx].astype(np.float32, copy=True)
    inserts = base_pool[insert_idx].astype(np.float32, copy=True)
    queries = choose_queries(query_pool, args.n_query, seed=args.seed + 17)

    all_vectors = np.vstack([initial, inserts]).astype(np.float32)
    alive = np.ones(all_vectors.shape[0], dtype=bool)
    alive[delete_ids.astype(np.int64)] = False
    dynamic_labels = np.flatnonzero(alive).astype(np.uint32)
    final_base = all_vectors[dynamic_labels]
    external_labels = np.arange(final_base.shape[0], dtype=np.uint32)

    print(f"Computing exact truth: final_n={final_base.shape[0]} queries={queries.shape[0]} k={args.gt_k}")
    truth_dynamic = exact_knn_labels(final_base, dynamic_labels, queries, args.gt_k, args.batch_size)
    truth_external = exact_knn_labels(final_base, external_labels, queries, args.gt_k, args.batch_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    anchors_dir = args.out_dir / "anchors"
    fixed = [medoid_label(initial, np.arange(initial.shape[0], dtype=np.uint32))]
    static = select_density_labels(
        initial,
        np.arange(initial.shape[0], dtype=np.uint32),
        count=args.anchors,
        seed=args.seed + 31,
        candidate_count=args.anchor_candidates,
        density_sample=args.density_sample,
        density_k=args.density_k,
    )
    dynamic = select_density_labels(
        final_base,
        dynamic_labels,
        count=args.anchors,
        seed=args.seed + 47,
        candidate_count=args.anchor_candidates,
        density_sample=args.density_sample,
        density_k=args.density_k,
    )

    paths = {
        "initial_fvecs": args.out_dir / "initial.fvecs",
        "initial_fbin": args.out_dir / "initial.fbin",
        "insert_fvecs": args.out_dir / "insert.fvecs",
        "insert_fbin": args.out_dir / "insert.fbin",
        "all_fbin": args.out_dir / "all_initial_insert.fbin",
        "query_fbin": args.out_dir / "query.fbin",
        "truth_dynamic_bin": args.out_dir / "truth_dynamic_labels.bin",
        "delete_u32": args.out_dir / "delete_ids.u32",
        "base_fvecs": args.out_dir / "base.fvecs",
        "query_fvecs": args.out_dir / "query.fvecs",
        "truth_ivecs": args.out_dir / "truth.ivecs",
        "base_fbin": args.out_dir / "base.fbin",
        "truth_bin": args.out_dir / "truth.bin",
        "nsg_knn_graph": args.out_dir / "base_hnsw_knn.graph",
        "initial_nsg_knn_graph": args.out_dir / "initial_hnsw_knn.graph",
        "all_nsg_knn_graph": args.out_dir / "all_hnsw_knn.graph",
    }
    write_fvecs(paths["initial_fvecs"], initial)
    write_diskann_fbin(paths["initial_fbin"], initial)
    write_fvecs(paths["insert_fvecs"], inserts)
    write_diskann_fbin(paths["insert_fbin"], inserts)
    write_diskann_fbin(paths["all_fbin"], all_vectors)
    write_diskann_fbin(paths["query_fbin"], queries)
    write_diskann_gt(paths["truth_dynamic_bin"], truth_dynamic)
    write_u32_list(paths["delete_u32"], delete_ids)

    write_fvecs(paths["base_fvecs"], final_base)
    write_fvecs(paths["query_fvecs"], queries)
    write_ivecs(paths["truth_ivecs"], truth_external.astype(np.int32))
    write_diskann_fbin(paths["base_fbin"], final_base)
    write_diskann_gt(paths["truth_bin"], truth_external)

    anchor_files = {
        "fixed_medoid": anchors_dir / "fixed_medoid.txt",
        "static_density": anchors_dir / "static_density.txt",
        "dynamic_density": anchors_dir / "dynamic_density.txt",
    }
    external_anchor_files = {
        "fixed_medoid": anchors_dir / "external_fixed_medoid.txt",
        "static_density": anchors_dir / "external_static_density.txt",
        "dynamic_density": anchors_dir / "external_dynamic_density.txt",
    }
    write_anchor_file(anchor_files["fixed_medoid"], fixed)
    write_anchor_file(anchor_files["static_density"], static)
    write_anchor_file(anchor_files["dynamic_density"], dynamic)
    write_anchor_file(external_anchor_files["fixed_medoid"], remap_anchors_to_compact(fixed, dynamic_labels))
    write_anchor_file(external_anchor_files["static_density"], remap_anchors_to_compact(static, dynamic_labels))
    write_anchor_file(external_anchor_files["dynamic_density"], remap_anchors_to_compact(dynamic, dynamic_labels))

    manifest = {
        "dataset": "sift1m_real_dynamic_subset",
        "n_initial": int(initial.shape[0]),
        "n_insert": int(inserts.shape[0]),
        "n_delete": int(delete_ids.size),
        "n_base": int(final_base.shape[0]),
        "n_query": int(queries.shape[0]),
        "dim": int(final_base.shape[1]),
        "gt_k": int(args.gt_k),
        "nsg_graph_k": int(args.nsg_graph_k),
        "seed": int(args.seed),
        "source": {
            "name": "SIFT1M / TexMex",
            "urls": SIFT_URLS,
            "raw_dir": str(args.raw_dir.resolve()),
            "base_pool_rows": int(pool_size),
            "drift_score": "mean of the first 16 SIFT dimensions; inserts are high-score vectors",
        },
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
        "anchor_files": {key: str(path.resolve()) for key, path in anchor_files.items()},
        "external_anchor_files": {key: str(path.resolve()) for key, path in external_anchor_files.items()},
        "manifest": str((args.out_dir / "manifest.json").resolve()),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    initial_manifest = {
        "dataset": "sift1m_real_initial_subset",
        "n_base": int(initial.shape[0]),
        "n_query": int(queries.shape[0]),
        "dim": int(initial.shape[1]),
        "gt_k": int(args.gt_k),
        "nsg_graph_k": int(args.nsg_graph_k),
        "seed": int(args.seed),
        "paths": {
            "base_fvecs": str(paths["initial_fvecs"].resolve()),
            "base_fbin": str(paths["initial_fbin"].resolve()),
            "query_fbin": str(paths["query_fbin"].resolve()),
            "truth_bin": str(paths["truth_dynamic_bin"].resolve()),
            "nsg_knn_graph": str(paths["initial_nsg_knn_graph"].resolve()),
        },
        "manifest": str((args.out_dir / "initial_manifest.json").resolve()),
    }
    (args.out_dir / "initial_manifest.json").write_text(json.dumps(initial_manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def prepare_hdf5_dynamic(args: argparse.Namespace) -> dict[str, Any]:
    hdf5_path = args.hdf5 or download_hdf5(args.dataset, args.raw_dir, force=args.force_download)
    train, test = read_hdf5_arrays(hdf5_path)
    if args.normalize:
        train = l2_normalize(train)
        test = l2_normalize(test)

    pool_size = max(args.pool_size, args.n_initial + args.n_insert)
    if pool_size > train.shape[0]:
        raise ValueError(f"requested pool_size={pool_size} but train has only {train.shape[0]} rows")
    base_pool = train[:pool_size].astype(np.float32, copy=True)
    query_pool = test.astype(np.float32, copy=True)

    initial_idx, insert_idx, delete_ids = choose_dynamic_split(
        base_pool=base_pool,
        n_initial=args.n_initial,
        n_insert=args.n_insert,
        n_delete=args.n_delete,
        seed=args.seed,
    )
    initial = base_pool[initial_idx].astype(np.float32, copy=True)
    inserts = base_pool[insert_idx].astype(np.float32, copy=True)
    queries = choose_queries(query_pool, args.n_query, seed=args.seed + 17)

    all_vectors = np.vstack([initial, inserts]).astype(np.float32)
    alive = np.ones(all_vectors.shape[0], dtype=bool)
    alive[delete_ids.astype(np.int64)] = False
    dynamic_labels = np.flatnonzero(alive).astype(np.uint32)
    final_base = all_vectors[dynamic_labels]
    external_labels = np.arange(final_base.shape[0], dtype=np.uint32)

    print(f"Computing exact truth: dataset={args.dataset} final_n={final_base.shape[0]} queries={queries.shape[0]} k={args.gt_k}")
    truth_dynamic = exact_knn_labels(final_base, dynamic_labels, queries, args.gt_k, args.batch_size)
    truth_external = exact_knn_labels(final_base, external_labels, queries, args.gt_k, args.batch_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    anchors_dir = args.out_dir / "anchors"
    fixed = [medoid_label(initial, np.arange(initial.shape[0], dtype=np.uint32))]
    static = select_density_labels(
        initial,
        np.arange(initial.shape[0], dtype=np.uint32),
        count=args.anchors,
        seed=args.seed + 31,
        candidate_count=args.anchor_candidates,
        density_sample=args.density_sample,
        density_k=args.density_k,
    )
    dynamic = select_density_labels(
        final_base,
        dynamic_labels,
        count=args.anchors,
        seed=args.seed + 47,
        candidate_count=args.anchor_candidates,
        density_sample=args.density_sample,
        density_k=args.density_k,
    )

    paths = {
        "initial_fvecs": args.out_dir / "initial.fvecs",
        "initial_fbin": args.out_dir / "initial.fbin",
        "insert_fvecs": args.out_dir / "insert.fvecs",
        "insert_fbin": args.out_dir / "insert.fbin",
        "all_fbin": args.out_dir / "all_initial_insert.fbin",
        "query_fbin": args.out_dir / "query.fbin",
        "truth_dynamic_bin": args.out_dir / "truth_dynamic_labels.bin",
        "delete_u32": args.out_dir / "delete_ids.u32",
        "base_fvecs": args.out_dir / "base.fvecs",
        "query_fvecs": args.out_dir / "query.fvecs",
        "truth_ivecs": args.out_dir / "truth.ivecs",
        "base_fbin": args.out_dir / "base.fbin",
        "truth_bin": args.out_dir / "truth.bin",
        "nsg_knn_graph": args.out_dir / "base_hnsw_knn.graph",
        "initial_nsg_knn_graph": args.out_dir / "initial_hnsw_knn.graph",
        "all_nsg_knn_graph": args.out_dir / "all_hnsw_knn.graph",
    }
    write_fvecs(paths["initial_fvecs"], initial)
    write_diskann_fbin(paths["initial_fbin"], initial)
    write_fvecs(paths["insert_fvecs"], inserts)
    write_diskann_fbin(paths["insert_fbin"], inserts)
    write_diskann_fbin(paths["all_fbin"], all_vectors)
    write_diskann_fbin(paths["query_fbin"], queries)
    write_diskann_gt(paths["truth_dynamic_bin"], truth_dynamic)
    write_u32_list(paths["delete_u32"], delete_ids)

    write_fvecs(paths["base_fvecs"], final_base)
    write_fvecs(paths["query_fvecs"], queries)
    write_ivecs(paths["truth_ivecs"], truth_external.astype(np.int32))
    write_diskann_fbin(paths["base_fbin"], final_base)
    write_diskann_gt(paths["truth_bin"], truth_external)

    anchor_files = {
        "fixed_medoid": anchors_dir / "fixed_medoid.txt",
        "static_density": anchors_dir / "static_density.txt",
        "dynamic_density": anchors_dir / "dynamic_density.txt",
    }
    external_anchor_files = {
        "fixed_medoid": anchors_dir / "external_fixed_medoid.txt",
        "static_density": anchors_dir / "external_static_density.txt",
        "dynamic_density": anchors_dir / "external_dynamic_density.txt",
    }
    write_anchor_file(anchor_files["fixed_medoid"], fixed)
    write_anchor_file(anchor_files["static_density"], static)
    write_anchor_file(anchor_files["dynamic_density"], dynamic)
    write_anchor_file(external_anchor_files["fixed_medoid"], remap_anchors_to_compact(fixed, dynamic_labels))
    write_anchor_file(external_anchor_files["static_density"], remap_anchors_to_compact(static, dynamic_labels))
    write_anchor_file(external_anchor_files["dynamic_density"], remap_anchors_to_compact(dynamic, dynamic_labels))

    manifest = {
        "dataset": f"{args.dataset}_dynamic_subset",
        "n_initial": int(initial.shape[0]),
        "n_insert": int(inserts.shape[0]),
        "n_delete": int(delete_ids.size),
        "n_base": int(final_base.shape[0]),
        "n_query": int(queries.shape[0]),
        "dim": int(final_base.shape[1]),
        "gt_k": int(args.gt_k),
        "nsg_graph_k": int(args.nsg_graph_k),
        "seed": int(args.seed),
        "source": {
            "name": f"ANN-Benchmarks {args.dataset}",
            "url": ANN_BENCHMARK_URLS.get(args.dataset),
            "hdf5": str(hdf5_path.resolve()),
            "pool_rows": int(pool_size),
            "normalize": bool(args.normalize),
            "drift_score": "mean of the first 16 dimensions; inserts are high-score vectors",
        },
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
        "anchor_files": {key: str(path.resolve()) for key, path in anchor_files.items()},
        "external_anchor_files": {key: str(path.resolve()) for key, path in external_anchor_files.items()},
        "manifest": str((args.out_dir / "manifest.json").resolve()),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    initial_manifest = {
        "dataset": f"{args.dataset}_initial_subset",
        "n_base": int(initial.shape[0]),
        "n_query": int(queries.shape[0]),
        "dim": int(initial.shape[1]),
        "gt_k": int(args.gt_k),
        "nsg_graph_k": int(args.nsg_graph_k),
        "seed": int(args.seed),
        "paths": {
            "base_fvecs": str(paths["initial_fvecs"].resolve()),
            "base_fbin": str(paths["initial_fbin"].resolve()),
            "query_fbin": str(paths["query_fbin"].resolve()),
            "truth_bin": str(paths["truth_dynamic_bin"].resolve()),
            "nsg_knn_graph": str(paths["initial_nsg_knn_graph"].resolve()),
        },
        "manifest": str((args.out_dir / "initial_manifest.json").resolve()),
    }
    (args.out_dir / "initial_manifest.json").write_text(json.dumps(initial_manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def compile_knn_builder(force: bool, timeout: int) -> Path:
    source = ROOT / "tools" / "hnsw_knn_graph_builder.cpp"
    binary = ROOT / "build" / ("hnsw_knn_graph_builder.exe" if os.name != "nt" else "hnsw_knn_graph_builder")
    binary.parent.mkdir(parents=True, exist_ok=True)
    if not force and binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary
    if os.name == "nt":
        root_wsl = to_wsl_path(str(ROOT.resolve()))
        command = (
            f"cd {shlex.quote(root_wsl)} && "
            "mkdir -p build && "
            "g++ -std=c++17 -O3 -march=native -fopenmp "
            "-Ithird_party/hnswlib "
            "tools/hnsw_knn_graph_builder.cpp -o build/hnsw_knn_graph_builder"
        )
        result = run_command(["wsl", "bash", "-lc", command], timeout=timeout)
    else:
        result = run_command(
            [
                "g++",
                "-std=c++17",
                "-O3",
                "-march=native",
                "-fopenmp",
                "-Ithird_party/hnswlib",
                str(source),
                "-o",
                str(binary),
            ],
            cwd=ROOT,
            timeout=timeout,
        )
    save_command(ROOT / "build" / "hnsw_knn_graph_builder_build.json", result)
    if not result.ok:
        raise RuntimeError("failed to compile hnsw_knn_graph_builder; see build/hnsw_knn_graph_builder_build.json")
    return binary


def build_nsg_graph(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    builder = compile_knn_builder(force=args.force_build, timeout=args.timeout)
    command = [
        str(builder.resolve()),
        "--base-fbin",
        manifest["paths"]["base_fbin"],
        "--out-graph",
        manifest["paths"]["nsg_knn_graph"],
        "--k",
        str(args.k or manifest["nsg_graph_k"]),
        "--M",
        str(args.hnsw_m),
        "--ef-construction",
        str(args.ef_construction),
        "--ef-search",
        str(args.ef_search),
        "--seed",
        str(args.seed),
    ]
    start = time.perf_counter()
    result = run_external_binary(command, timeout=args.timeout)
    save_command(args.manifest.parent / "hnsw_knn_graph_builder_log.json", result)
    if not result.ok:
        raise RuntimeError("hnsw KNN graph builder failed; see hnsw_knn_graph_builder_log.json")
    print(result.stdout.rstrip())
    print(f"Graph build wall seconds: {time.perf_counter() - start:.3f}")


def build_knn_graph(
    builder: Path,
    base_fbin: str,
    out_graph: str,
    k: int,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
    seed: int,
    timeout: int,
    log_path: Path,
) -> None:
    command = [
        str(builder.resolve()),
        "--base-fbin",
        base_fbin,
        "--out-graph",
        out_graph,
        "--k",
        str(k),
        "--M",
        str(hnsw_m),
        "--ef-construction",
        str(ef_construction),
        "--ef-search",
        str(ef_search),
        "--seed",
        str(seed),
    ]
    result = run_external_binary(command, timeout=timeout)
    save_command(log_path, result)
    if not result.ok:
        raise RuntimeError(f"hnsw KNN graph builder failed; see {log_path}")
    print(result.stdout.rstrip())


def build_dynamic_nsg_assets(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    builder = compile_knn_builder(force=args.force_build, timeout=args.timeout)
    graph_k = int(args.k or manifest["nsg_graph_k"])

    build_knn_graph(
        builder=builder,
        base_fbin=paths["initial_fbin"],
        out_graph=paths["initial_nsg_knn_graph"],
        k=graph_k,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        seed=args.seed,
        timeout=args.timeout,
        log_path=args.manifest.parent / "initial_hnsw_knn_graph_builder_log.json",
    )
    build_knn_graph(
        builder=builder,
        base_fbin=paths["all_fbin"],
        out_graph=paths["all_nsg_knn_graph"],
        k=graph_k,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        seed=args.seed + 1,
        timeout=args.timeout,
        log_path=args.manifest.parent / "all_hnsw_knn_graph_builder_log.json",
    )

    initial_manifest = json.loads((args.manifest.parent / "initial_manifest.json").read_text(encoding="utf-8"))
    index_path = args.index or (ROOT / "results" / f"{args.manifest.parent.name}_dynamic_nsg" / "initial.nsg")
    result = run_nsg_build(
        initial_manifest,
        index_path=index_path,
        L=args.nsg_L,
        R=args.nsg_R,
        C=args.nsg_C,
        timeout=args.timeout,
    )
    save_command(index_path.parent / "initial_nsg_build_log.json", result)
    if not result.ok:
        raise RuntimeError(f"initial NSG build failed; see {index_path.parent / 'initial_nsg_build_log.json'}")

    manifest.setdefault("paths", {})["initial_nsg"] = str(index_path.resolve())
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Initial NSG: {index_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare real ANN datasets.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download-sift", help="Download and extract the official TexMex SIFT1M dataset.")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "sift")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("download-hdf5", help="Download an ANN-Benchmarks HDF5 dataset.")
    p.add_argument("--dataset", choices=sorted(ANN_BENCHMARK_URLS), required=True)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "ann_benchmarks")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("prepare-sift-dynamic", help="Prepare a real SIFT dynamic-drift benchmark subset.")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "sift")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "sift100k_dynamic")
    p.add_argument("--pool-size", type=int, default=200_000)
    p.add_argument("--n-initial", type=int, default=100_000)
    p.add_argument("--n-insert", type=int, default=20_000)
    p.add_argument("--n-delete", type=int, default=10_000)
    p.add_argument("--n-query", type=int, default=1_000)
    p.add_argument("--gt-k", type=int, default=100)
    p.add_argument("--nsg-graph-k", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--anchors", type=int, default=128)
    p.add_argument("--anchor-candidates", type=int, default=12_000)
    p.add_argument("--density-sample", type=int, default=2_048)
    p.add_argument("--density-k", type=int, default=16)
    p.add_argument("--seed", type=int, default=41)

    p = sub.add_parser("prepare-hdf5-dynamic", help="Prepare a dynamic benchmark subset from ANN-Benchmarks HDF5.")
    p.add_argument("--dataset", choices=sorted(ANN_BENCHMARK_URLS), required=True)
    p.add_argument("--hdf5", type=Path, default=None)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "ann_benchmarks")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pool-size", type=int, default=200_000)
    p.add_argument("--n-initial", type=int, default=100_000)
    p.add_argument("--n-insert", type=int, default=20_000)
    p.add_argument("--n-delete", type=int, default=10_000)
    p.add_argument("--n-query", type=int, default=1_000)
    p.add_argument("--gt-k", type=int, default=100)
    p.add_argument("--nsg-graph-k", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--anchors", type=int, default=128)
    p.add_argument("--anchor-candidates", type=int, default=12_000)
    p.add_argument("--density-sample", type=int, default=2_048)
    p.add_argument("--density-k", type=int, default=16)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--force-download", action="store_true")

    p = sub.add_parser("build-nsg-graph", help="Build an approximate KNN graph for NSG/GATE with hnswlib.")
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift100k_dynamic" / "manifest.json")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--hnsw-m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=300)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--force-build", action="store_true")

    p = sub.add_parser("build-dynamic-nsg-assets", help="Build initial/all KNN graphs and initial NSG for a dynamic manifest.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--hnsw-m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=300)
    p.add_argument("--nsg-L", type=int, default=40)
    p.add_argument("--nsg-R", type=int, default=50)
    p.add_argument("--nsg-C", type=int, default=500)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--force-build", action="store_true")

    args = parser.parse_args()
    if args.cmd == "download-sift":
        root = download_sift(args.raw_dir, force=args.force)
        print(f"SIFT root: {root}")
    elif args.cmd == "download-hdf5":
        path = download_hdf5(args.dataset, raw_dir=args.raw_dir, force=args.force)
        print(f"HDF5 path: {path}")
    elif args.cmd == "prepare-sift-dynamic":
        prepare_sift_dynamic(args)
    elif args.cmd == "prepare-hdf5-dynamic":
        prepare_hdf5_dynamic(args)
    elif args.cmd == "build-nsg-graph":
        build_nsg_graph(args)
    elif args.cmd == "build-dynamic-nsg-assets":
        build_dynamic_nsg_assets(args)


if __name__ == "__main__":
    main()
