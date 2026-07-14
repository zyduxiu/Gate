from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import real_datasets as rd
from run_gate_adapter import build_gate_hubs


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def estimate_density_batched(vectors: np.ndarray, refs: np.ndarray, density_k: int, batch_size: int) -> np.ndarray:
    density = np.empty(vectors.shape[0], dtype=np.float64)
    k = min(max(1, density_k), refs.shape[0])
    ref_norm = np.sum(refs * refs, axis=1)
    for start in range(0, vectors.shape[0], batch_size):
        stop = min(start + batch_size, vectors.shape[0])
        batch = vectors[start:stop]
        dists = np.sum(batch * batch, axis=1, keepdims=True) + ref_norm.reshape(1, -1) - 2.0 * batch @ refs.T
        nearest = np.partition(np.maximum(dists, 0.0), kth=k - 1, axis=1)[:, :k]
        density[start:stop] = 1.0 / (np.mean(np.sqrt(nearest), axis=1) + 1e-6)
    return density


def choose_dense_insert_region(
    base_pool: np.ndarray,
    n_insert: int,
    high_pool: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = rd.projection_score(base_pool)
    high_pool = min(max(high_pool, n_insert), base_pool.shape[0])
    high_ids = np.argsort(scores)[-high_pool:]
    high_vectors = base_pool[high_ids]
    probe_size = min(2048, high_vectors.shape[0])
    probe = high_vectors[rng.choice(high_vectors.shape[0], size=probe_size, replace=False)]
    center = np.mean(probe, axis=0, dtype=np.float64).astype(np.float32)
    dists = np.sum((high_vectors - center.reshape(1, -1)) ** 2, axis=1)
    chosen = high_ids[np.argsort(dists)[:n_insert]]
    return chosen.astype(np.int64)


def choose_initial_pool(
    base_pool: np.ndarray,
    insert_ids: np.ndarray,
    n_initial: int,
    initial_quantile: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = rd.projection_score(base_pool)
    threshold = np.quantile(scores, initial_quantile)
    insert_set = set(int(x) for x in insert_ids)
    candidates = np.asarray(
        [idx for idx, score in enumerate(scores) if score <= threshold and idx not in insert_set],
        dtype=np.int64,
    )
    if candidates.size < n_initial:
        candidates = np.asarray([idx for idx in range(base_pool.shape[0]) if idx not in insert_set], dtype=np.int64)
    if candidates.size < n_initial:
        raise ValueError("not enough initial candidates after excluding insert region")
    return rng.choice(candidates, size=n_initial, replace=False).astype(np.int64)


def choose_high_density_deletes(
    initial: np.ndarray,
    n_delete: int,
    density_sample: int,
    density_k: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ref_count = min(density_sample, initial.shape[0])
    refs = initial[rng.choice(initial.shape[0], size=ref_count, replace=False)]
    density = normalize(estimate_density_batched(initial, refs, density_k=density_k, batch_size=batch_size))
    projection = normalize(rd.projection_score(initial))
    score = 0.75 * density + 0.25 * projection
    delete_ids = np.argsort(score)[-n_delete:]
    return np.sort(delete_ids).astype(np.uint32)


def choose_stress_queries(
    query_pool: np.ndarray,
    insert_vectors: np.ndarray,
    n_query: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    center = np.mean(insert_vectors, axis=0, dtype=np.float64).astype(np.float32)
    dists = np.sum((query_pool - center.reshape(1, -1)) ** 2, axis=1)
    near_count = n_query // 2
    near = np.argsort(dists)[:near_count]
    high_count = n_query // 4
    high = np.argsort(rd.projection_score(query_pool))[-high_count:]
    used = set(int(x) for x in np.concatenate([near, high]))
    remaining = np.asarray([idx for idx in range(query_pool.shape[0]) if idx not in used], dtype=np.int64)
    random_count = n_query - len(used)
    random_part = rng.choice(remaining, size=random_count, replace=False) if random_count > 0 else np.empty(0, dtype=np.int64)
    chosen = np.asarray(list(used) + [int(x) for x in random_part], dtype=np.int64)
    rng.shuffle(chosen)
    return query_pool[chosen].astype(np.float32, copy=True)


def parse_gate_hubs(hub_path: Path, label_map: np.ndarray | None = None) -> list[int]:
    lines = [line.strip() for line in hub_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = lines[0].split()
    stage1 = int(header[0])
    stage2 = int(header[1])
    hub_rows = lines[1 + stage1 : 1 + stage1 + stage1]
    anchors: list[int] = []
    seen: set[int] = set()
    for row in hub_rows:
        for value in row.split()[:stage2]:
            anchor = int(value)
            if label_map is not None:
                anchor = int(label_map[anchor])
            if anchor not in seen:
                seen.add(anchor)
                anchors.append(anchor)
    return anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a harsh SIFT streaming-drift benchmark for entry staleness.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "sift")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "sift_stress_drift")
    parser.add_argument("--pool-size", type=int, default=300000)
    parser.add_argument("--n-initial", type=int, default=100000)
    parser.add_argument("--n-insert", type=int, default=50000)
    parser.add_argument("--n-delete", type=int, default=30000)
    parser.add_argument("--n-query", type=int, default=1000)
    parser.add_argument("--gt-k", type=int, default=100)
    parser.add_argument("--nsg-graph-k", type=int, default=100)
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--anchor-candidates", type=int, default=30000)
    parser.add_argument("--density-sample", type=int, default=6000)
    parser.add_argument("--density-k", type=int, default=16)
    parser.add_argument("--initial-quantile", type=float, default=0.65)
    parser.add_argument("--insert-high-pool", type=int, default=120000)
    parser.add_argument("--gate-stage1", type=int, default=16)
    parser.add_argument("--gate-stage2", type=int, default=8)
    parser.add_argument("--gate-iterations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=91)
    args = parser.parse_args()

    sift_root = rd.find_sift_root(args.raw_dir)
    base_pool = rd.read_fvecs_slice(sift_root / "sift_base.fvecs", limit=args.pool_size)
    query_pool = rd.read_fvecs_slice(sift_root / "sift_query.fvecs", limit=max(10000, args.n_query))

    insert_idx = choose_dense_insert_region(base_pool, args.n_insert, args.insert_high_pool, args.seed)
    initial_idx = choose_initial_pool(base_pool, insert_idx, args.n_initial, args.initial_quantile, args.seed + 1)
    initial = base_pool[initial_idx].astype(np.float32, copy=True)
    inserts = base_pool[insert_idx].astype(np.float32, copy=True)
    delete_ids = choose_high_density_deletes(
        initial,
        args.n_delete,
        density_sample=args.density_sample,
        density_k=args.density_k,
        batch_size=args.batch_size,
        seed=args.seed + 2,
    )
    queries = choose_stress_queries(query_pool, inserts, args.n_query, args.seed + 3)

    all_vectors = np.vstack([initial, inserts]).astype(np.float32)
    alive = np.ones(all_vectors.shape[0], dtype=bool)
    alive[delete_ids.astype(np.int64)] = False
    dynamic_labels = np.flatnonzero(alive).astype(np.uint32)
    final_base = all_vectors[dynamic_labels]
    external_labels = np.arange(final_base.shape[0], dtype=np.uint32)

    truth_dynamic = rd.exact_knn_labels(final_base, dynamic_labels, queries, args.gt_k, args.batch_size)
    truth_external = rd.exact_knn_labels(final_base, external_labels, queries, args.gt_k, args.batch_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    anchors_dir = args.out_dir / "anchors"
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
    rd.write_fvecs(paths["initial_fvecs"], initial)
    rd.write_diskann_fbin(paths["initial_fbin"], initial)
    rd.write_fvecs(paths["insert_fvecs"], inserts)
    rd.write_diskann_fbin(paths["insert_fbin"], inserts)
    rd.write_diskann_fbin(paths["all_fbin"], all_vectors)
    rd.write_diskann_fbin(paths["query_fbin"], queries)
    rd.write_diskann_gt(paths["truth_dynamic_bin"], truth_dynamic)
    rd.write_u32_list(paths["delete_u32"], delete_ids)
    rd.write_fvecs(paths["base_fvecs"], final_base)
    rd.write_fvecs(paths["query_fvecs"], queries)
    rd.write_ivecs(paths["truth_ivecs"], truth_external.astype(np.int32))
    rd.write_diskann_fbin(paths["base_fbin"], final_base)
    rd.write_diskann_gt(paths["truth_bin"], truth_external)

    fixed = [rd.medoid_label(initial, np.arange(initial.shape[0], dtype=np.uint32))]
    static = rd.select_density_labels(
        initial,
        np.arange(initial.shape[0], dtype=np.uint32),
        count=args.anchors,
        seed=args.seed + 4,
        candidate_count=args.anchor_candidates,
        density_sample=args.density_sample,
        density_k=args.density_k,
    )
    dynamic = rd.select_density_labels(
        final_base,
        dynamic_labels,
        count=args.anchors,
        seed=args.seed + 5,
        candidate_count=args.anchor_candidates,
        density_sample=args.density_sample,
        density_k=args.density_k,
    )

    gate_dir = args.out_dir / "gate_hubs"
    initial_hub_file = gate_dir / "gate_initial_hubs.txt"
    initial_emb_file = gate_dir / "gate_initial_hub_embeddings.fvecs"
    final_hub_file = gate_dir / "gate_refreshed_hubs.txt"
    final_emb_file = gate_dir / "gate_refreshed_hub_embeddings.fvecs"
    build_gate_hubs(initial, initial_hub_file, initial_emb_file, args.gate_stage1, args.gate_stage2, args.gate_iterations, args.seed + 6)
    build_gate_hubs(final_base, final_hub_file, final_emb_file, args.gate_stage1, args.gate_stage2, args.gate_iterations, args.seed + 7)
    gate_initial = parse_gate_hubs(initial_hub_file)
    gate_refreshed = parse_gate_hubs(final_hub_file, label_map=dynamic_labels)

    anchor_files = {
        "fixed_medoid": anchors_dir / "fixed_medoid.txt",
        "static_density": anchors_dir / "static_density.txt",
        "dynamic_density": anchors_dir / "dynamic_density.txt",
        "gate_initial_hub": anchors_dir / "gate_initial_hub.txt",
        "gate_refreshed_hub": anchors_dir / "gate_refreshed_hub.txt",
    }
    external_anchor_files = {
        "fixed_medoid": anchors_dir / "external_fixed_medoid.txt",
        "static_density": anchors_dir / "external_static_density.txt",
        "dynamic_density": anchors_dir / "external_dynamic_density.txt",
        "gate_initial_hub": anchors_dir / "external_gate_initial_hub.txt",
        "gate_refreshed_hub": anchors_dir / "external_gate_refreshed_hub.txt",
    }
    anchors_by_name = {
        "fixed_medoid": fixed,
        "static_density": static,
        "dynamic_density": dynamic,
        "gate_initial_hub": gate_initial,
        "gate_refreshed_hub": gate_refreshed,
    }
    for key, values in anchors_by_name.items():
        rd.write_anchor_file(anchor_files[key], values)
        rd.write_anchor_file(external_anchor_files[key], rd.remap_anchors_to_compact(values, dynamic_labels))

    manifest = {
        "dataset": "sift1m_stress_streaming_drift",
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
            "raw_dir": str(args.raw_dir.resolve()),
            "base_pool_rows": int(args.pool_size),
            "drift": "dense high-projection insert region; high-density initial-region deletion; queries biased toward inserted region",
        },
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
        "anchor_files": {key: str(path.resolve()) for key, path in anchor_files.items()},
        "external_anchor_files": {key: str(path.resolve()) for key, path in external_anchor_files.items()},
        "gate_hub_files": {
            "gate_initial_hub_file": str(initial_hub_file.resolve()),
            "gate_refreshed_hub_file": str(final_hub_file.resolve()),
        },
        "manifest": str((args.out_dir / "manifest.json").resolve()),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    initial_manifest = {
        "dataset": "sift1m_stress_initial_subset",
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


if __name__ == "__main__":
    main()
