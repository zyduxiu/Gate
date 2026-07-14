from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def read_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
        if header.size != 2:
            raise ValueError(f"invalid fbin header: {path}")
        rows, dim = int(header[0]), int(header[1])
        values = np.fromfile(handle, dtype="<f4", count=rows * dim)
    if values.size != rows * dim:
        raise ValueError(f"truncated fbin payload: {path}")
    return values.reshape(rows, dim).astype(np.float32, copy=False)


def read_u32_list(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u4")
    if raw.size == 0:
        return np.empty(0, dtype=np.uint32)
    count = int(raw[0])
    values = raw[1 : 1 + count]
    if values.size != count:
        raise ValueError(f"truncated u32 list: {path}")
    return values.astype(np.uint32, copy=False)


def read_anchor_file(path: Path) -> list[int]:
    if not path.exists():
        return []
    anchors: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            anchors.append(int(line))
    return anchors


def write_anchor_file(path: Path, anchors: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(x)) for x in anchors) + "\n", encoding="utf-8")


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.sum(a * a, axis=1, keepdims=True) + np.sum(b * b, axis=1).reshape(1, -1) - 2.0 * a @ b.T


def min_squared_distance_to_refs(vectors: np.ndarray, refs: np.ndarray, batch_size: int) -> np.ndarray:
    result = np.empty(vectors.shape[0], dtype=np.float64)
    for start in range(0, vectors.shape[0], batch_size):
        stop = min(start + batch_size, vectors.shape[0])
        dists = squared_distances(vectors[start:stop], refs)
        result[start:stop] = np.min(dists, axis=1)
    return result


def estimate_density(vectors: np.ndarray, refs: np.ndarray, density_k: int, batch_size: int) -> np.ndarray:
    density = np.empty(vectors.shape[0], dtype=np.float64)
    k = min(max(1, density_k), refs.shape[0])
    for start in range(0, vectors.shape[0], batch_size):
        stop = min(start + batch_size, vectors.shape[0])
        dists = np.maximum(squared_distances(vectors[start:stop], refs), 0.0)
        nearest = np.partition(dists, kth=k - 1, axis=1)[:, :k]
        density[start:stop] = 1.0 / (np.mean(np.sqrt(nearest), axis=1) + 1e-6)
    return density


def read_per_query_costs(path: Path, query_count: int) -> np.ndarray:
    if not path.exists():
        return np.ones(query_count, dtype=np.float64)
    costs = np.ones(query_count, dtype=np.float64)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qi = int(row["query_id"])
            if 0 <= qi < query_count:
                costs[qi] = float(row.get("distance_computations") or row.get("latency_us") or 1.0)
    return costs


def query_affinity(
    candidates: np.ndarray,
    queries: np.ndarray,
    query_weights: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    query_weights = np.asarray(query_weights, dtype=np.float64)
    affinity = np.empty(candidates.shape[0], dtype=np.float64)
    for start in range(0, candidates.shape[0], batch_size):
        stop = min(start + batch_size, candidates.shape[0])
        dists = np.maximum(squared_distances(candidates[start:stop], queries), 0.0)
        closeness = 1.0 / (np.sqrt(dists) + 1e-3)
        affinity[start:stop] = closeness @ query_weights
    return affinity


def ridge_fit(features: np.ndarray, target: np.ndarray, regularization: float) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    xtx = x.T @ x
    penalty = np.eye(xtx.shape[0], dtype=np.float64) * float(regularization)
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, x.T @ y)


def choose_candidates(
    labels: np.ndarray,
    initial_rows: int,
    anchors: list[int],
    candidate_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alive_set = set(int(y) for y in labels)
    label_set = set(int(x) for x in anchors if int(x) in alive_set)
    inserted = labels[labels >= initial_rows]
    inserted_take = min(inserted.size, max(candidate_count // 4, 1))
    if inserted_take:
        label_set.update(int(x) for x in rng.choice(inserted, size=inserted_take, replace=False))
    remaining_count = max(0, candidate_count - len(label_set))
    if remaining_count:
        remaining = np.asarray([int(x) for x in labels if int(x) not in label_set], dtype=np.uint32)
        take = min(remaining_count, remaining.size)
        if take:
            label_set.update(int(x) for x in rng.choice(remaining, size=take, replace=False))
    return np.asarray(sorted(label_set), dtype=np.uint32)


def greedy_select(
    candidate_labels: np.ndarray,
    candidate_vectors: np.ndarray,
    learned_score: np.ndarray,
    density: np.ndarray,
    stale_distance: np.ndarray,
    count: int,
    coverage_weight: float,
    learned_weight: float,
    density_weight: float,
    stale_weight: float,
    initial_rows: int,
    initial_anchors: list[int] | None = None,
    max_inserted_fraction: float = 1.0,
) -> list[int]:
    label_to_pos = {int(label): pos for pos, label in enumerate(candidate_labels)}
    selected_pos: list[int] = []
    seen: set[int] = set()
    for anchor in initial_anchors or []:
        pos = label_to_pos.get(int(anchor))
        if pos is None or pos in seen:
            continue
        selected_pos.append(pos)
        seen.add(pos)
        if len(selected_pos) >= count:
            break
    if not selected_pos:
        selected_pos = [int(np.argmax(learned_score))]
        seen.add(selected_pos[0])

    max_inserted = int(np.floor(float(count) * float(max_inserted_fraction)))
    max_inserted = max(0, min(int(count), max_inserted))
    inserted_mask = candidate_labels >= initial_rows
    min_dist = np.min(squared_distances(candidate_vectors, candidate_vectors[selected_pos]), axis=1)
    while len(selected_pos) < min(count, candidate_labels.size):
        coverage = normalize(min_dist)
        score = (
            coverage_weight * coverage
            + learned_weight * learned_score
            + density_weight * density
            + stale_weight * stale_distance
        )
        score[np.asarray(selected_pos, dtype=np.int64)] = -1.0
        selected_inserted = int(np.sum(inserted_mask[np.asarray(selected_pos, dtype=np.int64)]))
        if max_inserted_fraction < 1.0 and selected_inserted >= max_inserted:
            constrained = score.copy()
            constrained[inserted_mask] = -1.0
            if float(np.max(constrained)) >= 0.0:
                score = constrained
        next_pos = int(np.argmax(score))
        if next_pos in seen or float(score[next_pos]) < 0.0:
            break
        selected_pos.append(next_pos)
        seen.add(next_pos)
        min_dist = np.minimum(min_dist, squared_distances(candidate_vectors, candidate_vectors[[next_pos]]).reshape(-1))
    return [int(candidate_labels[pos]) for pos in selected_pos]


def remap_to_compact(anchors: list[int], dynamic_labels: np.ndarray) -> list[int]:
    mapping = {int(label): pos for pos, label in enumerate(dynamic_labels)}
    compact: list[int] = []
    seen: set[int] = set()
    for anchor in anchors:
        mapped = mapping.get(int(anchor))
        if mapped is None or mapped in seen:
            continue
        seen.add(mapped)
        compact.append(int(mapped))
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn a workload-aware dynamic entry-anchor scorer.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift100k_dynamic" / "manifest.json")
    parser.add_argument("--method-name", default="learned_dynamic")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--external-out", type=Path, default=None)
    parser.add_argument("--baseline-perq", type=Path, default=ROOT / "results" / "sift100k_dynamic_nsg_entry" / "nsg_ep_L240_perq.csv")
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--candidate-count", type=int, default=24000)
    parser.add_argument("--density-sample", type=int, default=5000)
    parser.add_argument("--density-k", type=int, default=16)
    parser.add_argument("--train-query-fraction", type=float, default=0.60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--coverage-weight", type=float, default=0.45)
    parser.add_argument("--learned-weight", type=float, default=0.40)
    parser.add_argument("--density-weight", type=float, default=0.10)
    parser.add_argument("--stale-weight", type=float, default=0.05)
    parser.add_argument("--keep-anchor-source", choices=["none", "static_density", "dynamic_density"], default="none")
    parser.add_argument("--keep-anchors", type=int, default=0)
    parser.add_argument("--max-inserted-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--update-manifest", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    anchor_files = manifest.get("anchor_files", {})
    anchors_dir = Path(anchor_files.get("dynamic_density", args.manifest.parent / "anchors" / "dynamic_density.txt")).parent
    out_path = args.out or (anchors_dir / f"{args.method_name}.txt")
    meta_path = args.meta or out_path.with_suffix(".meta.json")
    external_path = args.external_out or (anchors_dir / f"external_{args.method_name}.txt")

    initial = read_fbin(Path(paths["initial_fbin"]))
    inserts = read_fbin(Path(paths["insert_fbin"]))
    queries = read_fbin(Path(paths["query_fbin"]))
    deletes = read_u32_list(Path(paths["delete_u32"]))

    all_vectors = np.vstack([initial, inserts]).astype(np.float32, copy=False)
    alive = np.ones(all_vectors.shape[0], dtype=bool)
    alive[deletes.astype(np.int64)] = False
    dynamic_labels = np.flatnonzero(alive).astype(np.uint32)

    static_anchors = read_anchor_file(Path(anchor_files.get("static_density", ""))) if "static_density" in anchor_files else []
    dynamic_anchors = read_anchor_file(Path(anchor_files.get("dynamic_density", ""))) if "dynamic_density" in anchor_files else []
    fixed_anchors = read_anchor_file(Path(anchor_files.get("fixed_medoid", ""))) if "fixed_medoid" in anchor_files else []

    candidate_labels = choose_candidates(
        dynamic_labels,
        initial_rows=initial.shape[0],
        anchors=static_anchors + dynamic_anchors + fixed_anchors,
        candidate_count=args.candidate_count,
        seed=args.seed,
    )
    candidate_vectors = all_vectors[candidate_labels]

    rng = np.random.default_rng(args.seed + 1)
    ref_labels = rng.choice(dynamic_labels, size=min(args.density_sample, dynamic_labels.size), replace=False)
    ref_vectors = all_vectors[ref_labels]
    density = normalize(estimate_density(candidate_vectors, ref_vectors, args.density_k, args.batch_size))

    if static_anchors:
        static_vectors = all_vectors[np.asarray(static_anchors, dtype=np.uint32)]
        stale_distance = normalize(min_squared_distance_to_refs(candidate_vectors, static_vectors, args.batch_size))
    else:
        stale_distance = np.zeros(candidate_vectors.shape[0], dtype=np.float64)

    if fixed_anchors:
        fixed_vectors = all_vectors[np.asarray(fixed_anchors, dtype=np.uint32)]
        medoid_distance = normalize(min_squared_distance_to_refs(candidate_vectors, fixed_vectors, args.batch_size))
    else:
        medoid_distance = np.zeros(candidate_vectors.shape[0], dtype=np.float64)

    inserted_flag = (candidate_labels >= initial.shape[0]).astype(np.float64)
    projection = normalize(np.mean(candidate_vectors[:, : min(16, candidate_vectors.shape[1])], axis=1))

    costs = read_per_query_costs(args.baseline_perq, queries.shape[0])
    order = np.argsort(costs)
    train_count = max(1, min(queries.shape[0], int(round(queries.shape[0] * args.train_query_fraction))))
    train_ids = np.sort(order[-train_count:])
    train_queries = queries[train_ids]
    train_costs = normalize(costs[train_ids])
    query_weights = np.maximum(train_costs - float(np.median(train_costs)), 0.0) ** 2
    if float(np.sum(query_weights)) <= 1e-12:
        query_weights = np.ones(train_queries.shape[0], dtype=np.float64)
    query_weights = query_weights / float(np.sum(query_weights))
    affinity = normalize(query_affinity(candidate_vectors, train_queries, query_weights, args.batch_size))

    features = np.column_stack(
        [
            np.ones(candidate_vectors.shape[0], dtype=np.float64),
            density,
            stale_distance,
            inserted_flag,
            projection,
            medoid_distance,
        ]
    )
    weights = ridge_fit(features, affinity, regularization=args.regularization)
    learned_score = normalize(features @ weights)

    if args.keep_anchor_source == "dynamic_density":
        warm_source = dynamic_anchors
    elif args.keep_anchor_source == "static_density":
        warm_source = static_anchors
    else:
        warm_source = []
    warm_anchors = warm_source[: max(0, args.keep_anchors)]

    selected = greedy_select(
        candidate_labels=candidate_labels,
        candidate_vectors=candidate_vectors,
        learned_score=learned_score,
        density=density,
        stale_distance=stale_distance,
        count=args.anchors,
        coverage_weight=args.coverage_weight,
        learned_weight=args.learned_weight,
        density_weight=args.density_weight,
        stale_weight=args.stale_weight,
        initial_rows=initial.shape[0],
        initial_anchors=warm_anchors,
        max_inserted_fraction=args.max_inserted_fraction,
    )
    write_anchor_file(out_path, selected)
    write_anchor_file(external_path, remap_to_compact(selected, dynamic_labels))

    meta: dict[str, Any] = {
        "method": args.method_name,
        "manifest": str(args.manifest.resolve()),
        "out": str(out_path.resolve()),
        "external_out": str(external_path.resolve()),
        "baseline_perq": str(args.baseline_perq.resolve()),
        "anchors": len(selected),
        "candidate_count": int(candidate_labels.size),
        "train_queries": int(train_queries.shape[0]),
        "feature_names": ["bias", "density", "stale_static_distance", "inserted_flag", "projection", "medoid_distance"],
        "ridge_weights": [float(x) for x in weights],
        "selected_inserted": int(sum(1 for anchor in selected if anchor >= initial.shape[0])),
        "selected_from_static": int(sum(1 for anchor in selected if anchor in set(static_anchors))),
        "selected_from_dynamic": int(sum(1 for anchor in selected if anchor in set(dynamic_anchors))),
        "warm_start_source": args.keep_anchor_source,
        "warm_start_requested": int(args.keep_anchors),
        "warm_start_used": int(sum(1 for anchor in selected if anchor in set(warm_anchors))),
        "score_summary": {
            "learned_min": float(np.min(learned_score)),
            "learned_mean": float(np.mean(learned_score)),
            "learned_max": float(np.max(learned_score)),
            "affinity_min": float(np.min(affinity)),
            "affinity_mean": float(np.mean(affinity)),
            "affinity_max": float(np.max(affinity)),
        },
        "config": {
            "density_k": args.density_k,
            "density_sample": args.density_sample,
            "train_query_fraction": args.train_query_fraction,
            "coverage_weight": args.coverage_weight,
            "learned_weight": args.learned_weight,
            "density_weight": args.density_weight,
            "stale_weight": args.stale_weight,
            "keep_anchor_source": args.keep_anchor_source,
            "keep_anchors": args.keep_anchors,
            "max_inserted_fraction": args.max_inserted_fraction,
            "seed": args.seed,
        },
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.update_manifest:
        manifest.setdefault("anchor_files", {})[args.method_name] = str(out_path.resolve())
        manifest.setdefault("external_anchor_files", {})[args.method_name] = str(external_path.resolve())
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Wrote {external_path}")
    print(f"Wrote {meta_path}")
    print(json.dumps({k: meta[k] for k in ["anchors", "candidate_count", "train_queries", "selected_inserted", "selected_from_static", "selected_from_dynamic"]}, indent=2))


if __name__ == "__main__":
    main()
