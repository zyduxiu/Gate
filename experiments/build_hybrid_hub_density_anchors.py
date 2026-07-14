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


def read_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
        if header.size != 2:
            raise ValueError(f"invalid fbin header: {path}")
        rows, dim = int(header[0]), int(header[1])
        data = np.fromfile(handle, dtype="<f4", count=rows * dim)
    if data.size != rows * dim:
        raise ValueError(f"truncated fbin: {path}")
    return data.reshape(rows, dim).astype(np.float32, copy=False)


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i4")
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    return rows[:, 1:].astype(np.int32, copy=False)


def read_u32_list(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u4")
    if raw.size == 0:
        return np.empty(0, dtype=np.uint32)
    count = int(raw[0])
    return raw[1 : 1 + count].astype(np.uint32, copy=False)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * a, axis=1, keepdims=True) + np.sum(b * b, axis=1).reshape(1, -1) - 2.0 * a @ b.T


def estimate_density(vectors: np.ndarray, refs: np.ndarray, density_k: int, batch_size: int) -> np.ndarray:
    out = np.empty(vectors.shape[0], dtype=np.float64)
    k = min(max(1, density_k), refs.shape[0])
    for start in range(0, vectors.shape[0], batch_size):
        stop = min(start + batch_size, vectors.shape[0])
        dists = np.maximum(squared_distances(vectors[start:stop], refs), 0.0)
        nearest = np.partition(dists, kth=k - 1, axis=1)[:, :k]
        out[start:stop] = 1.0 / (np.mean(np.sqrt(nearest), axis=1) + 1e-6)
    return out


def compute_indegree(knn: np.ndarray, alive: np.ndarray) -> np.ndarray:
    indegree = np.zeros(alive.size, dtype=np.float64)
    alive_rows = np.flatnonzero(alive)
    for row_id in alive_rows:
        for neigh in knn[int(row_id)]:
            if 0 <= int(neigh) < alive.size and alive[int(neigh)]:
                indegree[int(neigh)] += 1.0
    return indegree


def greedy_select(
    candidate_labels: np.ndarray,
    candidate_vectors: np.ndarray,
    base_score: np.ndarray,
    count: int,
    coverage_weight: float,
) -> list[int]:
    selected_pos = [int(np.argmax(base_score))]
    min_dist = np.maximum(squared_distances(candidate_vectors, candidate_vectors[selected_pos]), 0.0).reshape(-1)
    while len(selected_pos) < min(count, candidate_labels.size):
        score = coverage_weight * normalize(min_dist) + base_score
        score[np.asarray(selected_pos, dtype=np.int64)] = -1.0
        next_pos = int(np.argmax(score))
        if float(score[next_pos]) < 0.0:
            break
        selected_pos.append(next_pos)
        min_dist = np.minimum(min_dist, np.maximum(squared_distances(candidate_vectors, candidate_vectors[[next_pos]]), 0.0).reshape(-1))
    return [int(candidate_labels[pos]) for pos in selected_pos]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build density+hubness entry anchors for no-rebuild graph search.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift100k_dynamic" / "manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sift100k_dynamic" / "anchors" / "hybrid_hub_density.txt")
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--candidate-count", type=int, default=30000)
    parser.add_argument("--density-sample", type=int, default=6000)
    parser.add_argument("--density-k", type=int, default=16)
    parser.add_argument("--coverage-weight", type=float, default=0.45)
    parser.add_argument("--density-weight", type=float, default=0.20)
    parser.add_argument("--hub-weight", type=float, default=0.35)
    parser.add_argument("--insert-weight", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    initial = read_fbin(Path(paths["initial_fbin"]))
    inserts = read_fbin(Path(paths["insert_fbin"]))
    vectors = np.vstack([initial, inserts]).astype(np.float32, copy=False)
    deletes = read_u32_list(Path(paths["delete_u32"]))
    alive = np.ones(vectors.shape[0], dtype=bool)
    alive[deletes.astype(np.int64)] = False
    alive_labels = np.flatnonzero(alive).astype(np.int64)

    knn = read_ivecs(Path(paths["all_nsg_knn_graph"]))
    indegree = compute_indegree(knn, alive)

    pinned: list[int] = []
    for key in ("dynamic_density", "static_density"):
        anchor_path = manifest.get("anchor_files", {}).get(key)
        if anchor_path:
            for line in Path(anchor_path).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    value = int(line)
                    if 0 <= value < alive.size and alive[value]:
                        pinned.append(value)

    pinned = list(dict.fromkeys(pinned))
    remaining = np.asarray([x for x in alive_labels if int(x) not in set(pinned)], dtype=np.int64)
    take = min(max(0, args.candidate_count - len(pinned)), remaining.size)
    sampled = rng.choice(remaining, size=take, replace=False) if take else np.empty(0, dtype=np.int64)
    candidate_labels = np.asarray(pinned + [int(x) for x in sampled], dtype=np.int64)
    candidate_vectors = vectors[candidate_labels]

    refs = rng.choice(alive_labels, size=min(args.density_sample, alive_labels.size), replace=False)
    density = normalize(estimate_density(candidate_vectors, vectors[refs], args.density_k, args.batch_size))
    hubness = normalize(indegree[candidate_labels])
    inserted = (candidate_labels >= initial.shape[0]).astype(np.float64)
    base_score = args.density_weight * density + args.hub_weight * hubness + args.insert_weight * inserted

    selected = greedy_select(
        candidate_labels=candidate_labels,
        candidate_vectors=candidate_vectors,
        base_score=base_score,
        count=args.anchors,
        coverage_weight=args.coverage_weight,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(x) for x in selected) + "\n", encoding="ascii")
    meta_path = args.meta or args.out.with_suffix(".meta.json")
    meta = {
        "method": "hybrid_hub_density",
        "anchors": len(selected),
        "candidate_count": int(candidate_labels.size),
        "density_weight": args.density_weight,
        "hub_weight": args.hub_weight,
        "coverage_weight": args.coverage_weight,
        "insert_weight": args.insert_weight,
        "selected_inserted": int(sum(1 for x in selected if x >= initial.shape[0])),
        "selected_from_dynamic_density": int(
            sum(1 for x in selected if x in set(int(v) for v in pinned[: len(pinned) // 2]))
        ),
        "note": "Hybrid entry anchors selected by graph indegree hubness, local density, inserted-region bias, and greedy coverage.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
