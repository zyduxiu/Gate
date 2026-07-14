from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.io_formats import read_fvecs, write_ivecs


def read_anchor_file(path: Path) -> np.ndarray:
    return np.asarray([int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], dtype=np.int32)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def build_knn_graph(vectors: np.ndarray, degree: int) -> np.ndarray:
    vectors = normalize(vectors.astype(np.float32, copy=False))
    scores = vectors @ vectors.T
    np.fill_diagonal(scores, -np.inf)
    degree = min(degree, max(1, vectors.shape[0] - 1))
    part = np.argpartition(-scores, kth=degree - 1, axis=1)[:, :degree]
    row = np.arange(vectors.shape[0])[:, None]
    return part[row, np.argsort(-scores[row, part], axis=1)].astype(np.int32)


def navigate_one(
    query: np.ndarray,
    hub_vectors: np.ndarray,
    graph: np.ndarray,
    starts: np.ndarray,
    entries: int,
    ef: int,
    hops: int,
) -> list[int]:
    scores = hub_vectors @ query
    visited: set[int] = set()
    frontier: list[int] = [int(x) for x in starts]
    best: list[int] = []
    max_steps = max(1, int(hops)) if hops > 0 else ef
    steps = 0
    while frontier and len(visited) < ef and steps < max_steps:
        frontier = sorted((x for x in frontier if x not in visited), key=lambda x: float(scores[x]), reverse=True)
        if not frontier:
            break
        node = frontier.pop(0)
        visited.add(node)
        best.append(node)
        steps += 1
        for neigh in graph[node]:
            n = int(neigh)
            if n not in visited:
                frontier.append(n)
    if len(best) < entries:
        order = np.argsort(-scores)
        for idx in order:
            i = int(idx)
            if i not in best:
                best.append(i)
            if len(best) >= entries:
                break
    best = sorted(set(best), key=lambda x: float(scores[x]), reverse=True)[:entries]
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GATE high-level navigation query entries from learned hub/query embeddings.")
    parser.add_argument("--hub-embeddings", type=Path, required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--anchor-file", type=Path, required=True)
    parser.add_argument("--out-ivecs", type=Path, required=True)
    parser.add_argument("--out-graph", type=Path, default=None)
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--ef", type=int, default=64)
    parser.add_argument("--hops", type=int, default=0, help="Limit high-level GATE routing hops; 0 uses ef-only traversal.")
    parser.add_argument("--start-count", type=int, default=3)
    args = parser.parse_args()

    hub_embeddings = normalize(read_fvecs(args.hub_embeddings))
    query_embeddings = normalize(read_fvecs(args.query_embeddings))
    anchors = read_anchor_file(args.anchor_file)
    if hub_embeddings.shape[0] != anchors.size:
        raise ValueError(f"hub embedding rows {hub_embeddings.shape[0]} != anchor count {anchors.size}")
    graph = build_knn_graph(hub_embeddings, args.degree)
    global_scores = np.mean(hub_embeddings @ query_embeddings[: min(128, query_embeddings.shape[0])].T, axis=1)
    starts = np.argsort(-global_scores)[: max(1, args.start_count)].astype(np.int32)
    entries = np.empty((query_embeddings.shape[0], args.entries), dtype=np.int32)
    for qi, query in enumerate(query_embeddings):
        hub_local = navigate_one(query, hub_embeddings, graph, starts, args.entries, args.ef, args.hops)
        entries[qi] = anchors[np.asarray(hub_local, dtype=np.int64)]
    write_ivecs(args.out_ivecs, entries)
    if args.out_graph is not None:
        write_ivecs(args.out_graph, graph)
    meta = {
        "hub_embeddings": str(args.hub_embeddings.resolve()),
        "query_embeddings": str(args.query_embeddings.resolve()),
        "anchor_file": str(args.anchor_file.resolve()),
        "out_ivecs": str(args.out_ivecs.resolve()),
        "entries": args.entries,
        "degree": args.degree,
        "ef": args.ef,
        "hops": args.hops,
        "start_count": args.start_count,
        "note": "GATE high-level hub graph navigation over learned hub/query embeddings; outputs original graph entry ids.",
    }
    (args.out_ivecs.parent / "gate_navigation_entries_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
