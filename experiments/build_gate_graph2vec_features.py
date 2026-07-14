from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize as sk_normalize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.io_formats import read_fvecs, write_fvecs


def stable_hash(text: str, size: int = 16) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=size).hexdigest()


def read_nsg_index(path: Path) -> list[np.ndarray]:
    raw = np.fromfile(path, dtype="<i4")
    if raw.size < 2:
        raise ValueError(f"invalid NSG index: {path}")
    pos = 2
    graph: list[np.ndarray] = []
    while pos < raw.size:
        degree = int(raw[pos])
        pos += 1
        if degree < 0 or pos + degree > raw.size:
            raise ValueError(f"corrupt NSG graph at row {len(graph)} in {path}")
        graph.append(raw[pos : pos + degree].astype(np.int32, copy=True))
        pos += degree
    return graph


def read_gate_kmeans_eps(path: Path) -> tuple[int, int, int, np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    iter1_num, iter2_num, dim = map(int, lines[0].split())
    eps_lines = lines[1 + iter1_num : 1 + iter1_num + iter1_num]
    eps: list[int] = []
    for line in eps_lines:
        row = [int(x) for x in line.split()]
        if len(row) != iter2_num:
            raise ValueError(f"bad eps row width in {path}: expected {iter2_num}, got {len(row)}")
        eps.extend(row)
    return iter1_num, iter2_num, dim, np.asarray(eps, dtype=np.int32)


def projection_codes(base: np.ndarray, bits: int, seed: int) -> np.ndarray:
    if bits <= 0:
        return np.asarray([""] * base.shape[0], dtype=object)
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(base.shape[1], bits)).astype(np.float32)
    scores = base @ proj
    codes = []
    for row in scores > 0:
        codes.append("".join("1" if bool(x) else "0" for x in row))
    return np.asarray(codes, dtype=object)


def sample_neighbors(center: int, neighbors: np.ndarray, base: np.ndarray, sample_size: int) -> list[int]:
    valid = [int(x) for x in neighbors if 0 <= int(x) < base.shape[0] and int(x) != center]
    if len(valid) <= sample_size:
        return valid
    vec = base[center]
    neigh = base[np.asarray(valid, dtype=np.int64)]
    dists = np.sum((neigh - vec.reshape(1, -1)) ** 2, axis=1)
    order = np.argsort(dists)
    near = [valid[int(i)] for i in order[: sample_size // 2]]
    far = [valid[int(i)] for i in order[-(sample_size - len(near)) :]]
    merged: list[int] = []
    seen: set[int] = set()
    for node in near + far:
        if node not in seen:
            merged.append(node)
            seen.add(node)
    return merged


def sample_subgraph(
    ep: int,
    graph: list[np.ndarray],
    base: np.ndarray,
    max_hop: int,
    sample_size: int,
    max_nodes: int,
) -> tuple[list[int], dict[int, list[int]], dict[int, int]]:
    visited: set[int] = set()
    hops: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque([(int(ep), 0)])
    while queue and len(visited) < max_nodes:
        node, hop = queue.popleft()
        if node in visited or node < 0 or node >= len(graph) or hop > max_hop:
            continue
        visited.add(node)
        hops[node] = hop
        if hop == max_hop:
            continue
        for nxt in sample_neighbors(node, graph[node], base, sample_size):
            if nxt not in visited:
                queue.append((nxt, hop + 1))

    nodes = sorted(visited)
    node_set = set(nodes)
    adjacency: dict[int, list[int]] = {}
    for node in nodes:
        adjacency[node] = [int(x) for x in graph[node] if int(x) in node_set]
    return nodes, adjacency, hops


def bucket_degree(degree: int) -> str:
    return str(int(math.log2(max(1, degree)) + 0.5))


def wl_tokens(
    nodes: list[int],
    adjacency: dict[int, list[int]],
    hops: dict[int, int],
    graph: list[np.ndarray],
    vec_codes: np.ndarray,
    wl_iterations: int,
) -> list[str]:
    labels: dict[int, str] = {}
    for node in nodes:
        degree_bucket = bucket_degree(len(graph[node]))
        hop_bucket = str(hops.get(node, 99))
        code = str(vec_codes[node])
        labels[node] = stable_hash(f"deg={degree_bucket}|hop={hop_bucket}|vec={code}", size=8)

    tokens: list[str] = []
    for it in range(wl_iterations + 1):
        tokens.extend(f"wl{it}:{labels[node]}" for node in nodes)
        if it == wl_iterations:
            break
        new_labels: dict[int, str] = {}
        for node in nodes:
            neigh_labels = sorted(labels[n] for n in adjacency[node])
            payload = labels[node] + "|" + "|".join(neigh_labels)
            new_labels[node] = stable_hash(payload, size=8)
        labels = new_labels
    tokens.append(f"nodes:{bucket_degree(len(nodes))}")
    edge_count = sum(len(v) for v in adjacency.values())
    tokens.append(f"edges:{bucket_degree(edge_count)}")
    return tokens


def graph2vec_from_tokens(
    docs: list[list[str]],
    output_dim: int,
    hash_features: int,
    seed: int,
) -> np.ndarray:
    hasher = FeatureHasher(n_features=hash_features, input_type="string", alternate_sign=False)
    counts = hasher.transform(docs)
    tfidf = TfidfTransformer(norm="l2", sublinear_tf=True).fit_transform(counts)
    max_components = max(1, min(output_dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
    if max_components < 2:
        dense = tfidf.toarray().astype(np.float32)
        if dense.shape[1] < output_dim:
            dense = np.pad(dense, ((0, 0), (0, output_dim - dense.shape[1])))
        return sk_normalize(dense[:, :output_dim]).astype(np.float32)
    svd = TruncatedSVD(n_components=max_components, random_state=seed)
    emb = svd.fit_transform(tfidf).astype(np.float32)
    if emb.shape[1] < output_dim:
        emb = np.pad(emb, ((0, 0), (0, output_dim - emb.shape[1]))).astype(np.float32)
    return sk_normalize(emb[:, :output_dim]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GATE Graph2Vec-style topology features from NSG hub subgraphs.")
    parser.add_argument("--base-fvecs", type=Path, default=ROOT / "data" / "sift_stress_drift" / "base.fvecs")
    parser.add_argument("--nsg-index", type=Path, default=ROOT / "results" / "gate_official_selector_sift_stress_rerun" / "gate_final_rebuilt.nsg")
    parser.add_argument("--hub-path", type=Path, default=ROOT / "results" / "gate_paper_aligned_512_static" / "gate_hubs" / "gate512_refreshed_hubs.txt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "gate_paper_aligned_512_static" / "graph2vec_features")
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--hash-features", type=int, default=4096)
    parser.add_argument("--max-hop", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=384)
    parser.add_argument("--wl-iterations", type=int, default=2)
    parser.add_argument("--projection-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--write-subgraphs", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = read_fvecs(args.base_fvecs)
    graph = read_nsg_index(args.nsg_index)
    _, _, _, eps = read_gate_kmeans_eps(args.hub_path)
    if len(graph) != base.shape[0]:
        raise ValueError(f"graph/base mismatch: graph={len(graph)} base={base.shape[0]}")

    vec_codes = projection_codes(base, args.projection_bits, args.seed)
    docs: list[list[str]] = []
    subgraph_rows: list[dict[str, Any]] = []
    for hub_id, ep in enumerate(eps):
        nodes, adjacency, hops = sample_subgraph(
            int(ep),
            graph=graph,
            base=base,
            max_hop=args.max_hop,
            sample_size=args.sample_size,
            max_nodes=args.max_nodes,
        )
        docs.append(wl_tokens(nodes, adjacency, hops, graph, vec_codes, args.wl_iterations))
        subgraph_rows.append(
            {
                "hub_id": int(hub_id),
                "entry_point": int(ep),
                "nodes": len(nodes),
                "edges": int(sum(len(v) for v in adjacency.values())),
                "tokens": len(docs[-1]),
            }
        )
        if args.write_subgraphs:
            payload = {
                "hub_id": int(hub_id),
                "entry_point": int(ep),
                "nodes": nodes,
                "edges": [[int(src), int(dst)] for src, dsts in adjacency.items() for dst in dsts],
            }
            (args.out_dir / "subgraphs").mkdir(exist_ok=True)
            (args.out_dir / "subgraphs" / f"{hub_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    features = graph2vec_from_tokens(
        docs=docs,
        output_dim=args.output_dim,
        hash_features=args.hash_features,
        seed=args.seed,
    )
    out_fvecs = args.out_dir / "gate_graph2vec_features.fvecs"
    write_fvecs(out_fvecs, features)
    meta = {
        "method": "wl_graph2vec_topology_features",
        "base_fvecs": str(args.base_fvecs.resolve()),
        "nsg_index": str(args.nsg_index.resolve()),
        "hub_path": str(args.hub_path.resolve()),
        "out_fvecs": str(out_fvecs.resolve()),
        "hub_count": int(features.shape[0]),
        "output_dim": int(features.shape[1]),
        "hash_features": args.hash_features,
        "max_hop": args.max_hop,
        "sample_size": args.sample_size,
        "max_nodes": args.max_nodes,
        "wl_iterations": args.wl_iterations,
        "projection_bits": args.projection_bits,
        "subgraph_summary": {
            "nodes_mean": float(np.mean([x["nodes"] for x in subgraph_rows])),
            "nodes_max": int(max(x["nodes"] for x in subgraph_rows)),
            "edges_mean": float(np.mean([x["edges"] for x in subgraph_rows])),
            "tokens_mean": float(np.mean([x["tokens"] for x in subgraph_rows])),
        },
        "note": "Graph2Vec-style pipeline: sample hub-centered NSG subgraphs, generate WL rooted-subgraph tokens, then embed token documents with TF-IDF + SVD.",
    }
    (args.out_dir / "graph2vec_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (args.out_dir / "subgraph_summary.json").write_text(json.dumps(subgraph_rows, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
