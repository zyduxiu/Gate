from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.io_formats import write_fvecs, write_ivecs


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


def read_id_bin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
        if header.size != 2:
            raise ValueError(f"invalid id bin header: {path}")
        rows, cols = int(header[0]), int(header[1])
        data = np.fromfile(handle, dtype="<u4", count=rows * cols)
    if data.size != rows * cols:
        raise ValueError(f"truncated id bin: {path}")
    return data.reshape(rows, cols).astype(np.int32, copy=False)


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i4")
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    return rows[:, 1:].astype(np.int32, copy=False)


def read_anchor_file(path: Path) -> list[int]:
    anchors: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            anchors.append(int(line))
    return anchors


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (
        np.sum(a * a, axis=1, keepdims=True)
        + np.sum(b * b, axis=1).reshape(1, -1)
        - 2.0 * a @ b.T
    )


def sampled_subgraph(
    graph: np.ndarray,
    vectors: np.ndarray,
    anchor: int,
    max_hop: int,
    sample_size: int,
    max_nodes: int,
) -> dict[int, int]:
    hops: dict[int, int] = {int(anchor): 0}
    queue: deque[tuple[int, int]] = deque([(int(anchor), 0)])
    while queue and len(hops) < max_nodes:
        node, hop = queue.popleft()
        if hop >= max_hop or node < 0 or node >= graph.shape[0]:
            continue
        neigh = [int(x) for x in graph[node] if 0 <= int(x) < vectors.shape[0]]
        if not neigh:
            continue
        if len(neigh) > sample_size:
            center = vectors[node].reshape(1, -1)
            neigh_arr = np.asarray(neigh, dtype=np.int64)
            dists = np.sum((vectors[neigh_arr] - center) ** 2, axis=1)
            order = np.argsort(dists)
            half = max(1, sample_size // 2)
            picked = np.concatenate([order[:half], order[-(sample_size - half) :]])
            neigh = [int(neigh_arr[i]) for i in picked]
        for nxt in neigh:
            if nxt not in hops:
                hops[nxt] = hop + 1
                queue.append((nxt, hop + 1))
                if len(hops) >= max_nodes:
                    break
    return hops


def build_topology_embeddings(
    vectors: np.ndarray,
    graph: np.ndarray,
    hub_ids: np.ndarray,
    out_dim: int,
    max_hop: int,
    sample_size: int,
    max_nodes: int,
    method: str,
) -> tuple[np.ndarray, list[dict[int, int]], dict[str, float]]:
    raw_features: list[np.ndarray] = []
    hop_maps: list[dict[int, int]] = []
    node_counts: list[int] = []
    for hub in hub_ids:
        hop_map = sampled_subgraph(graph, vectors, int(hub), max_hop, sample_size, max_nodes)
        hop_maps.append(hop_map)
        nodes = np.fromiter(hop_map.keys(), dtype=np.int64)
        node_counts.append(int(nodes.size))
        sub = vectors[nodes]
        center = vectors[int(hub)]
        if method == "wl_hash":
            raw_features.append(wl_subtree_features(graph, hop_map, out_dim, iterations=2))
        else:
            degrees = np.asarray(
                [np.count_nonzero((graph[int(node)] >= 0) & (graph[int(node)] < vectors.shape[0])) for node in nodes],
                dtype=np.float32,
            )
            hops = np.asarray(list(hop_map.values()), dtype=np.float32)
            stats = np.asarray(
                [
                    nodes.size,
                    float(np.mean(degrees)) if degrees.size else 0.0,
                    float(np.std(degrees)) if degrees.size else 0.0,
                    float(np.max(degrees)) if degrees.size else 0.0,
                    float(np.mean(hops)) if hops.size else 0.0,
                    float(np.max(hops)) if hops.size else 0.0,
                    float(np.mean(np.sum((sub - center.reshape(1, -1)) ** 2, axis=1))) if sub.size else 0.0,
                    float(np.std(np.sum((sub - center.reshape(1, -1)) ** 2, axis=1))) if sub.size else 0.0,
                ],
                dtype=np.float32,
            )
            raw_features.append(
                np.concatenate(
                    [
                        center.astype(np.float32),
                        np.mean(sub, axis=0).astype(np.float32),
                        np.std(sub, axis=0).astype(np.float32),
                        (np.mean(sub, axis=0) - center).astype(np.float32),
                        stats,
                    ]
                )
            )

    raw = np.vstack(raw_features).astype(np.float32)
    if raw.shape[1] == out_dim:
        reduced = normalize_rows(raw).astype(np.float32, copy=False)
    else:
        mean = np.mean(raw, axis=0, keepdims=True)
        std = np.std(raw, axis=0, keepdims=True)
        raw_norm = (raw - mean) / np.maximum(std, 1e-6)
        u, s, _ = np.linalg.svd(raw_norm, full_matrices=False)
        reduced = u[:, : min(out_dim, u.shape[1])] * s[: min(out_dim, s.shape[0])]
        if reduced.shape[1] < out_dim:
            reduced = np.pad(reduced, ((0, 0), (0, out_dim - reduced.shape[1])))
        reduced = reduced.astype(np.float32, copy=False)
    meta = {
        "topology_embedding": method,
        "avg_sampled_subgraph_nodes": float(np.mean(node_counts)) if node_counts else 0.0,
        "min_sampled_subgraph_nodes": float(np.min(node_counts)) if node_counts else 0.0,
        "max_sampled_subgraph_nodes": float(np.max(node_counts)) if node_counts else 0.0,
    }
    return reduced, hop_maps, meta


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def wl_subtree_features(graph: np.ndarray, hop_map: dict[int, int], out_dim: int, iterations: int) -> np.ndarray:
    nodes = set(hop_map.keys())
    labels: dict[int, str] = {}
    for node, hop in hop_map.items():
        degree = sum(1 for n in graph[int(node)] if int(n) in nodes)
        labels[int(node)] = f"h{hop}_d{min(degree, 64)}"
    features = np.zeros(out_dim, dtype=np.float32)
    for _ in range(iterations + 1):
        for label in labels.values():
            features[stable_hash(label) % out_dim] += 1.0
        new_labels: dict[int, str] = {}
        for node in labels:
            neigh_labels = sorted(labels[int(n)] for n in graph[int(node)] if int(n) in labels)
            new_labels[node] = f"{labels[node]}|{'_'.join(neigh_labels[:32])}"
        labels = new_labels
    total = float(np.linalg.norm(features))
    if total > 1e-12:
        features /= total
    return features


def build_query_samples(
    queries: np.ndarray,
    truth: np.ndarray,
    hub_ids: np.ndarray,
    hub_vectors: np.ndarray,
    hop_maps: list[dict[int, int]],
    train_count: int,
    truth_k: int,
    tolerance: float,
    pos_per_query: int,
    neg_per_query: int,
    rng: np.random.Generator,
) -> tuple[list[list[int]], list[list[int]], np.ndarray, dict[str, float]]:
    hub_count = int(hub_ids.size)
    dists = squared_distances(queries[:train_count], hub_vectors)
    d_min = np.min(dists, axis=1, keepdims=True)
    d_max = np.max(dists, axis=1, keepdims=True)
    d_norm = (dists - d_min) / np.maximum(d_max - d_min, 1e-6)

    hop_score = np.full((train_count, hub_count), 99.0, dtype=np.float32)
    targets = truth[:train_count, :truth_k]
    for h, hop_map in enumerate(hop_maps):
        for q in range(train_count):
            best = 99
            for target in targets[q]:
                best = min(best, hop_map.get(int(target), 99))
            hop_score[q, h] = float(best)
    fallback = np.min(hop_score, axis=1) >= 99.0
    score = hop_score + 0.15 * d_norm.astype(np.float32)
    score[fallback] = d_norm[fallback].astype(np.float32)

    pos_sample: list[list[int]] = []
    neg_sample: list[list[int]] = []
    for q in range(train_count):
        row = score[q]
        best = float(np.min(row))
        pos = np.flatnonzero(row <= best + tolerance).astype(int).tolist()
        if len(pos) < pos_per_query:
            pos = np.argsort(row)[:pos_per_query].astype(int).tolist()
        else:
            pos = pos[:pos_per_query]
        neg = np.argsort(row)[-neg_per_query:].astype(int).tolist()
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos_sample.append(pos)
        neg_sample.append(neg)

    meta = {
        "train_queries": int(train_count),
        "hub_count": int(hub_count),
        "fallback_query_fraction": float(np.mean(fallback)) if fallback.size else 0.0,
        "avg_positive_hubs_per_query": float(np.mean([len(x) for x in pos_sample])) if pos_sample else 0.0,
        "avg_negative_hubs_per_query": float(np.mean([len(x) for x in neg_sample])) if neg_sample else 0.0,
    }
    return pos_sample, neg_sample, score, meta


def read_anchor_cost_scores(path: Path, train_count: int, hub_count: int) -> np.ndarray:
    scores = np.full((train_count, hub_count), np.inf, dtype=np.float32)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            query_id = int(row["query_id"])
            anchor_rank = int(row["anchor_rank"])
            if query_id >= train_count or anchor_rank >= hub_count:
                continue
            recall = float(row["recall_at_10"])
            expanded = float(row["expanded"])
            comps = float(row["distance_computations"])
            if comps >= 4_000_000_000:
                score = 1e9
            else:
                score = comps + 0.25 * expanded + (1.0 - recall) * 100_000.0
            scores[query_id, anchor_rank] = score
    if not np.all(np.isfinite(scores)):
        finite = scores[np.isfinite(scores)]
        fill = float(np.max(finite)) if finite.size else 1e9
        scores[~np.isfinite(scores)] = fill
    return scores


def samples_from_scores(
    scores: np.ndarray,
    pos_per_query: int,
    neg_per_query: int,
    rng: np.random.Generator,
) -> tuple[list[list[int]], list[list[int]], dict[str, float]]:
    pos_sample: list[list[int]] = []
    neg_sample: list[list[int]] = []
    for q in range(scores.shape[0]):
        row = scores[q]
        pos = np.argsort(row)[:pos_per_query].astype(int).tolist()
        neg = np.argsort(row)[-neg_per_query:].astype(int).tolist()
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos_sample.append(pos)
        neg_sample.append(neg)
    meta = {
        "train_queries": int(scores.shape[0]),
        "hub_count": int(scores.shape[1]),
        "fallback_query_fraction": 0.0,
        "avg_positive_hubs_per_query": float(np.mean([len(x) for x in pos_sample])) if pos_sample else 0.0,
        "avg_negative_hubs_per_query": float(np.mean([len(x) for x in neg_sample])) if neg_sample else 0.0,
        "label_source": "actual_per_anchor_search_cost",
    }
    return pos_sample, neg_sample, meta


def build_ep_pairs(
    pos_sample: list[list[int]],
    neg_sample: list[list[int]],
    hub_count: int,
    pairs_per_hub: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    ep_pos: list[list[int]] = [[] for _ in range(hub_count)]
    ep_neg: list[list[int]] = [[] for _ in range(hub_count)]
    for query_id, hubs in enumerate(pos_sample):
        for hub in hubs:
            ep_pos[int(hub)].append(query_id)
    for query_id, hubs in enumerate(neg_sample):
        for hub in hubs:
            ep_neg[int(hub)].append(query_id)

    all_queries = list(range(len(pos_sample)))
    pairs: list[tuple[int, int, int]] = []
    for hub in range(hub_count):
        positives = ep_pos[hub]
        negatives = ep_neg[hub]
        if not positives:
            continue
        if not negatives:
            negatives = all_queries
        for _ in range(min(pairs_per_hub, len(positives) * max(1, len(negatives)))):
            pairs.append((hub, rng.choice(positives), rng.choice(negatives)))
    rng.shuffle(pairs)
    return pairs


class FusionAugmentation(nn.Module):
    def __init__(self, node_dim: int, graph_dim: int, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.graph_fc = nn.Linear(graph_dim, node_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=node_dim,
            nhead=heads,
            dim_feedforward=max(hidden_dim, node_dim * 2),
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out = nn.Linear(node_dim, hidden_dim)

    def forward(self, node_feature: torch.Tensor, graph_feature: torch.Tensor) -> torch.Tensor:
        graph_token = self.graph_fc(graph_feature)
        x = torch.stack([node_feature, graph_token], dim=1)
        fused = self.encoder(x)[:, -1, :]
        return self.out(fused)


class GateHubTower(nn.Module):
    def __init__(
        self,
        node_dim: int,
        graph_dim: int,
        hidden_dim: int,
        output_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fusion = FusionAugmentation(node_dim, graph_dim, hidden_dim, heads, dropout)
        self.project = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, node_feature: torch.Tensor, graph_feature: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.project(self.fusion(node_feature, graph_feature)), dim=1)


class QueryTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.project(query), dim=1)


def train_model(
    node_features: np.ndarray,
    graph_features: np.ndarray,
    queries: np.ndarray,
    pairs: list[tuple[int, int, int]],
    hidden_dim: int,
    heads: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    margin: float,
    seed: int,
    use_query_tower: bool,
) -> tuple[GateHubTower, QueryTower | None, float]:
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GateHubTower(
        node_dim=node_features.shape[1],
        graph_dim=graph_features.shape[1],
        hidden_dim=hidden_dim,
        output_dim=queries.shape[1],
        heads=heads,
        dropout=dropout,
    ).to(device)
    query_tower = QueryTower(queries.shape[1], hidden_dim, queries.shape[1], dropout).to(device) if use_query_tower else None
    node_tensor = torch.tensor(node_features, dtype=torch.float32, device=device)
    graph_tensor = torch.tensor(graph_features, dtype=torch.float32, device=device)
    query_tensor = F.normalize(torch.tensor(queries, dtype=torch.float32, device=device), dim=1)
    params = list(model.parameters())
    if query_tower is not None:
        params.extend(query_tower.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
        margin=margin,
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    pair_indices = list(range(len(pairs)))
    for epoch in range(epochs):
        random.shuffle(pair_indices)
        total = 0.0
        seen = 0
        model.train()
        for start in range(0, len(pair_indices), batch_size):
            batch_ids = pair_indices[start : start + batch_size]
            batch = [pairs[i] for i in batch_ids]
            hubs = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
            pos = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
            neg = torch.tensor([x[2] for x in batch], dtype=torch.long, device=device)
            hub_emb = model(node_tensor[hubs], graph_tensor[hubs])
            if query_tower is None:
                pos_emb = query_tensor[pos]
                neg_emb = query_tensor[neg]
            else:
                pos_emb = query_tower(query_tensor[pos])
                neg_emb = query_tower(query_tensor[neg])
            loss = criterion(hub_emb, pos_emb, neg_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
        avg = total / max(1, seen)
        if avg < best_loss:
            best_loss = avg
            best_state = {
                "hub": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "query": {k: v.detach().cpu().clone() for k, v in query_tower.state_dict().items()} if query_tower is not None else None,
            }
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 25 == 0:
            print(f"epoch={epoch + 1} loss={avg:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state["hub"])
        if query_tower is not None and best_state["query"] is not None:
            query_tower.load_state_dict(best_state["query"])
    return model, query_tower, best_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GATE paper-aligned hub selector.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_drift" / "manifest.json")
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--anchor-file", type=Path, required=True)
    parser.add_argument("--anchor-costs-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--truth-k", type=int, default=10)
    parser.add_argument("--train-query-fraction", type=float, default=0.6)
    parser.add_argument("--max-hop", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--max-subgraph-nodes", type=int, default=512)
    parser.add_argument("--topology-embedding", choices=["wl_hash", "stats_svd"], default="wl_hash")
    parser.add_argument("--positive-tolerance", type=float, default=0.0)
    parser.add_argument("--pos-per-query", type=int, default=2)
    parser.add_argument("--neg-per-query", type=int, default=16)
    parser.add_argument("--pairs-per-hub", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--query-tower", action="store_true")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_manifest: dict[str, Any] = json.loads(args.train_manifest.read_text(encoding="utf-8")) if args.train_manifest is not None else manifest
    paths = manifest["paths"]
    train_paths = train_manifest["paths"]
    initial = read_fbin(Path(paths["initial_fbin"]))
    inserts = read_fbin(Path(paths["insert_fbin"]))
    queries = read_fbin(Path(paths["query_fbin"]))
    train_queries = read_fbin(Path(train_paths["query_fbin"]))
    train_truth = read_id_bin(Path(train_paths["truth_dynamic_bin"]))
    graph = read_ivecs(Path(paths["all_nsg_knn_graph"]))
    all_vectors = np.vstack([initial, inserts]).astype(np.float32, copy=False)
    hub_ids = np.asarray(read_anchor_file(args.anchor_file), dtype=np.int64)
    hub_ids = hub_ids[(hub_ids >= 0) & (hub_ids < all_vectors.shape[0])]
    if hub_ids.size == 0:
        raise ValueError("empty or invalid hub set")

    graph_features, hop_maps, topo_meta = build_topology_embeddings(
        all_vectors,
        graph,
        hub_ids,
        out_dim=queries.shape[1],
        max_hop=args.max_hop,
        sample_size=args.sample_size,
        max_nodes=args.max_subgraph_nodes,
        method=args.topology_embedding,
    )
    train_count = max(1, min(train_queries.shape[0], int(round(train_queries.shape[0] * args.train_query_fraction))))
    if args.anchor_costs_csv is not None and args.anchor_costs_csv.exists():
        score = read_anchor_cost_scores(args.anchor_costs_csv, train_count, int(hub_ids.size))
        pos_sample, neg_sample, sample_meta = samples_from_scores(
            score,
            pos_per_query=args.pos_per_query,
            neg_per_query=args.neg_per_query,
            rng=rng,
        )
    else:
        pos_sample, neg_sample, score, sample_meta = build_query_samples(
            queries=train_queries,
            truth=train_truth,
            hub_ids=hub_ids,
            hub_vectors=all_vectors[hub_ids],
            hop_maps=hop_maps,
            train_count=train_count,
            truth_k=args.truth_k,
            tolerance=args.positive_tolerance,
            pos_per_query=args.pos_per_query,
            neg_per_query=args.neg_per_query,
            rng=rng,
        )
    pairs = build_ep_pairs(pos_sample, neg_sample, int(hub_ids.size), args.pairs_per_hub, args.seed)
    if not pairs:
        raise ValueError("no contrastive pairs generated")

    model, query_tower, best_loss = train_model(
        node_features=all_vectors[hub_ids],
        graph_features=graph_features,
        queries=train_queries[:train_count],
        pairs=pairs,
        hidden_dim=args.hidden,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        margin=args.margin,
        seed=args.seed,
        use_query_tower=args.query_tower,
    )

    device = next(model.parameters()).device
    model.eval()
    if query_tower is not None:
        query_tower.eval()
    with torch.no_grad():
        node_tensor = torch.tensor(all_vectors[hub_ids], dtype=torch.float32, device=device)
        graph_tensor = torch.tensor(graph_features, dtype=torch.float32, device=device)
        hub_embeddings = model(node_tensor, graph_tensor).cpu().numpy().astype(np.float32)
        if query_tower is None:
            query_embeddings = torch.tensor(normalize_rows(queries.astype(np.float32, copy=False)), dtype=torch.float32).cpu().numpy()
        else:
            query_embeddings = query_tower(
                torch.tensor(normalize_rows(queries.astype(np.float32, copy=False)), dtype=torch.float32, device=device)
            ).cpu().numpy().astype(np.float32)
    scores = query_embeddings @ hub_embeddings.T
    top = np.argpartition(-scores, kth=min(args.entries - 1, scores.shape[1] - 1), axis=1)[:, : args.entries]
    row = np.arange(top.shape[0])[:, None]
    top = top[row, np.argsort(-scores[row, top], axis=1)]
    entries = hub_ids[top].astype(np.int32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_ivecs(args.out_dir / "gate_paper_entries.ivecs", entries)
    write_fvecs(args.out_dir / "gate_paper_hub_embeddings.fvecs", hub_embeddings)
    write_fvecs(args.out_dir / "gate_paper_query_embeddings.fvecs", query_embeddings)
    write_fvecs(args.out_dir / "gate_paper_graph_features.fvecs", graph_features)
    torch.save(
        {
            "hub_tower": model.state_dict(),
            "query_tower": query_tower.state_dict() if query_tower is not None else None,
            "use_query_tower": args.query_tower,
        },
        args.out_dir / "gate_paper_selector.pt",
    )
    (args.out_dir / "pos_sample.txt").write_text(
        "\n".join(" ".join(map(str, row)) for row in pos_sample) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "neg_sample.txt").write_text(
        "\n".join(" ".join(map(str, row)) for row in neg_sample) + "\n",
        encoding="utf-8",
    )
    meta = {
        "method": "gate_paper_aligned_selector",
        "manifest": str(args.manifest.resolve()),
        "train_manifest": str(args.train_manifest.resolve()) if args.train_manifest is not None else str(args.manifest.resolve()),
        "anchor_file": str(args.anchor_file.resolve()),
        "anchor_costs_csv": str(args.anchor_costs_csv.resolve()) if args.anchor_costs_csv is not None else None,
        "hub_count": int(hub_ids.size),
        "entries": int(args.entries),
        "train_queries": int(train_count),
        "total_queries": int(queries.shape[0]),
        "pairs": int(len(pairs)),
        "best_loss": float(best_loss),
        "query_tower": bool(args.query_tower),
        "topology": topo_meta,
        "samples": sample_meta,
        "score_min": float(np.min(score)),
        "score_max": float(np.max(score)),
        "note": (
            "GATE-aligned implementation: hierarchical hub candidate set is supplied by anchor_file; "
            "sampled hub subgraphs provide topological features; positive/negative query samples "
            "are generated by actual per-anchor search costs when available, otherwise by "
            "hop-to-ground-truth labels with vector-distance fallback; a fusion "
            "augmentation plus triplet cosine loss produces hub embeddings in query-vector space. "
            "The wl_hash topology mode is a lightweight Graph2Vec-style WL-subtree feature hashing implementation."
        ),
    }
    (args.out_dir / "gate_paper_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
