from __future__ import annotations

import argparse
import csv
import json
import sys
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
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_query_weights(path: Path | None, rows: int) -> np.ndarray:
    weights = np.ones(rows, dtype=np.float32)
    if path is None or not path.exists():
        return weights
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qi = int(row["query_id"])
            if 0 <= qi < rows:
                weights[qi] = max(float(row.get("distance_computations", 1.0)), 1.0)
    median = float(np.median(weights))
    weights = np.maximum(weights - median, 0.0).astype(np.float32) + 1.0
    weights /= float(np.mean(weights))
    return weights


def local_graph_features(vectors: np.ndarray, anchors: np.ndarray, knn: np.ndarray, graph_k: int) -> np.ndarray:
    features: list[np.ndarray] = []
    for anchor in anchors:
        neigh = [int(x) for x in knn[int(anchor), :graph_k] if 0 <= int(x) < vectors.shape[0]]
        if not neigh:
            neigh = [int(anchor)]
        neigh_vectors = vectors[np.asarray(neigh, dtype=np.int64)]
        center = vectors[int(anchor)]
        mean_delta = np.mean(neigh_vectors - center.reshape(1, -1), axis=0)
        spread = np.std(neigh_vectors, axis=0)
        distances = np.sum((neigh_vectors - center.reshape(1, -1)) ** 2, axis=1)
        stats = np.asarray(
            [
                len(neigh) / max(1, graph_k),
                float(np.mean(distances)),
                float(np.std(distances)),
                float(np.min(distances)),
                float(np.max(distances)),
            ],
            dtype=np.float32,
        )
        features.append(np.concatenate([center, mean_delta, spread, stats]).astype(np.float32))
    return np.vstack(features)


class TwoTower(nn.Module):
    def __init__(self, query_dim: int, hub_dim: int, hidden: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_tower = nn.Sequential(
            nn.Linear(query_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        self.hub_tower = nn.Sequential(
            nn.Linear(hub_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def encode_queries(self, queries: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_tower(queries), dim=1)

    def encode_hubs(self, hubs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.hub_tower(hubs), dim=1)


def choose_positive_hubs(vectors: np.ndarray, truth: np.ndarray, hub_ids: np.ndarray, topk: int) -> np.ndarray:
    hub_vectors = vectors[hub_ids]
    centers = np.mean(vectors[truth[:, :topk]], axis=1)
    distances = (
        np.sum(centers * centers, axis=1, keepdims=True)
        + np.sum(hub_vectors * hub_vectors, axis=1).reshape(1, -1)
        - 2.0 * centers @ hub_vectors.T
    )
    return np.argmin(distances, axis=1).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GATE-like two-tower query-to-entry selector.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift100k_dynamic" / "manifest.json")
    parser.add_argument("--anchor-file", type=Path, default=ROOT / "results" / "entry_topology_backend_matrix" / "anchors" / "gate_initial_hub.txt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "entry_topology_backend_matrix" / "gate_two_tower")
    parser.add_argument("--baseline-perq", type=Path, default=ROOT / "results" / "entry_topology_backend_matrix" / "nsg" / "initial_gate_static_no_repair" / "nsg_ep_L240_perq.csv")
    parser.add_argument("--entries", type=int, default=8)
    parser.add_argument("--truth-k", type=int, default=10)
    parser.add_argument("--graph-k", type=int, default=16)
    parser.add_argument("--train-query-fraction", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    initial = read_fbin(Path(paths["initial_fbin"]))
    inserts = read_fbin(Path(paths["insert_fbin"]))
    queries = read_fbin(Path(paths["query_fbin"]))
    truth = read_id_bin(Path(paths["truth_dynamic_bin"]))
    all_vectors = np.vstack([initial, inserts]).astype(np.float32, copy=False)
    all_knn = read_ivecs(Path(paths["all_nsg_knn_graph"]))
    hub_ids = np.asarray(read_anchor_file(args.anchor_file), dtype=np.int64)
    if hub_ids.size == 0:
        raise ValueError("empty hub anchor file")

    hub_features = local_graph_features(all_vectors, hub_ids, all_knn, args.graph_k)
    positives = choose_positive_hubs(all_vectors, truth, hub_ids, args.truth_k)
    weights = read_query_weights(args.baseline_perq, queries.shape[0])
    train_count = max(1, min(queries.shape[0], int(round(queries.shape[0] * args.train_query_fraction))))

    q_train = torch.tensor(queries[:train_count], dtype=torch.float32)
    pos_train = torch.tensor(positives[:train_count], dtype=torch.long)
    w_train = torch.tensor(weights[:train_count], dtype=torch.float32)
    hub_tensor = torch.tensor(hub_features, dtype=torch.float32)

    model = TwoTower(queries.shape[1], hub_features.shape[1], args.hidden, args.embed_dim, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    indices = torch.arange(train_count)
    best_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        perm = indices[torch.randperm(train_count)]
        total = 0.0
        seen = 0
        model.train()
        for start in range(0, train_count, args.batch_size):
            batch = perm[start : start + args.batch_size]
            q_emb = model.encode_queries(q_train[batch])
            h_emb = model.encode_hubs(hub_tensor)
            logits = q_emb @ h_emb.T / args.temperature
            losses = F.cross_entropy(logits, pos_train[batch], reduction="none")
            loss = torch.mean(losses * w_train[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(batch.numel())
            seen += int(batch.numel())
        avg = total / max(1, seen)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 40 == 0:
            print(f"epoch={epoch+1} loss={avg:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        query_emb = model.encode_queries(torch.tensor(queries, dtype=torch.float32)).cpu().numpy()
        hub_emb = model.encode_hubs(hub_tensor).cpu().numpy()
    scores = query_emb @ hub_emb.T
    top = np.argpartition(-scores, kth=args.entries - 1, axis=1)[:, : args.entries]
    row = np.arange(top.shape[0])[:, None]
    top = top[row, np.argsort(-scores[row, top], axis=1)]
    entries = hub_ids[top].astype(np.int32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries_path = args.out_dir / "gate_two_tower_entries.ivecs"
    hub_emb_path = args.out_dir / "gate_two_tower_hub_embeddings.fvecs"
    write_ivecs(entries_path, entries)
    write_fvecs(hub_emb_path, hub_emb.astype(np.float32))
    torch.save(model.state_dict(), args.out_dir / "gate_two_tower.pt")
    meta = {
        "method": "gate_two_tower_proxy",
        "anchor_file": str(args.anchor_file.resolve()),
        "hub_count": int(hub_ids.size),
        "entries": args.entries,
        "train_queries": train_count,
        "total_queries": int(queries.shape[0]),
        "best_loss": best_loss,
        "epochs": args.epochs,
        "note": "PyTorch two-tower query/hub selector with local graph features; closer to paper-faithful GATE than ridge, but not the original Graph2Vec artifact pipeline.",
    }
    (args.out_dir / "gate_two_tower_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
