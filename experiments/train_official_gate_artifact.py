from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.io_formats import read_fvecs, write_fvecs


def read_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
        rows, dim = int(header[0]), int(header[1])
        data = np.fromfile(handle, dtype="<f4", count=rows * dim)
    return data.reshape(rows, dim).astype(np.float32, copy=False)


def read_anchor_file(path: Path) -> list[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_samples(path: Path) -> list[list[int]]:
    return [list(map(int, line.split())) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_graph_feature_csv(path: Path, features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(features.astype(np.float32)).to_csv(path)


def model_name(params: dict[str, Any]) -> str:
    pieces = [str(params["name"])]
    for key, value in params.items():
        if key != "name":
            pieces.append(f"{key}{value}")
    return "_".join(pieces)


class OfficialContrastiveDataset(torch.utils.data.Dataset):
    """Mirrors GATE's cluster.model.train.dataset.ContrastiveDataset."""

    def __init__(
        self,
        graph_features: np.ndarray,
        ep_vectors: np.ndarray,
        query_vectors: np.ndarray,
        pos_sample: list[list[int]],
        neg_sample: list[list[int]],
        filter_num: int,
        sample_num: int,
    ) -> None:
        self.ep_graph_features = torch.tensor(graph_features, dtype=torch.float32)
        self.ep_node_features = torch.tensor(ep_vectors, dtype=torch.float32)
        self.query_node_features = torch.tensor(query_vectors, dtype=torch.float32)

        ep2query_pos: list[list[int]] = [[] for _ in range(len(graph_features))]
        ep2query_neg: list[list[int]] = [[] for _ in range(len(graph_features))]
        dummy_query_ids: set[int] = set()
        for query_id, (poses, negs) in enumerate(zip(pos_sample, neg_sample)):
            if len(negs) > filter_num or len(negs) < 10:
                dummy_query_ids.add(query_id)
                continue
            for ep_id in poses:
                if 0 <= ep_id < len(ep2query_pos):
                    ep2query_pos[ep_id].append(query_id)

        for query_id, negs in enumerate(neg_sample):
            if query_id in dummy_query_ids:
                continue
            for ep_id in negs:
                if 0 <= ep_id < len(ep2query_neg):
                    ep2query_neg[ep_id].append(query_id)

        pairs: list[tuple[int, int, int]] = []
        rng = random.Random(13)
        for ep_id, (poses, negs) in enumerate(zip(ep2query_pos, ep2query_neg)):
            poses = list(poses)
            negs = list(negs)
            rng.shuffle(poses)
            rng.shuffle(negs)
            poses = poses[:sample_num]
            negs = negs[:sample_num]
            for neg in negs:
                for pos in poses:
                    pairs.append((ep_id, pos, neg))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ep_id, pos_query_id, neg_query_id = self.pairs[idx]
        return (
            self.ep_node_features[ep_id],
            self.ep_graph_features[ep_id],
            self.query_node_features[pos_query_id],
            self.query_node_features[neg_query_id],
        )


def evaluate_top1_accuracy(model: torch.nn.Module, ep_vectors: np.ndarray, graph_features: np.ndarray, queries: np.ndarray, pos_sample: list[list[int]], device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        ep_tensor = torch.tensor(ep_vectors, dtype=torch.float32, device=device)
        graph_tensor = torch.tensor(graph_features, dtype=torch.float32, device=device)
        query_tensor = torch.tensor(queries, dtype=torch.float32, device=device)
        hub_emb = model(ep_tensor, graph_tensor)
        hits = 0
        for query, positives in zip(query_tensor, pos_sample):
            scores = F.cosine_similarity(hub_emb, query.unsqueeze(0), dim=1)
            pred = int(torch.argmax(scores).item())
            if pred in positives:
                hits += 1
    return hits / max(1, len(pos_sample))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the official GATE artifact model on prepared local files.")
    parser.add_argument("--gate-root", type=Path, default=ROOT / "third_party" / "GATE" / "jacksondca-gate-78334b4")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "gate_paper_aligned_512_static" / "static_manifest_gate512_paper_aligned.json")
    parser.add_argument("--anchor-file", type=Path, required=True)
    parser.add_argument("--graph-features-fvecs", type=Path, required=True)
    parser.add_argument("--pos-sample", type=Path, required=True)
    parser.add_argument("--neg-sample", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=2024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--filter-num", type=int, default=64)
    parser.add_argument("--sample-num", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    sys.path.insert(0, str(args.gate_root.resolve()))
    from cluster.model.train.model import Model  # type: ignore

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    base = read_fbin(Path(paths["initial_fbin"]))
    queries = read_fbin(Path(paths["query_fbin"]))
    anchors = np.asarray(read_anchor_file(args.anchor_file), dtype=np.int64)
    ep_vectors = base[anchors].astype(np.float32)
    graph_features = read_fvecs(args.graph_features_fvecs).astype(np.float32)
    if graph_features.shape[0] != ep_vectors.shape[0]:
        raise ValueError(f"graph rows {graph_features.shape[0]} != hubs {ep_vectors.shape[0]}")
    pos_sample = read_samples(args.pos_sample)
    neg_sample = read_samples(args.neg_sample)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_graph_feature_csv(args.out_dir / "official_graph_features.csv", graph_features)
    write_fvecs(args.out_dir / "official_ep_vectors.fvecs", ep_vectors)

    model_params = {
        "name": "gate_official_artifact",
        "input_dim": int(ep_vectors.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "output_dim": int(queries.shape[1]),
        "dropout": float(args.dropout),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "epochs": int(args.epochs),
    }
    model = Model(model_params["input_dim"], model_params["hidden_dim"], model_params["output_dim"], model_params["dropout"]).to(device)
    criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    dataset = OfficialContrastiveDataset(
        graph_features=graph_features,
        ep_vectors=ep_vectors,
        query_vectors=queries,
        pos_sample=pos_sample,
        neg_sample=neg_sample,
        filter_num=args.filter_num,
        sample_num=args.sample_num,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        seen = 0
        for node_batch, graph_batch, pos_batch, neg_batch in loader:
            node_batch = node_batch.to(device)
            graph_batch = graph_batch.to(device)
            pos_batch = pos_batch.to(device)
            neg_batch = neg_batch.to(device)
            emb = model(node_batch, graph_batch)
            loss = criterion(emb, pos_batch, neg_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * int(node_batch.shape[0])
            seen += int(node_batch.shape[0])
        avg_loss = total / max(1, seen)
        acc = evaluate_top1_accuracy(model, ep_vectors, graph_features, queries, pos_sample, device)
        history.append({"epoch": float(epoch + 1), "loss": avg_loss, "acc": acc})
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model, args.out_dir / f"{model_name(model_params)}.pth")
        if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 25 == 0:
            print(f"epoch={epoch + 1} loss={avg_loss:.6f} acc={acc:.4f} best_acc={best_acc:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ep_tensor = torch.tensor(ep_vectors, dtype=torch.float32, device=device)
        graph_tensor = torch.tensor(graph_features, dtype=torch.float32, device=device)
        embeddings = model(ep_tensor, graph_tensor).detach().cpu().numpy().astype(np.float32)
    write_fvecs(args.out_dir / "official_gate_hub_embeddings.fvecs", embeddings)
    meta = {
        "method": "official_gate_artifact_model_adapter",
        "gate_root": str(args.gate_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "anchor_file": str(args.anchor_file.resolve()),
        "graph_features_fvecs": str(args.graph_features_fvecs.resolve()),
        "pos_sample": str(args.pos_sample.resolve()),
        "neg_sample": str(args.neg_sample.resolve()),
        "model_params": model_params,
        "device": str(device),
        "pairs": len(dataset),
        "best_acc": best_acc,
        "history": history,
        "note": "Uses the official GATE artifact Model and contrastive dataset logic, with local adapter paths.",
    }
    (args.out_dir / "official_gate_training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
