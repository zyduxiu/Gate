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

from dynanchor.io_formats import write_ivecs
from run_entry_topology_matrix import dynamic_labels_from_manifest


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


def parse_gate_hub_ids(hub_path: Path, dynamic_labels: list[int]) -> list[int]:
    lines = [line.strip() for line in hub_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stage1, stage2, _dim = [int(value) for value in lines[0].split()[:3]]
    ids: list[int] = []
    seen: set[int] = set()
    for line in lines[1 + stage1 : 1 + stage1 + stage1]:
        values = [int(value) for value in line.split()]
        if len(values) != stage2:
            raise ValueError(f"invalid GATE hub row in {hub_path}")
        for value in values:
            mapped = int(dynamic_labels[value])
            if mapped not in seen:
                seen.add(mapped)
                ids.append(mapped)
    return ids


def read_query_weights(path: Path | None, rows: int) -> np.ndarray:
    weights = np.ones(rows, dtype=np.float64)
    if path is None or not path.exists():
        return weights
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qi = int(row["query_id"])
            if 0 <= qi < rows:
                weights[qi] = max(float(row.get("distance_computations", 1.0)), 1.0)
    median = float(np.median(weights))
    weights = np.maximum(weights - median, 0.0) + 1.0
    return weights


def fit_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge: float) -> np.ndarray:
    x_aug = np.hstack([x.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)])
    w = np.sqrt(weights).reshape(-1, 1)
    xw = x_aug * w
    yw = y.astype(np.float64) * w
    reg = ridge * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight GATE-like query-to-hub entry selector.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift100k_dynamic" / "manifest.json")
    parser.add_argument("--gate-hub-path", type=Path, default=ROOT / "results" / "sift100k_gate_adapter" / "gate_hubs.txt")
    parser.add_argument("--out-ivecs", type=Path, default=ROOT / "results" / "entry_topology_backend_matrix" / "gate_like_learned_entries.ivecs")
    parser.add_argument("--out-anchor-txt", type=Path, default=ROOT / "results" / "entry_topology_backend_matrix" / "anchors" / "gate_static_hub_mapped.txt")
    parser.add_argument("--baseline-perq", type=Path, default=ROOT / "results" / "entry_topology_backend_matrix" / "nsg" / "no_repair" / "nsg_ep_L240_perq.csv")
    parser.add_argument("--entries", type=int, default=8)
    parser.add_argument("--truth-k", type=int, default=10)
    parser.add_argument("--train-query-fraction", type=float, default=0.6)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args()

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    initial = read_fbin(Path(paths["initial_fbin"]))
    inserts = read_fbin(Path(paths["insert_fbin"]))
    queries = read_fbin(Path(paths["query_fbin"]))
    truth = read_id_bin(Path(paths["truth_dynamic_bin"]))
    all_vectors = np.vstack([initial, inserts]).astype(np.float32, copy=False)
    dynamic_labels = dynamic_labels_from_manifest(manifest)
    hub_ids = parse_gate_hub_ids(args.gate_hub_path, dynamic_labels)
    hub_vectors = all_vectors[np.asarray(hub_ids, dtype=np.int64)]

    train_count = max(1, min(queries.shape[0], int(round(queries.shape[0] * args.train_query_fraction))))
    train_queries = queries[:train_count]
    train_truth = truth[:train_count, : args.truth_k]
    target_centers = np.mean(all_vectors[train_truth], axis=1)
    distances = (
        np.sum(target_centers * target_centers, axis=1, keepdims=True)
        + np.sum(hub_vectors * hub_vectors, axis=1).reshape(1, -1)
        - 2.0 * target_centers @ hub_vectors.T
    )
    target_hubs = hub_vectors[np.argmin(distances, axis=1)]
    weights = read_query_weights(args.baseline_perq, queries.shape[0])[:train_count]
    beta = fit_ridge(train_queries, target_hubs, weights, args.ridge)

    query_aug = np.hstack([queries.astype(np.float64), np.ones((queries.shape[0], 1), dtype=np.float64)])
    predicted = (query_aug @ beta).astype(np.float32)
    pred_distances = (
        np.sum(predicted * predicted, axis=1, keepdims=True)
        + np.sum(hub_vectors * hub_vectors, axis=1).reshape(1, -1)
        - 2.0 * predicted @ hub_vectors.T
    )
    top = np.argpartition(pred_distances, kth=args.entries - 1, axis=1)[:, : args.entries]
    row = np.arange(top.shape[0])[:, None]
    top = top[row, np.argsort(pred_distances[row, top], axis=1)]
    entries = np.asarray(hub_ids, dtype=np.int32)[top]

    args.out_ivecs.parent.mkdir(parents=True, exist_ok=True)
    write_ivecs(args.out_ivecs, entries)
    args.out_anchor_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_anchor_txt.write_text("\n".join(str(value) for value in hub_ids) + "\n", encoding="utf-8")
    meta = {
        "method": "gate_like_learned_query_entries",
        "hub_count": len(hub_ids),
        "entries": args.entries,
        "train_queries": train_count,
        "total_queries": int(queries.shape[0]),
        "truth_k": args.truth_k,
        "ridge": args.ridge,
        "note": "Lightweight query-to-hub ridge selector; closer to GATE's learned entry selection than static hubs, but not paper-faithful two-tower GATE.",
    }
    args.out_ivecs.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
