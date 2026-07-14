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

from dynanchor.io_formats import read_ivecs, write_ivecs


def read_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
        rows, dim = int(header[0]), int(header[1])
        data = np.fromfile(handle, dtype="<f4", count=rows * dim)
    return data.reshape(rows, dim).astype(np.float32, copy=False)


def read_anchors(path: Path) -> np.ndarray:
    return np.asarray([int(x.strip()) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()], dtype=np.int64)


def nearest_anchors(vectors: np.ndarray, queries: np.ndarray, anchors: np.ndarray, take: int) -> np.ndarray:
    anchor_vecs = vectors[anchors]
    dists = (
        np.sum(queries * queries, axis=1, keepdims=True)
        + np.sum(anchor_vecs * anchor_vecs, axis=1).reshape(1, -1)
        - 2.0 * queries @ anchor_vecs.T
    )
    top = np.argpartition(dists, kth=min(take - 1, anchors.size - 1), axis=1)[:, :take]
    row = np.arange(top.shape[0])[:, None]
    top = top[row, np.argsort(dists[row, top], axis=1)]
    return anchors[top].astype(np.int32)


def merge_entries(nearest: np.ndarray, gate: np.ndarray, entries: int, nearest_take: int, gate_take: int, order: str) -> np.ndarray:
    rows: list[list[int]] = []
    for i in range(nearest.shape[0]):
        if order == "nearest_first":
            candidates = list(nearest[i, :nearest_take]) + list(gate[i, :gate_take])
        elif order == "gate_first":
            candidates = list(gate[i, :gate_take]) + list(nearest[i, :nearest_take])
        else:
            candidates = []
            for j in range(max(nearest_take, gate_take)):
                if j < nearest_take:
                    candidates.append(int(nearest[i, j]))
                if j < gate_take:
                    candidates.append(int(gate[i, j]))
        row: list[int] = []
        seen: set[int] = set()
        for item in candidates:
            item = int(item)
            if item < 0 or item in seen:
                continue
            seen.add(item)
            row.append(item)
            if len(row) >= entries:
                break
        while len(row) < entries:
            for item in nearest[i]:
                item = int(item)
                if item not in seen:
                    seen.add(item)
                    row.append(item)
                    break
            else:
                row.append(row[-1] if row else 0)
        rows.append(row[:entries])
    return np.asarray(rows, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid query entries from nearest online anchors and GATE predictions.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-file", type=Path, required=True)
    parser.add_argument("--gate-entries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--nearest-take", type=int, default=2)
    parser.add_argument("--gate-take", type=int, default=2)
    parser.add_argument("--order", choices=["nearest_first", "gate_first", "interleave"], default="nearest_first")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    initial = read_fbin(Path(manifest["paths"]["initial_fbin"]))
    inserts = read_fbin(Path(manifest["paths"]["insert_fbin"]))
    queries = read_fbin(Path(manifest["paths"]["query_fbin"]))
    vectors = np.vstack([initial, inserts]).astype(np.float32, copy=False)
    anchors = read_anchors(args.anchor_file)
    gate = read_ivecs(args.gate_entries)
    if gate.shape[0] != queries.shape[0]:
        raise ValueError(f"query row mismatch: gate={gate.shape[0]} queries={queries.shape[0]}")
    nearest = nearest_anchors(vectors, queries, anchors, max(args.entries, args.nearest_take))
    hybrid = merge_entries(nearest, gate, args.entries, args.nearest_take, args.gate_take, args.order)
    write_ivecs(args.out, hybrid)
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "rows": int(hybrid.shape[0]),
                "entries": int(hybrid.shape[1]),
                "nearest_take": args.nearest_take,
                "gate_take": args.gate_take,
                "order": args.order,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
