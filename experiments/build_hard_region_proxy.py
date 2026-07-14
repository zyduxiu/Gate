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


def read_per_query_costs(path: Path, rows: int) -> np.ndarray:
    costs = np.zeros(rows, dtype=np.float64)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qi = int(row["query_id"])
            if 0 <= qi < rows:
                costs[qi] = float(row["distance_computations"])
    return costs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hard-query region repair proxy from stale-entry per-query costs.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift100k_dynamic" / "manifest.json")
    parser.add_argument("--baseline-perq", type=Path, default=ROOT / "results" / "entry_topology_backend_matrix" / "nsg" / "no_repair" / "nsg_ep_L240_perq.csv")
    parser.add_argument("--out-ivecs", type=Path, default=ROOT / "results" / "entry_topology_backend_matrix" / "hard_region_proxy.ivecs")
    parser.add_argument("--train-query-fraction", type=float, default=0.6)
    parser.add_argument("--top-fraction", type=float, default=0.25)
    parser.add_argument("--truth-k", type=int, default=24)
    args = parser.parse_args()

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    truth = read_id_bin(Path(manifest["paths"]["truth_dynamic_bin"]))
    costs = read_per_query_costs(args.baseline_perq, truth.shape[0])
    train_count = max(1, min(truth.shape[0], int(round(truth.shape[0] * args.train_query_fraction))))
    hard_count = max(1, int(round(train_count * args.top_fraction)))
    train_costs = costs[:train_count]
    hard_query_ids = np.argsort(train_costs)[-hard_count:][::-1]
    regions = truth[hard_query_ids, : args.truth_k]

    args.out_ivecs.parent.mkdir(parents=True, exist_ok=True)
    write_ivecs(args.out_ivecs, regions)
    meta = {
        "method": "hard_query_region_proxy",
        "train_queries": train_count,
        "hard_queries": hard_count,
        "top_fraction": args.top_fraction,
        "truth_k": args.truth_k,
        "source_perq": str(args.baseline_perq),
        "note": "Uses stale-entry hard training queries and their exact-NN vicinity as an RFix/NGFix-like repair proxy; not a paper-faithful implementation.",
    }
    args.out_ivecs.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
