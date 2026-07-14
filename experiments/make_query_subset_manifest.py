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

from dynanchor.io_formats import write_diskann_fbin, write_diskann_gt, write_fvecs, write_ivecs


def read_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
        if header.size != 2:
            raise ValueError(f"invalid fbin header: {path}")
        rows, dim = int(header[0]), int(header[1])
        data = np.fromfile(handle, dtype="<f4", count=rows * dim)
    return data.reshape(rows, dim).astype(np.float32, copy=False)


def read_id_bin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
        if header.size != 2:
            raise ValueError(f"invalid id bin header: {path}")
        rows, cols = int(header[0]), int(header[1])
        data = np.fromfile(handle, dtype="<u4", count=rows * cols)
    return data.reshape(rows, cols).astype(np.uint32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an extreme query-subset manifest from an existing dynamic manifest.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_drift" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--criterion", choices=["top1_inserted", "insert_fraction", "hard_baseline"], default="top1_inserted")
    parser.add_argument("--truth-k", type=int, default=10)
    parser.add_argument("--min-insert-fraction", type=float, default=0.8)
    parser.add_argument("--perq-csv", type=Path, default=None)
    parser.add_argument("--hardest-fraction", type=float, default=0.5)
    args = parser.parse_args()

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    queries = read_fbin(Path(paths["query_fbin"]))
    truth = read_id_bin(Path(paths["truth_dynamic_bin"]))
    initial_rows = int(manifest["n_initial"])

    if args.criterion == "top1_inserted":
        selected = np.flatnonzero(truth[:, 0] >= initial_rows)
    elif args.criterion == "insert_fraction":
        k = min(args.truth_k, truth.shape[1])
        frac = np.mean(truth[:, :k] >= initial_rows, axis=1)
        selected = np.flatnonzero(frac >= args.min_insert_fraction)
    else:
        if args.perq_csv is None:
            raise ValueError("--perq-csv is required for hard_baseline")
        import csv

        costs: list[tuple[int, float]] = []
        with args.perq_csv.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                costs.append((int(row["query_id"]), float(row["distance_computations"])))
        costs.sort(key=lambda x: -x[1])
        take = max(1, int(round(len(costs) * args.hardest_fraction)))
        selected = np.asarray([qid for qid, _ in costs[:take]], dtype=np.int64)

    if selected.size == 0:
        raise ValueError("query subset is empty")
    selected = np.sort(selected.astype(np.int64))
    subset_queries = queries[selected]
    subset_truth = truth[selected]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset_paths = {
        "query_fbin": args.out_dir / "query.fbin",
        "truth_dynamic_bin": args.out_dir / "truth_dynamic_labels.bin",
        "query_fvecs": args.out_dir / "query.fvecs",
        "truth_ivecs": args.out_dir / "truth.ivecs",
        "truth_bin": args.out_dir / "truth.bin",
        "query_ids": args.out_dir / "query_ids.u32",
    }
    write_diskann_fbin(subset_paths["query_fbin"], subset_queries)
    write_diskann_gt(subset_paths["truth_dynamic_bin"], subset_truth)
    write_fvecs(subset_paths["query_fvecs"], subset_queries)
    write_ivecs(subset_paths["truth_ivecs"], subset_truth.astype(np.int32))
    write_diskann_gt(subset_paths["truth_bin"], subset_truth)
    with subset_paths["query_ids"].open("wb") as handle:
        arr = selected.astype("<u4")
        handle.write(np.asarray([arr.size], dtype="<u4").tobytes())
        handle.write(arr.tobytes())

    new_manifest = dict(manifest)
    new_manifest["dataset"] = f"{manifest.get('dataset', 'dynamic')}_{args.criterion}"
    new_manifest["n_query"] = int(selected.size)
    new_manifest["query_subset"] = {
        "source_manifest": str(args.manifest.resolve()),
        "criterion": args.criterion,
        "truth_k": int(args.truth_k),
        "min_insert_fraction": float(args.min_insert_fraction),
        "hardest_fraction": float(args.hardest_fraction),
        "query_count": int(selected.size),
    }
    new_paths = dict(paths)
    for key, value in subset_paths.items():
        new_paths[key] = str(value.resolve())
    new_manifest["paths"] = new_paths
    new_manifest["manifest"] = str((args.out_dir / "manifest.json").resolve())
    (args.out_dir / "manifest.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    print(json.dumps(new_manifest["query_subset"], indent=2))


if __name__ == "__main__":
    main()
