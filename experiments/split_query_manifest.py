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
        values = np.fromfile(handle, dtype="<f4", count=rows * dim)
    if values.size != rows * dim:
        raise ValueError(f"truncated fbin payload: {path}")
    return values.reshape(rows, dim).astype(np.float32, copy=False)


def read_id_bin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
        if header.size != 2:
            raise ValueError(f"invalid id bin header: {path}")
        rows, cols = int(header[0]), int(header[1])
        values = np.fromfile(handle, dtype="<u4", count=rows * cols)
    if values.size != rows * cols:
        raise ValueError(f"truncated id bin payload: {path}")
    return values.reshape(rows, cols).astype(np.uint32, copy=False)


def write_u32_list(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype="<u4")
    with path.open("wb") as handle:
        handle.write(np.asarray([arr.size], dtype="<u4").tobytes())
        handle.write(arr.tobytes())


def write_subset_manifest(source: dict[str, Any], out_dir: Path, name: str, ids: np.ndarray, queries: np.ndarray, truth: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_queries = queries[ids]
    subset_truth = truth[ids]
    subset_paths = {
        "query_fbin": out_dir / "query.fbin",
        "truth_dynamic_bin": out_dir / "truth_dynamic_labels.bin",
        "query_fvecs": out_dir / "query.fvecs",
        "truth_ivecs": out_dir / "truth.ivecs",
        "truth_bin": out_dir / "truth.bin",
        "query_ids": out_dir / "query_ids.u32",
    }
    write_diskann_fbin(subset_paths["query_fbin"], subset_queries)
    write_diskann_gt(subset_paths["truth_dynamic_bin"], subset_truth)
    write_fvecs(subset_paths["query_fvecs"], subset_queries)
    write_ivecs(subset_paths["truth_ivecs"], subset_truth.astype(np.int32))
    write_diskann_gt(subset_paths["truth_bin"], subset_truth)
    write_u32_list(subset_paths["query_ids"], ids)

    manifest = dict(source)
    manifest["dataset"] = f"{source.get('dataset', 'dynamic')}_{name}"
    manifest["n_query"] = int(ids.size)
    manifest["query_split"] = {
        "source_manifest": source.get("manifest", ""),
        "name": name,
        "query_count": int(ids.size),
    }
    paths = dict(source["paths"])
    for key, value in subset_paths.items():
        paths[key] = str(value.resolve())
    manifest["paths"] = paths
    manifest["manifest"] = str((out_dir / "manifest.json").resolve())
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a dynamic query manifest into train/eval query manifests.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_insert80_queries" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "sift_stress_insert80_train_eval")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be in (0, 1)")
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    queries = read_fbin(Path(paths["query_fbin"]))
    truth = read_id_bin(Path(paths["truth_dynamic_bin"]))
    if queries.shape[0] != truth.shape[0]:
        raise ValueError("query/truth row mismatch")

    rng = np.random.default_rng(args.seed)
    order = np.arange(queries.shape[0], dtype=np.uint32)
    rng.shuffle(order)
    train_count = max(1, min(order.size - 1, int(round(order.size * args.train_fraction))))
    train_ids = np.sort(order[:train_count])
    eval_ids = np.sort(order[train_count:])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = write_subset_manifest(manifest, args.out_dir / "train", "train", train_ids, queries, truth)
    eval_manifest = write_subset_manifest(manifest, args.out_dir / "eval", "eval", eval_ids, queries, truth)
    meta = {
        "source_manifest": str(args.manifest.resolve()),
        "train_manifest": str(train_manifest.resolve()),
        "eval_manifest": str(eval_manifest.resolve()),
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "train_queries": int(train_ids.size),
        "eval_queries": int(eval_ids.size),
    }
    (args.out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
