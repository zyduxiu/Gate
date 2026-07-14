from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.external import CommandResult, gate_paths, run_external_binary
from dynanchor.io_formats import load_manifest, read_fvecs, recall_from_ivecs, write_fvecs


def save_command(path: Path, result: CommandResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "command": result.command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def squared_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return (
        np.sum(points * points, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1).reshape(1, -1)
        - 2.0 * points @ centers.T
    )


def kmeans(points: np.ndarray, k: int, iterations: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    n, dim = points.shape
    if n == 0:
        raise ValueError("cannot cluster empty points")
    k = max(1, min(int(k), n))
    init = rng.choice(n, size=k, replace=False)
    centers = points[init].copy()
    labels = np.zeros(n, dtype=np.int32)

    for _ in range(max(1, int(iterations))):
        distances = squared_distances(points, centers)
        labels = np.argmin(distances, axis=1).astype(np.int32)
        nearest = np.min(distances, axis=1)
        for cluster_id in range(k):
            members = labels == cluster_id
            if np.any(members):
                centers[cluster_id] = np.mean(points[members], axis=0)
            else:
                farthest = int(np.argmax(nearest))
                centers[cluster_id] = points[farthest]
                labels[farthest] = cluster_id
                nearest[farthest] = 0.0
    labels = np.argmin(squared_distances(points, centers), axis=1).astype(np.int32)
    return labels, centers.astype(np.float32)


def medoid_id(points: np.ndarray, ids: np.ndarray, center: np.ndarray) -> int:
    distances = np.sum((points - center.reshape(1, -1)) ** 2, axis=1)
    return int(ids[int(np.argmin(distances))])


def build_gate_hubs(
    base: np.ndarray,
    hub_path: Path,
    embedding_path: Path,
    stage1: int,
    stage2: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    base = np.asarray(base, dtype=np.float32)
    labels1, centers1 = kmeans(base, stage1, iterations, rng)
    dim = base.shape[1]
    eps_by_stage1: list[list[int]] = []
    embeddings: list[np.ndarray] = []
    global_ids = np.arange(base.shape[0], dtype=np.int32)

    for cluster_id in range(centers1.shape[0]):
        ids1 = global_ids[labels1 == cluster_id]
        if ids1.size == 0:
            fallback = medoid_id(base, global_ids, centers1[cluster_id])
            ids1 = np.asarray([fallback], dtype=np.int32)
        points1 = base[ids1]
        sub_k = min(stage2, points1.shape[0])
        labels2, centers2 = kmeans(points1, sub_k, iterations, rng)
        eps: list[int] = []
        sub_embeddings: list[np.ndarray] = []
        for sub_id in range(stage2):
            if sub_id < sub_k:
                ids2 = ids1[labels2 == sub_id]
                if ids2.size == 0:
                    ep = medoid_id(points1, ids1, centers2[sub_id])
                    centroid = base[ep]
                else:
                    centroid = np.mean(base[ids2], axis=0).astype(np.float32)
                    ep = medoid_id(base[ids2], ids2, centroid)
            else:
                ep = eps[-1] if eps else medoid_id(points1, ids1, centers1[cluster_id])
                centroid = base[ep]
            eps.append(int(ep))
            sub_embeddings.append(np.asarray(centroid, dtype=np.float32))
        eps_by_stage1.append(eps)
        embeddings.extend(sub_embeddings)

    hub_path.parent.mkdir(parents=True, exist_ok=True)
    with hub_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{centers1.shape[0]} {stage2} {dim}\n")
        for center in centers1:
            handle.write(" ".join(f"{float(x):.9g}" for x in center) + "\n")
        for eps in eps_by_stage1:
            handle.write(" ".join(str(x) for x in eps) + "\n")

    embedding_matrix = np.vstack(embeddings).astype(np.float32)
    write_fvecs(embedding_path, embedding_matrix)
    return {
        "stage1": int(centers1.shape[0]),
        "stage2": int(stage2),
        "anchors": int(embedding_matrix.shape[0]),
        "dim": int(dim),
        "hub_path": str(hub_path.resolve()),
        "embedding_path": str(embedding_path.resolve()),
    }


def parse_gate_stdout(stdout: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, pattern in {
        "search_time_seconds": r"search time:\s*([0-9.eE+-]+)",
        "qps": r"QPS:\s*([0-9.eE+-]+)",
    }.items():
        match = re.search(pattern, stdout)
        if match:
            parsed[key] = float(match.group(1))
    return parsed


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "method",
        "search_l",
        "recall_at_10",
        "qps",
        "search_time_seconds",
        "wall_seconds",
        "anchors",
        "stage1",
        "stage2",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GATE-style static hub entry overlay baseline.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "fair_10k" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "gate_adapter_10k")
    parser.add_argument("--stage1", type=int, default=16)
    parser.add_argument("--stage2", type=int, default=8)
    parser.add_argument("--kmeans-iters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--search-l", default="80,120,240")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--build-l", type=int, default=40)
    parser.add_argument("--build-r", type=int, default=50)
    parser.add_argument("--build-c", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--reuse-index", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    paths = gate_paths()
    nsg_index_exe = paths["nsg_index_exe"]
    gate_search_exe = paths["gate_search_exe"]
    if nsg_index_exe is None or gate_search_exe is None:
        raise FileNotFoundError("GATE executables missing; run .\\scripts\\build_external_baselines.ps1 -Baseline gate -UseWsl")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = read_fvecs(Path(manifest["paths"]["base_fvecs"]))
    hub_path = args.out_dir / "gate_hubs.txt"
    embedding_path = args.out_dir / "gate_centroid_embeddings.fvecs"
    hub_meta = build_gate_hubs(
        base=base,
        hub_path=hub_path,
        embedding_path=embedding_path,
        stage1=args.stage1,
        stage2=args.stage2,
        iterations=args.kmeans_iters,
        seed=args.seed,
    )

    index_path = args.out_dir / "gate_final.nsg"
    if not args.reuse_index or not index_path.exists():
        result = run_external_binary(
            [
                nsg_index_exe,
                manifest["paths"]["base_fvecs"],
                manifest["paths"]["nsg_knn_graph"],
                str(args.build_l),
                str(args.build_r),
                str(args.build_c),
                str(index_path.resolve()),
            ],
            timeout=args.timeout,
        )
        save_command(args.out_dir / "gate_build_log.json", result)
        if not result.ok:
            raise RuntimeError("GATE NSG build failed; see gate_build_log.json")

    rows: list[dict[str, Any]] = []
    search_l_values = [int(x.strip()) for x in args.search_l.split(",") if x.strip()]
    for search_l in search_l_values:
        result_path = args.out_dir / f"gate_result_l{search_l}.ivecs"
        start = time.perf_counter()
        result = run_external_binary(
            [
                gate_search_exe,
                manifest["paths"]["base_fvecs"],
                manifest["paths"]["query_fvecs"],
                str(index_path.resolve()),
                str(search_l),
                str(args.topk),
                str(hub_path.resolve()),
                str(embedding_path.resolve()),
                str(result_path.resolve()),
            ],
            timeout=args.timeout,
        )
        wall = time.perf_counter() - start
        save_command(args.out_dir / f"gate_search_l{search_l}_log.json", result)
        if not result.ok:
            raise RuntimeError(f"GATE search failed for L={search_l}; see gate_search_l{search_l}_log.json")
        metrics = recall_from_ivecs(result_path, Path(manifest["paths"]["truth_ivecs"]), args.topk)
        parsed = parse_gate_stdout(result.stdout)
        row = {
            "backend": "gate_adapter",
            "method": "static_centroid_hub_entry",
            "search_l": search_l,
            "recall_at_10": metrics["recall"],
            "qps": parsed.get("qps", ""),
            "search_time_seconds": parsed.get("search_time_seconds", ""),
            "wall_seconds": wall,
            "anchors": hub_meta["anchors"],
            "stage1": hub_meta["stage1"],
            "stage2": hub_meta["stage2"],
            "note": "GATE C++ search with generated static two-level centroid hubs; no two-tower training",
        }
        rows.append(row)
        print(row)

    write_rows(args.out_dir / "gate_adapter_results.csv", rows)
    (args.out_dir / "gate_adapter_meta.json").write_text(
        json.dumps(
            {
                "config": vars(args) | {"manifest": str(args.manifest), "out_dir": str(args.out_dir)},
                "hub_meta": hub_meta,
                "gate_paths": paths,
                "rows": rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_dir / 'gate_adapter_results.csv'}")


if __name__ == "__main__":
    main()
