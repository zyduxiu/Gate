from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor import DensityAnchorRouter, DynamicKNNGraph, MedoidRouter, RandomAnchorRouter
from dynanchor.data import make_streaming_drift_scenario
from dynanchor.evaluation import evaluate_router
from dynanchor.external import CommandResult, run_command, run_external_binary, to_wsl_path
from dynanchor.io_formats import write_diskann_fbin, write_diskann_gt


def write_u32_list(path: Path, values: list[int] | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype="<u4")
    header = np.asarray([arr.size], dtype="<u4")
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(arr.tobytes())


def exact_knn_labels(
    base: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    k: int,
    batch_size: int = 128,
) -> np.ndarray:
    base = np.asarray(base, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint32)
    queries = np.asarray(queries, dtype=np.float32)
    result = np.empty((queries.shape[0], k), dtype=np.uint32)
    base_norm = np.sum(base * base, axis=1)
    for start in range(0, queries.shape[0], batch_size):
        stop = min(start + batch_size, queries.shape[0])
        q = queries[start:stop]
        dists = np.sum(q * q, axis=1, keepdims=True) + base_norm.reshape(1, -1) - 2.0 * q @ base.T
        part = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        row = np.arange(part.shape[0])[:, None]
        ordered = part[row, np.argsort(dists[row, part], axis=1)]
        result[start:stop] = labels[ordered]
    return result


def build_routers(anchor_count: int, entry_count: int, seed: int):
    return [
        MedoidRouter(),
        RandomAnchorRouter(m=anchor_count, s=entry_count, seed=seed + 11),
        DensityAnchorRouter(m=anchor_count, s=entry_count, dynamic=False),
        DensityAnchorRouter(m=anchor_count, s=entry_count, dynamic=True),
    ]


def write_anchor_file(path: Path, anchors: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(x)) for x in anchors) + "\n", encoding="utf-8")


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


def compile_runner(force: bool = False, timeout: int = 600) -> Path:
    source = ROOT / "tools" / "hnsw_anchor_runner.cpp"
    binary = ROOT / "build" / ("hnsw_anchor_runner.exe" if os.name != "nt" else "hnsw_anchor_runner")
    binary.parent.mkdir(parents=True, exist_ok=True)
    if (
        not force
        and binary.exists()
        and binary.stat().st_mtime >= source.stat().st_mtime
    ):
        return binary

    if os.name == "nt":
        root_wsl = to_wsl_path(str(ROOT.resolve()))
        command = (
            f"cd {shlex.quote(root_wsl)} && "
            "mkdir -p build && "
            "g++ -std=c++17 -O3 -march=native -fopenmp "
            "-Ithird_party/hnswlib "
            "tools/hnsw_anchor_runner.cpp -o build/hnsw_anchor_runner"
        )
        result = run_command(["wsl", "bash", "-lc", command], timeout=timeout)
    else:
        result = run_command(
            [
                "g++",
                "-std=c++17",
                "-O3",
                "-march=native",
                "-fopenmp",
                "-Ithird_party/hnswlib",
                str(source),
                "-o",
                str(binary),
            ],
            cwd=ROOT,
            timeout=timeout,
        )
    save_command(ROOT / "build" / "hnsw_anchor_runner_build.json", result)
    if not result.ok:
        raise RuntimeError("failed to compile hnsw_anchor_runner; see build/hnsw_anchor_runner_build.json")
    return binary


def run_runner(
    runner: Path,
    data_paths: dict[str, Path],
    out_json: Path,
    mode: str,
    ef: int,
    entries: int,
    topk: int,
    m: int,
    ef_construction: int,
    seed: int,
    timeout: int,
    anchors_txt: Path | None = None,
) -> dict[str, Any]:
    command = [
        str(runner.resolve()),
        "--initial-fbin",
        str(data_paths["initial"].resolve()),
        "--insert-fbin",
        str(data_paths["insert"].resolve()),
        "--query-fbin",
        str(data_paths["queries"].resolve()),
        "--truth-bin",
        str(data_paths["truth"].resolve()),
        "--delete-u32",
        str(data_paths["deletes"].resolve()),
        "--mode",
        mode,
        "--out-json",
        str(out_json.resolve()),
        "--ef",
        str(ef),
        "--entries",
        str(entries),
        "--topk",
        str(topk),
        "--M",
        str(m),
        "--ef-construction",
        str(ef_construction),
        "--seed",
        str(seed),
    ]
    if anchors_txt is not None:
        command.extend(["--anchors-txt", str(anchors_txt.resolve())])
    result = run_external_binary(command, timeout=timeout)
    save_command(out_json.with_suffix(".log.json"), result)
    if not result.ok:
        raise RuntimeError(f"hnsw runner failed for {mode} ef={ef}; see {out_json.with_suffix('.log.json')}")
    return json.loads(out_json.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "method",
        "ef",
        "entries",
        "anchors",
        "recall_at_10",
        "avg_cmps",
        "avg_hops",
        "avg_latency_us",
        "p99_latency_us",
        "qps",
        "build_seconds",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dynamic-anchor entry routing on a compiled hnswlib backend.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-initial", type=int, default=10_000)
    parser.add_argument("--n-insert", type=int, default=2_000)
    parser.add_argument("--n-delete", type=int, default=1_000)
    parser.add_argument("--n-queries", type=int, default=1_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--graph-k", type=int, default=16)
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--entries", type=int, default=8)
    parser.add_argument("--efs", default="20,40,80,120,240")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "hnsw_anchor_10k")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "hnsw_anchor_10k")
    args = parser.parse_args()

    efs = [int(x.strip()) for x in args.efs.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    scenario = make_streaming_drift_scenario(
        n_initial=args.n_initial,
        n_insert=args.n_insert,
        n_delete=args.n_delete,
        n_queries=args.n_queries,
        dim=args.dim,
        seed=args.seed,
    )
    inserts = np.vstack(scenario.insert_batches).astype(np.float32)
    all_vectors = np.vstack([scenario.initial, inserts]).astype(np.float32)
    alive = np.ones(all_vectors.shape[0], dtype=bool)
    alive[np.asarray(scenario.delete_ids, dtype=np.int64)] = False
    alive_labels = np.flatnonzero(alive).astype(np.uint32)
    truth = exact_knn_labels(
        all_vectors[alive],
        alive_labels,
        scenario.queries_after,
        k=args.topk,
    )

    data_paths = {
        "initial": args.data_dir / "initial.fbin",
        "insert": args.data_dir / "insert.fbin",
        "queries": args.data_dir / "query_after.fbin",
        "truth": args.data_dir / "truth_labels.bin",
        "deletes": args.data_dir / "delete_ids.u32",
    }
    write_diskann_fbin(data_paths["initial"], scenario.initial)
    write_diskann_fbin(data_paths["insert"], inserts)
    write_diskann_fbin(data_paths["queries"], scenario.queries_after)
    write_diskann_gt(data_paths["truth"], truth)
    write_u32_list(data_paths["deletes"], scenario.delete_ids)

    graph = DynamicKNNGraph(scenario.initial, k=args.graph_k, seed=args.seed)
    routers = build_routers(args.anchors, args.entries, args.seed)
    for router in routers:
        router.fit(graph)

    # Warm the adaptive router statistics on the pre-drift workload, matching run_fair_comparison.py.
    for router in routers:
        evaluate_router(
            graph,
            router,
            scenario.queries_before,
            phase="before_drift",
            k=args.topk,
            ef=max(efs),
            s=args.entries,
            early_stop=False,
        )

    for batch in scenario.insert_batches:
        graph.insert(batch)
    graph.delete(scenario.delete_ids)
    maintenance = {router.name: [router.maintain(graph) for _ in range(3)] for router in routers}
    (args.out_dir / "maintenance.json").write_text(json.dumps(maintenance, indent=2), encoding="utf-8")

    anchor_files: dict[str, Path] = {}
    for router in routers:
        if router.name not in {"fixed_medoid", "random_multi", "static_density", "dynamic_density"}:
            continue
        anchor_path = args.out_dir / "anchors" / f"{router.name}.txt"
        write_anchor_file(anchor_path, router.anchor_ids())
        anchor_files[router.name] = anchor_path

    manifest = {
        "n_initial": args.n_initial,
        "n_insert": args.n_insert,
        "n_delete": args.n_delete,
        "n_queries": args.n_queries,
        "dim": args.dim,
        "topk": args.topk,
        "seed": args.seed,
        "paths": {key: str(path.resolve()) for key, path in data_paths.items()},
        "anchor_files": {key: str(path.resolve()) for key, path in anchor_files.items()},
    }
    (args.data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    runner = compile_runner(force=args.force_build, timeout=args.timeout)
    rows: list[dict[str, Any]] = []

    for ef in efs:
        payload = run_runner(
            runner=runner,
            data_paths=data_paths,
            out_json=args.out_dir / f"builtin_ef{ef}.json",
            mode="builtin",
            ef=ef,
            entries=1,
            topk=args.topk,
            m=args.hnsw_m,
            ef_construction=args.hnsw_ef_construction,
            seed=args.seed,
            timeout=args.timeout,
        )
        rows.append(
            {
                "backend": "hnswlib_cpp",
                "method": "builtin_hnsw_entry",
                "ef": ef,
                "entries": 1,
                "anchors": 0,
                "recall_at_10": payload["recall_at_10"],
                "avg_cmps": payload["avg_cmps"],
                "avg_hops": payload["avg_hops"],
                "avg_latency_us": payload["avg_latency_us"],
                "p99_latency_us": payload["p99_latency_us"],
                "qps": payload["qps"],
                "build_seconds": payload["build_seconds"],
                "note": "standard hnswlib search after dynamic insert/delete; no rebuild",
            }
        )

        for method in ["fixed_medoid", "static_density", "dynamic_density"]:
            payload = run_runner(
                runner=runner,
                data_paths=data_paths,
                out_json=args.out_dir / f"{method}_ef{ef}.json",
                mode="anchors",
                ef=ef,
                entries=args.entries,
                topk=args.topk,
                m=args.hnsw_m,
                ef_construction=args.hnsw_ef_construction,
                seed=args.seed,
                timeout=args.timeout,
                anchors_txt=anchor_files[method],
            )
            rows.append(
                {
                    "backend": "hnswlib_cpp_base_layer",
                    "method": method,
                    "ef": ef,
                    "entries": args.entries,
                    "anchors": payload["anchors"],
                    "recall_at_10": payload["recall_at_10"],
                    "avg_cmps": payload["avg_cmps"],
                    "avg_hops": payload["avg_hops"],
                    "avg_latency_us": payload["avg_latency_us"],
                    "p99_latency_us": payload["p99_latency_us"],
                    "qps": payload["qps"],
                    "build_seconds": payload["build_seconds"],
                    "note": "anchor-selected hnswlib layer-0 search after dynamic insert/delete; no rebuild",
                }
            )

    write_rows(args.out_dir / "hnsw_anchor_fair.csv", rows)
    (args.out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print(f"Wrote {args.out_dir / 'hnsw_anchor_fair.csv'}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
