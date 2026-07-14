from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor import DensityAnchorRouter, DynamicKNNGraph, MedoidRouter, RandomAnchorRouter
from dynanchor.data import make_streaming_drift_scenario
from dynanchor.evaluation import evaluate_router
from dynanchor.external import (
    CommandResult,
    run_diskann_benchmark,
    run_nsg_build,
    run_nsg_search,
    run_sptag_build,
    run_sptag_search,
    write_diskann_benchmark_config,
)
from dynanchor.io_formats import (
    recall_from_ivecs,
    recall_from_sptag_txt,
    write_external_dataset_from_arrays,
)


def build_routers(anchor_count: int, entry_count: int, seed: int):
    return [
        MedoidRouter(),
        RandomAnchorRouter(m=anchor_count, s=entry_count, seed=seed + 11),
        DensityAnchorRouter(m=anchor_count, s=entry_count, dynamic=False),
        DensityAnchorRouter(m=anchor_count, s=entry_count, dynamic=True),
    ]


def save_command(prefix: Path, result: CommandResult) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": result.command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_diskann_summary(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for job in payload:
        for item in job["results"]["search"]["Topk"]:
            rows.append(
                {
                    "backend": "diskann",
                    "setting": f"search_l={item['search_l']}",
                    "recall_at_10": item["recall"]["average"],
                    "avg_cmps": item["mean_cmps"],
                    "avg_hops": item["mean_hops"],
                    "p99_latency_us_best": min(item["p99_latencies"]),
                    "qps_best": max(item["qps"]),
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a conservative fair comparison on one drift workload.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-initial", type=int, default=10_000)
    parser.add_argument("--n-insert", type=int, default=2_000)
    parser.add_argument("--n-delete", type=int, default=1_000)
    parser.add_argument("--n-queries", type=int, default=1_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--graph-k", type=int, default=16)
    parser.add_argument("--ef", type=int, default=240)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--entries", type=int, default=8)
    parser.add_argument("--gt-k", type=int, default=100)
    parser.add_argument("--nsg-graph-k", type=int, default=100)
    parser.add_argument("--nsg-search-l", type=int, default=120)
    parser.add_argument("--sptag-maxcheck", type=int, default=1024)
    parser.add_argument("--diskann-search-l", default="20,40,80,120,240")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "fair_10k")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "fair_10k")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    scenario = make_streaming_drift_scenario(
        n_initial=args.n_initial,
        n_insert=args.n_insert,
        n_delete=args.n_delete,
        n_queries=args.n_queries,
        dim=args.dim,
        seed=args.seed,
    )
    graph = DynamicKNNGraph(scenario.initial, k=args.graph_k, seed=args.seed)
    routers = build_routers(args.anchors, args.entries, args.seed)
    for router in routers:
        router.fit(graph)

    own_rows = []
    for router in routers:
        summary = evaluate_router(
            graph,
            router,
            scenario.queries_before,
            phase="before_drift",
            k=args.topk,
            ef=args.ef,
            s=args.entries,
            early_stop=False,
        ).to_dict()
        own_rows.append(summary)

    for batch in scenario.insert_batches:
        graph.insert(batch)
    graph.delete(scenario.delete_ids)
    maintenance = {router.name: [router.maintain(graph) for _ in range(3)] for router in routers}

    for router in routers:
        summary = evaluate_router(
            graph,
            router,
            scenario.queries_after,
            phase="after_drift",
            k=args.topk,
            ef=args.ef,
            s=args.entries,
            early_stop=False,
        ).to_dict()
        own_rows.append(summary)

    write_rows(args.out_dir / "own_dynamic_anchor.csv", own_rows)
    (args.out_dir / "maintenance.json").write_text(json.dumps(maintenance, indent=2), encoding="utf-8")
    timings["own_dynamic_anchor_seconds"] = time.perf_counter() - t0

    alive = graph.alive_ids()
    final_base = graph.vectors[alive]
    manifest = write_external_dataset_from_arrays(
        args.data_dir,
        final_base,
        scenario.queries_after,
        gt_k=args.gt_k,
        nsg_graph_k=args.nsg_graph_k,
        metadata={
            "source": "streaming final alive vectors",
            "n_initial": args.n_initial,
            "n_insert": args.n_insert,
            "n_delete": args.n_delete,
            "seed": args.seed,
        },
    )
    timings["export_final_dataset_seconds"] = time.perf_counter() - t0 - timings["own_dynamic_anchor_seconds"]

    external_rows: list[dict[str, Any]] = []
    if not args.skip_external:
        nsg_dir = args.out_dir / "nsg"
        nsg_index = nsg_dir / "final.nsg"
        nsg_result = nsg_dir / "result.ivecs"
        result = run_nsg_build(manifest, nsg_index, L=40, R=50, C=500, timeout=args.timeout)
        save_command(nsg_dir / "build_log", result)
        if not result.ok:
            raise RuntimeError("NSG build failed")
        result = run_nsg_search(
            manifest,
            nsg_index,
            nsg_result,
            search_l=args.nsg_search_l,
            search_k=args.topk,
            timeout=args.timeout,
        )
        save_command(nsg_dir / "search_log", result)
        if not result.ok:
            raise RuntimeError("NSG search failed")
        nsg_metrics = recall_from_ivecs(nsg_result, Path(manifest["paths"]["truth_ivecs"]), args.topk)
        (nsg_dir / "metrics.json").write_text(json.dumps(nsg_metrics, indent=2), encoding="utf-8")
        external_rows.append(
            {
                "backend": "nsg",
                "setting": f"search_l={args.nsg_search_l}",
                "recall_at_10": nsg_metrics["recall"],
            }
        )

        sptag_dir = args.out_dir / "sptag"
        sptag_index = sptag_dir / "index"
        sptag_result = sptag_dir / "result.txt"
        result = run_sptag_build(manifest, sptag_index, algo="BKT", threads=args.threads, timeout=args.timeout)
        save_command(sptag_dir / "build_log", result)
        if not result.ok:
            raise RuntimeError("SPTAG build failed")
        result = run_sptag_search(
            manifest,
            sptag_index,
            sptag_result,
            maxcheck=args.sptag_maxcheck,
            k=args.topk,
            threads=1,
            timeout=args.timeout,
        )
        save_command(sptag_dir / "search_log", result)
        if not result.ok:
            raise RuntimeError("SPTAG search failed")
        sptag_metrics = recall_from_sptag_txt(sptag_result, Path(manifest["paths"]["truth_ivecs"]), args.topk)
        (sptag_dir / "metrics.json").write_text(json.dumps(sptag_metrics, indent=2), encoding="utf-8")
        external_rows.append(
            {
                "backend": "sptag",
                "setting": f"maxcheck={args.sptag_maxcheck}",
                "recall_at_10": sptag_metrics["recall"],
            }
        )

        diskann_dir = args.out_dir / "diskann"
        diskann_config = args.out_dir / "diskann_benchmark.json"
        diskann_output = args.out_dir / "diskann_output.json"
        search_l = [int(x.strip()) for x in args.diskann_search_l.split(",") if x.strip()]
        write_diskann_benchmark_config(
            manifest=manifest,
            config_path=diskann_config,
            output_dir=diskann_dir,
            search_l=search_l,
            max_degree=32,
            l_build=80,
            alpha=1.2,
            threads=args.threads,
            reps=3,
            topk=args.topk,
            start_point_strategy="medoid",
        )
        result = run_diskann_benchmark(diskann_config, diskann_output, timeout=args.timeout)
        save_command(args.out_dir / "diskann_run_log", result)
        if not result.ok:
            raise RuntimeError("DiskANN run failed")
        external_rows.extend(parse_diskann_summary(diskann_output))

    write_rows(args.out_dir / "external_baselines.csv", external_rows)

    fair_rows: list[dict[str, Any]] = []
    for row in own_rows:
        if row["phase"] == "after_drift":
            fair_rows.append(
                {
                    "backend": "own",
                    "setting": row["router"],
                    "recall_at_10": row["recall_at_k"],
                    "avg_visited": row["avg_visited"],
                    "p99_latency_ms": row["p99_latency_ms"],
                    "anchors": row["anchors"],
                }
            )
    fair_rows.extend(external_rows)
    write_rows(args.out_dir / "fair_summary.csv", fair_rows)

    timings["total_seconds"] = time.perf_counter() - t0
    run_config = vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir), "timings": timings}
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {args.out_dir / 'fair_summary.csv'}")
    for row in fair_rows:
        print(row)


if __name__ == "__main__":
    main()
