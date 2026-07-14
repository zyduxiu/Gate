from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor import DensityAnchorRouter, DynamicKNNGraph, MedoidRouter
from dynanchor.anchors import BaseRouter
from dynanchor.data import make_streaming_drift_scenario
from dynanchor.metrics import percentile, recall_at_k, safe_mean


def build_routers(anchor_count: int, entry_count: int):
    return [
        MedoidRouter(),
        DensityAnchorRouter(m=anchor_count, s=entry_count, dynamic=False),
        DensityAnchorRouter(m=anchor_count, s=entry_count, dynamic=True),
    ]


def compute_truth(graph: DynamicKNNGraph, queries: Iterable[np.ndarray], k: int) -> list[list[int]]:
    return [graph.exact_search(query, k=k)[0] for query in queries]


def evaluate_with_truth(
    graph: DynamicKNNGraph,
    router: BaseRouter,
    queries: Iterable[np.ndarray],
    truths: list[list[int]],
    phase: str,
    topology: str,
    k: int,
    ef: int,
    entries: int,
) -> dict[str, Any]:
    recalls: list[float] = []
    visited: list[float] = []
    depths: list[float] = []
    latencies: list[float] = []
    query_count = 0

    for query, truth in zip(queries, truths):
        query_count += 1
        selected_entries = router.select(query, graph, s=entries)
        result = graph.search(query, selected_entries, topk=k, ef=ef, early_stop=False)
        recall = recall_at_k(result.ids, truth, k)
        router.after_query(query, graph, result, recall)
        recalls.append(recall)
        visited.append(float(result.visited))
        depths.append(float(result.max_depth))
        latencies.append(float(result.latency_ms))

    return {
        "topology": topology,
        "phase": phase,
        "router": router.name,
        "ef": ef,
        "recall_at_k": safe_mean(recalls),
        "avg_visited": safe_mean(visited),
        "p95_visited": percentile(visited, 95),
        "avg_depth": safe_mean(depths),
        "avg_latency_ms": safe_mean(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "load_gini": router.load_gini(),
        "anchors": len(router.anchor_ids()),
        "queries": query_count,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "topology",
        "phase",
        "router",
        "ef",
        "recall_at_k",
        "avg_visited",
        "p95_visited",
        "avg_depth",
        "avg_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "load_gini",
        "anchors",
        "queries",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(args: argparse.Namespace, topology: str, ef: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start = time.perf_counter()
    scenario = make_streaming_drift_scenario(
        n_initial=args.n_initial,
        n_insert=args.n_insert,
        n_delete=args.n_delete,
        n_queries=args.n_queries,
        dim=args.dim,
        seed=args.seed,
    )
    graph = DynamicKNNGraph(scenario.initial, k=args.graph_k, seed=args.seed)
    routers = build_routers(args.anchors, args.entries)
    for router in routers:
        router.fit(graph)

    rows: list[dict[str, Any]] = []
    truth_before = compute_truth(graph, scenario.queries_before, args.topk)
    for router in routers:
        rows.append(
            evaluate_with_truth(
                graph,
                router,
                scenario.queries_before,
                truth_before,
                phase="before_drift",
                topology=topology,
                k=args.topk,
                ef=ef,
                entries=args.entries,
            )
        )

    inserted_ids: list[int] = []
    for batch in scenario.insert_batches:
        inserted_ids.extend(graph.insert(batch))
    graph.delete(scenario.delete_ids)

    repair_report: dict[str, Any] = {"mode": topology}
    if topology == "local_repair":
        repair_report |= graph.repair_after_delete(
            scenario.delete_ids,
            inserted_ids=inserted_ids,
            target_degree=args.repair_degree or args.graph_k,
        )
    elif topology != "no_repair":
        raise ValueError("topology must be no_repair or local_repair")

    maintenance = {router.name: [router.maintain(graph) for _ in range(args.maintenance_rounds)] for router in routers}
    truth_after = compute_truth(graph, scenario.queries_after, args.topk)
    for router in routers:
        rows.append(
            evaluate_with_truth(
                graph,
                router,
                scenario.queries_after,
                truth_after,
                phase="after_drift",
                topology=topology,
                k=args.topk,
                ef=ef,
                entries=args.entries,
            )
        )

    meta = {
        "topology": topology,
        "ef": ef,
        "seconds": time.perf_counter() - start,
        "repair": repair_report,
        "maintenance": maintenance,
    }
    return rows, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="2x2 ablation for graph repair and dynamic entry anchors.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-initial", type=int, default=10_000)
    parser.add_argument("--n-insert", type=int, default=2_000)
    parser.add_argument("--n-delete", type=int, default=1_000)
    parser.add_argument("--n-queries", type=int, default=1_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--graph-k", type=int, default=16)
    parser.add_argument("--repair-degree", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--entries", type=int, default=8)
    parser.add_argument("--efs", default="240")
    parser.add_argument("--topologies", default="no_repair,local_repair")
    parser.add_argument("--maintenance-rounds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "topology_entry_10k")
    args = parser.parse_args()

    efs = [int(x.strip()) for x in args.efs.split(",") if x.strip()]
    topologies = [x.strip() for x in args.topologies.split(",") if x.strip()]
    all_rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []

    for ef in efs:
        for topology in topologies:
            rows, run_meta = run_one(args, topology=topology, ef=ef)
            all_rows.extend(rows)
            meta.append(run_meta)
            after = {row["router"]: row for row in rows if row["phase"] == "after_drift"}
            print(
                f"ef={ef} topology={topology} "
                f"fixed={after['fixed_medoid']['recall_at_k']:.4f} "
                f"static={after['static_density']['recall_at_k']:.4f} "
                f"dynamic={after['dynamic_density']['recall_at_k']:.4f} "
                f"repair_nodes={run_meta['repair'].get('repaired_nodes', 0)}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "topology_entry_ablation.csv"
    json_path = args.out_dir / "topology_entry_ablation.json"
    write_rows(csv_path, all_rows)
    json_path.write_text(
        json.dumps({"config": vars(args), "runs": meta, "results": all_rows}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
