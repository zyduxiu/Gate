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


def build_routers(anchor_count: int, entry_count: int, seed: int):
    return [
        MedoidRouter(),
        RandomAnchorRouter(m=anchor_count, s=entry_count, seed=seed + 11),
        DensityAnchorRouter(m=anchor_count, s=entry_count, dynamic=False),
        DensityAnchorRouter(m=anchor_count, s=entry_count, dynamic=True),
    ]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one_ef(args: argparse.Namespace, ef: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    routers = build_routers(args.anchors, args.entries, args.seed)
    for router in routers:
        router.fit(graph)

    rows: list[dict[str, Any]] = []
    for router in routers:
        row = evaluate_router(
            graph,
            router,
            scenario.queries_before,
            phase="before_drift",
            k=args.topk,
            ef=ef,
            s=args.entries,
            early_stop=False,
        ).to_dict()
        row["ef"] = ef
        rows.append(row)

    for batch in scenario.insert_batches:
        graph.insert(batch)
    graph.delete(scenario.delete_ids)
    maintenance = {router.name: [router.maintain(graph) for _ in range(args.maintenance_rounds)] for router in routers}

    for router in routers:
        row = evaluate_router(
            graph,
            router,
            scenario.queries_after,
            phase="after_drift",
            k=args.topk,
            ef=ef,
            s=args.entries,
            early_stop=False,
        ).to_dict()
        row["ef"] = ef
        rows.append(row)

    meta = {
        "ef": ef,
        "seconds": time.perf_counter() - start,
        "maintenance": maintenance,
    }
    return rows, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ef for dynamic-anchor fair comparison.")
    parser.add_argument("--efs", default="20,40,80,120,240")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-initial", type=int, default=10_000)
    parser.add_argument("--n-insert", type=int, default=2_000)
    parser.add_argument("--n-delete", type=int, default=1_000)
    parser.add_argument("--n-queries", type=int, default=1_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--graph-k", type=int, default=16)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--entries", type=int, default=8)
    parser.add_argument("--maintenance-rounds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "fair_10k")
    args = parser.parse_args()

    efs = [int(x.strip()) for x in args.efs.split(",") if x.strip()]
    all_rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    for ef in efs:
        rows, ef_meta = run_one_ef(args, ef)
        all_rows.extend(rows)
        meta.append(ef_meta)
        after_dynamic = [
            row for row in rows if row["phase"] == "after_drift" and row["router"] == "dynamic_density"
        ][0]
        print(
            f"ef={ef} dynamic_density after_drift "
            f"recall={after_dynamic['recall_at_k']:.4f} "
            f"visited={after_dynamic['avg_visited']:.1f} "
            f"p99_ms={after_dynamic['p99_latency_ms']:.3f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "own_ef_sweep.csv", all_rows)
    (args.out_dir / "own_ef_sweep_meta.json").write_text(
        json.dumps({"config": vars(args), "efs": efs, "runs": meta}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_dir / 'own_ef_sweep.csv'}")


if __name__ == "__main__":
    main()
