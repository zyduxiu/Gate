from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a streaming drift demo for dynamic anchors.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-initial", type=int, default=900)
    parser.add_argument("--n-insert", type=int, default=240)
    parser.add_argument("--n-delete", type=int, default=120)
    parser.add_argument("--n-queries", type=int, default=240)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--graph-k", type=int, default=14)
    parser.add_argument("--ef", type=int, default=90)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--anchors", type=int, default=32)
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

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

    rows = []
    for router in routers:
        rows.append(
            evaluate_router(
                graph,
                router,
                scenario.queries_before,
                phase="before_drift",
                k=args.topk,
                ef=args.ef,
                s=args.entries,
            ).to_dict()
        )

    for batch in scenario.insert_batches:
        graph.insert(batch)
    graph.delete(scenario.delete_ids)

    maintenance = {}
    for router in routers:
        reports = []
        for _ in range(3):
            reports.append(router.maintain(graph))
        maintenance[router.name] = reports

    for router in routers:
        rows.append(
            evaluate_router(
                graph,
                router,
                scenario.queries_after,
                phase="after_drift",
                k=args.topk,
                ef=args.ef,
                s=args.entries,
            ).to_dict()
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "streaming_demo_results.json"
    csv_path = args.out_dir / "streaming_demo_results.csv"

    payload = {
        "config": vars(args) | {"out_dir": str(args.out_dir)},
        "maintenance": maintenance,
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    for row in rows:
        print(
            f"{row['phase']:12s} {row['router']:16s} "
            f"recall={row['recall_at_k']:.3f} "
            f"visited={row['avg_visited']:.1f} "
            f"p99_ms={row['p99_latency_ms']:.3f} "
            f"anchors={row['anchors']} "
            f"load_gini={row['load_gini']:.3f}"
        )


if __name__ == "__main__":
    main()
