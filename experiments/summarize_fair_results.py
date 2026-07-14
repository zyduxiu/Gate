from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "method",
        "budget_name",
        "budget",
        "recall_at_10",
        "avg_visited_or_cmps",
        "avg_hops",
        "p99_latency",
        "latency_unit",
        "qps_best",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fair comparison recall-cost results.")
    parser.add_argument("--fair-dir", type=Path, default=Path("results/fair_10k"))
    parser.add_argument("--hnsw-dir", type=Path, default=Path("results/hnsw_anchor_10k"))
    parser.add_argument("--gate-dir", type=Path, default=Path("results/gate_adapter_10k"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    own = read_csv(args.fair_dir / "own_ef_sweep.csv")
    for row in own:
        if row["phase"] != "after_drift":
            continue
        if row["router"] not in {"fixed_medoid", "static_density", "dynamic_density"}:
            continue
        rows.append(
            {
                "backend": "own_python_graph",
                "method": row["router"],
                "budget_name": "ef",
                "budget": int(float(row["ef"])),
                "recall_at_10": float(row["recall_at_k"]),
                "avg_visited_or_cmps": float(row["avg_visited"]),
                "avg_hops": float(row["avg_depth"]),
                "p99_latency": float(row["p99_latency_ms"]),
                "latency_unit": "ms",
                "qps_best": "",
                "note": "dynamic update workload; no graph rebuild; Python prototype",
            }
        )

    external = read_csv(args.fair_dir / "external_baselines.csv")
    for row in external:
        backend = row["backend"]
        if backend == "diskann":
            budget = int(row["setting"].split("=")[1])
            rows.append(
                {
                    "backend": "diskann",
                    "method": "medoid_start_rebuilt_final_index",
                    "budget_name": "search_l",
                    "budget": budget,
                    "recall_at_10": float(row["recall_at_10"]),
                    "avg_visited_or_cmps": float(row["avg_cmps"]),
                    "avg_hops": float(row["avg_hops"]),
                    "p99_latency": float(row["p99_latency_us_best"]),
                    "latency_unit": "us",
                    "qps_best": float(row["qps_best"]),
                    "note": "rebuilt on final alive data; favorable to baseline",
                }
            )
        elif backend == "nsg":
            rows.append(
                {
                    "backend": "nsg",
                    "method": "rebuilt_final_index",
                    "budget_name": "search_l",
                    "budget": int(row["setting"].split("=")[1]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "avg_visited_or_cmps": "",
                    "avg_hops": "",
                    "p99_latency": "",
                    "latency_unit": "",
                    "qps_best": "",
                    "note": "rebuilt on final alive data; favorable to baseline",
                }
            )
        elif backend == "sptag":
            rows.append(
                {
                    "backend": "sptag",
                    "method": "rebuilt_final_index",
                    "budget_name": "maxcheck",
                    "budget": int(row["setting"].split("=")[1]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "avg_visited_or_cmps": "",
                    "avg_hops": "",
                    "p99_latency": "",
                    "latency_unit": "",
                    "qps_best": "",
                    "note": "rebuilt on final alive data; favorable to baseline",
                }
            )

    hnsw_path = args.hnsw_dir / "hnsw_anchor_fair.csv"
    if hnsw_path.exists():
        for row in read_csv(hnsw_path):
            rows.append(
                {
                    "backend": row["backend"],
                    "method": row["method"],
                    "budget_name": "ef",
                    "budget": int(row["ef"]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "avg_visited_or_cmps": float(row["avg_cmps"]),
                    "avg_hops": float(row["avg_hops"]),
                    "p99_latency": float(row["p99_latency_us"]),
                    "latency_unit": "us",
                    "qps_best": float(row["qps"]),
                    "note": row["note"],
                }
            )

    gate_path = args.gate_dir / "gate_adapter_results.csv"
    if gate_path.exists():
        for row in read_csv(gate_path):
            rows.append(
                {
                    "backend": row["backend"],
                    "method": row["method"],
                    "budget_name": "search_l",
                    "budget": int(row["search_l"]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "avg_visited_or_cmps": "",
                    "avg_hops": "",
                    "p99_latency": "",
                    "latency_unit": "",
                    "qps_best": float(row["qps"]) if row["qps"] else "",
                    "note": row["note"],
                }
            )

    rows.sort(key=lambda r: (str(r["backend"]), str(r["method"]), int(r["budget"])))
    write_csv(args.fair_dir / "fair_recall_cost.csv", rows)
    (args.fair_dir / "fair_recall_cost.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {args.fair_dir / 'fair_recall_cost.csv'}")


if __name__ == "__main__":
    main()
