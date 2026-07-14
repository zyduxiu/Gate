from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


def read_perq(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty per-query csv: {path}")
    return {
        "query_id": np.asarray([int(row["query_id"]) for row in rows], dtype=np.int32),
        "recall_at_10": np.asarray([float(row["recall_at_10"]) for row in rows], dtype=np.float64),
        "expanded": np.asarray([float(row["expanded"]) for row in rows], dtype=np.float64),
        "distance_computations": np.asarray([float(row["distance_computations"]) for row in rows], dtype=np.float64),
        "latency_us": np.asarray([float(row["latency_us"]) for row in rows], dtype=np.float64),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bin",
        "hardest_fraction",
        "queries",
        "method",
        "recall_at_10",
        "avg_expanded",
        "avg_distance_computations",
        "avg_latency_us",
        "p95_latency_us",
        "p99_latency_us",
        "delta_comps_vs_baseline",
        "pct_comps_vs_baseline",
        "delta_latency_vs_baseline_us",
        "pct_latency_vs_baseline",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, float]:
    return {
        "recall_at_10": float(np.mean(values["recall_at_10"][indices])),
        "avg_expanded": float(np.mean(values["expanded"][indices])),
        "avg_distance_computations": float(np.mean(values["distance_computations"][indices])),
        "avg_latency_us": float(np.mean(values["latency_us"][indices])),
        "p95_latency_us": float(np.percentile(values["latency_us"][indices], 95)),
        "p99_latency_us": float(np.percentile(values["latency_us"][indices], 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hard-query bins from per-query NSG runner CSVs.")
    parser.add_argument("--dir", type=Path, default=Path("results/stress_hard_bins_perq"))
    parser.add_argument("--methods", default="nsg_ep,gate_initial_hub,gate_refreshed_hub,online_hub_density")
    parser.add_argument("--baseline", default="gate_initial_hub")
    parser.add_argument("--search-l", type=int, default=240)
    parser.add_argument("--fractions", default="1.0,0.5,0.2,0.1,0.05")
    parser.add_argument("--out", type=Path, default=Path("results/stress_hard_bins_perq/hard_query_bins.csv"))
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    data: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        path = args.dir / f"{method}_L{args.search_l}_perq.csv"
        data[method] = read_perq(path)
    if args.baseline not in data:
        raise KeyError(f"baseline {args.baseline} not in methods")
    baseline = data[args.baseline]
    order = np.argsort(-baseline["distance_computations"])
    rows: list[dict[str, Any]] = []
    for frac in [float(item.strip()) for item in args.fractions.split(",") if item.strip()]:
        take = max(1, int(round(len(order) * frac)))
        indices = order[:take]
        base_summary = summarize(baseline, indices)
        bin_name = "all" if frac >= 0.999 else f"top_{int(round(frac * 100))}pct_hard"
        for method in methods:
            stats = summarize(data[method], indices)
            delta_comps = stats["avg_distance_computations"] - base_summary["avg_distance_computations"]
            delta_latency = stats["avg_latency_us"] - base_summary["avg_latency_us"]
            rows.append(
                {
                    "bin": bin_name,
                    "hardest_fraction": frac,
                    "queries": int(take),
                    "method": method,
                    **stats,
                    "delta_comps_vs_baseline": float(delta_comps),
                    "pct_comps_vs_baseline": float(delta_comps / max(base_summary["avg_distance_computations"], 1e-9) * 100.0),
                    "delta_latency_vs_baseline_us": float(delta_latency),
                    "pct_latency_vs_baseline": float(delta_latency / max(base_summary["avg_latency_us"], 1e-9) * 100.0),
                }
            )
    write_csv(args.out, rows)
    print(f"Wrote {args.out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
