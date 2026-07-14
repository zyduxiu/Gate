from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (q / 100.0) * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "hardness_source",
        "bin",
        "method",
        "queries",
        "baseline_min_distance_computations",
        "baseline_max_distance_computations",
        "recall_at_10",
        "avg_distance_computations",
        "avg_expanded",
        "avg_latency_us",
        "p99_latency_us",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-query results by baseline hard-query bins.")
    parser.add_argument("--dir", type=Path, default=Path("results/sift100k_dynamic_nsg_entry"))
    parser.add_argument("--baseline", default="nsg_ep")
    parser.add_argument("--methods", default="nsg_ep,fixed_medoid,static_density,dynamic_density")
    parser.add_argument("--search-l", type=int, default=240)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/sift100k_dynamic_nsg_entry/hard_bins_l240.csv"))
    args = parser.parse_args()

    baseline_path = args.dir / f"{args.baseline}_L{args.search_l}_perq.csv"
    baseline_rows = read_csv(baseline_path)
    baseline_costs = [float(row["distance_computations"]) for row in baseline_rows]
    order = sorted(range(len(baseline_costs)), key=lambda idx: baseline_costs[idx])
    bins: list[list[int]] = []
    for bin_id in range(args.bins):
        start = (len(order) * bin_id) // args.bins
        stop = (len(order) * (bin_id + 1)) // args.bins
        bins.append(order[start:stop])

    rows: list[dict[str, Any]] = []
    for method in [item.strip() for item in args.methods.split(",") if item.strip()]:
        method_rows = read_csv(args.dir / f"{method}_L{args.search_l}_perq.csv")
        for bin_id, indices in enumerate(bins, start=1):
            recalls = [float(method_rows[idx]["recall_at_10"]) for idx in indices]
            distances = [float(method_rows[idx]["distance_computations"]) for idx in indices]
            expanded = [float(method_rows[idx]["expanded"]) for idx in indices]
            latencies = [float(method_rows[idx]["latency_us"]) for idx in indices]
            baseline_bin_costs = [baseline_costs[idx] for idx in indices]
            rows.append(
                {
                    "hardness_source": f"{args.baseline}_distance_computations_L{args.search_l}",
                    "bin": f"Q{bin_id}",
                    "method": method,
                    "queries": len(indices),
                    "baseline_min_distance_computations": min(baseline_bin_costs),
                    "baseline_max_distance_computations": max(baseline_bin_costs),
                    "recall_at_10": mean(recalls),
                    "avg_distance_computations": mean(distances),
                    "avg_expanded": mean(expanded),
                    "avg_latency_us": mean(latencies),
                    "p99_latency_us": percentile(latencies, 99.0),
                }
            )

    write_csv(args.out, rows)
    print(f"Wrote {args.out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
