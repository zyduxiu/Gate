from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


def as_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_json_result(path: Path, dataset: str, workload: str) -> dict[str, Any] | None:
    if path.name.endswith(".log.json"):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if "mode" not in payload or "recall_at_10" not in payload:
        return None
    maintenance = payload.get("maintenance", {})
    return {
        "dataset": dataset,
        "workload": workload,
        "backend": payload.get("backend", ""),
        "method": payload["mode"],
        "search_l": int(payload.get("search_l", 0)),
        "entries": int(payload.get("entries", 0)),
        "recall_at_10": as_float(payload.get("recall_at_10")),
        "qps": as_float(payload.get("qps")),
        "avg_latency_us": as_float(payload.get("avg_latency_us")),
        "p99_latency_us": as_float(payload.get("p99_latency_us")),
        "avg_distance_computations": as_float(payload.get("avg_distance_computations")),
        "avg_expanded": as_float(payload.get("avg_expanded")),
        "maintenance_seconds": as_float(maintenance.get("seconds"), 0.0),
        "maintenance_distance_computations": as_float(maintenance.get("distance_computations"), 0.0),
    }


def read_summary_csv(path: Path, dataset: str, workload: str, default_search_l: int, default_entries: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row.get("method")
            if not method:
                continue
            rows.append(
                {
                    "dataset": row.get("dataset", dataset) or dataset,
                    "workload": workload,
                    "backend": row.get("backend", ""),
                    "method": method,
                    "search_l": int(as_float(row.get("search_l"), float(default_search_l))),
                    "entries": int(as_float(row.get("entries"), float(default_entries))),
                    "recall_at_10": as_float(row.get("recall_at_k") or row.get("recall_at_10") or row.get("recall_med") or row.get("recall_median")),
                    "qps": as_float(row.get("qps") or row.get("qps_med") or row.get("qps_median")),
                    "qps_min_source": as_float(row.get("qps_min")),
                    "qps_max_source": as_float(row.get("qps_max")),
                    "avg_latency_us": as_float(row.get("avg_latency_us") or row.get("avg_latency_us_med")),
                    "p99_latency_us": as_float(row.get("p99_latency_us") or row.get("p99_us_med") or row.get("p99_median")),
                    "avg_distance_computations": as_float(
                        row.get("avg_distance_computations") or row.get("comps_med") or row.get("comps_median")
                    ),
                    "avg_expanded": as_float(row.get("avg_expanded")),
                    "maintenance_seconds": as_float(row.get("maintenance_seconds") or row.get("maintenance_seconds_median"), 0.0),
                    "maintenance_distance_computations": as_float(row.get("maintenance_distance_computations"), 0.0),
                    "source_runs": int(as_float(row.get("runs"), 1.0)),
                }
            )
    return rows


def collect_inputs(inputs: list[Path], dataset: str, workload: str, default_search_l: int, default_entries: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in inputs:
        if item.is_dir():
            for path in sorted(item.glob("*.json")):
                parsed = read_json_result(path, dataset, workload)
                if parsed is not None:
                    rows.append(parsed)
            for name in ("summary.csv", "median_summary.csv", "median_with_hybrid_summary.csv"):
                path = item / name
                if path.exists():
                    rows.extend(read_summary_csv(path, dataset, workload, default_search_l, default_entries))
        elif item.suffix.lower() == ".json":
            parsed = read_json_result(item, dataset, workload)
            if parsed is not None:
                rows.append(parsed)
        elif item.suffix.lower() == ".csv":
            rows.extend(read_summary_csv(item, dataset, workload, default_search_l, default_entries))
    return rows


def finite(values: list[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["workload"], row["backend"], row["method"], row["search_l"], row["entries"])
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for key, items in groups.items():
        dataset, workload, backend, method, search_l, entries = key

        def stats(name: str) -> tuple[float, float, float, float, float]:
            vals = finite([as_float(item.get(name)) for item in items])
            if not vals:
                return (math.nan, math.nan, math.nan, math.nan, math.nan)
            return (mean(vals), median(vals), pstdev(vals) if len(vals) > 1 else 0.0, min(vals), max(vals))

        recall = stats("recall_at_10")
        qps = stats("qps")
        p99 = stats("p99_latency_us")
        avg_lat = stats("avg_latency_us")
        comps = stats("avg_distance_computations")
        expanded = stats("avg_expanded")
        maint = stats("maintenance_seconds")
        maint_comps = stats("maintenance_distance_computations")
        source_runs = [int(item.get("source_runs", 1)) for item in items]
        qps_min_sources = finite([as_float(item.get("qps_min_source")) for item in items])
        qps_max_sources = finite([as_float(item.get("qps_max_source")) for item in items])

        out.append(
            {
                "dataset": dataset,
                "workload": workload,
                "backend": backend,
                "method": method,
                "search_l": search_l,
                "entries": entries,
                "runs": sum(source_runs),
                "recall_mean": recall[0],
                "recall_median": recall[1],
                "qps_mean": qps[0],
                "qps_median": qps[1],
                "qps_std": qps[2],
                "qps_min": min(qps_min_sources) if qps_min_sources else qps[3],
                "qps_max": max(qps_max_sources) if qps_max_sources else qps[4],
                "avg_latency_us_mean": avg_lat[0],
                "p99_latency_us_mean": p99[0],
                "p99_latency_us_median": p99[1],
                "avg_distance_computations_mean": comps[0],
                "avg_distance_computations_median": comps[1],
                "avg_expanded_mean": expanded[0],
                "maintenance_seconds_mean": maint[0],
                "maintenance_distance_computations_mean": maint_comps[0],
                "note": "GATE-aligned summary: QPS is the main efficiency metric; avg_expanded is a search-work proxy, not exact routing hops.",
            }
        )
    out.sort(key=lambda row: (str(row["dataset"]), str(row["workload"]), int(row["search_l"]), str(row["method"])))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "workload",
        "backend",
        "method",
        "search_l",
        "entries",
        "runs",
        "recall_mean",
        "recall_median",
        "qps_mean",
        "qps_median",
        "qps_std",
        "qps_min",
        "qps_max",
        "avg_latency_us_mean",
        "p99_latency_us_mean",
        "p99_latency_us_median",
        "avg_distance_computations_mean",
        "avg_distance_computations_median",
        "avg_expanded_mean",
        "maintenance_seconds_mean",
        "maintenance_distance_computations_mean",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a GATE-style recall-QPS summary from dynamic-entry result files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Result dirs or CSV/JSON files.")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--workload", default="full")
    parser.add_argument("--default-search-l", type=int, default=0)
    parser.add_argument("--default-entries", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/gate_aligned_summary.csv"))
    args = parser.parse_args()

    raw_rows = collect_inputs(args.inputs, args.dataset, args.workload, args.default_search_l, args.default_entries)
    if not raw_rows:
        raise SystemExit("No result rows found.")
    rows = aggregate(raw_rows)
    write_csv(args.out, rows)
    print(f"Wrote {args.out} with {len(rows)} rows from {len(raw_rows)} raw result rows.")


if __name__ == "__main__":
    main()
