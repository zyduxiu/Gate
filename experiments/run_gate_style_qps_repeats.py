from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_METHODS = "nsg_ep,gate_initial_hub,gate_refreshed_hub,online_hub_density,online_density,static_density"
DEFAULT_L1 = "20,40,60,80,100,120,160,200,240"
DEFAULT_L100 = "100,120,160,200,240,280,320,400"


def run_command(command: list[str], timeout: int) -> None:
    print("RUN", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with code {result.returncode}: {' '.join(command)}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite(values: list[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def stats(values: list[float]) -> dict[str, float]:
    vals = finite(values)
    if not vals:
        return {"mean": math.nan, "median": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "workload",
        "backend",
        "method",
        "recall_metric",
        "search_l",
        "entries",
        "runs",
        "recall_at_k_mean",
        "recall_at_k_median",
        "recall_at_k_std",
        "qps_mean",
        "qps_median",
        "qps_std",
        "qps_min",
        "qps_max",
        "avg_latency_us_mean",
        "avg_latency_us_median",
        "p99_latency_us_mean",
        "p99_latency_us_median",
        "avg_distance_computations_mean",
        "avg_distance_computations_median",
        "avg_expanded_mean",
        "avg_expanded_median",
        "maintenance_seconds_mean",
        "maintenance_seconds_median",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summaries(summary_paths: list[Path], workload: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for path in summary_paths:
        for row in read_csv(path):
            recall_metric = row.get("recall_metric") or "recall@10"
            key = (
                row.get("dataset", ""),
                workload,
                row.get("backend", ""),
                row.get("method", ""),
                recall_metric,
                int(to_float(row.get("search_l"), 0.0)),
                int(to_float(row.get("entries"), 0.0)),
            )
            groups[key].append(row)

    out: list[dict[str, Any]] = []
    for key, items in groups.items():
        dataset, workload_name, backend, method, recall_metric, search_l, entries = key

        def column(name: str) -> dict[str, float]:
            return stats([to_float(item.get(name)) for item in items])

        recall = column("recall_at_k")
        qps = column("qps")
        avg_latency = column("avg_latency_us")
        p99 = column("p99_latency_us")
        comps = column("avg_distance_computations")
        expanded = column("avg_expanded")
        maintenance = column("maintenance_seconds")
        out.append(
            {
                "dataset": dataset,
                "workload": workload_name,
                "backend": backend,
                "method": method,
                "recall_metric": recall_metric,
                "search_l": search_l,
                "entries": entries,
                "runs": len(items),
                "recall_at_k_mean": recall["mean"],
                "recall_at_k_median": recall["median"],
                "recall_at_k_std": recall["std"],
                "qps_mean": qps["mean"],
                "qps_median": qps["median"],
                "qps_std": qps["std"],
                "qps_min": qps["min"],
                "qps_max": qps["max"],
                "avg_latency_us_mean": avg_latency["mean"],
                "avg_latency_us_median": avg_latency["median"],
                "p99_latency_us_mean": p99["mean"],
                "p99_latency_us_median": p99["median"],
                "avg_distance_computations_mean": comps["mean"],
                "avg_distance_computations_median": comps["median"],
                "avg_expanded_mean": expanded["mean"],
                "avg_expanded_median": expanded["median"],
                "maintenance_seconds_mean": maintenance["mean"],
                "maintenance_seconds_median": maintenance["median"],
            }
        )
    out.sort(key=lambda row: (row["recall_metric"], int(row["search_l"]), str(row["method"])))
    return out


def run_repeats(args: argparse.Namespace, topk: int, search_l: str, label: str) -> Path:
    top_dir = args.out_dir / label
    summary_paths: list[Path] = []
    for repeat in range(1, args.repeats + 1):
        run_dir = top_dir / f"run_{repeat:02d}"
        summary_path = run_dir / "summary.csv"
        if args.skip_existing and summary_path.exists():
            print(f"SKIP existing {summary_path}", flush=True)
            summary_paths.append(summary_path)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
                sys.executable,
                str((ROOT / "experiments" / "run_nsg_dynamic_entry.py").resolve()),
                "--manifest",
                str(args.manifest.resolve()),
                "--initial-nsg",
                str(args.initial_nsg.resolve()),
                "--runner",
                str(args.runner.resolve()),
                "--out-dir",
                str(run_dir.resolve()),
                "--methods",
                args.methods,
                "--search-l",
                search_l,
                "--entries",
                str(args.entries),
                "--topk",
                str(topk),
                "--maintain-sample",
                str(args.maintain_sample),
                "--maintain-changes",
                str(args.maintain_changes),
                "--maintain-batch",
                str(args.maintain_batch),
                "--timeout",
                str(args.timeout),
                "--per-query-l",
                "",
            ]
        if args.query_entries_ivecs is not None:
            command.extend(["--query-entries-ivecs", str(args.query_entries_ivecs.resolve())])
        run_command(
            command,
            timeout=args.timeout * max(2, len(search_l.split(",")) * len(args.methods.split(","))),
        )
        run_command(
            [
                sys.executable,
                str((ROOT / "experiments" / "summarize_nsg_entry_results.py").resolve()),
                "--dir",
                str(run_dir.resolve()),
                "--manifest",
                str(args.manifest.resolve()),
                "--out",
                str(summary_path.resolve()),
            ],
            timeout=600,
        )
        summary_paths.append(summary_path)

    aggregate_path = top_dir / f"aggregate_{args.repeats}runs.csv"
    rows = aggregate_summaries(summary_paths, label)
    write_csv(aggregate_path, rows)
    legacy_path = top_dir / "aggregate_10runs.csv"
    if legacy_path != aggregate_path:
        shutil.copyfile(aggregate_path, legacy_path)
    return aggregate_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GATE-style Recall@1/Recall@100 QPS repeats and aggregate median/mean curves.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_insert80_queries" / "manifest.json")
    parser.add_argument("--initial-nsg", type=Path, default=ROOT / "results" / "sift_stress_drift_dynamic_nsg" / "initial.nsg")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "stress_insert80_gate_style_10repeats")
    parser.add_argument("--query-entries-ivecs", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--search-l-1", default=DEFAULT_L1)
    parser.add_argument("--search-l-100", default=DEFAULT_L100)
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--maintain-sample", type=int, default=512)
    parser.add_argument("--maintain-changes", type=int, default=4)
    parser.add_argument("--maintain-batch", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "manifest": str(args.manifest.resolve()),
        "initial_nsg": str(args.initial_nsg.resolve()),
        "repeats": args.repeats,
        "methods": args.methods,
        "search_l_1": args.search_l_1,
        "search_l_100": args.search_l_100,
        "entries": args.entries,
        "query_entries_ivecs": str(args.query_entries_ivecs.resolve()) if args.query_entries_ivecs else None,
        "note": "GATE-style repeated QPS curves. Aggregates use mean/median over repeated single-thread query runs.",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    recall1 = run_repeats(args, topk=1, search_l=args.search_l_1, label="recall1")
    recall100 = run_repeats(args, topk=100, search_l=args.search_l_100, label="recall100")

    combined_rows = read_csv(recall1) + read_csv(recall100)
    combined_path = args.out_dir / f"aggregate_recall1_recall100_{args.repeats}runs.csv"
    write_csv(combined_path, combined_rows)
    legacy_combined_path = args.out_dir / "aggregate_recall1_recall100_10runs.csv"
    if legacy_combined_path != combined_path:
        shutil.copyfile(combined_path, legacy_combined_path)

    if args.plot:
        run_command(
            [
                sys.executable,
                str((ROOT / "experiments" / "plot_gate_style_curves.py").resolve()),
                "--inputs",
                str(combined_path.resolve()),
                "--out",
                str((args.out_dir / "gate_style_recall1_recall100_qps_median_pareto.png").resolve()),
                "--recall-col",
                "recall_at_k_median",
                "--qps-col",
                "qps_median",
                "--dataset-col",
                "dataset",
                "--panel-col",
                "recall_metric",
                "--methods",
                args.methods,
                "--title",
                f"SIFT insert-heavy dynamic no-rebuild: {args.repeats}-run median QPS curves",
                "--xlabel",
                "Recall",
                "--scale-y-thousands",
                "--min-recall",
                "0.90",
                "--pareto-frontier",
            ],
            timeout=600,
        )

    print(
        json.dumps(
            {
                "recall1": str(recall1),
                "recall100": str(recall100),
                "combined": str(combined_path),
                "legacy_combined": str(legacy_combined_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
