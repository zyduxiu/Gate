from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[Any], timeout: int) -> None:
    print("RUN", " ".join(str(x) for x in command), flush=True)
    result = subprocess.run(
        [str(x) for x in command],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(str(x) for x in command)}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair static GATE-entry vs NSG-entry comparison under one runner.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "static_fair_gate_vs_nsg" / "static_manifest.json")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "static_fair_gate_vs_nsg" / "runs")
    parser.add_argument("--search-l", default="40,80,120,160,240")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--entries", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--schedule", choices=["grouped", "interleaved", "interleaved_random"], default="grouped")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    initial_nsg = Path(manifest["paths"]["initial_nsg"])
    rows: list[dict[str, Any]] = []

    specs = [("nsg_ep", "nsg_ep", None)]
    for label, path in manifest.get("query_entry_files", {}).items():
        specs.append((f"query_entries_{label}", f"query_entries_{label}", Path(path)))

    rng = random.Random(args.seed)
    jobs: list[tuple[str, str, Path | None, int]] = []
    if args.schedule == "grouped":
        for alias, method, query_entries in specs:
            for repeat in range(1, args.repeats + 1):
                jobs.append((alias, method, query_entries, repeat))
    else:
        for repeat in range(1, args.repeats + 1):
            repeat_specs = list(specs)
            if args.schedule == "interleaved_random":
                rng.shuffle(repeat_specs)
            for alias, method, query_entries in repeat_specs:
                jobs.append((alias, method, query_entries, repeat))

    for alias, method, query_entries, repeat in jobs:
            run_dir = args.out_dir / alias / f"run_{repeat:02d}"
            command = [
                sys.executable,
                ROOT / "experiments" / "run_nsg_dynamic_entry.py",
                "--manifest",
                args.manifest,
                "--initial-nsg",
                initial_nsg,
                "--runner",
                args.runner,
                "--out-dir",
                run_dir,
                "--methods",
                method,
                "--search-l",
                args.search_l,
                "--entries",
                str(args.entries),
                "--topk",
                str(args.topk),
                "--insert-degree",
                "0",
                "--maintain-sample",
                "1",
                "--maintain-changes",
                "1",
                "--maintain-batch",
                "1000000",
                "--timeout",
                str(args.timeout),
            ]
            if query_entries is not None:
                command.extend(["--query-entries-ivecs", query_entries])
            run(command, timeout=args.timeout)
            summary_path = run_dir / "summary.csv"
            run(
                [
                    sys.executable,
                    ROOT / "experiments" / "summarize_nsg_entry_results.py",
                    "--dir",
                    run_dir,
                    "--manifest",
                    args.manifest,
                    "--out",
                    summary_path,
                ],
                timeout=600,
            )
            for row in read_csv(summary_path):
                row["method"] = alias
                row["repeat"] = repeat
                row["note"] = "Pure static same-runner entry comparison: same final/rebuilt NSG graph and search code; only entry initialization differs. Query-entry selector time is excluded."
                rows.append(row)

    write_csv(args.out_dir / "static_fair_entry_raw.csv", rows)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["search_l"]), []).append(row)
    agg: list[dict[str, Any]] = []
    for (method, search_l), items in sorted(groups.items(), key=lambda x: (int(x[0][1]), x[0][0])):
        def mean(name: str) -> float:
            vals = [float(x[name]) for x in items if x.get(name) not in (None, "")]
            return sum(vals) / len(vals) if vals else float("nan")

        def median(name: str) -> float:
            vals = sorted(float(x[name]) for x in items if x.get(name) not in (None, ""))
            if not vals:
                return float("nan")
            mid = len(vals) // 2
            if len(vals) % 2:
                return vals[mid]
            return (vals[mid - 1] + vals[mid]) / 2.0

        def pstdev(name: str) -> float:
            vals = [float(x[name]) for x in items if x.get(name) not in (None, "")]
            if not vals:
                return float("nan")
            avg = sum(vals) / len(vals)
            return (sum((x - avg) ** 2 for x in vals) / len(vals)) ** 0.5

        qps_mean = mean("qps")
        qps_std = pstdev("qps")
        agg.append(
            {
                "method": method,
                "search_l": int(search_l),
                "runs": len(items),
                "recall_mean": mean("recall_at_k"),
                "qps_mean": qps_mean,
                "qps_median": median("qps"),
                "qps_std": qps_std,
                "qps_cv": qps_std / qps_mean if qps_mean == qps_mean and qps_mean else float("nan"),
                "avg_latency_us_mean": mean("avg_latency_us"),
                "avg_latency_us_median": median("avg_latency_us"),
                "p99_latency_us_mean": mean("p99_latency_us"),
                "avg_distance_computations_mean": mean("avg_distance_computations"),
                "note": items[0]["note"],
            }
        )
    write_csv(args.out_dir / "static_fair_entry_aggregate.csv", agg)
    print(json.dumps({"raw": str(args.out_dir / "static_fair_entry_raw.csv"), "aggregate": str(args.out_dir / "static_fair_entry_aggregate.csv")}, indent=2))


if __name__ == "__main__":
    main()
