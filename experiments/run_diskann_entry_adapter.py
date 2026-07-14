from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.external import run_diskann_benchmark, write_diskann_benchmark_config
from dynanchor.io_formats import load_manifest


def parse_topk(output_path: Path, method: str) -> list[dict[str, object]]:
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    topk = payload[0]["results"]["search"]["Topk"]
    rows: list[dict[str, object]] = []
    for item in topk:
        rows.append(
            {
                "method": method,
                "search_l": item["search_l"],
                "recall_at_10": item["recall"]["average"],
                "mean_cmps": item["mean_cmps"],
                "mean_hops": item["mean_hops"],
                "avg_latency_us": mean(item["mean_latencies"]),
                "p99_latency_us": mean(item["p99_latencies"]),
                "max_p99_latency_us": max(item["p99_latencies"]),
                "qps": mean(item["qps"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_config(config: Path, output: Path, timeout: int | None) -> None:
    result = run_diskann_benchmark(config, output, timeout=timeout)
    log_path = output.with_suffix(".log.json")
    log_path.write_text(
        json.dumps(
            {
                "command": result.command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"DiskANN benchmark failed for {config}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiskANN/Vamana with search-time entry-anchor adapters.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift100k_dynamic" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "sift100k_diskann_entry")
    parser.add_argument("--methods", default="fixed_medoid,static_density,dynamic_density")
    parser.add_argument("--search-l", default="20,40,80,120,240")
    parser.add_argument("--max-degree", type=int, default=32)
    parser.add_argument("--l-build", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--entries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    search_l = [int(x.strip()) for x in args.search_l.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    build_dir = args.out_dir / "shared_medoid_graph"
    build_config = args.out_dir / "build_shared_graph.json"
    shared_index = build_dir / "diskann_graph_index"

    if not args.skip_build:
        write_diskann_benchmark_config(
            manifest=manifest,
            config_path=build_config,
            output_dir=build_dir,
            search_l=search_l,
            max_degree=args.max_degree,
            l_build=args.l_build,
            alpha=args.alpha,
            threads=args.threads,
            reps=args.reps,
            topk=args.topk,
            start_point_strategy="medoid",
        )
        run_config(build_config, args.out_dir / "build_shared_graph_output.json", timeout=args.timeout)

    rows: list[dict[str, object]] = []

    baseline_config = args.out_dir / "builtin_topk_load.json"
    baseline_output = args.out_dir / "builtin_topk_load_output.json"
    write_diskann_benchmark_config(
        manifest=manifest,
        config_path=baseline_config,
        output_dir=args.out_dir / "builtin_topk_load",
        search_l=search_l,
        max_degree=args.max_degree,
        l_build=args.l_build,
        alpha=args.alpha,
        threads=args.threads,
        reps=args.reps,
        topk=args.topk,
        start_point_strategy="medoid",
        load_path=shared_index,
    )
    run_config(baseline_config, baseline_output, timeout=args.timeout)
    rows.extend(parse_topk(baseline_output, "builtin_topk"))

    anchor_files = manifest.get("external_anchor_files", {})
    for method in methods:
        if method not in anchor_files:
            available = ", ".join(sorted(anchor_files))
            raise KeyError(f"no external_anchor_files entry for {method!r}; available: {available}")
        config = args.out_dir / f"{method}_entry_load.json"
        output = args.out_dir / f"{method}_entry_load_output.json"
        write_diskann_benchmark_config(
            manifest=manifest,
            config_path=config,
            output_dir=args.out_dir / f"{method}_entry_load",
            search_l=search_l,
            max_degree=args.max_degree,
            l_build=args.l_build,
            alpha=args.alpha,
            threads=args.threads,
            reps=args.reps,
            topk=args.topk,
            start_point_strategy="medoid",
            entry_anchor_file=anchor_files[method],
            entries=args.entries,
            load_path=shared_index,
        )
        run_config(config, output, timeout=args.timeout)
        rows.extend(parse_topk(output, method))

    summary = args.out_dir / "diskann_entry_adapter_summary.csv"
    write_csv(summary, rows)
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
