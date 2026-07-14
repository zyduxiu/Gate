from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_hnsw_anchor_fair import compile_runner, run_runner


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "dataset",
        "method",
        "ef",
        "entries",
        "anchors",
        "recall_at_10",
        "avg_cmps",
        "avg_hops",
        "avg_latency_us",
        "p99_latency_us",
        "qps",
        "build_seconds",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compiled hnswlib anchor search on a real dynamic dataset manifest.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift100k_dynamic" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "sift100k_hnsw_anchor")
    parser.add_argument("--efs", default="40,80,120,240")
    parser.add_argument("--entries", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--force-build", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    efs = [int(x.strip()) for x in args.efs.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_paths = {
        "initial": Path(manifest["paths"]["initial_fbin"]),
        "insert": Path(manifest["paths"]["insert_fbin"]),
        "queries": Path(manifest["paths"]["query_fbin"]),
        "truth": Path(manifest["paths"]["truth_dynamic_bin"]),
        "deletes": Path(manifest["paths"]["delete_u32"]),
    }
    anchor_files = {key: Path(value) for key, value in manifest["anchor_files"].items()}
    runner = compile_runner(force=args.force_build, timeout=args.timeout)
    rows: list[dict[str, Any]] = []

    for ef in efs:
        payload = run_runner(
            runner=runner,
            data_paths=data_paths,
            out_json=args.out_dir / f"builtin_ef{ef}.json",
            mode="builtin",
            ef=ef,
            entries=1,
            topk=args.topk,
            m=args.hnsw_m,
            ef_construction=args.hnsw_ef_construction,
            seed=manifest["seed"],
            timeout=args.timeout,
        )
        rows.append(
            {
                "backend": "hnswlib_cpp",
                "dataset": manifest["dataset"],
                "method": "builtin_hnsw_entry",
                "ef": ef,
                "entries": 1,
                "anchors": 0,
                "recall_at_10": payload["recall_at_10"],
                "avg_cmps": payload["avg_cmps"],
                "avg_hops": payload["avg_hops"],
                "avg_latency_us": payload["avg_latency_us"],
                "p99_latency_us": payload["p99_latency_us"],
                "qps": payload["qps"],
                "build_seconds": payload["build_seconds"],
                "note": "standard hnswlib search on real SIFT dynamic subset; no rebuild after insert/delete",
            }
        )

        for method in ["fixed_medoid", "static_density", "dynamic_density"]:
            payload = run_runner(
                runner=runner,
                data_paths=data_paths,
                out_json=args.out_dir / f"{method}_ef{ef}.json",
                mode="anchors",
                ef=ef,
                entries=args.entries,
                topk=args.topk,
                m=args.hnsw_m,
                ef_construction=args.hnsw_ef_construction,
                seed=manifest["seed"],
                timeout=args.timeout,
                anchors_txt=anchor_files[method],
            )
            rows.append(
                {
                    "backend": "hnswlib_cpp_base_layer",
                    "dataset": manifest["dataset"],
                    "method": method,
                    "ef": ef,
                    "entries": args.entries,
                    "anchors": payload["anchors"],
                    "recall_at_10": payload["recall_at_10"],
                    "avg_cmps": payload["avg_cmps"],
                    "avg_hops": payload["avg_hops"],
                    "avg_latency_us": payload["avg_latency_us"],
                    "p99_latency_us": payload["p99_latency_us"],
                    "qps": payload["qps"],
                    "build_seconds": payload["build_seconds"],
                    "note": "anchor-selected hnswlib layer-0 search on real SIFT dynamic subset; no graph rebuild",
                }
            )

    write_rows(args.out_dir / "real_hnsw_anchor.csv", rows)
    (args.out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'real_hnsw_anchor.csv'}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
