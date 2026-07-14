from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.external import CommandResult, gate_paths, run_external_binary
from dynanchor.io_formats import recall_from_ivecs


def save_log(path: Path, result: CommandResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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


def parse_gate_stdout(stdout: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, pattern in {
        "search_time_seconds": r"search time:\s*([0-9.eE+-]+)",
        "qps": r"QPS:\s*([0-9.eE+-]+)",
    }.items():
        match = re.search(pattern, stdout)
        if match:
            parsed[key] = float(match.group(1))
    return parsed


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "method",
        "search_l",
        "recall_at_10",
        "qps",
        "search_time_seconds",
        "wall_seconds",
        "hub_path",
        "embedding_path",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official GATE C++ search path with selectable hub embeddings.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_drift" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "gate_official_selector_sift_stress")
    parser.add_argument("--hub-path", type=Path, default=ROOT / "data" / "sift_stress_drift" / "gate_hubs" / "gate_refreshed_hubs.txt")
    parser.add_argument(
        "--embeddings",
        default=(
            "centroid_refreshed:data/sift_stress_drift/gate_hubs/gate_refreshed_hub_embeddings.fvecs,"
            "paper_gate_costlabel:results/gate_paper_costlabel_sift_stress/selectors/gate_refreshed_hub/gate_paper_hub_embeddings.fvecs"
        ),
    )
    parser.add_argument("--search-l", type=int, default=240)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--build-l", type=int, default=40)
    parser.add_argument("--build-r", type=int, default=50)
    parser.add_argument("--build-c", type=int, default=500)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--reuse-index", action="store_true")
    args = parser.parse_args()

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    gate = gate_paths()
    nsg_index_exe = gate["nsg_index_exe"]
    gate_search_exe = gate["gate_search_exe"]
    if nsg_index_exe is None or gate_search_exe is None:
        raise FileNotFoundError("GATE executables missing; run scripts/build_external_baselines.ps1 -Baseline gate -UseWsl")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index_path = args.index_path if args.index_path is not None else args.out_dir / "gate_final_rebuilt.nsg"
    if args.index_path is None and (not args.reuse_index or not index_path.exists()):
        result = run_external_binary(
            [
                nsg_index_exe,
                paths["base_fvecs"],
                paths["nsg_knn_graph"],
                str(args.build_l),
                str(args.build_r),
                str(args.build_c),
                str(index_path.resolve()),
            ],
            timeout=args.timeout,
        )
        save_log(args.out_dir / "build.log.json", result)
        if not result.ok:
            raise RuntimeError(f"GATE NSG build failed; see {args.out_dir / 'build.log.json'}")

    rows: list[dict[str, Any]] = []
    embedding_specs = [item.strip() for item in args.embeddings.split(",") if item.strip()]
    for spec in embedding_specs:
        label, _, raw_path = spec.partition(":")
        if not label or not raw_path:
            raise ValueError(f"invalid embedding spec: {spec}")
        embedding_path = (ROOT / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path)
        result_path = args.out_dir / f"{label}_L{args.search_l}.ivecs"
        start = time.perf_counter()
        result = run_external_binary(
            [
                gate_search_exe,
                paths["base_fvecs"],
                paths["query_fvecs"],
                str(index_path.resolve()),
                str(args.search_l),
                str(args.topk),
                str(args.hub_path.resolve()),
                str(embedding_path.resolve()),
                str(result_path.resolve()),
            ],
            timeout=args.timeout,
        )
        wall = time.perf_counter() - start
        save_log(args.out_dir / f"{label}_L{args.search_l}.log.json", result)
        if not result.ok:
            raise RuntimeError(f"GATE search failed for {label}; see {args.out_dir / f'{label}_L{args.search_l}.log.json'}")
        metrics = recall_from_ivecs(result_path, Path(paths["truth_ivecs"]), args.topk)
        parsed = parse_gate_stdout(result.stdout)
        row = {
            "backend": "official_gate_test_gate_search",
            "method": label,
            "search_l": args.search_l,
            "recall_at_10": metrics["recall"],
            "qps": parsed.get("qps", ""),
            "search_time_seconds": parsed.get("search_time_seconds", ""),
            "wall_seconds": wall,
            "hub_path": str(args.hub_path.resolve()),
            "embedding_path": str(embedding_path.resolve()),
            "note": "Official GATE C++ search path on rebuilt final/static NSG; dynamic insert/delete no-rebuild is not represented here.",
        }
        rows.append(row)
        print(row)

    write_rows(args.out_dir / "official_gate_selector_results.csv", rows)
    (args.out_dir / "official_gate_selector_meta.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "index_path": str(index_path.resolve()),
                "hub_path": str(args.hub_path.resolve()),
                "rows": rows,
                "gate_paths": gate,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_dir / 'official_gate_selector_results.csv'}")


if __name__ == "__main__":
    main()
