from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.external import gate_paths, nsg_paths, run_external_binary
from dynanchor.io_formats import recall_from_ivecs


def parse_search_time(stdout: str) -> float:
    match = re.search(r"search time:\s*([0-9.eE+-]+)", stdout)
    return float(match.group(1)) if match else math.nan


def parse_qps(stdout: str, queries: int) -> float:
    match = re.search(r"QPS:\s*([0-9.eE+-]+)", stdout)
    if match:
        return float(match.group(1))
    seconds = parse_search_time(stdout)
    return queries / seconds if seconds and not math.isnan(seconds) else math.nan


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "method",
        "search_l",
        "topk",
        "queries",
        "recall",
        "qps",
        "search_time_seconds",
        "index_path",
        "hub_path",
        "embedding_path",
        "gate_mode",
        "entry_count",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pure static official GATE and official NSG on the same rebuilt index.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_drift" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "static_gate_vs_nsg")
    parser.add_argument("--index-path", type=Path, default=ROOT / "results" / "gate_official_selector_sift_stress_rerun" / "gate_final_rebuilt.nsg")
    parser.add_argument("--gate-hub-path", type=Path, default=ROOT / "data" / "sift_stress_drift" / "gate_hubs" / "gate_refreshed_hubs.txt")
    parser.add_argument("--search-l", default="40,80,120,160,240")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--gate-mode", choices=["optimized", "slow_artifact"], default="optimized")
    parser.add_argument("--entry-count", type=int, default=3)
    parser.add_argument(
        "--gate-embeddings",
        default=(
            "gate_centroid_refreshed:data/sift_stress_drift/gate_hubs/gate_refreshed_hub_embeddings.fvecs,"
            "gate_paper_costlabel:results/gate_paper_costlabel_sift_stress/selectors/gate_refreshed_hub/gate_paper_hub_embeddings.fvecs"
        ),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    queries = int(manifest["n_query"])
    search_ls = [int(x.strip()) for x in args.search_l.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    nsg = nsg_paths()
    gate = gate_paths()
    if nsg["search_exe"] is None:
        raise FileNotFoundError("official NSG search executable not found")
    if gate["gate_search_exe"] is None:
        raise FileNotFoundError("official GATE search executable not found")
    if args.gate_mode == "optimized" and gate["gate_optimized_search_exe"] is None:
        raise FileNotFoundError("optimized GATE search executable not found; rebuild the GATE artifact")
    if not args.index_path.exists():
        raise FileNotFoundError(f"index not found: {args.index_path}")

    gate_exe = gate["gate_optimized_search_exe"] if args.gate_mode == "optimized" else gate["gate_search_exe"]

    rows: list[dict[str, Any]] = []
    for search_l in search_ls:
        nsg_result_path = args.out_dir / f"official_nsg_L{search_l}_K{args.topk}.ivecs"
        nsg_result = run_external_binary(
            [
                nsg["search_exe"],
                paths["base_fvecs"],
                paths["query_fvecs"],
                str(args.index_path.resolve()),
                str(search_l),
                str(args.topk),
                str(nsg_result_path.resolve()),
            ],
            timeout=args.timeout,
        )
        save_log(
            args.out_dir / f"official_nsg_L{search_l}_K{args.topk}.log.json",
            {"command": nsg_result.command, "returncode": nsg_result.returncode, "stdout": nsg_result.stdout, "stderr": nsg_result.stderr},
        )
        if not nsg_result.ok:
            raise RuntimeError(f"official NSG search failed at L={search_l}")
        nsg_metrics = recall_from_ivecs(nsg_result_path, Path(paths["truth_ivecs"]), args.topk)
        nsg_time = parse_search_time(nsg_result.stdout)
        rows.append(
            {
                "backend": "official_nsg_static",
                "method": "NSG optimized search",
                "search_l": search_l,
                "topk": args.topk,
                "queries": queries,
                "recall": nsg_metrics["recall"],
                "qps": queries / nsg_time if nsg_time and not math.isnan(nsg_time) else math.nan,
                "search_time_seconds": nsg_time,
                "index_path": str(args.index_path.resolve()),
                "hub_path": "",
                "embedding_path": "",
                "gate_mode": "",
                "entry_count": "",
                "note": "Pure static official NSG optimized search on final/rebuilt index.",
            }
        )

        for spec in [item.strip() for item in args.gate_embeddings.split(",") if item.strip()]:
            label, _, raw_embedding = spec.partition(":")
            if not label or not raw_embedding:
                raise ValueError(f"bad gate embedding spec: {spec}")
            embedding_path = Path(raw_embedding)
            if not embedding_path.is_absolute():
                embedding_path = ROOT / embedding_path
            gate_result_path = args.out_dir / f"{label}_L{search_l}_K{args.topk}.ivecs"
            gate_command = [
                gate_exe,
                paths["base_fvecs"],
                paths["query_fvecs"],
                str(args.index_path.resolve()),
                str(search_l),
                str(args.topk),
                str(args.gate_hub_path.resolve()),
                str(embedding_path.resolve()),
                str(gate_result_path.resolve()),
            ]
            if args.gate_mode == "optimized":
                gate_command.append(str(args.entry_count))
            gate_result = run_external_binary(gate_command, timeout=args.timeout)
            save_log(
                args.out_dir / f"{label}_L{search_l}_K{args.topk}.log.json",
                {
                    "command": gate_result.command,
                    "returncode": gate_result.returncode,
                    "stdout": gate_result.stdout,
                    "stderr": gate_result.stderr,
                },
            )
            if not gate_result.ok:
                raise RuntimeError(f"official GATE search failed for {label} at L={search_l}")
            gate_metrics = recall_from_ivecs(gate_result_path, Path(paths["truth_ivecs"]), args.topk)
            rows.append(
                {
                    "backend": "official_gate_static",
                    "method": label,
                    "search_l": search_l,
                    "topk": args.topk,
                    "queries": queries,
                    "recall": gate_metrics["recall"],
                    "qps": parse_qps(gate_result.stdout, queries),
                    "search_time_seconds": parse_search_time(gate_result.stdout),
                    "index_path": str(args.index_path.resolve()),
                    "hub_path": str(args.gate_hub_path.resolve()),
                    "embedding_path": str(embedding_path.resolve()),
                    "gate_mode": args.gate_mode,
                    "entry_count": args.entry_count if args.gate_mode == "optimized" else "",
                    "note": (
                        "GATE selector on the optimized NSG graph path."
                        if args.gate_mode == "optimized"
                        else "Pure static official GATE test_gate_search slow artifact path; not the paper's optimized benchmark harness."
                    ),
                }
            )

    out_csv = args.out_dir / "static_gate_vs_nsg.csv"
    write_csv(out_csv, rows)
    save_log(
        args.out_dir / "static_gate_vs_nsg_meta.json",
        {
            "manifest": str(args.manifest.resolve()),
            "index_path": str(args.index_path.resolve()),
            "gate_mode": args.gate_mode,
            "entry_count": args.entry_count,
            "rows": rows,
        },
    )
    print(f"Wrote {out_csv}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
