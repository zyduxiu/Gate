from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "backend",
        "method",
        "budget_name",
        "budget",
        "recall_at_10",
        "avg_cmps",
        "avg_hops",
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
    parser = argparse.ArgumentParser(description="Summarize real SIFT benchmark results.")
    parser.add_argument("--manifest", type=Path, default=Path("data/sift100k_dynamic/manifest.json"))
    parser.add_argument("--hnsw-dir", type=Path, default=Path("results/sift100k_hnsw_anchor"))
    parser.add_argument("--external-dir", type=Path, default=Path("results/sift100k_external"))
    parser.add_argument("--gate-dir", type=Path, default=Path("results/sift100k_gate_adapter"))
    parser.add_argument("--out", type=Path, default=Path("results/real_sift100k_summary.csv"))
    args = parser.parse_args()

    dataset = "sift100k_dynamic"
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        dataset = manifest.get("dataset", dataset)

    rows: list[dict[str, Any]] = []
    for row in read_csv(args.hnsw_dir / "real_hnsw_anchor.csv"):
        rows.append(
            {
                "dataset": dataset,
                "backend": row["backend"],
                "method": row["method"],
                "budget_name": "ef",
                "budget": int(row["ef"]),
                "recall_at_10": float(row["recall_at_10"]),
                "avg_cmps": float(row["avg_cmps"]),
                "avg_hops": float(row["avg_hops"]),
                "p99_latency_us": float(row["p99_latency_us"]),
                "qps": float(row["qps"]),
                "build_seconds": float(row["build_seconds"]),
                "note": row["note"],
            }
        )

    diskann_path = args.external_dir / "diskann_output.json"
    if diskann_path.exists():
        payload = json.loads(diskann_path.read_text(encoding="utf-8"))
        for item in payload[0]["results"]["search"]["Topk"]:
            rows.append(
                {
                    "dataset": dataset,
                    "backend": "diskann",
                    "method": "medoid_start_rebuilt_final_index",
                    "budget_name": "search_l",
                    "budget": int(item["search_l"]),
                    "recall_at_10": float(item["recall"]["average"]),
                    "avg_cmps": float(item["mean_cmps"]),
                    "avg_hops": float(item["mean_hops"]),
                    "p99_latency_us": float(min(item["p99_latencies"])),
                    "qps": float(max(item["qps"])),
                    "build_seconds": "",
                    "note": "rebuilt on final alive SIFT subset; favorable to baseline",
                }
            )

    for name in ["nsg", "sptag"]:
        metrics_path = args.external_dir / name / "metrics.json" if name == "nsg" else args.external_dir / "sptag_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "dataset": dataset,
                    "backend": name,
                    "method": "rebuilt_final_index",
                    "budget_name": "search_l" if name == "nsg" else "maxcheck",
                    "budget": 120 if name == "nsg" else 1024,
                    "recall_at_10": float(metrics["recall"]),
                    "avg_cmps": "",
                    "avg_hops": "",
                    "p99_latency_us": "",
                    "qps": "",
                    "build_seconds": "",
                    "note": "rebuilt on final alive SIFT subset; NSG uses hnswlib-generated KNN graph" if name == "nsg" else "rebuilt on final alive SIFT subset",
                }
            )

    for row in read_csv(args.gate_dir / "gate_adapter_results.csv"):
        rows.append(
            {
                "dataset": dataset,
                "backend": "gate_adapter",
                "method": row["method"],
                "budget_name": "search_l",
                "budget": int(row["search_l"]),
                "recall_at_10": float(row["recall_at_10"]),
                "avg_cmps": "",
                "avg_hops": "",
                "p99_latency_us": "",
                "qps": float(row["qps"]) if row["qps"] else "",
                "build_seconds": "",
                "note": row["note"],
            }
        )

    rows.sort(key=lambda row: (str(row["backend"]), str(row["method"]), int(row["budget"])))
    write_csv(args.out, rows)
    print(f"Wrote {args.out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
