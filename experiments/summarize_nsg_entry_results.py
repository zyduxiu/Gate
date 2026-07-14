from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "backend",
        "method",
        "search_l",
        "entries",
        "anchors",
        "repair_degree",
        "repair_touched_nodes",
        "hard_repair_local_degree",
        "hard_repair_bridge_degree",
        "hard_repair_touched_nodes",
        "maintenance_added",
        "maintenance_retired",
        "maintenance_removed_dead",
        "maintenance_rounds",
        "maintenance_distance_computations",
        "maintenance_seconds",
        "recall_metric",
        "recall_at_k",
        "recall_at_10",
        "avg_expanded",
        "avg_distance_computations",
        "avg_latency_us",
        "p99_latency_us",
        "qps",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize same-graph NSG entry replacement results.")
    parser.add_argument("--dir", type=Path, default=Path("results/sift100k_nsg_entry"))
    parser.add_argument("--manifest", type=Path, default=Path("data/sift100k_dynamic/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("results/sift100k_nsg_entry/nsg_entry_summary.csv"))
    args = parser.parse_args()

    dataset = "sift100k_dynamic"
    if args.manifest.exists():
        dataset = json.loads(args.manifest.read_text(encoding="utf-8")).get("dataset", dataset)

    rows: list[dict[str, Any]] = []
    for path in sorted(args.dir.glob("*.json")):
        if path.name.endswith(".log.json") or path.name.endswith("_perq.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        mode = payload["mode"]
        backend = payload["backend"]
        if backend == "nsg_dynamic_no_rebuild_entry_runner":
            if int(payload.get("repair_degree", 0)) > 0:
                note = "initial NSG graph plus local insert edges, deleted-node masking, and local topology repair; entry initialization varies"
            else:
                note = "initial NSG graph plus local insert edges and deleted-node masking; only entry initialization changes"
        else:
            note = "same rebuilt NSG graph; only entry initialization changes"
        maintenance = payload.get("maintenance", {})
        rows.append(
            {
                "dataset": dataset,
                "backend": backend,
                "method": mode,
                "search_l": int(payload["search_l"]),
                "entries": int(payload["entries"]),
                "anchors": int(payload["anchors"]),
                "repair_degree": int(payload.get("repair_degree", 0)),
                "repair_touched_nodes": int(payload.get("repair_touched_nodes", 0)),
                "hard_repair_local_degree": int(payload.get("hard_repair_local_degree", 0)),
                "hard_repair_bridge_degree": int(payload.get("hard_repair_bridge_degree", 0)),
                "hard_repair_touched_nodes": int(payload.get("hard_repair_touched_nodes", 0)),
                "maintenance_added": int(maintenance.get("added", 0)),
                "maintenance_retired": int(maintenance.get("retired", 0)),
                "maintenance_removed_dead": int(maintenance.get("removed_dead", 0)),
                "maintenance_rounds": int(maintenance.get("rounds", 0)),
                "maintenance_distance_computations": float(maintenance.get("distance_computations", 0.0)),
                "maintenance_seconds": float(maintenance.get("seconds", 0.0)),
                "recall_metric": payload.get("recall_metric", "recall@10"),
                "recall_at_k": float(payload.get("recall_at_k", payload["recall_at_10"])),
                "recall_at_10": float(payload["recall_at_10"]),
                "avg_expanded": float(payload["avg_expanded"]),
                "avg_distance_computations": float(payload["avg_distance_computations"]),
                "avg_latency_us": float(payload["avg_latency_us"]),
                "p99_latency_us": float(payload["p99_latency_us"]),
                "qps": float(payload["qps"]),
                "note": note,
            }
        )
    rows.sort(key=lambda row: (int(row["search_l"]), str(row["method"])))
    write_csv(args.out, rows)
    print(f"Wrote {args.out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
