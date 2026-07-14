from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
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


def run_command(command: list[str], timeout: int | None = None) -> None:
    print(" ".join(command))
    result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with return code {result.returncode}: {' '.join(command)}")


def read_u32_list(path: Path) -> list[int]:
    import numpy as np

    raw = np.fromfile(path, dtype="<u4")
    if raw.size == 0:
        return []
    count = int(raw[0])
    return [int(value) for value in raw[1 : 1 + count]]


def dynamic_labels_from_manifest(manifest: dict[str, Any]) -> list[int]:
    n_total = int(manifest["n_initial"]) + int(manifest["n_insert"])
    deleted = set(read_u32_list(Path(manifest["paths"]["delete_u32"])))
    return [idx for idx in range(n_total) if idx not in deleted]


def export_gate_hub_anchors(hub_path: Path, out_path: Path, dynamic_labels: list[int] | None = None) -> list[int]:
    if not hub_path.exists():
        raise FileNotFoundError(f"GATE hub file not found: {hub_path}")
    lines = [line.strip() for line in hub_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty GATE hub file: {hub_path}")
    header = lines[0].split()
    if len(header) < 3:
        raise ValueError(f"invalid GATE hub header: {lines[0]}")
    stage1 = int(header[0])
    stage2 = int(header[1])
    id_lines = lines[1 + stage1 : 1 + stage1 + stage1]
    anchors: list[int] = []
    seen: set[int] = set()
    for line in id_lines:
        ids = [int(value) for value in line.split()]
        if len(ids) != stage2:
            raise ValueError(f"expected {stage2} hub ids per line in {hub_path}, got {len(ids)}")
        for value in ids:
            mapped = int(dynamic_labels[value]) if dynamic_labels is not None and 0 <= value < len(dynamic_labels) else value
            if mapped not in seen:
                seen.add(mapped)
                anchors.append(mapped)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(value) for value in anchors) + "\n", encoding="utf-8")
    return anchors


def prepare_manifest_with_gate_anchor(manifest_path: Path, gate_hub_path: Path, out_dir: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    anchor_path = out_dir / "anchors" / "gate_static_hub.txt"
    anchors = export_gate_hub_anchors(gate_hub_path, anchor_path, dynamic_labels_from_manifest(manifest))
    manifest.setdefault("anchor_files", {})["gate_static_hub"] = str(anchor_path.resolve())
    manifest["entry_topology_matrix"] = {
        "gate_static_hub_anchor_count": len(anchors),
        "gate_static_hub_source": str(gate_hub_path.resolve()),
        "gate_static_hub_note": "Parsed leaf hub data-point ids from the generated GATE static centroid-hub file.",
    }
    out_manifest = out_dir / "manifest_with_gate_static_hub.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_manifest


def repair_specs(value: str) -> list[tuple[str, int, str]]:
    specs: list[tuple[str, int, str]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"repair spec must be name:degree, got {item!r}")
        name, degree = item.split(":", 1)
        note = "no graph repair"
        if int(degree) > 0:
            note = "local KNN topology repair proxy; not paper-faithful RFix/NGFix"
        specs.append((name.strip(), int(degree), note))
    return specs


def run_nsg_matrix(args: argparse.Namespace, manifest_with_gate: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topology, degree, topology_note in repair_specs(args.repairs):
        out_dir = args.out_dir / "nsg" / topology
        run_command(
            [
                sys.executable,
                "experiments/run_nsg_dynamic_entry.py",
                "--manifest",
                str(manifest_with_gate),
                "--out-dir",
                str(out_dir),
                "--methods",
                args.methods,
                "--search-l",
                args.search_l,
                "--entries",
                str(args.entries),
                "--repair-degree",
                str(degree),
                "--maintain-batch",
                str(args.maintain_batch),
                "--maintain-changes",
                str(args.maintain_changes),
                "--maintain-sample",
                str(args.maintain_sample),
                "--timeout",
                str(args.timeout),
                "--per-query-l",
                args.per_query_l,
            ],
            timeout=args.timeout + 60,
        )
        summary = out_dir / "summary.csv"
        run_command(
            [
                sys.executable,
                "experiments/summarize_nsg_entry_results.py",
                "--dir",
                str(out_dir),
                "--manifest",
                str(manifest_with_gate),
                "--out",
                str(summary),
            ],
            timeout=120,
        )
        for row in read_csv(summary):
            row = dict(row)
            row.update(
                {
                    "graph_backend": "NSG",
                    "dynamic_graph_regime": "no_full_rebuild",
                    "topology": topology,
                    "topology_note": topology_note,
                    "comparison_status": "executed",
                }
            )
            rows.append(row)
    return rows


def add_diskann_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.diskann_summary is None:
        return []
    rows: list[dict[str, Any]] = []
    for row in read_csv(args.diskann_summary):
        row = dict(row)
        row.update(
            {
                "dataset": args.diskann_dataset,
                "backend": "diskann_entry_adapter",
                "graph_backend": "DiskANN/Vamana",
                "dynamic_graph_regime": "rebuilt_or_loaded_shared_graph",
                "topology": "same_rebuilt_graph",
                "entries": str(args.entries),
                "anchors": "",
                "repair_degree": "",
                "repair_touched_nodes": "",
                "maintenance_added": "",
                "maintenance_retired": "",
                "maintenance_removed_dead": "",
                "maintenance_rounds": "",
                "maintenance_distance_computations": "",
                "avg_expanded": "",
                "avg_distance_computations": row.get("mean_cmps", ""),
                "topology_note": "DiskANN/Vamana adapter keeps the same rebuilt/shared graph and changes search-time entry initialization; not a no-rebuild dynamic graph result.",
                "comparison_status": "executed_adapter_only",
                "note": "orthogonality adapter; dynamic no-rebuild DiskANN/Vamana remains future work",
            }
        )
        rows.append(row)
    return rows


def add_paper_only_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.include_paper_only:
        return []
    base = {
        "dataset": args.diskann_dataset,
        "recall_at_10": "",
        "avg_distance_computations": "",
        "avg_latency_us": "",
        "p99_latency_us": "",
        "qps": "",
    }
    return [
        {
            **base,
            "backend": "rfix_ngfix",
            "method": "RFix/NGFix",
            "graph_backend": "NSG/DiskANN-style graph",
            "dynamic_graph_regime": "paper_only",
            "topology": "paper_faithful_rfix_ngfix",
            "comparison_status": "blocked_no_public_code_or_reimplementation",
            "topology_note": "True RFix/NGFix repairs entry-to-vicinity and local hard regions; current matrix only includes local KNN repair proxies.",
            "note": "must cite/discuss; implement or obtain artifact for paper-faithful numbers",
        },
        {
            **base,
            "backend": "diskann_plus_plus",
            "method": "DiskANN++",
            "graph_backend": "DiskANN/Vamana",
            "dynamic_graph_regime": "paper_only_static_query_sensitive_entry",
            "topology": "same_or_page_optimized_graph",
            "comparison_status": "blocked_no_public_code_or_reimplementation",
            "topology_note": "Query-sensitive entry vertex and page layout; overlaps with entry choice, but not online entry-set maintenance under drift.",
            "note": "must cite/discuss; implement entry-vertex strategy if artifact is unavailable",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/assemble the entry strategy x topology repair x backend matrix.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift100k_dynamic" / "manifest.json")
    parser.add_argument("--gate-hub-path", type=Path, default=ROOT / "results" / "sift100k_gate_adapter" / "gate_hubs.txt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "entry_topology_backend_matrix")
    parser.add_argument("--methods", default="nsg_ep,fixed_medoid,static_density,gate_static_hub,online_density")
    parser.add_argument("--search-l", default="240")
    parser.add_argument("--per-query-l", default="240")
    parser.add_argument("--entries", type=int, default=8)
    parser.add_argument("--repairs", default="no_repair:0,local_repair32:32,rfix_like_local64_proxy:64")
    parser.add_argument("--maintain-batch", type=int, default=1000)
    parser.add_argument("--maintain-changes", type=int, default=4)
    parser.add_argument("--maintain-sample", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--diskann-summary", type=Path, default=ROOT / "results" / "sift100k_diskann_entry_shared" / "diskann_entry_adapter_summary.csv")
    parser.add_argument("--diskann-dataset", default="sift1m_real_dynamic_subset")
    parser.add_argument("--skip-nsg", action="store_true")
    parser.add_argument("--skip-diskann-existing", action="store_true")
    parser.add_argument("--include-paper-only", action="store_true", default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_with_gate = prepare_manifest_with_gate_anchor(args.manifest, args.gate_hub_path, args.out_dir)

    rows: list[dict[str, Any]] = []
    if not args.skip_nsg:
        rows.extend(run_nsg_matrix(args, manifest_with_gate))
    if not args.skip_diskann_existing:
        rows.extend(add_diskann_rows(args))
    rows.extend(add_paper_only_rows(args))

    matrix_csv = args.out_dir / "entry_topology_backend_matrix.csv"
    write_csv(matrix_csv, rows)
    meta = {
        "manifest": str(args.manifest.resolve()),
        "manifest_with_gate": str(manifest_with_gate.resolve()),
        "methods": args.methods,
        "search_l": args.search_l,
        "repairs": args.repairs,
        "diskann_summary": str(args.diskann_summary.resolve()) if args.diskann_summary else None,
        "note": "RFix-like rows are local KNN repair proxies unless comparison_status says paper-only.",
    }
    (args.out_dir / "matrix_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {matrix_csv}")


if __name__ == "__main__":
    main()
