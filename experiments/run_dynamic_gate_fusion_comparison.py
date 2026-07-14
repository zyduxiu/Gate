from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.io_formats import read_fvecs, write_ivecs
from dynanchor.external import run_external_binary


def run(command: list[Any], timeout: int) -> None:
    print("RUN", " ".join(str(x) for x in command), flush=True)
    if Path(str(command[0])).name == "nsg_dynamic_entry_runner":
        ext = run_external_binary([str(x) for x in command], timeout=timeout)
        if ext.stdout:
            print(ext.stdout.rstrip(), flush=True)
        if ext.stderr:
            print(ext.stderr.rstrip(), file=sys.stderr, flush=True)
        if ext.returncode != 0:
            raise RuntimeError(f"command failed ({ext.returncode}): {' '.join(str(x) for x in command)}")
        return
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


def read_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
        if header.size != 2:
            raise ValueError(f"bad fbin header: {path}")
        rows, dim = int(header[0]), int(header[1])
        data = np.fromfile(handle, dtype="<f4", count=rows * dim)
    if data.size != rows * dim:
        raise ValueError(f"truncated fbin: {path}")
    return data.reshape(rows, dim)


def parse_gate_eps(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    iter1, iter2, _dim = [int(x) for x in lines[0].split()[:3]]
    eps_lines = lines[1 + iter1 : 1 + iter1 + iter1]
    eps: list[int] = []
    for line in eps_lines:
        values = [int(x) for x in line.split()]
        if len(values) != iter2:
            raise ValueError(f"expected {iter2} eps per row in {path}")
        eps.extend(values)
    return np.asarray(eps, dtype=np.int32)


def cosine_top_entries(queries: np.ndarray, embeddings: np.ndarray, ids: np.ndarray, top: int) -> np.ndarray:
    queries = np.asarray(queries, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int32)
    q_norm = np.linalg.norm(queries, axis=1, keepdims=True)
    e_norm = np.linalg.norm(embeddings, axis=1, keepdims=True).T
    scores = queries @ embeddings.T
    scores = scores / np.maximum(q_norm * e_norm, 1e-12)
    top = min(top, embeddings.shape[0])
    top_idx = np.argpartition(-scores, kth=top - 1, axis=1)[:, :top]
    row = np.arange(top_idx.shape[0])[:, None]
    top_idx = top_idx[row, np.argsort(-scores[row, top_idx], axis=1)]
    return ids[top_idx].astype(np.int32)


def read_anchor_ids(path: Path) -> np.ndarray:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            values.append(int(line.split()[0]))
    if not values:
        raise ValueError(f"empty anchor file: {path}")
    return np.asarray(values, dtype=np.int32)


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


def export_online_anchors(args: argparse.Namespace, manifest: dict[str, Any], out_dir: Path) -> Path:
    paths = manifest["paths"]
    out_json = out_dir / "online_export_probe.json"
    out_anchors = out_dir / "ours_online_maintained_E.txt"
    if args.skip_existing and out_anchors.exists():
        return out_anchors
    seed_anchors = Path(manifest["anchor_files"]["static_density"])
    command = [
        args.runner,
        "--initial-fbin",
        paths["initial_fbin"],
        "--insert-fbin",
        paths["insert_fbin"],
        "--query-fbin",
        paths["query_fbin"],
        "--truth-bin",
        paths["truth_dynamic_bin"],
        "--delete-u32",
        paths["delete_u32"],
        "--initial-nsg",
        paths["initial_nsg"],
        "--all-knn-graph",
        paths["all_nsg_knn_graph"],
        "--mode",
        "online_hub_density",
        "--anchors-txt",
        seed_anchors,
        "--search-l",
        str(args.export_search_l),
        "--entries",
        str(args.entries),
        "--topk",
        str(args.topk),
        "--insert-degree",
        str(args.insert_degree),
        "--max-degree",
        str(args.max_degree),
        "--maintain-batch",
        str(args.maintain_batch),
        "--maintain-changes",
        str(args.maintain_changes),
        "--maintain-sample",
        str(args.maintain_sample),
        "--online-score-weights",
        args.online_score_weights,
        "--online-query-gain-weight",
        str(args.online_query_gain_weight),
        "--online-query-load-weight",
        str(args.online_query_load_weight),
        "--anchor-cost-query-limit",
        str(args.anchor_cost_query_limit),
        "--out-json",
        out_json,
        "--out-anchors-txt",
        out_anchors,
    ]
    run(command, timeout=args.timeout)
    return out_anchors


def prepare_entries(args: argparse.Namespace, manifest: dict[str, Any], out_dir: Path, our_anchors: Path) -> dict[str, Path]:
    paths = manifest["paths"]
    queries = read_fvecs(Path(paths["query_fvecs"]))
    all_vectors = read_fbin(Path(paths["all_fbin"]))

    gate_hub_file = Path(manifest["gate_hub_files"]["gate_refreshed_hub_file"])
    gate_embedding_file = Path(
        manifest.get("gate_hub_files", {}).get(
            "gate_refreshed_hub_embeddings",
            str(gate_hub_file.with_name("gate_refreshed_hub_embeddings.fvecs")),
        )
    )
    gate_eps = parse_gate_eps(gate_hub_file)
    gate_embeddings = read_fvecs(gate_embedding_file)
    gate_entries = cosine_top_entries(queries, gate_embeddings, gate_eps, args.entries)
    gate_entries_path = out_dir / f"gate_refresh_entries_top{args.entries}.ivecs"
    write_ivecs(gate_entries_path, gate_entries)

    our_ids = read_anchor_ids(our_anchors)
    our_embeddings = all_vectors[our_ids]
    our_entries = cosine_top_entries(queries, our_embeddings, our_ids, args.entries)
    our_entries_path = out_dir / f"gate_over_ours_online_E_entries_top{args.entries}.ivecs"
    write_ivecs(our_entries_path, our_entries)

    return {
        "gate_refresh": gate_entries_path,
        "gate_over_ours_online_E": our_entries_path,
    }


def run_method(
    args: argparse.Namespace,
    manifest_path: Path,
    manifest: dict[str, Any],
    alias: str,
    method: str,
    out_dir: Path,
    query_entries: Path | None,
    repeat: int,
) -> list[dict[str, Any]]:
    run_dir = out_dir / alias / f"run_{repeat:02d}"
    command = [
        sys.executable,
        ROOT / "experiments" / "run_nsg_dynamic_entry.py",
        "--manifest",
        manifest_path,
        "--initial-nsg",
        manifest["paths"]["initial_nsg"],
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
        str(args.insert_degree),
        "--max-degree",
        str(args.max_degree),
        "--maintain-batch",
        str(args.maintain_batch),
        "--maintain-changes",
        str(args.maintain_changes),
        "--maintain-sample",
        str(args.maintain_sample),
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
            manifest_path,
            "--out",
            summary_path,
        ],
        timeout=600,
    )
    rows: list[dict[str, Any]] = []
    for row in read_csv(summary_path):
        row["method"] = alias
        row["repeat"] = repeat
        row["note"] = "Dynamic no-rebuild same-kernel comparison. GATE-refresh and GATE-over-ours entries are precomputed; selector cost is excluded for both."
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["search_l"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (method, search_l), items in sorted(groups.items(), key=lambda x: (int(x[0][1]), x[0][0])):
        def mean(name: str) -> float:
            vals = [float(item[name]) for item in items if item.get(name) not in (None, "")]
            return sum(vals) / len(vals) if vals else float("nan")

        out.append(
            {
                "method": method,
                "search_l": int(search_l),
                "runs": len(items),
                "recall_mean": mean("recall_at_k"),
                "qps_mean": mean("qps"),
                "avg_latency_us_mean": mean("avg_latency_us"),
                "p99_latency_us_mean": mean("p99_latency_us"),
                "avg_distance_computations_mean": mean("avg_distance_computations"),
                "maintenance_seconds_mean": mean("maintenance_seconds"),
                "note": items[0]["note"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse online maintained E with a GATE-style selector under the same dynamic NSG kernel.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_drift" / "manifest.json")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "dynamic_gate_fusion_full1000")
    parser.add_argument("--search-l", default="40,80,120,160,240")
    parser.add_argument("--export-search-l", type=int, default=120)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--entries", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--insert-degree", type=int, default=32)
    parser.add_argument("--max-degree", type=int, default=64)
    parser.add_argument("--maintain-batch", type=int, default=1000)
    parser.add_argument("--maintain-changes", type=int, default=4)
    parser.add_argument("--maintain-sample", type=int, default=512)
    parser.add_argument("--online-score-weights", default="0.000000,0.326441,0.580062,0.093497")
    parser.add_argument("--online-query-gain-weight", type=float, default=0.3)
    parser.add_argument("--online-query-load-weight", type=float, default=0.3)
    parser.add_argument("--anchor-cost-query-limit", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    our_anchors = export_online_anchors(args, manifest, args.out_dir)
    entry_files = prepare_entries(args, manifest, args.out_dir, our_anchors)

    meta = {
        "manifest": str(args.manifest.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "our_anchors": str(our_anchors.resolve()),
        "entry_files": {k: str(v.resolve()) for k, v in entry_files.items()},
        "search_l": args.search_l,
        "topk": args.topk,
        "entries": args.entries,
        "note": "GATE-refresh and GATE-over-ours use the same cosine top-entry generation. Both exclude selector overhead.",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    specs = [
        ("nsg_ep", "nsg_ep", None),
        ("gate_refresh", "query_entries_gate_refresh", entry_files["gate_refresh"]),
        ("gate_over_ours_online_E", "query_entries_gate_over_ours_online_E", entry_files["gate_over_ours_online_E"]),
    ]
    for repeat in range(1, args.repeats + 1):
        for alias, method, entries in specs:
            rows.extend(run_method(args, args.manifest, manifest, alias, method, args.out_dir / "runs", entries, repeat))

    raw = args.out_dir / "dynamic_gate_fusion_raw.csv"
    agg = args.out_dir / "dynamic_gate_fusion_aggregate.csv"
    write_csv(raw, rows)
    write_csv(agg, aggregate(rows))
    print(json.dumps({"raw": str(raw), "aggregate": str(agg)}, indent=2))


if __name__ == "__main__":
    main()
