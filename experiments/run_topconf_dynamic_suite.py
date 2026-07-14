from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.external import run_external_binary
from dynanchor.io_formats import read_fvecs, write_ivecs


DEFAULT_DATASETS = {
    "sift": ROOT / "data" / "sift_stress_drift" / "manifest.json",
    "glove": ROOT / "data" / "glove100k_dynamic" / "manifest.json",
    "fashion": ROOT / "data" / "fashion55k_dynamic" / "manifest.json",
}

WEIGHT_CONFIGS = {
    "default": "0.000000,0.326441,0.580062,0.093497",
    "no_hubness": "0.000000,0.750000,0.000000,0.250000",
    "hub_heavy": "0.000000,0.200000,0.700000,0.100000",
    "coverage_hub": "0.000000,0.400000,0.500000,0.100000",
    "fresh_heavy": "0.000000,0.250000,0.400000,0.350000",
    "query_free": "0.000000,0.326441,0.580062,0.093497",
}


def run(command: list[Any], timeout: int, cwd: Path = ROOT) -> None:
    print("RUN", " ".join(str(item) for item in command), flush=True)
    if Path(str(command[0])).name in {"nsg_dynamic_entry_runner", "hnsw_anchor_runner"}:
        result = run_external_binary([str(item) for item in command], timeout=timeout)
        if result.stdout:
            print(result.stdout.rstrip(), flush=True)
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr, flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"command failed ({result.returncode}): {' '.join(str(x) for x in command)}")
        return
    result = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
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


def read_anchor_ids(path: Path) -> np.ndarray:
    ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(int(line.split()[0]))
    if not ids:
        raise ValueError(f"empty anchor file: {path}")
    return np.asarray(ids, dtype=np.int32)


def parse_gate_eps(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    iter1, iter2, _dim = [int(x) for x in lines[0].split()[:3]]
    eps_lines = lines[1 + iter1 : 1 + iter1 + iter1]
    eps: list[int] = []
    for line in eps_lines:
        values = [int(x) for x in line.split()]
        if len(values) != iter2:
            raise ValueError(f"expected {iter2} entries in {path}")
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


def split_manifest(source_manifest: Path, out_dir: Path, seed: int, train_fraction: float, timeout: int) -> tuple[Path, Path]:
    meta_path = out_dir / "split_meta.json"
    train_manifest = out_dir / "train" / "manifest.json"
    eval_manifest = out_dir / "eval" / "manifest.json"
    if train_manifest.exists() and eval_manifest.exists():
        return train_manifest, eval_manifest
    run(
        [
            sys.executable,
            ROOT / "experiments" / "split_query_manifest.py",
            "--manifest",
            source_manifest,
            "--out-dir",
            out_dir,
            "--seed",
            seed,
            "--train-fraction",
            train_fraction,
        ],
        timeout=timeout,
    )
    if not meta_path.exists():
        raise RuntimeError(f"split did not write {meta_path}")
    return train_manifest, eval_manifest


def gate_candidate_set(
    manifest: dict[str, Any],
    kind: str,
    all_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    gate_files = manifest.get("gate_hub_files", {})
    if kind == "static" and gate_files.get("gate_initial_hub_file"):
        hub_file = Path(gate_files["gate_initial_hub_file"])
        emb_file = Path(gate_files.get("gate_initial_hub_embeddings", hub_file.with_name("gate_initial_hub_embeddings.fvecs")))
        return parse_gate_eps(hub_file), read_fvecs(emb_file), "official_gate_initial_hubs"
    if kind == "refresh" and gate_files.get("gate_refreshed_hub_file"):
        hub_file = Path(gate_files["gate_refreshed_hub_file"])
        emb_file = Path(gate_files.get("gate_refreshed_hub_embeddings", hub_file.with_name("gate_refreshed_hub_embeddings.fvecs")))
        return parse_gate_eps(hub_file), read_fvecs(emb_file), "official_gate_refreshed_hubs"

    anchor_key = "static_density" if kind == "static" else "dynamic_density"
    ids = read_anchor_ids(Path(manifest["anchor_files"][anchor_key]))
    return ids, all_vectors[ids], f"gate_style_{anchor_key}"


def export_online_anchors(
    args: argparse.Namespace,
    train_manifest: Path,
    train: dict[str, Any],
    out_dir: Path,
    weight_name: str,
    weights: str,
    query_gain_weight: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = (out_dir / f"online_export_{weight_name}.json").resolve()
    out_anchors = (out_dir / f"ours_online_E_{weight_name}.txt").resolve()
    if args.skip_existing and out_anchors.exists():
        return out_anchors
    paths = train["paths"]
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
        train["anchor_files"]["static_density"],
        "--search-l",
        args.export_search_l,
        "--entries",
        args.entries,
        "--topk",
        args.topk,
        "--insert-degree",
        args.insert_degree,
        "--max-degree",
        args.max_degree,
        "--maintain-batch",
        args.maintain_batch,
        "--maintain-changes",
        args.maintain_changes,
        "--maintain-sample",
        args.maintain_sample,
        "--online-score-weights",
        weights,
        "--online-query-gain-weight",
        query_gain_weight,
        "--online-query-load-weight",
        args.online_query_load_weight,
        "--anchor-cost-query-limit",
        args.anchor_cost_query_limit,
        "--out-json",
        out_json,
        "--out-anchors-txt",
        out_anchors,
    ]
    run(command, timeout=args.timeout)
    return out_anchors


def prepare_query_entries(
    args: argparse.Namespace,
    eval_manifest: dict[str, Any],
    out_dir: Path,
    online_anchor_file: Path,
    suffix: str,
) -> tuple[dict[str, Path], dict[str, str]]:
    paths = eval_manifest["paths"]
    queries = read_fvecs(Path(paths["query_fvecs"]))
    all_vectors = read_fbin(Path(paths["all_fbin"]))

    entry_files: dict[str, Path] = {}
    provenance: dict[str, str] = {}
    for alias, kind in [("gate_static_E", "static"), ("gate_refresh_E", "refresh")]:
        ids, embeddings, source = gate_candidate_set(eval_manifest, kind, all_vectors)
        entries = cosine_top_entries(queries, embeddings, ids, args.entries)
        out_path = out_dir / f"{alias}_{suffix}_top{args.entries}.ivecs"
        write_ivecs(out_path, entries)
        entry_files[alias] = out_path
        provenance[alias] = source

    ids = read_anchor_ids(online_anchor_file)
    entries = cosine_top_entries(queries, all_vectors[ids], ids, args.entries)
    out_path = out_dir / f"gate_over_ours_online_E_{suffix}_top{args.entries}.ivecs"
    write_ivecs(out_path, entries)
    entry_files["gate_over_ours_online_E"] = out_path
    provenance["gate_over_ours_online_E"] = "ours_online_maintained_E_then_gate_style_selector"
    return entry_files, provenance


def run_nsg_entry(
    args: argparse.Namespace,
    eval_manifest_path: Path,
    eval_manifest: dict[str, Any],
    out_dir: Path,
    alias: str,
    method: str,
    query_entries: Path | None,
    repeat: int,
) -> list[dict[str, Any]]:
    run_dir = out_dir / alias / f"run_{repeat:02d}"
    command = [
        sys.executable,
        ROOT / "experiments" / "run_nsg_dynamic_entry.py",
        "--manifest",
        eval_manifest_path,
        "--initial-nsg",
        eval_manifest["paths"]["initial_nsg"],
        "--runner",
        args.runner,
        "--out-dir",
        run_dir,
        "--methods",
        method,
        "--search-l",
        args.search_l,
        "--entries",
        args.entries,
        "--topk",
        args.topk,
        "--insert-degree",
        args.insert_degree,
        "--max-degree",
        args.max_degree,
        "--maintain-batch",
        args.maintain_batch,
        "--maintain-changes",
        args.maintain_changes,
        "--maintain-sample",
        args.maintain_sample,
        "--timeout",
        args.timeout,
    ]
    if query_entries is not None:
        command.extend(["--query-entries-ivecs", query_entries])
    run(command, timeout=args.timeout)
    summary = run_dir / "summary.csv"
    run(
        [
            sys.executable,
            ROOT / "experiments" / "summarize_nsg_entry_results.py",
            "--dir",
            run_dir,
            "--manifest",
            eval_manifest_path,
            "--out",
            summary,
        ],
        timeout=600,
    )
    rows: list[dict[str, Any]] = []
    for row in read_csv(summary):
        row["method"] = alias
        row["repeat"] = repeat
        row["backend_group"] = "NSG dynamic no-rebuild"
        row["query_split"] = "train-maintain/eval-search"
        rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def aggregate(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)
    out: list[dict[str, Any]] = []
    metrics = [
        "recall_at_k",
        "recall_at_10",
        "qps",
        "avg_latency_us",
        "p99_latency_us",
        "avg_distance_computations",
        "maintenance_seconds",
        "maintenance_distance_computations",
    ]
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        row: dict[str, Any] = {name: value for name, value in zip(group_keys, key)}
        row["runs"] = len(items)
        for metric in metrics:
            vals: list[float] = []
            for item in items:
                value = item.get(metric, "")
                if value not in (None, ""):
                    vals.append(float(value))
            if vals:
                row[f"{metric}_mean"] = mean(vals)
                row[f"{metric}_std"] = stdev(vals)
        out.append(row)
    return out


def add_speedups(rows: list[dict[str, Any]]) -> None:
    by_dataset_l: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_dataset_l.setdefault((str(row.get("dataset")), str(row.get("search_l"))), {})[str(row.get("method"))] = row
    for (_dataset, _l), methods in by_dataset_l.items():
        nsg = methods.get("nsg_ep")
        gate_refresh = methods.get("gate_refresh_E")
        for row in methods.values():
            qps = float(row.get("qps_mean", "nan"))
            if nsg and float(nsg.get("qps_mean", "nan")) > 0:
                row["qps_speedup_vs_nsg_ep"] = qps / float(nsg["qps_mean"])
            if gate_refresh and float(gate_refresh.get("qps_mean", "nan")) > 0:
                row["qps_speedup_vs_gate_refresh_E"] = qps / float(gate_refresh["qps_mean"])


def run_hnsw_controls(args: argparse.Namespace, manifest_path: Path, out_dir: Path, dataset_alias: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.no_hnsw:
        return rows
    from run_hnsw_anchor_fair import compile_runner, run_runner

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_paths = {
        "initial": Path(manifest["paths"]["initial_fbin"]),
        "insert": Path(manifest["paths"]["insert_fbin"]),
        "queries": Path(manifest["paths"]["query_fbin"]),
        "truth": Path(manifest["paths"]["truth_dynamic_bin"]),
        "deletes": Path(manifest["paths"]["delete_u32"]),
    }
    runner = compile_runner(timeout=args.timeout)
    efs = [int(item.strip()) for item in args.hnsw_efs.split(",") if item.strip()]
    for repeat in range(1, args.repeats + 1):
        for ef in efs:
            run_dir = out_dir / "hnsw" / dataset_alias / f"run_{repeat:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            payload = run_runner(
                runner=runner,
                data_paths=data_paths,
                out_json=run_dir / f"builtin_ef{ef}.json",
                mode="builtin",
                ef=ef,
                entries=1,
                topk=args.topk,
                m=args.hnsw_m,
                ef_construction=args.hnsw_ef_construction,
                seed=int(manifest.get("seed", 7)) + repeat,
                timeout=args.timeout,
            )
            rows.append(
                {
                    "dataset_alias": dataset_alias,
                    "dataset": manifest["dataset"],
                    "backend": "hnswlib_cpp",
                    "method": "builtin_hnsw_entry",
                    "ef": ef,
                    "entries": 1,
                    "anchors": 0,
                    "recall_at_10": payload["recall_at_10"],
                    "recall_at_k": payload["recall_at_10"],
                    "avg_cmps": payload["avg_cmps"],
                    "avg_hops": payload["avg_hops"],
                    "avg_latency_us": payload["avg_latency_us"],
                    "p99_latency_us": payload["p99_latency_us"],
                    "qps": payload["qps"],
                    "build_seconds": payload["build_seconds"],
                    "repeat": repeat,
                    "backend_group": "HNSW dynamic control",
                    "note": "standard hnswlib hierarchical entry on the same dynamic train/eval manifest",
                }
            )
    return rows


def run_dataset(args: argparse.Namespace, alias: str, manifest_path: Path, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    split_dir = out_dir / "splits" / alias
    train_manifest_path, eval_manifest_path = split_manifest(
        manifest_path,
        split_dir,
        seed=args.split_seed,
        train_fraction=args.train_fraction,
        timeout=args.timeout,
    )
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    online_dir = out_dir / "online_E" / alias
    online_anchor = export_online_anchors(
        args,
        train_manifest_path,
        train_manifest,
        online_dir,
        "default",
        args.online_score_weights,
        args.online_query_gain_weight,
    )
    entry_files, provenance = prepare_query_entries(args, eval_manifest, out_dir / "entry_files" / alias, online_anchor, "default")

    specs = [
        ("nsg_ep", "nsg_ep", None),
        ("gate_static_E", "query_entries_gate_static_E", entry_files["gate_static_E"]),
        ("gate_refresh_E", "query_entries_gate_refresh_E", entry_files["gate_refresh_E"]),
        ("gate_over_ours_online_E", "query_entries_gate_over_ours_online_E", entry_files["gate_over_ours_online_E"]),
    ]
    nsg_rows: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        for method_alias, runner_method, entries in specs:
            nsg_rows.extend(
                run_nsg_entry(
                    args,
                    eval_manifest_path,
                    eval_manifest,
                    out_dir / "runs" / alias,
                    method_alias,
                    runner_method,
                    entries,
                    repeat,
                )
            )
    hnsw_rows = run_hnsw_controls(args, eval_manifest_path, out_dir, alias)
    meta = {
        "alias": alias,
        "source_manifest": str(manifest_path.resolve()),
        "train_manifest": str(train_manifest_path.resolve()),
        "eval_manifest": str(eval_manifest_path.resolve()),
        "online_anchor_file": str(online_anchor.resolve()),
        "entry_files": {key: str(value.resolve()) for key, value in entry_files.items()},
        "entry_provenance": provenance,
    }
    return nsg_rows, hnsw_rows, meta


def run_sensitivity(args: argparse.Namespace, manifest_path: Path, out_dir: Path) -> list[dict[str, Any]]:
    if args.no_sensitivity:
        return []
    split_dir = out_dir / "splits" / "sensitivity_sift"
    train_manifest_path, eval_manifest_path = split_manifest(
        manifest_path,
        split_dir,
        seed=args.split_seed + 17,
        train_fraction=args.train_fraction,
        timeout=args.timeout,
    )
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    selected = [item.strip() for item in args.sensitivity_configs.split(",") if item.strip()]
    old_search_l = args.search_l
    args.search_l = args.sensitivity_search_l
    try:
        for config in selected:
            if config not in WEIGHT_CONFIGS:
                raise KeyError(f"unknown sensitivity config {config}; available={sorted(WEIGHT_CONFIGS)}")
            query_gain = 0.0 if config == "query_free" else args.online_query_gain_weight
            anchor_file = export_online_anchors(
                args,
                train_manifest_path,
                train_manifest,
                out_dir / "sensitivity" / config,
                config,
                WEIGHT_CONFIGS[config],
                query_gain,
            )
            entry_files, provenance = prepare_query_entries(
                args,
                eval_manifest,
                out_dir / "sensitivity" / config,
                anchor_file,
                config,
            )
            for repeat in range(1, args.repeats + 1):
                method_rows = run_nsg_entry(
                    args,
                    eval_manifest_path,
                    eval_manifest,
                    out_dir / "sensitivity" / config,
                    f"gate_over_ours_online_E[{config}]",
                    f"query_entries_ours_{config}",
                    entry_files["gate_over_ours_online_E"],
                    repeat,
                )
                for row in method_rows:
                    row["sensitivity_config"] = config
                    row["online_score_weights"] = WEIGHT_CONFIGS[config]
                    row["online_query_gain_weight"] = query_gain
                    row["entry_provenance"] = provenance["gate_over_ours_online_E"]
                    rows.append(row)
    finally:
        args.search_l = old_search_l
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-conference-style dynamic entry maintenance suite.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "topconf_dynamic_entry_suite")
    parser.add_argument("--datasets", default="sift,glove,fashion")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--search-l", default="80,120,160,240")
    parser.add_argument("--export-search-l", type=int, default=120)
    parser.add_argument("--hnsw-efs", default="80,120,240")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--entries", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=20260714)
    parser.add_argument("--insert-degree", type=int, default=32)
    parser.add_argument("--max-degree", type=int, default=64)
    parser.add_argument("--maintain-batch", type=int, default=1000)
    parser.add_argument("--maintain-changes", type=int, default=4)
    parser.add_argument("--maintain-sample", type=int, default=512)
    parser.add_argument("--online-score-weights", default=WEIGHT_CONFIGS["default"])
    parser.add_argument("--online-query-gain-weight", type=float, default=0.3)
    parser.add_argument("--online-query-load-weight", type=float, default=0.3)
    parser.add_argument("--anchor-cost-query-limit", type=int, default=128)
    parser.add_argument("--sensitivity-search-l", default="160")
    parser.add_argument("--sensitivity-configs", default="default,no_hubness,hub_heavy,coverage_hub,fresh_heavy,query_free")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-hnsw", action="store_true")
    parser.add_argument("--no-sensitivity", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_aliases = [item.strip() for item in args.datasets.split(",") if item.strip()]
    dataset_paths = {alias: DEFAULT_DATASETS[alias] for alias in dataset_aliases}

    all_nsg_rows: list[dict[str, Any]] = []
    all_hnsw_rows: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    for alias, manifest_path in dataset_paths.items():
        nsg_rows, hnsw_rows, meta = run_dataset(args, alias, manifest_path, args.out_dir)
        for row in nsg_rows:
            row["dataset_alias"] = alias
        all_nsg_rows.extend(nsg_rows)
        all_hnsw_rows.extend(hnsw_rows)
        metas.append(meta)

    sensitivity_rows = run_sensitivity(args, DEFAULT_DATASETS["sift"], args.out_dir)

    write_csv(args.out_dir / "raw_nsg_gate_ours.csv", all_nsg_rows)
    write_csv(args.out_dir / "raw_hnsw.csv", all_hnsw_rows)
    write_csv(args.out_dir / "raw_sensitivity.csv", sensitivity_rows)

    nsg_agg = aggregate(all_nsg_rows, ["dataset_alias", "dataset", "method", "search_l"])
    add_speedups(nsg_agg)
    hnsw_agg = aggregate(all_hnsw_rows, ["dataset_alias", "dataset", "method", "ef"])
    sensitivity_agg = aggregate(sensitivity_rows, ["dataset", "method", "search_l", "sensitivity_config"])

    write_csv(args.out_dir / "aggregate_nsg_gate_ours_5runs.csv", nsg_agg)
    write_csv(args.out_dir / "aggregate_hnsw_5runs.csv", hnsw_agg)
    write_csv(args.out_dir / "aggregate_sensitivity_5runs.csv", sensitivity_agg)
    (args.out_dir / "meta.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "datasets": {alias: str(path.resolve()) for alias, path in dataset_paths.items()},
                "dataset_runs": metas,
                "notes": [
                    "NSG/GATE/ours use the same dynamic no-rebuild NSG search kernel; only entry initialization changes.",
                    "Online E is maintained using train queries only; search metrics are evaluated on held-out eval queries.",
                    "SIFT uses existing GATE hub artifacts when present. Other datasets use gate-style refreshed/static E from density anchors.",
                    "Selector cost is excluded for gate_refresh_E and gate_over_ours_online_E equally; maintenance/export cost is reported separately.",
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "nsg_gate_ours": str((args.out_dir / "aggregate_nsg_gate_ours_5runs.csv").resolve()),
                "hnsw": str((args.out_dir / "aggregate_hnsw_5runs.csv").resolve()),
                "sensitivity": str((args.out_dir / "aggregate_sensitivity_5runs.csv").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
