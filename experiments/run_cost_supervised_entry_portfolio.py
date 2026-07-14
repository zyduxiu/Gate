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
from dynanchor.io_formats import write_ivecs


def run(command: list[Any], timeout: int) -> None:
    print("RUN", " ".join(str(x) for x in command), flush=True)
    if Path(str(command[0])).name == "nsg_dynamic_entry_runner":
        result = run_external_binary([str(x) for x in command], timeout=timeout)
        if result.stdout:
            print(result.stdout.rstrip(), flush=True)
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr, flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"command failed ({result.returncode}): {' '.join(str(x) for x in command)}")
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
    return data.reshape(rows, dim).astype(np.float32, copy=False)


def read_id_bin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
        if header.size != 2:
            raise ValueError(f"bad id bin header: {path}")
        rows, cols = int(header[0]), int(header[1])
        data = np.fromfile(handle, dtype="<u4", count=rows * cols)
    if data.size != rows * cols:
        raise ValueError(f"truncated id bin: {path}")
    return data.reshape(rows, cols).astype(np.int32, copy=False)


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i4")
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    return rows[:, 1:].astype(np.int32, copy=False)


def read_anchor_file(path: Path) -> list[int]:
    if not path.exists():
        return []
    out: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(int(line.split()[0]))
    return out


def write_anchor_file(path: Path, anchors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(x)) for x in anchors) + "\n", encoding="utf-8")


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.min(values)) if values.size else 0.0
    hi = float(np.max(values)) if values.size else 1.0
    if hi <= lo + 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def row_norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def alive_mask(rows: int, delete_path: Path) -> np.ndarray:
    alive = np.ones(rows, dtype=bool)
    if not delete_path.exists():
        return alive
    raw = np.fromfile(delete_path, dtype="<u4")
    if raw.size == 0:
        return alive
    count = int(raw[0])
    ids = raw[1 : 1 + count].astype(np.int64, copy=False)
    ids = ids[(ids >= 0) & (ids < rows)]
    alive[ids] = False
    return alive


def graph_stats(graph: np.ndarray, vectors: np.ndarray, graph_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = vectors.shape[0]
    indegree = np.zeros(rows, dtype=np.float64)
    valid = (graph >= 0) & (graph < rows)
    for neigh in graph[valid]:
        indegree[int(neigh)] += 1.0
    density = np.zeros(rows, dtype=np.float64)
    degree = np.sum(valid[:, :graph_k], axis=1).astype(np.float64)
    for i in range(rows):
        neigh = graph[i, :graph_k]
        neigh = neigh[(neigh >= 0) & (neigh < rows)]
        if neigh.size:
            d = np.sum((vectors[neigh.astype(np.int64)] - vectors[i : i + 1]) ** 2, axis=1)
            density[i] = 1.0 / (float(np.mean(d)) + 1e-6)
    return normalize(indegree), normalize(density), normalize(degree)


def build_candidates(
    manifest: dict[str, Any],
    vectors: np.ndarray,
    graph: np.ndarray,
    truth: np.ndarray,
    train_count: int,
    candidate_count: int,
    graph_k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    rows = vectors.shape[0]
    alive = alive_mask(rows, Path(manifest["paths"]["delete_u32"]))
    hub, density, degree = graph_stats(graph, vectors, graph_k)
    support = np.zeros(rows, dtype=np.float64)
    hard_support = np.zeros(rows, dtype=np.float64)
    truth_k = min(20, truth.shape[1])
    for qi in range(min(train_count, truth.shape[0])):
        weight = 1.0 + qi / max(1, train_count)
        for rank, node in enumerate(truth[qi, :truth_k]):
            node = int(node)
            if 0 <= node < rows and alive[node]:
                gain = weight / math.sqrt(rank + 1.0)
                support[node] += gain
                if qi >= int(0.7 * train_count):
                    hard_support[node] += 2.0 * gain
    support = normalize(support)
    hard_support = normalize(hard_support)

    mandatory: list[int] = []
    for key in ("gate_refreshed_hub", "gate_initial_hub", "dynamic_density", "static_density"):
        value = manifest.get("anchor_files", {}).get(key)
        if value:
            mandatory.extend(read_anchor_file(Path(value)))
    mandatory = [x for x in dict.fromkeys(mandatory) if 0 <= x < rows and alive[x]]

    inserted = np.zeros(rows, dtype=np.float64)
    n_initial = int(manifest.get("n_initial", rows))
    if n_initial < rows:
        inserted[n_initial:] = 1.0
    score = (
        0.34 * support
        + 0.22 * hard_support
        + 0.20 * hub
        + 0.16 * density
        + 0.05 * degree
        + 0.03 * inserted
    )
    score[~alive] = -1.0
    pool = np.argsort(-score)[: max(candidate_count * 12, candidate_count + 256)].astype(np.int64)
    pool = [int(x) for x in pool if score[int(x)] >= 0.0]
    selected: list[int] = []
    mandatory_budget = min(len(mandatory), max(16, candidate_count // 4))
    mandatory_sorted = sorted(mandatory, key=lambda x: float(score[int(x)]), reverse=True)
    for node in mandatory_sorted[:mandatory_budget]:
        if node not in selected:
            selected.append(int(node))

    # Score-first selection with light diversity against near-duplicates.
    norm_vectors = row_norm(vectors)
    for node in pool:
        if len(selected) >= candidate_count:
            break
        if node in selected:
            continue
        if selected:
            sims = norm_vectors[np.asarray(selected[-128:], dtype=np.int64)] @ norm_vectors[node]
            if float(np.max(sims)) > 0.985 and rng.random() > 0.15:
                continue
        selected.append(int(node))
    if len(selected) < candidate_count:
        remaining = np.flatnonzero(alive)
        rng.shuffle(remaining)
        for node in remaining:
            if len(selected) >= candidate_count:
                break
            if int(node) not in selected:
                selected.append(int(node))

    candidates = np.asarray(selected[:candidate_count], dtype=np.int32)
    scalars = {
        "hub": hub[candidates].astype(np.float32),
        "density": density[candidates].astype(np.float32),
        "degree": degree[candidates].astype(np.float32),
        "support": support[candidates].astype(np.float32),
        "hard_support": hard_support[candidates].astype(np.float32),
        "inserted": inserted[candidates].astype(np.float32),
    }
    meta = {
        "candidate_count": int(candidates.size),
        "mandatory_count": len(mandatory),
        "from_gate_refreshed": int(sum(x in set(read_anchor_file(Path(manifest.get("anchor_files", {}).get("gate_refreshed_hub", "")))) for x in candidates)) if manifest.get("anchor_files", {}).get("gate_refreshed_hub") else 0,
        "score_weights": "support=.34 hard_support=.22 hub=.20 density=.16 degree=.05 inserted=.03",
    }
    return candidates, meta, scalars


def read_anchor_costs(path: Path, query_count: int, candidate_count: int) -> tuple[np.ndarray, np.ndarray]:
    cost = np.full((query_count, candidate_count), np.inf, dtype=np.float64)
    recall = np.zeros((query_count, candidate_count), dtype=np.float64)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qi = int(row["query_id"])
            ai = int(row["anchor_rank"])
            if 0 <= qi < query_count and 0 <= ai < candidate_count:
                cost[qi, ai] = float(row["distance_computations"])
                recall[qi, ai] = float(row["recall_at_10"])
    finite = np.isfinite(cost)
    if not np.all(finite):
        fill = float(np.max(cost[finite])) if np.any(finite) else 1e9
        cost[~finite] = fill
    return cost, recall


def feature_tensor(
    queries: np.ndarray,
    vectors: np.ndarray,
    candidates: np.ndarray,
    scalars: dict[str, np.ndarray],
) -> np.ndarray:
    q = queries.astype(np.float32, copy=False)
    cvec = vectors[candidates.astype(np.int64)].astype(np.float32, copy=False)
    qn = np.sum(q * q, axis=1, keepdims=True)
    cn = np.sum(cvec * cvec, axis=1).reshape(1, -1)
    dot = q @ cvec.T
    dist = qn + cn - 2.0 * dot
    dist = np.maximum(dist, 0.0)
    qnorm = np.sqrt(np.maximum(qn, 1e-12))
    cnorm = np.sqrt(np.maximum(cn, 1e-12))
    cosine = dot / np.maximum(qnorm * cnorm, 1e-12)
    dist_norm = (dist - np.min(dist, axis=1, keepdims=True)) / np.maximum(
        np.max(dist, axis=1, keepdims=True) - np.min(dist, axis=1, keepdims=True),
        1e-6,
    )
    rank = np.argsort(np.argsort(dist, axis=1), axis=1).astype(np.float32) / max(1, candidates.size - 1)
    fields = [
        dist_norm,
        cosine,
        rank,
    ]
    for name in ("hub", "density", "degree", "support", "hard_support", "inserted"):
        value = scalars[name].reshape(1, -1).astype(np.float32)
        fields.append(np.repeat(value, q.shape[0], axis=0))
        fields.append(dist_norm * value)
    return np.stack(fields, axis=2).astype(np.float32)


def fit_ridge_ranker(features: np.ndarray, cost: np.ndarray, recall: np.ndarray, train_count: int, penalty: float, ridge: float) -> dict[str, Any]:
    x = features[:train_count].reshape(-1, features.shape[2]).astype(np.float64)
    y = np.log1p(cost[:train_count].reshape(-1))
    y = y + penalty * np.maximum(0.0, 1.0 - recall[:train_count].reshape(-1))
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    xz = (x - mean) / np.maximum(std, 1e-6)
    x_aug = np.concatenate([xz, np.ones((xz.shape[0], 1), dtype=np.float64)], axis=1)
    reg = ridge * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = ridge * 1e-3
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    return {"mean": mean, "std": std, "weights": weights}


def predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x = features.reshape(-1, features.shape[2]).astype(np.float64)
    xz = (x - model["mean"]) / np.maximum(model["std"], 1e-6)
    x_aug = np.concatenate([xz, np.ones((xz.shape[0], 1), dtype=np.float64)], axis=1)
    y = x_aug @ model["weights"]
    return y.reshape(features.shape[0], features.shape[1])


def top_entries_from_scores(
    scores: np.ndarray,
    candidates: np.ndarray,
    candidate_vectors: np.ndarray,
    entries: int,
    diverse: bool,
) -> np.ndarray:
    out = np.empty((scores.shape[0], entries), dtype=np.int32)
    norm = row_norm(candidate_vectors)
    for qi in range(scores.shape[0]):
        order = np.argsort(scores[qi])
        picked: list[int] = []
        for idx in order:
            idx = int(idx)
            if diverse and picked:
                sims = norm[np.asarray(picked, dtype=np.int64)] @ norm[idx]
                if float(np.max(sims)) > 0.985:
                    continue
            picked.append(idx)
            if len(picked) >= entries:
                break
        if len(picked) < entries:
            for idx in order:
                idx = int(idx)
                if idx not in picked:
                    picked.append(idx)
                if len(picked) >= entries:
                    break
        out[qi] = candidates[np.asarray(picked[:entries], dtype=np.int64)]
    return out


def combine_entry_matrices(primary: np.ndarray, secondary: np.ndarray, entries: int) -> np.ndarray:
    rows = primary.shape[0]
    out = np.empty((rows, entries), dtype=np.int32)
    for qi in range(rows):
        picked: list[int] = []
        for source in (primary[qi], secondary[qi]):
            for node in source:
                node = int(node)
                if node not in picked:
                    picked.append(node)
                if len(picked) >= entries:
                    break
            if len(picked) >= entries:
                break
        while len(picked) < entries:
            picked.append(int(primary[qi, min(len(picked), primary.shape[1] - 1)]))
        out[qi] = np.asarray(picked[:entries], dtype=np.int32)
    return out


def run_runner(
    args: argparse.Namespace,
    manifest_path: Path,
    manifest: dict[str, Any],
    out_dir: Path,
    mode: str,
    query_entries: Path | None = None,
    anchors: Path | None = None,
    out_anchor_costs: Path | None = None,
    anchor_cost_query_limit: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = manifest["paths"]
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
        mode,
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
        "--out-json",
        (out_dir / f"{mode}_L{args.search_l}.json").resolve(),
    ]
    if anchors is not None:
        command.extend(["--anchors-txt", anchors.resolve()])
    if query_entries is not None:
        command.extend(["--query-entries-ivecs", query_entries.resolve()])
    if out_anchor_costs is not None:
        out_anchor_costs.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["--out-anchor-costs-csv", out_anchor_costs.resolve()])
        if anchor_cost_query_limit:
            command.extend(["--anchor-cost-query-limit", anchor_cost_query_limit])
    run(command, timeout=args.timeout)


def summarize(run_dir: Path, manifest_path: Path) -> list[dict[str, str]]:
    out = run_dir / "summary.csv"
    run(
        [
            sys.executable,
            ROOT / "experiments" / "summarize_nsg_entry_results.py",
            "--dir",
            run_dir,
            "--manifest",
            manifest_path,
            "--out",
            out,
        ],
        timeout=600,
    )
    with out.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost-supervised static entry portfolio for ANN graph search.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "static_fair_gate_vs_nsg" / "static_manifest.json")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "cost_supervised_entry_portfolio_static")
    parser.add_argument("--candidate-count", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--search-l", type=int, default=120)
    parser.add_argument("--entries", type=int, default=3)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--graph-k", type=int, default=32)
    parser.add_argument("--insert-degree", type=int, default=0)
    parser.add_argument("--max-degree", type=int, default=64)
    parser.add_argument("--recall-penalty", type=float, default=4.0)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    initial = read_fbin(Path(paths["initial_fbin"]))
    inserts = read_fbin(Path(paths["insert_fbin"]))
    vectors = np.vstack([initial, inserts]).astype(np.float32, copy=False)
    queries = read_fbin(Path(paths["query_fbin"]))
    truth = read_id_bin(Path(paths["truth_dynamic_bin"]))
    graph = read_ivecs(Path(paths["all_nsg_knn_graph"]))
    train_count = max(1, min(queries.shape[0], int(round(queries.shape[0] * args.train_fraction))))

    candidates, candidate_meta, scalars = build_candidates(
        manifest=manifest,
        vectors=vectors,
        graph=graph,
        truth=truth,
        train_count=train_count,
        candidate_count=args.candidate_count,
        graph_k=args.graph_k,
        rng=rng,
    )
    candidate_file = args.out_dir / "cost_portfolio_candidates.txt"
    write_anchor_file(candidate_file, candidates)

    costs_csv = args.out_dir / "candidate_anchor_costs.csv"
    probe_dir = args.out_dir / "cost_probe"
    run_runner(
        args,
        args.manifest,
        manifest,
        probe_dir,
        mode="static_density",
        anchors=candidate_file,
        out_anchor_costs=costs_csv,
        anchor_cost_query_limit=queries.shape[0],
    )
    cost, recall = read_anchor_costs(costs_csv, queries.shape[0], candidates.size)
    features = feature_tensor(queries, vectors, candidates, scalars)
    model = fit_ridge_ranker(features, cost, recall, train_count, args.recall_penalty, args.ridge)
    predicted = predict(model, features)
    candidate_vectors = vectors[candidates.astype(np.int64)]

    nearest_scores = features[:, :, 0]
    oracle_scores = np.log1p(cost) + args.recall_penalty * np.maximum(0.0, 1.0 - recall)
    entries = {
        "query_entries_cost_portfolio": top_entries_from_scores(predicted, candidates, candidate_vectors, args.entries, diverse=False),
        "query_entries_cost_portfolio_diverse": top_entries_from_scores(predicted, candidates, candidate_vectors, args.entries, diverse=True),
        "query_entries_nearest_portfolio": top_entries_from_scores(nearest_scores, candidates, candidate_vectors, args.entries, diverse=True),
        "query_entries_oracle_portfolio": top_entries_from_scores(oracle_scores, candidates, candidate_vectors, args.entries, diverse=True),
    }
    gate_centroid_path = manifest.get("query_entry_files", {}).get("gate_centroid_refreshed")
    if gate_centroid_path:
        gate_centroid = read_ivecs(Path(gate_centroid_path))
        nearest = entries["query_entries_nearest_portfolio"]
        if gate_centroid.shape[0] == nearest.shape[0]:
            entries["query_entries_portfolio_then_gate"] = combine_entry_matrices(nearest, gate_centroid, args.entries)
            entries["query_entries_gate_then_portfolio"] = combine_entry_matrices(gate_centroid, nearest, args.entries)
    entry_files: dict[str, str] = {}
    for label, matrix in entries.items():
        path = args.out_dir / f"{label}.ivecs"
        write_ivecs(path, matrix)
        entry_files[label] = str(path.resolve())

    specs: list[tuple[str, str, Path | None, Path | None]] = [
        ("nsg_ep", "nsg_ep", None, None),
        ("portfolio_anchor_nearest", "static_density", None, candidate_file),
    ]
    for label, value in manifest.get("query_entry_files", {}).items():
        specs.append((f"query_entries_{label}", f"query_entries_{label}", Path(value), None))
    for key in ("gate_initial_hub", "gate_refreshed_hub"):
        value = manifest.get("anchor_files", {}).get(key)
        if value:
            specs.append((key, key, None, Path(value)))
    for label, path in entry_files.items():
        specs.append((label, label, Path(path), None))

    rows: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        for alias, mode, query_entries, anchors in specs:
            run_dir = args.out_dir / "runs" / alias / f"run_{repeat:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_runner(
                args,
                args.manifest,
                manifest,
                run_dir,
                mode=mode,
                query_entries=query_entries,
                anchors=anchors,
            )
            for row in summarize(run_dir, args.manifest):
                row["method"] = alias
                row["repeat"] = repeat
                row["train_fraction"] = args.train_fraction
                row["candidate_count"] = args.candidate_count
                rows.append(row)
    write_csv(args.out_dir / "cost_portfolio_raw.csv", rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["search_l"])), []).append(row)
    agg: list[dict[str, Any]] = []
    for (method, search_l), items in sorted(groups.items(), key=lambda item: item[0]):
        out: dict[str, Any] = {"method": method, "search_l": int(search_l), "runs": len(items)}
        for metric in ("recall_at_10", "recall_at_k", "qps", "avg_latency_us", "p99_latency_us", "avg_distance_computations"):
            vals = [float(row[metric]) for row in items if row.get(metric) not in (None, "")]
            out[f"{metric}_mean"] = sum(vals) / len(vals)
            if len(vals) > 1:
                m = out[f"{metric}_mean"]
                out[f"{metric}_std"] = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
        agg.append(out)
    write_csv(args.out_dir / "cost_portfolio_aggregate.csv", agg)

    meta = {
        "method": "cost_supervised_entry_portfolio",
        "manifest": str(args.manifest.resolve()),
        "candidate_file": str(candidate_file.resolve()),
        "candidate_meta": candidate_meta,
        "entry_files": entry_files,
        "train_queries": train_count,
        "total_queries": int(queries.shape[0]),
        "objective": "ridge regression predicts log(1 + single-anchor distance_computations) plus recall-loss penalty; query selects the lowest predicted cost entries",
        "note": "This is deliberately not GATE: no high-level navigation graph and no Graph2Vec/two-tower. It is a direct cost-supervised entry portfolio.",
    }
    (args.out_dir / "cost_portfolio_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": str((args.out_dir / "cost_portfolio_aggregate.csv").resolve()), "meta": meta}, indent=2))


if __name__ == "__main__":
    main()
