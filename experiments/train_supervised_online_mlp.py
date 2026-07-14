from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.external import run_external_binary


def read_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
        if header.size != 2:
            raise ValueError(f"invalid fbin header: {path}")
        rows, dim = int(header[0]), int(header[1])
        values = np.fromfile(handle, dtype="<f4", count=rows * dim)
    if values.size != rows * dim:
        raise ValueError(f"truncated fbin payload: {path}")
    return values.reshape(rows, dim).astype(np.float32, copy=False)


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i4")
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    return rows[:, 1:].astype(np.int32, copy=True)


def read_u32_list(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u4")
    if raw.size == 0:
        return np.empty(0, dtype=np.uint32)
    count = int(raw[0])
    return raw[1 : 1 + count].astype(np.uint32, copy=False)


def read_anchor_file(path: Path) -> list[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_anchor_file(path: Path, anchors: list[int] | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(x)) for x in anchors) + "\n", encoding="utf-8")


def l2_matrix(a: np.ndarray, b: np.ndarray, batch_size: int = 512) -> np.ndarray:
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
    b_norm = np.sum(b * b, axis=1).reshape(1, -1)
    for start in range(0, a.shape[0], batch_size):
        stop = min(start + batch_size, a.shape[0])
        block = a[start:stop]
        out[start:stop] = np.sum(block * block, axis=1, keepdims=True) + b_norm - 2.0 * block @ b.T
    return out


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def sample_candidates(
    initial_rows: int,
    total_rows: int,
    alive: np.ndarray,
    static_anchors: list[int],
    candidate_count: int,
    inserted_fraction: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    inserted = np.flatnonzero(alive[initial_rows:]).astype(np.uint32) + initial_rows
    old = np.flatnonzero(alive[:initial_rows]).astype(np.uint32)
    take_inserted = min(inserted.size, int(round(candidate_count * inserted_fraction)))
    take_old = min(old.size, max(0, candidate_count - take_inserted))
    chosen: set[int] = set(int(x) for x in static_anchors if 0 <= int(x) < total_rows and alive[int(x)])
    if take_inserted:
        chosen.update(int(x) for x in rng.choice(inserted, size=take_inserted, replace=False))
    if take_old:
        chosen.update(int(x) for x in rng.choice(old, size=take_old, replace=False))
    if len(chosen) > candidate_count:
        keep_static = [x for x in static_anchors if x in chosen]
        rest = np.asarray([x for x in chosen if x not in set(keep_static)], dtype=np.uint32)
        take_rest = max(0, candidate_count - len(keep_static))
        if rest.size > take_rest:
            rest = rng.choice(rest, size=take_rest, replace=False)
        chosen = set(keep_static) | set(int(x) for x in rest)
    return np.asarray(sorted(chosen), dtype=np.uint32)


def compute_features(
    all_vectors: np.ndarray,
    all_knn: np.ndarray,
    alive: np.ndarray,
    candidates: np.ndarray,
    static_anchors: list[int],
    initial_rows: int,
) -> np.ndarray:
    anchor_vecs = all_vectors[np.asarray(static_anchors, dtype=np.uint32)]
    cand_vecs = all_vectors[candidates]
    far = np.min(l2_matrix(cand_vecs, anchor_vecs), axis=1)

    density = np.zeros(candidates.size, dtype=np.float64)
    for i, node in enumerate(candidates):
        dists: list[float] = []
        for other in all_knn[int(node), :]:
            oid = int(other)
            if oid < 0 or oid >= all_vectors.shape[0] or oid == int(node) or not alive[oid]:
                continue
            diff = all_vectors[int(node)] - all_vectors[oid]
            dists.append(float(np.sqrt(np.dot(diff, diff))))
            if len(dists) >= 16:
                break
        density[i] = 0.0 if not dists else 1.0 / (float(np.mean(dists)) + 1e-6)

    # Approximate navigational hubness by alive-neighbor indegree in the available KNN graph.
    indegree = np.zeros(all_vectors.shape[0], dtype=np.float64)
    for row_id in range(min(all_knn.shape[0], all_vectors.shape[0])):
        if not alive[row_id]:
            continue
        for other in all_knn[row_id, :64]:
            oid = int(other)
            if 0 <= oid < all_vectors.shape[0] and alive[oid]:
                indegree[oid] += 1.0
    hubness = indegree[candidates]
    inserted = (candidates >= initial_rows).astype(np.float64)
    return np.column_stack([normalize(far), normalize(density), normalize(hubness), inserted]).astype(np.float64)


def run_anchor_costs(
    manifest: dict[str, Any],
    initial_nsg: Path,
    runner: Path,
    anchors_txt: Path,
    out_dir: Path,
    search_l: int,
    topk: int,
    query_limit: int,
    timeout: int,
) -> Path:
    paths = manifest["paths"]
    out_dir.mkdir(parents=True, exist_ok=True)
    costs_csv = out_dir / "candidate_anchor_costs.csv"
    out_json = out_dir / "candidate_anchor_probe.json"
    command = [
        str(runner.resolve()),
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
        str(initial_nsg.resolve()),
        "--all-knn-graph",
        paths["all_nsg_knn_graph"],
        "--mode",
        "static_density",
        "--anchors-txt",
        str(anchors_txt.resolve()),
        "--search-l",
        str(search_l),
        "--entries",
        "4",
        "--topk",
        str(topk),
        "--insert-degree",
        "32",
        "--max-degree",
        "64",
        "--maintain-batch",
        "1000",
        "--maintain-changes",
        "4",
        "--maintain-sample",
        "512",
        "--out-json",
        str(out_json.resolve()),
        "--out-anchor-costs-csv",
        str(costs_csv.resolve()),
        "--anchor-cost-query-limit",
        str(query_limit),
    ]
    result = run_external_binary(command, timeout=timeout)
    (out_dir / "candidate_anchor_probe.log.json").write_text(
        json.dumps({"command": result.command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, indent=2),
        encoding="utf-8",
    )
    if not result.ok:
        raise RuntimeError(f"anchor cost probe failed; see {out_dir / 'candidate_anchor_probe.log.json'}")
    return costs_csv


def read_anchor_labels(costs_csv: Path, candidates: np.ndarray, objective: str) -> tuple[np.ndarray, dict[str, float]]:
    per_anchor: dict[int, list[tuple[float, float]]] = {int(anchor): [] for anchor in candidates}
    with costs_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            anchor = int(row["anchor_id"])
            if anchor in per_anchor:
                per_anchor[anchor].append((float(row["recall_at_10"]), float(row["distance_computations"])))
    mean_recall = np.zeros(candidates.size, dtype=np.float64)
    mean_dist = np.zeros(candidates.size, dtype=np.float64)
    for i, anchor in enumerate(candidates):
        values = per_anchor[int(anchor)]
        if not values:
            mean_recall[i] = 0.0
            mean_dist[i] = 1e9
        else:
            mean_recall[i] = float(np.mean([x[0] for x in values]))
            mean_dist[i] = float(np.mean([x[1] for x in values]))
    good_dist = 1.0 - normalize(mean_dist)
    if objective == "distance":
        labels = good_dist
    elif objective == "recall":
        labels = normalize(mean_recall)
    elif objective == "utility":
        # Use raw recall as a small guardrail. Normalizing recall here is harmful when
        # recall varies only in the fourth decimal place.
        labels = 0.25 * mean_recall + 0.75 * good_dist
        labels = normalize(labels)
    else:
        raise ValueError(f"unknown label objective: {objective}")
    return labels.astype(np.float64), {
        "mean_recall_min": float(np.min(mean_recall)),
        "mean_recall_max": float(np.max(mean_recall)),
        "mean_dist_min": float(np.min(mean_dist)),
        "mean_dist_max": float(np.max(mean_dist)),
        "label_min": float(np.min(labels)),
        "label_max": float(np.max(labels)),
    }


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict[str, np.ndarray | float | list[float]]:
    rng = np.random.default_rng(seed)
    n, dim = x.shape
    w1 = rng.normal(scale=0.25, size=(hidden, dim))
    b1 = np.zeros(hidden, dtype=np.float64)
    w2 = rng.normal(scale=0.20, size=hidden)
    b2 = 0.0
    m = {name: np.zeros_like(value) for name, value in {"w1": w1, "b1": b1, "w2": w2}.items()}
    v = {name: np.zeros_like(value) for name, value in {"w1": w1, "b1": b1, "w2": w2}.items()}
    mb2 = 0.0
    vb2 = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        z1 = x @ w1.T + b1.reshape(1, -1)
        h = np.maximum(z1, 0.0)
        pred = h @ w2 + b2
        err = pred - y
        loss = float(np.mean(err * err) + weight_decay * (np.sum(w1 * w1) + np.sum(w2 * w2)))
        losses.append(loss)
        dpred = (2.0 / n) * err
        gw2 = h.T @ dpred + 2.0 * weight_decay * w2
        gb2 = float(np.sum(dpred))
        dh = dpred.reshape(-1, 1) @ w2.reshape(1, -1)
        dz1 = dh * (z1 > 0.0)
        gw1 = dz1.T @ x + 2.0 * weight_decay * w1
        gb1 = np.sum(dz1, axis=0)
        grads = {"w1": gw1, "b1": gb1, "w2": gw2}
        params = {"w1": w1, "b1": b1, "w2": w2}
        for name, grad in grads.items():
            m[name] = beta1 * m[name] + (1.0 - beta1) * grad
            v[name] = beta2 * v[name] + (1.0 - beta2) * (grad * grad)
            mh = m[name] / (1.0 - beta1**epoch)
            vh = v[name] / (1.0 - beta2**epoch)
            params[name] -= lr * mh / (np.sqrt(vh) + eps)
        mb2 = beta1 * mb2 + (1.0 - beta1) * gb2
        vb2 = beta2 * vb2 + (1.0 - beta2) * (gb2 * gb2)
        b2 -= lr * (mb2 / (1.0 - beta1**epoch)) / (math.sqrt(vb2 / (1.0 - beta2**epoch)) + eps)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": float(b2), "losses": losses}


def write_model(path: Path, model: dict[str, np.ndarray | float | list[float]]) -> None:
    w1 = np.asarray(model["w1"], dtype=np.float64)
    b1 = np.asarray(model["b1"], dtype=np.float64)
    w2 = np.asarray(model["w2"], dtype=np.float64)
    b2 = float(model["b2"])
    values: list[str] = [str(w1.shape[0])]
    values.extend(f"{value:.12g}" for value in w1.reshape(-1))
    values.extend(f"{value:.12g}" for value in b1)
    values.extend(f"{value:.12g}" for value in w2)
    values.append(f"{b2:.12g}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised MLP online candidate scorer from actual anchor search costs.")
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "results" / "online_mlp_train_eval_insert80" / "query_split" / "train" / "manifest.json")
    parser.add_argument("--initial-nsg", type=Path, default=ROOT / "results" / "sift_stress_drift_dynamic_nsg" / "initial.nsg")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "supervised_online_mlp_insert80")
    parser.add_argument("--candidate-count", type=int, default=512)
    parser.add_argument("--inserted-fraction", type=float, default=0.80)
    parser.add_argument("--search-l", type=int, default=280)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--label-query-limit", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-objective", choices=["distance", "recall", "utility"], default="distance")
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    initial = read_fbin(Path(paths["initial_fbin"]))
    inserts = read_fbin(Path(paths["insert_fbin"]))
    all_vectors = np.vstack([initial, inserts]).astype(np.float32, copy=False)
    all_knn = read_ivecs(Path(paths["all_nsg_knn_graph"]))
    deletes = read_u32_list(Path(paths["delete_u32"]))
    alive = np.ones(all_vectors.shape[0], dtype=bool)
    alive[deletes.astype(np.int64)] = False
    static_anchors = read_anchor_file(Path(manifest["anchor_files"]["static_density"]))
    candidates = sample_candidates(
        initial_rows=initial.shape[0],
        total_rows=all_vectors.shape[0],
        alive=alive,
        static_anchors=static_anchors,
        candidate_count=args.candidate_count,
        inserted_fraction=args.inserted_fraction,
        seed=args.seed,
    )
    candidates_txt = args.out_dir / "candidate_anchors.txt"
    write_anchor_file(candidates_txt, candidates)
    features = compute_features(all_vectors, all_knn, alive, candidates, static_anchors, initial.shape[0])
    costs_csv = run_anchor_costs(
        manifest=manifest,
        initial_nsg=args.initial_nsg,
        runner=args.runner,
        anchors_txt=candidates_txt,
        out_dir=args.out_dir,
        search_l=args.search_l,
        topk=args.topk,
        query_limit=args.label_query_limit,
        timeout=args.timeout,
    )
    labels, label_summary = read_anchor_labels(costs_csv, candidates, args.label_objective)
    model = train_mlp(features, labels, args.hidden, args.epochs, args.lr, args.weight_decay, args.seed + 1)
    model_path = args.out_dir / "supervised_online_mlp.txt"
    write_model(model_path, model)
    pred = np.maximum(features @ np.asarray(model["w1"]).T + np.asarray(model["b1"]).reshape(1, -1), 0.0) @ np.asarray(model["w2"]) + float(model["b2"])
    corr = float(np.corrcoef(pred, labels)[0, 1]) if np.std(pred) > 1e-12 and np.std(labels) > 1e-12 else 0.0
    meta = {
        "train_manifest": str(args.train_manifest.resolve()),
        "model": str(model_path.resolve()),
        "candidate_count": int(candidates.size),
        "feature_names": ["coverage_far", "density", "hubness", "inserted_bonus"],
        "search_l": args.search_l,
        "topk": args.topk,
        "label_query_limit": args.label_query_limit,
        "hidden": args.hidden,
        "label_objective": args.label_objective,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "label_summary": label_summary,
        "train_mse_final": float(model["losses"][-1]),
        "train_mse_initial": float(model["losses"][0]),
        "prediction_label_corr": corr,
    }
    (args.out_dir / "supervised_online_mlp_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savetxt(args.out_dir / "candidate_features_labels.csv", np.column_stack([candidates, features, labels, pred]), delimiter=",", header="anchor,coverage_far,density,hubness,inserted_bonus,label,prediction", comments="")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
