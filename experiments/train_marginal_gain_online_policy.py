from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from dynanchor.external import run_external_binary
from train_lightweight_online_policy import fit_simplex_linear, fit_unconstrained_ridge, project_simplex, score_summary


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


def read_anchor_file(path: Path) -> list[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_anchor_file(path: Path, anchors: list[int] | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(x)) for x in anchors) + "\n", encoding="utf-8")


def read_feature_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "anchor": float(row["anchor"]),
                    "coverage_far": float(row["coverage_far"]),
                    "density": float(row["density"]),
                    "hubness": float(row["hubness"]),
                    "inserted_bonus": float(row["inserted_bonus"]),
                }
            )
    if not rows:
        raise ValueError(f"empty feature file: {path}")
    return rows


def l2_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * a, axis=1, keepdims=True) + np.sum(b * b, axis=1).reshape(1, -1) - 2.0 * a @ b.T


def choose_candidates(rows: list[dict[str, float]], existing: set[int], limit: int, seed: int) -> list[dict[str, float]]:
    pool = [row for row in rows if int(row["anchor"]) not in existing]
    if len(pool) <= limit:
        return pool
    rng = np.random.default_rng(seed)
    # Keep high prior-score points and random exploratory points.
    scored = sorted(
        pool,
        key=lambda row: -(
            0.25 * row["coverage_far"]
            + 0.15 * row["density"]
            + 0.45 * row["hubness"]
            + 0.15 * row["inserted_bonus"]
        ),
    )
    exploit = scored[: max(1, limit // 2)]
    exploit_ids = {int(row["anchor"]) for row in exploit}
    rest = [row for row in pool if int(row["anchor"]) not in exploit_ids]
    explore_take = max(0, limit - len(exploit))
    if rest and explore_take:
        idx = rng.choice(len(rest), size=min(explore_take, len(rest)), replace=False)
        exploit.extend(rest[int(i)] for i in idx)
    return exploit[:limit]


def anchor_query_loads(all_vectors: np.ndarray, queries: np.ndarray, anchors: list[int]) -> np.ndarray:
    anchor_vecs = all_vectors[np.asarray(anchors, dtype=np.int64)]
    dists = l2_matrix(queries, anchor_vecs)
    assignment = np.argmin(dists, axis=1)
    return np.bincount(assignment, minlength=len(anchors))


def least_load_anchor(all_vectors: np.ndarray, queries: np.ndarray, anchors: list[int]) -> int:
    counts = anchor_query_loads(all_vectors, queries, anchors)
    return int(anchors[int(np.argmin(counts))])


def nearest_anchor_to_candidate(all_vectors: np.ndarray, anchors: list[int], candidate: int) -> int:
    anchor_vecs = all_vectors[np.asarray(anchors, dtype=np.int64)]
    cand = all_vectors[[candidate]]
    dists = l2_matrix(cand, anchor_vecs).reshape(-1)
    return int(anchors[int(np.argmin(dists))])


def retire_pool_for_candidate(
    all_vectors: np.ndarray,
    queries: np.ndarray,
    anchors: list[int],
    candidate: int,
    per_rule: int,
) -> list[int]:
    """Small retire set that approximates max_r cost(E)-cost(E+c-r)."""
    anchor_vecs = all_vectors[np.asarray(anchors, dtype=np.int64)]
    cand = all_vectors[[candidate]]
    cand_dists = l2_matrix(cand, anchor_vecs).reshape(-1)
    loads = anchor_query_loads(all_vectors, queries, anchors)
    per_rule = max(1, min(per_rule, len(anchors)))

    indices: list[int] = []
    # Redundant with the candidate: if candidate covers this place, retire nearby old entry.
    indices.extend(int(i) for i in np.argsort(cand_dists)[:per_rule])
    # Low utility under the current query distribution.
    indices.extend(int(i) for i in np.argsort(loads)[:per_rule])
    # Very isolated anchors can also become stale under drift/deletion.
    indices.extend(int(i) for i in np.argsort(-cand_dists)[:per_rule])

    seen: set[int] = set()
    pool: list[int] = []
    for idx in indices:
        anchor = int(anchors[idx])
        if anchor not in seen:
            seen.add(anchor)
            pool.append(anchor)
    return pool


def query_coverage_features(
    all_vectors: np.ndarray,
    queries: np.ndarray,
    anchors: list[int],
    candidates: list[int],
    query_limit: int,
) -> dict[int, tuple[float, float]]:
    rows = min(query_limit, queries.shape[0]) if query_limit > 0 else queries.shape[0]
    if rows <= 0 or not candidates:
        return {candidate: (0.0, 0.0) for candidate in candidates}
    q = queries[:rows]
    anchor_vecs = all_vectors[np.asarray(anchors, dtype=np.int64)]
    base_best = np.min(l2_matrix(q, anchor_vecs), axis=1)
    values: dict[int, tuple[float, float]] = {}
    for candidate in candidates:
        cand_dist = l2_matrix(q, all_vectors[[candidate]]).reshape(-1)
        improvement = np.maximum(0.0, base_best - cand_dist)
        load = cand_dist < base_best
        values[candidate] = (float(np.mean(improvement)), float(np.mean(load)))
    return values


def normalize_columns(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for col in range(out.shape[1]):
        lo = float(np.min(out[:, col]))
        hi = float(np.max(out[:, col]))
        if hi - lo < 1e-12:
            out[:, col] = 0.0
        else:
            out[:, col] = (out[:, col] - lo) / (hi - lo)
    return out


def run_anchor_set(
    manifest: dict[str, Any],
    initial_nsg: Path,
    runner: Path,
    anchors: list[int],
    out_dir: Path,
    name: str,
    search_l: int,
    topk: int,
    entries: int,
    timeout: int,
) -> dict[str, Any]:
    anchor_path = out_dir / "anchors" / f"{name}.txt"
    out_json = out_dir / "runs" / f"{name}.json"
    write_anchor_file(anchor_path, anchors)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists():
        return json.loads(out_json.read_text(encoding="utf-8"))
    paths = manifest["paths"]
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
        str(anchor_path.resolve()),
        "--search-l",
        str(search_l),
        "--entries",
        str(entries),
        "--topk",
        str(topk),
        "--insert-degree",
        "32",
        "--max-degree",
        "64",
        "--out-json",
        str(out_json.resolve()),
    ]
    result = run_external_binary(command, timeout=timeout)
    log_path = out_dir / "runs" / f"{name}.log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps({"command": result.command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, indent=2),
        encoding="utf-8",
    )
    if not result.ok:
        raise RuntimeError(f"anchor set run failed for {name}; see {log_path}")
    return json.loads(out_json.read_text(encoding="utf-8"))


def normalize_gain(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn lightweight online E-maintenance parameters from marginal-gain labels.")
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "results" / "online_mlp_train_eval_insert80" / "query_split" / "train" / "manifest.json")
    parser.add_argument("--initial-nsg", type=Path, default=ROOT / "results" / "sift_stress_drift_dynamic_nsg" / "initial.nsg")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--features", type=Path, default=ROOT / "results" / "supervised_online_mlp_insert80_distance" / "candidate_features_labels.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "marginal_gain_policy_insert80")
    parser.add_argument("--candidate-limit", type=int, default=48)
    parser.add_argument("--search-l", type=int, default=280)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--retire-strategy", choices=["least_load", "nearest", "best_of_pool"], default="least_load")
    parser.add_argument("--retire-pool-per-rule", type=int, default=2)
    parser.add_argument("--query-feature-limit", type=int, default=512)
    parser.add_argument("--reuse-labels", type=Path, default=None)
    parser.add_argument("--recall-penalty", type=float, default=2000.0)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    initial = read_fbin(Path(paths["initial_fbin"]))
    inserts = read_fbin(Path(paths["insert_fbin"]))
    queries = read_fbin(Path(paths["query_fbin"]))
    all_vectors = np.vstack([initial, inserts]).astype(np.float32, copy=False)
    base_e = read_anchor_file(Path(manifest["anchor_files"]["static_density"]))
    existing = set(base_e)
    feature_rows = read_feature_rows(args.features)
    candidates = choose_candidates(feature_rows, existing, args.candidate_limit, args.seed)
    candidate_ids = [int(row["anchor"]) for row in candidates]
    query_features = query_coverage_features(
        all_vectors=all_vectors,
        queries=queries,
        anchors=base_e,
        candidates=candidate_ids,
        query_limit=args.query_feature_limit,
    )

    if args.reuse_labels is None:
        base_payload = run_anchor_set(
            manifest=manifest,
            initial_nsg=args.initial_nsg,
            runner=args.runner,
            anchors=base_e,
            out_dir=args.out_dir,
            name="base_E",
            search_l=args.search_l,
            topk=args.topk,
            entries=args.entries,
            timeout=args.timeout,
        )
        base_dist = float(base_payload["avg_distance_computations"])
        base_recall = float(base_payload["recall_at_k"])
    else:
        base_dist = 0.0
        base_recall = 0.0
    if args.retire_strategy == "least_load":
        default_retired = least_load_anchor(all_vectors, queries, base_e)
    else:
        default_retired = -1

    rows: list[dict[str, Any]] = []
    x_values: list[list[float]] = []
    raw_gains: list[float] = []
    if args.reuse_labels is not None:
        with args.reuse_labels.open(newline="", encoding="utf-8") as handle:
            label_rows = list(csv.DictReader(handle))
        row_by_candidate = {int(float(row["anchor"])): row for row in candidates}
        for label_row in label_rows:
            cand = int(label_row["candidate"])
            if cand not in row_by_candidate:
                continue
            row = row_by_candidate[cand]
            q_gain, q_load = query_features.get(cand, (0.0, 0.0))
            feature = [
                row["coverage_far"],
                row["density"],
                row["hubness"],
                row["inserted_bonus"],
                q_gain,
                q_load,
            ]
            gain = float(label_row["raw_marginal_gain"])
            x_values.append(feature)
            raw_gains.append(gain)
            merged = dict(label_row)
            merged.update(
                {
                    "coverage_far": row["coverage_far"],
                    "density": row["density"],
                    "hubness": row["hubness"],
                    "inserted_bonus": row["inserted_bonus"],
                    "query_gain": q_gain,
                    "query_load": q_load,
                }
            )
            rows.append(merged)
    for idx, row in enumerate([] if args.reuse_labels is not None else candidates):
        cand = int(row["anchor"])
        if args.retire_strategy == "nearest":
            retire_candidates = [nearest_anchor_to_candidate(all_vectors, base_e, cand)]
        elif args.retire_strategy == "best_of_pool":
            retire_candidates = retire_pool_for_candidate(
                all_vectors=all_vectors,
                queries=queries,
                anchors=base_e,
                candidate=cand,
                per_rule=args.retire_pool_per_rule,
            )
        else:
            retire_candidates = [default_retired]
        best_payload: dict[str, Any] | None = None
        best_gain = -float("inf")
        best_retired = -1
        for retire_idx, retired in enumerate(retire_candidates):
            after_e = [anchor for anchor in base_e if anchor != retired] + [cand]
            payload = run_anchor_set(
                manifest=manifest,
                initial_nsg=args.initial_nsg,
                runner=args.runner,
                anchors=after_e,
                out_dir=args.out_dir,
                name=f"cand_{idx:03d}_{cand}_retire_{retire_idx:02d}_{retired}",
                search_l=args.search_l,
                topk=args.topk,
                entries=args.entries,
                timeout=args.timeout,
            )
            after_dist_i = float(payload["avg_distance_computations"])
            after_recall_i = float(payload["recall_at_k"])
            # Positive means candidate improves E. Penalize recall drops in distance-computation units.
            gain_i = (base_dist - after_dist_i) + args.recall_penalty * (after_recall_i - base_recall)
            if gain_i > best_gain:
                best_gain = gain_i
                best_payload = payload
                best_retired = retired
        if best_payload is None:
            raise RuntimeError(f"empty retire candidate set for candidate {cand}")
        retired = best_retired
        after_dist = float(best_payload["avg_distance_computations"])
        after_recall = float(best_payload["recall_at_k"])
        gain = best_gain
        q_gain, q_load = query_features.get(cand, (0.0, 0.0))
        feature = [row["coverage_far"], row["density"], row["hubness"], row["inserted_bonus"], q_gain, q_load]
        x_values.append(feature)
        raw_gains.append(gain)
        rows.append(
            {
                "candidate": cand,
                "retired": retired,
                "retire_candidates_evaluated": len(retire_candidates),
                "coverage_far": row["coverage_far"],
                "density": row["density"],
                "hubness": row["hubness"],
                "inserted_bonus": row["inserted_bonus"],
                "query_gain": q_gain,
                "query_load": q_load,
                "base_recall": base_recall,
                "after_recall": after_recall,
                "base_distance": base_dist,
                "after_distance": after_dist,
                "raw_marginal_gain": gain,
            }
        )

    x = normalize_columns(np.asarray(x_values, dtype=np.float64))
    raw_gain = np.asarray(raw_gains, dtype=np.float64)
    y = normalize_gain(raw_gain)
    simplex_w, losses = fit_simplex_linear(x, y, lr=args.lr, epochs=args.epochs, l2=args.l2, seed=args.seed)
    ridge_w = fit_unconstrained_ridge(x, y, l2=args.l2)
    ridge_positive = project_simplex(np.maximum(ridge_w, 0.0))
    policies = [
        {
            "policy": "simplex_projected_gradient",
            "coverage": simplex_w[0],
            "density": simplex_w[1],
            "hubness": simplex_w[2],
            "inserted": simplex_w[3],
            "query_gain": simplex_w[4],
            "query_load": simplex_w[5],
            **score_summary(x, y, simplex_w),
        },
        {
            "policy": "ridge_positive_simplex",
            "coverage": ridge_positive[0],
            "density": ridge_positive[1],
            "hubness": ridge_positive[2],
            "inserted": ridge_positive[3],
            "query_gain": ridge_positive[4],
            "query_load": ridge_positive[5],
            **score_summary(x, y, ridge_positive),
        },
    ]
    best = min(policies, key=lambda item: float(item["mse"]))
    weights = [
        float(best["coverage"]),
        float(best["density"]),
        float(best["hubness"]),
        float(best["inserted"]),
        float(best["query_gain"]),
        float(best["query_load"]),
    ]
    weights_txt = ",".join(f"{value:.6f}" for value in weights)
    online_score_weights_txt = ",".join(f"{value:.6f}" for value in weights[:4])

    rows_path = args.out_dir / "marginal_gain_labels.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    policy_path = args.out_dir / "marginal_gain_policy.json"
    payload = {
        "train_manifest": str(args.train_manifest.resolve()),
        "features": str(args.features.resolve()),
        "base_recall": base_recall,
        "base_distance": base_dist,
        "candidate_limit": len(candidates),
        "retire_strategy": args.retire_strategy,
        "recall_penalty": args.recall_penalty,
        "query_feature_limit": args.query_feature_limit,
        "reuse_labels": str(args.reuse_labels.resolve()) if args.reuse_labels is not None else "",
        "selected_policy": best["policy"],
        "learned_six_feature_weights": weights_txt,
        "online_score_weights": online_score_weights_txt,
        "online_query_gain_weight": weights[4],
        "online_query_load_weight": weights[5],
        "feature_names": ["coverage_far", "density", "hubness", "inserted_bonus", "query_gain", "query_load"],
        "selected_metrics": best,
        "all_policies": policies,
        "raw_gain_min": float(np.min(raw_gain)),
        "raw_gain_mean": float(np.mean(raw_gain)),
        "raw_gain_max": float(np.max(raw_gain)),
        "positive_gain_count": int(np.sum(raw_gain > 0.0)),
        "labels_csv": str(rows_path.resolve()),
        "note": "Labels are marginal gain of replacing one anchor in E with a candidate: cost(E)-cost(E').",
    }
    policy_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
