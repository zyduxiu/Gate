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

from train_lightweight_online_policy import fit_simplex_linear, fit_unconstrained_ridge, project_simplex, score_summary
from train_marginal_gain_online_policy import (
    l2_matrix,
    normalize_columns,
    normalize_gain,
    query_coverage_features,
    read_anchor_file,
    read_fbin,
    read_feature_rows,
    retire_pool_for_candidate,
    run_anchor_set,
)


def normalize_rows(rows: list[list[float]]) -> np.ndarray:
    return normalize_columns(np.asarray(rows, dtype=np.float64))


def score_feature_rows(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.zeros(0, dtype=np.float64)
    return features @ weights


def parse_simplex_weights(value: str, expected: int) -> np.ndarray:
    weights = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if weights.size != expected:
        raise ValueError(f"expected {expected} comma-separated weights, got {weights.size}")
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return np.ones(expected, dtype=np.float64) / float(expected)
    return weights / total


def fit_prior_regularized_simplex(
    x: np.ndarray,
    y: np.ndarray,
    prior: np.ndarray,
    prior_lambda: float,
    lr: float,
    epochs: int,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    w = prior.copy()
    if not np.any(w):
        w = rng.uniform(0.0, 1.0, size=x.shape[1])
        w = project_simplex(w)
    losses: list[float] = []
    n = float(x.shape[0])
    for _ in range(epochs):
        pred = x @ w
        err = pred - y
        grad = (2.0 / n) * (x.T @ err) + 2.0 * l2 * w
        if prior_lambda > 0.0:
            grad += 2.0 * prior_lambda * (w - prior)
        w = project_simplex(w - lr * grad)
        loss = float(np.mean(err * err))
        if prior_lambda > 0.0:
            loss += float(prior_lambda * np.sum((w - prior) ** 2))
        losses.append(loss)
    return w, losses


def candidate_feature(
    row: dict[str, float],
    q_features: dict[int, tuple[float, float]],
) -> list[float]:
    cand = int(row["anchor"])
    q_gain, q_load = q_features.get(cand, (0.0, 0.0))
    return [
        float(row["coverage_far"]),
        float(row["density"]),
        float(row["hubness"]),
        float(row["inserted_bonus"]),
        q_gain,
        q_load,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train online entry-set policy from rollout-aligned marginal-gain labels.")
    parser.add_argument("--train-manifest", type=Path, default=ROOT / "results" / "online_mlp_train_eval_insert80" / "query_split" / "train" / "manifest.json")
    parser.add_argument("--initial-nsg", type=Path, default=ROOT / "results" / "sift_stress_drift_dynamic_nsg" / "initial.nsg")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--features", type=Path, default=ROOT / "results" / "supervised_online_mlp_insert80_distance" / "candidate_features_labels.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "rollout_marginal_policy_insert80")
    parser.add_argument("--maintain-batch", type=int, default=1000)
    parser.add_argument("--round-limit", type=int, default=12)
    parser.add_argument("--candidates-per-round", type=int, default=4)
    parser.add_argument("--low-candidates-per-round", type=int, default=0)
    parser.add_argument("--random-candidates-per-round", type=int, default=1)
    parser.add_argument("--retire-pool-per-rule", type=int, default=1)
    parser.add_argument("--query-feature-limit", type=int, default=128)
    parser.add_argument("--search-l", type=int, default=280)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--recall-penalty", type=float, default=2000.0)
    parser.add_argument("--rollout-weights", default="0.000000,0.326441,0.580062,0.093497,0.300000,0.300000")
    parser.add_argument("--prior-weights", default="0.000000,0.326441,0.580062,0.093497,0.300000,0.300000")
    parser.add_argument("--prior-lambda", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=7000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    initial = read_fbin(Path(paths["initial_fbin"]))
    inserts = read_fbin(Path(paths["insert_fbin"]))
    queries = read_fbin(Path(paths["query_fbin"]))
    all_vectors = np.vstack([initial, inserts]).astype(np.float32, copy=False)
    current_e = read_anchor_file(Path(manifest["anchor_files"]["static_density"]))
    feature_rows = read_feature_rows(args.features)
    inserted_rows = [row for row in feature_rows if int(row["anchor"]) >= initial.shape[0]]
    by_batch: dict[int, list[dict[str, float]]] = {}
    for row in inserted_rows:
        anchor = int(row["anchor"])
        batch = (anchor - initial.shape[0]) // args.maintain_batch
        by_batch.setdefault(batch, []).append(row)

    rollout_weights = parse_simplex_weights(args.rollout_weights, 6)
    prior_weights = parse_simplex_weights(args.prior_weights, 6)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    x_values: list[list[float]] = []
    raw_gains: list[float] = []
    evaluated_rounds = 0

    for batch in sorted(by_batch):
        cand_rows = by_batch[batch]
        if not cand_rows:
            continue
        cand_ids = [int(row["anchor"]) for row in cand_rows]
        q_features = query_coverage_features(all_vectors, queries, current_e, cand_ids, args.query_feature_limit)
        raw_features = [candidate_feature(row, q_features) for row in cand_rows]
        norm_features = normalize_rows(raw_features)
        scores = score_feature_rows(norm_features, rollout_weights)
        order = list(np.argsort(-scores))
        selected_indices: list[int] = []
        selected_indices.extend(order[: args.candidates_per_round])
        if args.low_candidates_per_round > 0:
            selected_indices.extend(order[-args.low_candidates_per_round :])
        seen_indices = set(selected_indices)
        remaining = [idx for idx in order if idx not in seen_indices]
        take = min(args.random_candidates_per_round, len(remaining))
        if take:
            selected_indices.extend(int(i) for i in rng.choice(remaining, size=take, replace=False))
        deduped_indices: list[int] = []
        seen_indices.clear()
        for idx in selected_indices:
            if idx not in seen_indices:
                seen_indices.add(idx)
                deduped_indices.append(idx)
        selected = [cand_rows[i] for i in deduped_indices]

        base_payload = run_anchor_set(
            manifest=manifest,
            initial_nsg=args.initial_nsg,
            runner=args.runner,
            anchors=current_e,
            out_dir=args.out_dir,
            name=f"round_{batch:03d}_base",
            search_l=args.search_l,
            topk=args.topk,
            entries=args.entries,
            timeout=args.timeout,
        )
        base_dist = float(base_payload["avg_distance_computations"])
        base_recall = float(base_payload["recall_at_k"])

        best_round_gain = -float("inf")
        best_round_candidate = -1
        best_round_retired = -1

        for local_idx, row in enumerate(selected):
            cand = int(row["anchor"])
            retire_candidates = retire_pool_for_candidate(
                all_vectors=all_vectors,
                queries=queries,
                anchors=current_e,
                candidate=cand,
                per_rule=args.retire_pool_per_rule,
            )
            best_gain = -float("inf")
            best_retired = -1
            best_after_dist = 0.0
            best_after_recall = 0.0
            for retire_idx, retired in enumerate(retire_candidates):
                after_e = [anchor for anchor in current_e if anchor != retired] + [cand]
                payload = run_anchor_set(
                    manifest=manifest,
                    initial_nsg=args.initial_nsg,
                    runner=args.runner,
                    anchors=after_e,
                    out_dir=args.out_dir,
                    name=f"round_{batch:03d}_cand_{local_idx:02d}_{cand}_retire_{retire_idx:02d}_{retired}",
                    search_l=args.search_l,
                    topk=args.topk,
                    entries=args.entries,
                    timeout=args.timeout,
                )
                after_dist = float(payload["avg_distance_computations"])
                after_recall = float(payload["recall_at_k"])
                gain = (base_dist - after_dist) + args.recall_penalty * (after_recall - base_recall)
                if gain > best_gain:
                    best_gain = gain
                    best_retired = retired
                    best_after_dist = after_dist
                    best_after_recall = after_recall

            feature = candidate_feature(row, q_features)
            x_values.append(feature)
            raw_gains.append(best_gain)
            rows.append(
                {
                    "round": batch,
                    "candidate": cand,
                    "retired": best_retired,
                    "retire_candidates_evaluated": len(retire_candidates),
                    "coverage_far": feature[0],
                    "density": feature[1],
                    "hubness": feature[2],
                    "inserted_bonus": feature[3],
                    "query_gain": feature[4],
                    "query_load": feature[5],
                    "base_recall": base_recall,
                    "after_recall": best_after_recall,
                    "base_distance": base_dist,
                    "after_distance": best_after_dist,
                    "raw_marginal_gain": best_gain,
                }
            )
            if best_gain > best_round_gain:
                best_round_gain = best_gain
                best_round_candidate = cand
                best_round_retired = best_retired

        if best_round_candidate >= 0 and best_round_gain > 0.0:
            current_e = [anchor for anchor in current_e if anchor != best_round_retired] + [best_round_candidate]

        evaluated_rounds += 1
        if evaluated_rounds >= args.round_limit:
            break

    if not rows:
        raise RuntimeError("no rollout labels were generated")

    x_raw = np.asarray(x_values, dtype=np.float64)
    x = normalize_columns(x_raw)
    raw_gain = np.asarray(raw_gains, dtype=np.float64)
    y = normalize_gain(raw_gain)
    if args.prior_lambda > 0.0:
        simplex_w, losses = fit_prior_regularized_simplex(
            x,
            y,
            prior=prior_weights,
            prior_lambda=args.prior_lambda,
            lr=args.lr,
            epochs=args.epochs,
            l2=args.l2,
            seed=args.seed,
        )
    else:
        simplex_w, losses = fit_simplex_linear(x, y, lr=args.lr, epochs=args.epochs, l2=args.l2, seed=args.seed)
    ridge_w = fit_unconstrained_ridge(x, y, l2=args.l2)
    ridge_positive = project_simplex(np.maximum(ridge_w, 0.0))
    feature_names = ["coverage_far", "density", "hubness", "inserted_bonus", "query_gain", "query_load"]

    def policy_row(name: str, w: np.ndarray) -> dict[str, float | str]:
        return {
            "policy": name,
            "coverage": float(w[0]),
            "density": float(w[1]),
            "hubness": float(w[2]),
            "inserted": float(w[3]),
            "query_gain": float(w[4]),
            "query_load": float(w[5]),
            **score_summary(x, y, w),
        }

    simplex_name = "prior_regularized_simplex" if args.prior_lambda > 0.0 else "simplex_projected_gradient"
    policies = [
        policy_row(simplex_name, simplex_w),
        policy_row("ridge_positive_simplex", ridge_positive),
    ]
    best = policies[0] if args.prior_lambda > 0.0 else min(policies, key=lambda item: float(item["mse"]))
    weights = [float(best["coverage"]), float(best["density"]), float(best["hubness"]), float(best["inserted"]), float(best["query_gain"]), float(best["query_load"])]
    weights_txt = ",".join(f"{value:.6f}" for value in weights)
    online_score_weights_txt = ",".join(f"{value:.6f}" for value in weights[:4])

    labels_path = args.out_dir / "rollout_marginal_labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "train_manifest": str(args.train_manifest.resolve()),
        "features": str(args.features.resolve()),
        "round_limit": args.round_limit,
        "evaluated_rounds": evaluated_rounds,
        "labels": len(rows),
        "candidates_per_round": args.candidates_per_round,
        "low_candidates_per_round": args.low_candidates_per_round,
        "random_candidates_per_round": args.random_candidates_per_round,
        "rollout_seed_weights": args.rollout_weights,
        "prior_weights": ",".join(f"{value:.6f}" for value in prior_weights),
        "prior_lambda": args.prior_lambda,
        "selected_policy": best["policy"],
        "learned_six_feature_weights": weights_txt,
        "online_score_weights": online_score_weights_txt,
        "online_query_gain_weight": weights[4],
        "online_query_load_weight": weights[5],
        "feature_names": feature_names,
        "selected_metrics": best,
        "all_policies": policies,
        "raw_gain_min": float(np.min(raw_gain)),
        "raw_gain_mean": float(np.mean(raw_gain)),
        "raw_gain_max": float(np.max(raw_gain)),
        "positive_gain_count": int(np.sum(raw_gain > 0.0)),
        "labels_csv": str(labels_path.resolve()),
        "note": "Labels are measured along an online rollout: cost(E_t)-cost(E_t+c-r), not only from the initial E.",
        "train_loss_initial": losses[0],
        "train_loss_final": losses[-1],
    }
    (args.out_dir / "rollout_marginal_policy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
