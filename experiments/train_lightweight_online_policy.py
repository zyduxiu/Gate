from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_feature_label_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                [
                    float(row["coverage_far"]),
                    float(row["density"]),
                    float(row["hubness"]),
                    float(row["inserted_bonus"]),
                ]
            )
            labels.append(float(row["label"]))
    if not rows:
        raise ValueError(f"empty feature-label file: {path}")
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Project a vector to the non-negative unit simplex."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    order = np.sort(values)[::-1]
    cssv = np.cumsum(order)
    rho_candidates = order * np.arange(1, values.size + 1) > (cssv - 1.0)
    if not np.any(rho_candidates):
        return np.ones_like(values) / float(values.size)
    rho = int(np.flatnonzero(rho_candidates)[-1])
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    projected = np.maximum(values - theta, 0.0)
    total = float(np.sum(projected))
    if total <= 1e-12:
        return np.ones_like(values) / float(values.size)
    return projected / total


def fit_simplex_linear(
    x: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.0, 1.0, size=x.shape[1])
    w = project_simplex(w)
    losses: list[float] = []
    n = float(x.shape[0])
    for _ in range(epochs):
        pred = x @ w
        err = pred - y
        grad = (2.0 / n) * (x.T @ err) + 2.0 * l2 * w
        w = project_simplex(w - lr * grad)
        losses.append(float(np.mean(err * err)))
    return w, losses


def fit_unconstrained_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    xtx = x.T @ x
    reg = np.eye(x.shape[1], dtype=np.float64) * l2
    return np.linalg.solve(xtx + reg, x.T @ y)


def score_summary(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, float]:
    pred = x @ w
    corr = 0.0
    if float(np.std(pred)) > 1e-12 and float(np.std(y)) > 1e-12:
        corr = float(np.corrcoef(pred, y)[0, 1])
    return {
        "mse": float(np.mean((pred - y) ** 2)),
        "corr": corr,
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn lightweight online E-maintenance scorer weights from supervised anchor labels.")
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "results" / "supervised_online_mlp_insert80_distance" / "candidate_features_labels.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "lightweight_online_policy_insert80")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()

    x, y = read_feature_label_csv(args.features)
    simplex_w, losses = fit_simplex_linear(x, y, lr=args.lr, epochs=args.epochs, l2=args.l2, seed=args.seed)
    ridge_w = fit_unconstrained_ridge(x, y, l2=args.l2)
    ridge_positive = project_simplex(np.maximum(ridge_w, 0.0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "policy": "simplex_projected_gradient",
            "coverage": simplex_w[0],
            "density": simplex_w[1],
            "hubness": simplex_w[2],
            "inserted": simplex_w[3],
            **score_summary(x, y, simplex_w),
        },
        {
            "policy": "ridge_positive_simplex",
            "coverage": ridge_positive[0],
            "density": ridge_positive[1],
            "hubness": ridge_positive[2],
            "inserted": ridge_positive[3],
            **score_summary(x, y, ridge_positive),
        },
    ]
    out_csv = args.out_dir / "learned_policy_weights.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda row: float(row["mse"]))
    weights = [float(best["coverage"]), float(best["density"]), float(best["hubness"]), float(best["inserted"])]
    weights_txt = ",".join(f"{value:.6f}" for value in weights)
    payload = {
        "features": str(args.features.resolve()),
        "selected_policy": best["policy"],
        "online_score_weights": weights_txt,
        "feature_names": ["coverage_far", "density", "hubness", "inserted_bonus"],
        "selected_metrics": best,
        "all_policies": rows,
        "train_loss_initial": losses[0],
        "train_loss_final": losses[-1],
        "note": "This learns the lightweight parameters of the online entry-set E maintenance scorer, not a query-time selector.",
    }
    (args.out_dir / "learned_policy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
