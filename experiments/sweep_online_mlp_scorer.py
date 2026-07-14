from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MlpConfig:
    name: str
    w1: list[list[float]]
    b1: list[float]
    w2: list[float]
    b2: float = 0.0


def identity_linear(name: str, weights: tuple[float, float, float, float]) -> MlpConfig:
    return MlpConfig(
        name=name,
        w1=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        b1=[0.0, 0.0, 0.0, 0.0],
        w2=list(weights),
    )


CONFIGS: dict[str, MlpConfig] = {
    # Sanity check: same feature family as the strongest linear sweep, implemented through MLP inference.
    "mlp_linear_query_fresh": identity_linear("mlp_linear_query_fresh", (0.25, 0.20, 0.35, 0.20)),
    # Boost points that are simultaneously uncovered and dense/hub-like, instead of rewarding each term alone.
    "mlp_uncovered_dense_hub": MlpConfig(
        name="mlp_uncovered_dense_hub",
        w1=[
            [1.0, 0.0, 0.0, 0.0],  # far
            [0.0, 1.0, 0.0, 0.0],  # density
            [0.0, 0.0, 1.0, 0.0],  # hubness
            [0.0, 0.0, 0.0, 1.0],  # inserted
            [1.0, 1.0, 0.0, 0.0],  # uncovered and dense
            [1.0, 0.0, 1.0, 0.0],  # uncovered and hub-like
            [0.0, 1.0, 1.0, 0.0],  # dense and hub-like
        ],
        b1=[0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
        w2=[0.18, 0.14, 0.22, 0.08, 0.20, 0.14, 0.10],
    ),
    # More aggressive for inserted dense regions; useful when queries are strongly biased to new clusters.
    "mlp_insert_region_gate": MlpConfig(
        name="mlp_insert_region_gate",
        w1=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],  # far inserted point
            [0.0, 1.0, 0.0, 1.0],  # dense inserted point
            [0.0, 0.0, 1.0, 1.0],  # inserted hub
        ],
        b1=[0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
        w2=[0.16, 0.14, 0.20, 0.08, 0.18, 0.14, 0.10],
    ),
    # Conservative: keep hubness strong, only add new-region candidates when coverage is clearly stale.
    "mlp_conservative_hub": MlpConfig(
        name="mlp_conservative_hub",
        w1=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        b1=[0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
        w2=[0.20, 0.12, 0.34, 0.05, 0.18, 0.11],
    ),
}


def write_model(path: Path, config: MlpConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hidden = len(config.w1)
    if len(config.b1) != hidden or len(config.w2) != hidden:
        raise ValueError(f"invalid config: {config.name}")
    values: list[str] = [str(hidden)]
    for row in config.w1:
        if len(row) != 4:
            raise ValueError(f"invalid row width in config: {config.name}")
        values.extend(f"{value:.12g}" for value in row)
    values.extend(f"{value:.12g}" for value in config.b1)
    values.extend(f"{value:.12g}" for value in config.w2)
    values.append(f"{config.b2:.12g}")
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def run(command: list[str], timeout: int) -> None:
    print("RUN", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep lightweight MLP scorers for online_hub_density candidate maintenance.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_insert80_queries" / "manifest.json")
    parser.add_argument("--initial-nsg", type=Path, default=ROOT / "results" / "sift_stress_drift_dynamic_nsg" / "initial.nsg")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "online_mlp_scorer_sweep_insert80")
    parser.add_argument("--search-l", default="240,280,320")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--maintain-sample", type=int, default=512)
    parser.add_argument("--maintain-changes", type=int, default=4)
    parser.add_argument("--maintain-batch", type=int, default=1000)
    parser.add_argument("--retire-weights", default="0.55,0.25,0.20")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    selected = [item.strip() for item in args.configs.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.out_dir / "models"
    all_rows: list[dict[str, str]] = []

    meta = {
        "manifest": str(args.manifest.resolve()),
        "initial_nsg": str(args.initial_nsg.resolve()),
        "search_l": args.search_l,
        "topk": args.topk,
        "entries": args.entries,
        "maintain_sample": args.maintain_sample,
        "maintain_changes": args.maintain_changes,
        "maintain_batch": args.maintain_batch,
        "retire_weights": args.retire_weights,
        "configs": selected,
        "note": "Small ReLU MLP scorer over normalized [coverage_far, density, graph_hubness, inserted_bonus].",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for name in selected:
        config = CONFIGS[name]
        model_path = models_dir / f"{name}.txt"
        write_model(model_path, config)
        run_dir = args.out_dir / name
        run(
            [
                sys.executable,
                str((ROOT / "experiments" / "run_nsg_dynamic_entry.py").resolve()),
                "--manifest",
                str(args.manifest.resolve()),
                "--initial-nsg",
                str(args.initial_nsg.resolve()),
                "--runner",
                str(args.runner.resolve()),
                "--out-dir",
                str(run_dir.resolve()),
                "--methods",
                "online_hub_density",
                "--search-l",
                args.search_l,
                "--entries",
                str(args.entries),
                "--topk",
                str(args.topk),
                "--maintain-sample",
                str(args.maintain_sample),
                "--maintain-changes",
                str(args.maintain_changes),
                "--maintain-batch",
                str(args.maintain_batch),
                "--online-retire-weights",
                args.retire_weights,
                "--online-mlp-model",
                str(model_path.resolve()),
                "--timeout",
                str(args.timeout),
            ],
            timeout=args.timeout,
        )
        summary = run_dir / "summary.csv"
        run(
            [
                sys.executable,
                str((ROOT / "experiments" / "summarize_nsg_entry_results.py").resolve()),
                "--dir",
                str(run_dir.resolve()),
                "--manifest",
                str(args.manifest.resolve()),
                "--out",
                str(summary.resolve()),
            ],
            timeout=600,
        )
        for row in read_csv(summary):
            row["scorer_config"] = name
            row["online_mlp_model"] = str(model_path.resolve())
            row["online_retire_weights"] = args.retire_weights
            all_rows.append(row)

    aggregate = args.out_dir / "aggregate.csv"
    write_csv(aggregate, all_rows)
    print(json.dumps({"aggregate": str(aggregate)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
