from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from sweep_online_mlp_scorer import CONFIGS, MlpConfig, write_model


@dataclass(frozen=True)
class RunSpec:
    alias: str
    method: str
    mlp_model: Path | None = None
    score_weights: str = ""


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def random_mlp_configs(count: int, seed: int) -> dict[str, MlpConfig]:
    import random

    rng = random.Random(seed)
    configs: dict[str, MlpConfig] = {}
    base_rows = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
    base_bias = [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    for idx in range(count):
        # Random search around interpretable feature gates.  This is validation-selected,
        # not gradient-trained, so train/eval separation is essential.
        w1 = [row[:] for row in base_rows]
        b1 = [b + rng.uniform(-0.15, 0.15) for b in base_bias]
        raw = [rng.uniform(0.02, 0.45) for _ in w1]
        total = sum(raw)
        w2 = [x / total for x in raw]
        configs[f"mlp_random_{idx:02d}"] = MlpConfig(name=f"mlp_random_{idx:02d}", w1=w1, b1=b1, w2=w2)
    return configs


def run_nsg(
    manifest: Path,
    initial_nsg: Path,
    runner: Path,
    out_dir: Path,
    spec: RunSpec,
    search_l: str,
    topk: int,
    entries: int,
    maintain_sample: int,
    maintain_changes: int,
    maintain_batch: int,
    retire_weights: str,
    timeout: int,
) -> Path:
    command = [
        sys.executable,
        str((ROOT / "experiments" / "run_nsg_dynamic_entry.py").resolve()),
        "--manifest",
        str(manifest.resolve()),
        "--initial-nsg",
        str(initial_nsg.resolve()),
        "--runner",
        str(runner.resolve()),
        "--out-dir",
        str(out_dir.resolve()),
        "--methods",
        spec.method,
        "--search-l",
        search_l,
        "--entries",
        str(entries),
        "--topk",
        str(topk),
        "--maintain-sample",
        str(maintain_sample),
        "--maintain-changes",
        str(maintain_changes),
        "--maintain-batch",
        str(maintain_batch),
        "--online-retire-weights",
        retire_weights,
        "--timeout",
        str(timeout),
    ]
    if spec.score_weights:
        command.extend(["--online-score-weights", spec.score_weights])
    if spec.mlp_model is not None:
        command.extend(["--online-mlp-model", str(spec.mlp_model.resolve())])
    run(command, timeout=timeout)

    summary = out_dir / "summary.csv"
    run(
        [
            sys.executable,
            str((ROOT / "experiments" / "summarize_nsg_entry_results.py").resolve()),
            "--dir",
            str(out_dir.resolve()),
            "--manifest",
            str(manifest.resolve()),
            "--out",
            str(summary.resolve()),
        ],
        timeout=600,
    )
    return summary


def select_best(train_rows: list[dict[str, str]], select_l: int, recall_slack: float, objective: str) -> dict[str, str]:
    candidates = [row for row in train_rows if int(row["search_l"]) == select_l]
    if not candidates:
        raise ValueError(f"no train rows for L={select_l}")
    best_recall = max(float(row["recall_at_k"]) for row in candidates)
    eligible = [row for row in candidates if float(row["recall_at_k"]) >= best_recall - recall_slack]
    if objective == "qps":
        return max(eligible, key=lambda row: (float(row["qps"]), -float(row["p99_latency_us"])))
    if objective == "distance":
        return min(eligible, key=lambda row: (float(row["avg_distance_computations"]), float(row["p99_latency_us"])))
    if objective == "p99":
        return min(eligible, key=lambda row: (float(row["p99_latency_us"]), float(row["avg_distance_computations"])))
    if objective == "composite":
        def normalized(values: list[float], value: float) -> float:
            lo = min(values)
            hi = max(values)
            if hi - lo < 1e-12:
                return 0.0
            return (value - lo) / (hi - lo)

        dists = [float(row["avg_distance_computations"]) for row in eligible]
        p99s = [float(row["p99_latency_us"]) for row in eligible]
        qps_values = [float(row["qps"]) for row in eligible]
        return min(
            eligible,
            key=lambda row: (
                0.55 * normalized(dists, float(row["avg_distance_computations"]))
                + 0.30 * normalized(p99s, float(row["p99_latency_us"]))
                + 0.15 * (1.0 - normalized(qps_values, float(row["qps"]))),
                float(row["avg_distance_computations"]),
            ),
        )
    raise ValueError(f"unknown selection objective: {objective}")


def aggregate_repeats(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    keys = sorted({(row["alias"], int(row["search_l"])) for row in rows}, key=lambda item: (item[1], item[0]))
    aggregate: list[dict[str, object]] = []
    for alias, search_l in keys:
        group = [row for row in rows if row["alias"] == alias and int(row["search_l"]) == search_l]

        def vals(column: str) -> list[float]:
            return [float(row[column]) for row in group]

        aggregate.append(
            {
                "alias": alias,
                "search_l": search_l,
                "runs": len(group),
                "recall_at_k_mean": mean(vals("recall_at_k")),
                "recall_at_k_median": median(vals("recall_at_k")),
                "qps_mean": mean(vals("qps")),
                "qps_median": median(vals("qps")),
                "qps_std": pstdev(vals("qps")) if len(group) > 1 else 0.0,
                "p99_latency_us_median": median(vals("p99_latency_us")),
                "avg_distance_computations_median": median(vals("avg_distance_computations")),
                "maintenance_seconds_median": median(vals("maintenance_seconds")),
            }
        )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/eval split selection for lightweight online MLP entry-set scorer.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_insert80_queries" / "manifest.json")
    parser.add_argument("--initial-nsg", type=Path, default=ROOT / "results" / "sift_stress_drift_dynamic_nsg" / "initial.nsg")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "online_mlp_train_eval_insert80")
    parser.add_argument("--search-l", default="240,280,320")
    parser.add_argument("--select-l", type=int, default=280)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--maintain-sample", type=int, default=512)
    parser.add_argument("--maintain-changes", type=int, default=4)
    parser.add_argument("--maintain-batch", type=int, default=1000)
    parser.add_argument("--retire-weights", default="0.55,0.25,0.20")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--random-configs", type=int, default=8)
    parser.add_argument("--recall-slack", type=float, default=0.00005)
    parser.add_argument("--selection-objective", choices=["distance", "p99", "qps", "composite"], default="distance")
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.out_dir / "query_split"
    split_meta_path = split_dir / "split_meta.json"
    if not split_meta_path.exists():
        run(
            [
                sys.executable,
                str((ROOT / "experiments" / "split_query_manifest.py").resolve()),
                "--manifest",
                str(args.manifest.resolve()),
                "--out-dir",
                str(split_dir.resolve()),
                "--train-fraction",
                str(args.train_fraction),
                "--seed",
                str(args.split_seed),
            ],
            timeout=600,
        )
    split_meta = json.loads(split_meta_path.read_text(encoding="utf-8"))
    train_manifest = Path(split_meta["train_manifest"])
    eval_manifest = Path(split_meta["eval_manifest"])

    configs = dict(CONFIGS)
    configs.update(random_mlp_configs(args.random_configs, seed=args.split_seed + 17))
    models_dir = args.out_dir / "models"

    train_rows: list[dict[str, str]] = []
    for name, config in configs.items():
        model_path = models_dir / f"{name}.txt"
        write_model(model_path, config)
        summary = run_nsg(
            manifest=train_manifest,
            initial_nsg=args.initial_nsg,
            runner=args.runner,
            out_dir=args.out_dir / "train" / name,
            spec=RunSpec(alias=name, method="online_hub_density", mlp_model=model_path),
            search_l=args.search_l,
            topk=args.topk,
            entries=args.entries,
            maintain_sample=args.maintain_sample,
            maintain_changes=args.maintain_changes,
            maintain_batch=args.maintain_batch,
            retire_weights=args.retire_weights,
            timeout=args.timeout,
        )
        for row in read_csv(summary):
            row["alias"] = name
            row["online_mlp_model"] = str(model_path.resolve())
            train_rows.append(row)

    train_aggregate = args.out_dir / "train_aggregate.csv"
    write_csv(train_aggregate, train_rows)
    selected = select_best(
        train_rows,
        select_l=args.select_l,
        recall_slack=args.recall_slack,
        objective=args.selection_objective,
    )
    selected_name = selected["alias"]
    selected_model = models_dir / f"{selected_name}.txt"
    selected_payload = {
        "selected_config": selected_name,
        "selected_model": str(selected_model.resolve()),
        "selection_l": args.select_l,
        "selection_objective": args.selection_objective,
        "recall_slack": args.recall_slack,
        "selected_train_row": selected,
        "train_aggregate": str(train_aggregate.resolve()),
        "split_meta": split_meta,
    }
    (args.out_dir / "selected_config.json").write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")

    eval_specs = [
        RunSpec(alias="nsg_ep", method="nsg_ep"),
        RunSpec(alias="gate_initial_hub", method="gate_initial_hub"),
        RunSpec(alias="gate_refreshed_hub", method="gate_refreshed_hub"),
        RunSpec(alias="online_default_linear", method="online_hub_density"),
        RunSpec(alias=f"selected_{selected_name}", method="online_hub_density", mlp_model=selected_model),
    ]
    eval_rows: list[dict[str, str]] = []
    for repeat in range(1, args.eval_repeats + 1):
        for spec in eval_specs:
            summary = run_nsg(
                manifest=eval_manifest,
                initial_nsg=args.initial_nsg,
                runner=args.runner,
                out_dir=args.out_dir / "eval" / f"repeat{repeat}" / spec.alias,
                spec=spec,
                search_l=args.search_l,
                topk=args.topk,
                entries=args.entries,
                maintain_sample=args.maintain_sample,
                maintain_changes=args.maintain_changes,
                maintain_batch=args.maintain_batch,
                retire_weights=args.retire_weights,
                timeout=args.timeout,
            )
            for row in read_csv(summary):
                row["alias"] = spec.alias
                row["repeat"] = str(repeat)
                eval_rows.append(row)

    write_csv(args.out_dir / "eval_all_runs.csv", eval_rows)
    eval_aggregate = aggregate_repeats(eval_rows)
    write_csv(args.out_dir / "eval_aggregate.csv", eval_aggregate)
    print(json.dumps({"selected": selected_payload, "eval_aggregate": str((args.out_dir / "eval_aggregate.csv").resolve())}, indent=2))


if __name__ == "__main__":
    main()
