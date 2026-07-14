from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CONFIGS: dict[str, str] = {
    "default": "0.45,0.20,0.30,0.05",
    "coverage_heavy": "0.60,0.15,0.20,0.05",
    "hub_heavy": "0.30,0.15,0.50,0.05",
    "density_heavy": "0.35,0.40,0.20,0.05",
    "balanced": "0.35,0.25,0.35,0.05",
    "fresh_heavy": "0.35,0.15,0.25,0.25",
    "coverage_hub": "0.50,0.10,0.35,0.05",
    "query_fresh_proxy": "0.25,0.20,0.35,0.20",
}


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
    parser = argparse.ArgumentParser(description="Sweep online candidate scorer weights for online_hub_density.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_insert80_queries" / "manifest.json")
    parser.add_argument("--initial-nsg", type=Path, default=ROOT / "results" / "sift_stress_drift_dynamic_nsg" / "initial.nsg")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "online_scorer_weight_sweep_insert80")
    parser.add_argument("--search-l", default="240,280,320")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--maintain-sample", type=int, default=512)
    parser.add_argument("--maintain-changes", type=int, default=4)
    parser.add_argument("--maintain-batch", type=int, default=1000)
    parser.add_argument("--retire-weights", default="0.55,0.25,0.20")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    args = parser.parse_args()

    selected = [item.strip() for item in args.configs.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
        "configs": {name: CONFIGS[name] for name in selected},
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for name in selected:
        weights = CONFIGS[name]
        run_dir = args.out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
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
                "--online-score-weights",
                weights,
                "--online-retire-weights",
                args.retire_weights,
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
            row["online_score_weights"] = weights
            row["online_retire_weights"] = args.retire_weights
            all_rows.append(row)

    aggregate = args.out_dir / "aggregate.csv"
    write_csv(aggregate, all_rows)
    print(json.dumps({"aggregate": str(aggregate)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
