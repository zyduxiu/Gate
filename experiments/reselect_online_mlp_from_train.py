from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from train_eval_online_mlp_scorer import select_best


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reselect an online MLP scorer from an existing train aggregate.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results" / "online_mlp_train_eval_insert80")
    parser.add_argument("--select-l", type=int, default=280)
    parser.add_argument("--recall-slack", type=float, default=0.00005)
    parser.add_argument("--selection-objective", choices=["distance", "p99", "qps", "composite"], default="distance")
    args = parser.parse_args()

    train_path = args.run_dir / "train_aggregate.csv"
    rows = read_csv(train_path)
    selected = select_best(
        rows,
        select_l=args.select_l,
        recall_slack=args.recall_slack,
        objective=args.selection_objective,
    )
    model_path = args.run_dir / "models" / f"{selected['alias']}.txt"
    payload = {
        "selected_config": selected["alias"],
        "selected_model": str(model_path.resolve()),
        "selection_l": args.select_l,
        "selection_objective": args.selection_objective,
        "recall_slack": args.recall_slack,
        "selected_train_row": selected,
        "train_aggregate": str(train_path.resolve()),
    }
    out = args.run_dir / f"selected_config_{args.selection_objective}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
