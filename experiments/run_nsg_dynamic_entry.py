from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.external import run_external_binary


def run_method(
    runner: Path,
    manifest: dict[str, Any],
    out_dir: Path,
    initial_nsg: Path,
    method: str,
    search_l: int,
    entries: int,
    topk: int,
    insert_degree: int,
    max_degree: int,
    repair_degree: int,
    maintain_batch: int,
    maintain_changes: int,
    maintain_sample: int,
    online_score_weights: str,
    online_retire_weights: str,
    online_retire_load_weight: float,
    online_query_gain_weight: float,
    online_query_load_weight: float,
    anchor_cost_query_limit: int,
    online_mlp_model: Path | None,
    query_entries_ivecs: Path | None,
    hard_repair_ivecs: Path | None,
    hard_repair_local_degree: int,
    hard_repair_bridge_degree: int,
    timeout: int,
    per_query: bool,
) -> None:
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
        method,
        "--search-l",
        str(search_l),
        "--entries",
        str(entries),
        "--topk",
        str(topk),
        "--insert-degree",
        str(insert_degree),
        "--max-degree",
        str(max_degree),
        "--repair-degree",
        str(repair_degree),
        "--maintain-batch",
        str(maintain_batch),
        "--maintain-changes",
        str(maintain_changes),
        "--maintain-sample",
        str(maintain_sample),
        "--out-json",
        str((out_dir / f"{method}_L{search_l}.json").resolve()),
    ]
    if online_score_weights:
        command.extend(["--online-score-weights", online_score_weights])
    if online_retire_weights:
        command.extend(["--online-retire-weights", online_retire_weights])
    if online_retire_load_weight > 0.0:
        command.extend(["--online-retire-load-weight", str(online_retire_load_weight)])
    if online_query_gain_weight > 0.0:
        command.extend(["--online-query-gain-weight", str(online_query_gain_weight)])
    if online_query_load_weight > 0.0:
        command.extend(["--online-query-load-weight", str(online_query_load_weight)])
    if anchor_cost_query_limit > 0:
        command.extend(["--anchor-cost-query-limit", str(anchor_cost_query_limit)])
    if online_mlp_model is not None:
        command.extend(["--online-mlp-model", str(online_mlp_model.resolve())])
    if query_entries_ivecs is not None:
        command.extend(["--query-entries-ivecs", str(query_entries_ivecs.resolve())])
    if hard_repair_ivecs is not None:
        command.extend(
            [
                "--hard-repair-ivecs",
                str(hard_repair_ivecs.resolve()),
                "--hard-repair-local-degree",
                str(hard_repair_local_degree),
                "--hard-repair-bridge-degree",
                str(hard_repair_bridge_degree),
            ]
        )
    if method != "nsg_ep":
        anchor_key = "static_density" if method in {"online_density", "online_hub_density"} else method
        anchor_file = manifest.get("anchor_files", {}).get(anchor_key)
        query_entries_mode = method == "query_entries" or method.startswith("query_entries_")
        if query_entries_mode:
            anchor_file = None
        elif not anchor_file:
            raise KeyError(f"manifest has no anchor_files entry for method={anchor_key}")
        if anchor_file:
            command.extend(["--anchors-txt", anchor_file])
    if per_query:
        command.extend(["--out-per-query", str((out_dir / f"{method}_L{search_l}_perq.csv").resolve())])

    result = run_external_binary(command, timeout=timeout)
    log_path = out_dir / f"{method}_L{search_l}.log.json"
    log_path.write_text(
        json.dumps(
            {
                "command": result.command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not result.ok:
        raise RuntimeError(f"{method} L={search_l} failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dynamic no-rebuild NSG entry runner for selected methods.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift100k_dynamic" / "manifest.json")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--initial-nsg", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "sift100k_dynamic_nsg_entry")
    parser.add_argument("--methods", default="learned_dynamic")
    parser.add_argument("--search-l", default="20,40,80,120,240")
    parser.add_argument("--entries", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--insert-degree", type=int, default=32)
    parser.add_argument("--max-degree", type=int, default=64)
    parser.add_argument("--repair-degree", type=int, default=0)
    parser.add_argument("--maintain-batch", type=int, default=1000)
    parser.add_argument("--maintain-changes", type=int, default=4)
    parser.add_argument("--maintain-sample", type=int, default=256)
    parser.add_argument("--online-score-weights", default="")
    parser.add_argument("--online-retire-weights", default="")
    parser.add_argument("--online-retire-load-weight", type=float, default=0.0)
    parser.add_argument("--online-query-gain-weight", type=float, default=0.0)
    parser.add_argument("--online-query-load-weight", type=float, default=0.0)
    parser.add_argument("--anchor-cost-query-limit", type=int, default=0)
    parser.add_argument("--online-mlp-model", type=Path, default=None)
    parser.add_argument("--query-entries-ivecs", type=Path, default=None)
    parser.add_argument("--hard-repair-ivecs", type=Path, default=None)
    parser.add_argument("--hard-repair-local-degree", type=int, default=0)
    parser.add_argument("--hard-repair-bridge-degree", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--per-query-l", default="240")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    initial_nsg = args.initial_nsg
    if initial_nsg is None:
        initial_nsg_value = manifest.get("paths", {}).get("initial_nsg")
        if initial_nsg_value:
            initial_nsg = Path(initial_nsg_value)
        else:
            initial_nsg = ROOT / "results" / "sift100k_dynamic_nsg" / "initial.nsg"
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    search_ls = [int(item.strip()) for item in args.search_l.split(",") if item.strip()]
    per_query_ls = {int(item.strip()) for item in args.per_query_l.split(",") if item.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for search_l in search_ls:
        for method in methods:
            print(f"Running method={method} L={search_l}")
            run_method(
                runner=args.runner,
                manifest=manifest,
                out_dir=args.out_dir,
                initial_nsg=initial_nsg,
                method=method,
                search_l=search_l,
                entries=args.entries,
                topk=args.topk,
                insert_degree=args.insert_degree,
                max_degree=args.max_degree,
                repair_degree=args.repair_degree,
                maintain_batch=args.maintain_batch,
                maintain_changes=args.maintain_changes,
                maintain_sample=args.maintain_sample,
                online_score_weights=args.online_score_weights,
                online_retire_weights=args.online_retire_weights,
                online_retire_load_weight=args.online_retire_load_weight,
                online_query_gain_weight=args.online_query_gain_weight,
                online_query_load_weight=args.online_query_load_weight,
                anchor_cost_query_limit=args.anchor_cost_query_limit,
                online_mlp_model=args.online_mlp_model,
                query_entries_ivecs=args.query_entries_ivecs,
                hard_repair_ivecs=args.hard_repair_ivecs,
                hard_repair_local_degree=args.hard_repair_local_degree,
                hard_repair_bridge_degree=args.hard_repair_bridge_degree,
                timeout=args.timeout,
                per_query=search_l in per_query_ls,
            )

    print(f"Wrote results under {args.out_dir}")


if __name__ == "__main__":
    main()
