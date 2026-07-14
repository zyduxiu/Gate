from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.external import CommandResult, run_command, run_external_binary


def write_log(path: Path, result: CommandResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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


def require_ok(label: str, result: CommandResult, log_path: Path) -> None:
    write_log(log_path, result)
    if not result.ok:
        raise RuntimeError(f"{label} failed; see {log_path}")


def run_runner(
    *,
    runner: Path,
    manifest: dict[str, Any],
    initial_nsg: Path,
    out_dir: Path,
    mode: str,
    search_l: int,
    entries: int,
    topk: int,
    insert_degree: int,
    max_degree: int,
    repair_degree: int,
    maintain_batch: int,
    maintain_changes: int,
    maintain_sample: int,
    timeout: int,
    anchors_txt: Path | None = None,
    query_entries_ivecs: Path | None = None,
    out_anchors_txt: Path | None = None,
    out_anchor_costs_csv: Path | None = None,
    anchor_cost_query_limit: int = 0,
    per_query: bool = False,
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
        mode,
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
        str((out_dir / f"{mode}_L{search_l}.json").resolve()),
    ]
    if anchors_txt is not None:
        command.extend(["--anchors-txt", str(anchors_txt.resolve())])
    if query_entries_ivecs is not None:
        command.extend(["--query-entries-ivecs", str(query_entries_ivecs.resolve())])
    if out_anchors_txt is not None:
        out_anchors_txt.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["--out-anchors-txt", str(out_anchors_txt.resolve())])
    if out_anchor_costs_csv is not None:
        out_anchor_costs_csv.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["--out-anchor-costs-csv", str(out_anchor_costs_csv.resolve())])
        if anchor_cost_query_limit > 0:
            command.extend(["--anchor-cost-query-limit", str(anchor_cost_query_limit)])
    if per_query:
        command.extend(["--out-per-query", str((out_dir / f"{mode}_L{search_l}_perq.csv").resolve())])
    result = run_external_binary(command, timeout=timeout)
    require_ok(mode, result, out_dir / f"{mode}_L{search_l}.log.json")


def train_gate_paper(
    *,
    manifest_path: Path,
    anchor_file: Path,
    out_dir: Path,
    entries: int,
    epochs: int,
    timeout: int,
    train_query_fraction: float,
    seed: int,
    anchor_costs_csv: Path | None,
    query_tower: bool,
    topology_embedding: str,
) -> Path:
    command = [
        sys.executable,
        str((ROOT / "experiments" / "train_gate_paper_selector.py").resolve()),
        "--manifest",
        str(manifest_path.resolve()),
        "--anchor-file",
        str(anchor_file.resolve()),
        "--out-dir",
        str(out_dir.resolve()),
        "--entries",
        str(entries),
        "--epochs",
        str(epochs),
        "--train-query-fraction",
        str(train_query_fraction),
        "--seed",
        str(seed),
        "--topology-embedding",
        topology_embedding,
    ]
    if query_tower:
        command.append("--query-tower")
    if anchor_costs_csv is not None:
        command.extend(["--anchor-costs-csv", str(anchor_costs_csv.resolve())])
    result = run_command(command, timeout=timeout)
    require_ok(f"train paper GATE {anchor_file.name}", result, out_dir / "train.log.json")
    entries_path = out_dir / "gate_paper_entries.ivecs"
    if not entries_path.exists():
        raise FileNotFoundError(f"training did not create {entries_path}")
    return entries_path


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-aligned GATE selector comparisons on the dynamic NSG backend.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_drift" / "manifest.json")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--initial-nsg", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "gate_paper_comparison_sift_stress")
    parser.add_argument("--search-l", type=int, default=240)
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--insert-degree", type=int, default=32)
    parser.add_argument("--max-degree", type=int, default=64)
    parser.add_argument("--repair-degree", type=int, default=0)
    parser.add_argument("--maintain-batch", type=int, default=1000)
    parser.add_argument("--maintain-changes", type=int, default=4)
    parser.add_argument("--maintain-sample", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-query-fraction", type=float, default=0.6)
    parser.add_argument("--query-tower", action="store_true")
    parser.add_argument("--topology-embedding", choices=["wl_hash", "stats_svd"], default="wl_hash")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    initial_nsg = args.initial_nsg
    if initial_nsg is None:
        initial_nsg = Path(manifest.get("paths", {}).get("initial_nsg", ROOT / "results" / "sift_stress_drift_dynamic_nsg" / "initial.nsg"))
    anchor_files = {key: Path(value) for key, value in manifest.get("anchor_files", {}).items()}
    static_density = anchor_files["static_density"]
    gate_initial = anchor_files["gate_initial_hub"]
    gate_refreshed = anchor_files["gate_refreshed_hub"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    anchors_dir = args.out_dir / "anchors"
    selectors_dir = args.out_dir / "selectors"
    online_e = anchors_dir / "online_hub_density_maintained_E.txt"
    costs_dir = args.out_dir / "anchor_costs"
    query_limit = max(1, min(int(manifest.get("n_query", 1000)), int(round(float(manifest.get("n_query", 1000)) * args.train_query_fraction))))

    run_runner(
        runner=args.runner,
        manifest=manifest,
        initial_nsg=initial_nsg,
        out_dir=args.out_dir,
        mode="nsg_ep",
        search_l=args.search_l,
        entries=args.entries,
        topk=args.topk,
        insert_degree=args.insert_degree,
        max_degree=args.max_degree,
        repair_degree=args.repair_degree,
        maintain_batch=args.maintain_batch,
        maintain_changes=args.maintain_changes,
        maintain_sample=args.maintain_sample,
        timeout=args.timeout,
    )
    for mode, anchor_file in [
        ("gate_initial_hub", gate_initial),
        ("gate_refreshed_hub", gate_refreshed),
    ]:
        run_runner(
            runner=args.runner,
            manifest=manifest,
            initial_nsg=initial_nsg,
            out_dir=args.out_dir,
            mode=mode,
            search_l=args.search_l,
            entries=args.entries,
            topk=args.topk,
            insert_degree=args.insert_degree,
            max_degree=args.max_degree,
            repair_degree=args.repair_degree,
            maintain_batch=args.maintain_batch,
            maintain_changes=args.maintain_changes,
            maintain_sample=args.maintain_sample,
            timeout=args.timeout,
            anchors_txt=anchor_file,
            out_anchor_costs_csv=costs_dir / f"{mode}_costs.csv",
            anchor_cost_query_limit=query_limit,
        )
    run_runner(
        runner=args.runner,
        manifest=manifest,
        initial_nsg=initial_nsg,
        out_dir=args.out_dir,
        mode="online_hub_density",
        search_l=args.search_l,
        entries=args.entries,
        topk=args.topk,
        insert_degree=args.insert_degree,
        max_degree=args.max_degree,
        repair_degree=args.repair_degree,
        maintain_batch=args.maintain_batch,
        maintain_changes=args.maintain_changes,
        maintain_sample=args.maintain_sample,
        timeout=args.timeout,
        anchors_txt=static_density,
        out_anchors_txt=online_e,
        out_anchor_costs_csv=costs_dir / "online_hub_density_E_costs.csv",
        anchor_cost_query_limit=query_limit,
    )

    selector_inputs = {
        "gate_initial_hub": (gate_initial, costs_dir / "gate_initial_hub_costs.csv"),
        "gate_refreshed_hub": (gate_refreshed, costs_dir / "gate_refreshed_hub_costs.csv"),
        "online_hub_density_E": (online_e, costs_dir / "online_hub_density_E_costs.csv"),
    }
    for offset, (label, (anchor_file, costs_file)) in enumerate(selector_inputs.items()):
        selector_out = selectors_dir / label
        entries_path = train_gate_paper(
            manifest_path=args.manifest,
            anchor_file=anchor_file,
            out_dir=selector_out,
            entries=args.entries,
            epochs=args.epochs,
            timeout=args.timeout,
            train_query_fraction=args.train_query_fraction,
            seed=args.seed + offset,
            anchor_costs_csv=costs_file,
            query_tower=args.query_tower,
            topology_embedding=args.topology_embedding,
        )
        run_runner(
            runner=args.runner,
            manifest=manifest,
            initial_nsg=initial_nsg,
            out_dir=args.out_dir,
            mode=f"query_entries_paper_gate_{label}",
            search_l=args.search_l,
            entries=args.entries,
            topk=args.topk,
            insert_degree=args.insert_degree,
            max_degree=args.max_degree,
            repair_degree=args.repair_degree,
            maintain_batch=args.maintain_batch,
            maintain_changes=args.maintain_changes,
            maintain_sample=args.maintain_sample,
            timeout=args.timeout,
            query_entries_ivecs=entries_path,
        )

    summary_path = args.out_dir / "summary.csv"
    summary_cmd = [
        sys.executable,
        str((ROOT / "experiments" / "summarize_nsg_entry_results.py").resolve()),
        "--dir",
        str(args.out_dir.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--out",
        str(summary_path.resolve()),
    ]
    summary_result = run_command(summary_cmd, timeout=300)
    require_ok("summarize", summary_result, args.out_dir / "summarize.log.json")
    meta = {
        "manifest": str(args.manifest.resolve()),
        "initial_nsg": str(initial_nsg.resolve()),
        "summary": str(summary_path.resolve()),
        "online_maintained_entry_file": str(online_e.resolve()),
        "search_l": args.search_l,
        "entries": args.entries,
        "epochs": args.epochs,
        "train_query_fraction": args.train_query_fraction,
        "query_tower": args.query_tower,
        "topology_embedding": args.topology_embedding,
        "anchor_cost_query_limit": query_limit,
        "note": (
            "Paper-aligned GATE selector uses sampled hub subgraph features, actual per-anchor search-cost positive/negative samples, "
            "and triplet cosine loss. The same dynamic NSG backend is used for all methods; only entry initialization differs."
        ),
    }
    (args.out_dir / "gate_paper_comparison_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    rows = read_summary(summary_path)
    print(json.dumps(meta, indent=2))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
