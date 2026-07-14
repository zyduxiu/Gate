from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DynamicSpec:
    alias: str
    method: str
    note: str
    online_score_weights: str = ""
    online_query_gain_weight: float = 0.0
    online_query_load_weight: float = 0.0
    anchor_cost_query_limit: int = 0
    query_entries_ivecs: Path | None = None


def run(command: list[str], cwd: Path = ROOT, timeout: int = 7200) -> subprocess.CompletedProcess[str]:
    print("RUN", " ".join(str(x) for x in command), flush=True)
    result = subprocess.run(
        [str(x) for x in command],
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(str(x) for x in command)}")
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def stat(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {"mean": math.nan, "median": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "std": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["backend"]),
            str(row["method"]),
            int(fnum(row["search_l"])),
            int(fnum(row.get("entries", 0))),
            str(row.get("recall_metric", "recall@100")),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (backend, method, search_l, entries, recall_metric), items in groups.items():
        rec = stat([fnum(x.get("recall_at_k", x.get("recall_at_10"))) for x in items])
        qps = stat([fnum(x.get("qps")) for x in items])
        p99 = stat([fnum(x.get("p99_latency_us")) for x in items])
        comps = stat([fnum(x.get("avg_distance_computations")) for x in items])
        maint = stat([fnum(x.get("maintenance_seconds")) for x in items])
        out.append(
            {
                "backend": backend,
                "method": method,
                "search_l": search_l,
                "entries": entries,
                "recall_metric": recall_metric,
                "runs": len(items),
                "recall_mean": rec["mean"],
                "recall_median": rec["median"],
                "qps_mean": qps["mean"],
                "qps_median": qps["median"],
                "qps_std": qps["std"],
                "p99_latency_us_mean": p99["mean"],
                "p99_latency_us_median": p99["median"],
                "avg_distance_computations_mean": comps["mean"],
                "avg_distance_computations_median": comps["median"],
                "maintenance_seconds_mean": maint["mean"],
                "maintenance_seconds_median": maint["median"],
                "note": items[0].get("note", ""),
            }
        )
    out.sort(key=lambda row: (int(row["search_l"]), str(row["backend"]), str(row["method"])))
    return out


def run_dynamic_spec(args: argparse.Namespace, spec: DynamicSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        run_dir = args.out_dir / "dynamic_no_rebuild" / spec.alias / f"run_{repeat:02d}"
        summary_path = run_dir / "summary.csv"
        if not (args.skip_existing and summary_path.exists()):
            command = [
                sys.executable,
                ROOT / "experiments" / "run_nsg_dynamic_entry.py",
                "--manifest",
                args.manifest,
                "--initial-nsg",
                args.initial_nsg,
                "--runner",
                args.runner,
                "--out-dir",
                run_dir,
                "--methods",
                spec.method,
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
                "--timeout",
                str(args.timeout),
                "--per-query-l",
                "",
            ]
            if spec.online_score_weights:
                command.extend(["--online-score-weights", spec.online_score_weights])
            if spec.online_query_gain_weight > 0:
                command.extend(["--online-query-gain-weight", str(spec.online_query_gain_weight)])
            if spec.online_query_load_weight > 0:
                command.extend(["--online-query-load-weight", str(spec.online_query_load_weight)])
            if spec.anchor_cost_query_limit > 0:
                command.extend(["--anchor-cost-query-limit", str(spec.anchor_cost_query_limit)])
            if spec.query_entries_ivecs is not None:
                command.extend(["--query-entries-ivecs", spec.query_entries_ivecs])
            run(command, timeout=args.timeout * 2)
            run(
                [
                    sys.executable,
                    ROOT / "experiments" / "summarize_nsg_entry_results.py",
                    "--dir",
                    run_dir,
                    "--manifest",
                    args.manifest,
                    "--out",
                    summary_path,
                ],
                timeout=600,
            )
        for row in read_csv(summary_path):
            row["method"] = spec.alias
            row["backend"] = "custom_dynamic_nsg_no_rebuild"
            row["repeat"] = repeat
            row["note"] = spec.note
            rows.append(row)
    return rows


def parse_official_search_time(stdout: str) -> float:
    match = re.search(r"search time:\s*([0-9.]+)", stdout)
    if not match:
        return math.nan
    return float(match.group(1))


def run_official_nsg(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir = args.out_dir / "official_nsg_final_rebuilt"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "base_L40_R50_C500.nsg"
    if not (args.skip_existing and index.exists()):
        run(
            [
                sys.executable,
                ROOT / "experiments" / "external_baseline.py",
                "nsg-build",
                "--manifest",
                args.manifest,
                "--index",
                index,
                "--L",
                "40",
                "--R",
                "50",
                "--C",
                "500",
                "--timeout",
                str(args.timeout),
            ],
            timeout=args.timeout,
        )

    truth_compact = out_dir / "truth_compact.ivecs"
    if not truth_compact.exists():
        run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import json, numpy as np; "
                    "from src.dynanchor.io_formats import read_ivecs, write_ivecs; "
                    f"m=json.loads(Path(r'{args.manifest}').read_text()); "
                    "truth=read_ivecs(Path(m['paths']['truth_ivecs'])); "
                    "raw=np.fromfile(m['paths']['delete_u32'], dtype='<u4'); "
                    "delete=raw[1:] if len(raw) and raw[0]==len(raw)-1 else raw; "
                    "alive=np.ones(int(m['n_initial'])+int(m['n_insert']), dtype=bool); "
                    "alive[delete.astype(np.int64)]=False; labels=np.flatnonzero(alive); "
                    "mp={int(v):i for i,v in enumerate(labels)}; "
                    "compact=np.vectorize(lambda x: mp[int(x)], otypes=[np.int32])(truth); "
                    f"write_ivecs(Path(r'{truth_compact}'), compact)"
                ),
            ],
            timeout=600,
        )

    rows: list[dict[str, Any]] = []
    for search_l in [int(x) for x in args.search_l.split(",") if x.strip()]:
        for repeat in range(1, args.repeats + 1):
            result_path = out_dir / f"official_nsg_L{search_l}_K{args.topk}_run{repeat:02d}.ivecs"
            log_path = out_dir / f"official_nsg_L{search_l}_K{args.topk}_run{repeat:02d}.log.json"
            if args.skip_existing and result_path.exists() and log_path.exists():
                log = json.loads(log_path.read_text(encoding="utf-8"))
                stdout = log.get("stdout", "")
            else:
                result = run(
                    [
                        sys.executable,
                        ROOT / "experiments" / "external_baseline.py",
                        "nsg-search",
                        "--manifest",
                        args.manifest,
                        "--index",
                        index,
                        "--result",
                        result_path,
                        "--search-l",
                        str(search_l),
                        "--search-k",
                        str(args.topk),
                        "--timeout",
                        str(args.timeout),
                    ],
                    timeout=args.timeout,
                )
                stdout = result.stdout
                log_path.write_text(
                    json.dumps(
                        {
                            "command": result.args,
                            "returncode": result.returncode,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            metric_result = run(
                [
                    sys.executable,
                    ROOT / "experiments" / "external_baseline.py",
                    "eval-ivecs",
                    "--result",
                    result_path,
                    "--truth",
                    truth_compact,
                    "--k",
                    str(args.topk),
                ],
                timeout=600,
            )
            metric = json.loads(metric_result.stdout.split("Wrote")[0])
            search_seconds = parse_official_search_time(stdout)
            qps = float(metric["queries"]) / search_seconds if search_seconds and not math.isnan(search_seconds) else math.nan
            rows.append(
                {
                    "dataset": "sift1m_stress_streaming_drift_insert_fraction_eval",
                    "backend": "official_nsg_final_rebuilt_static",
                    "method": "official_nsg_rebuilt",
                    "search_l": search_l,
                    "entries": 1,
                    "anchors": 0,
                    "recall_metric": f"recall@{args.topk}",
                    "recall_at_k": metric["recall"],
                    "recall_at_10": metric["recall"],
                    "avg_expanded": "",
                    "avg_distance_computations": "",
                    "avg_latency_us": search_seconds * 1e6 / float(metric["queries"]),
                    "p99_latency_us": "",
                    "qps": qps,
                    "maintenance_seconds": 0.0,
                    "repeat": repeat,
                    "note": "Official NSG optimized search on final/rebuilt compact base. This is a static rebuilt upper/reference baseline, not the same dynamic no-rebuild graph as the entry ablation.",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full entry baselines plus official NSG final/rebuilt reference.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "online_mlp_train_eval_insert80" / "query_split" / "eval" / "manifest.json")
    parser.add_argument("--initial-nsg", type=Path, default=ROOT / "results" / "sift_stress_drift_dynamic_nsg" / "initial.nsg")
    parser.add_argument("--runner", type=Path, default=ROOT / "build" / "nsg_dynamic_entry_runner")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "full_baseline_with_official_nsg")
    parser.add_argument("--search-l", default="240,280,320")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--entries", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maintain-sample", type=int, default=512)
    parser.add_argument("--maintain-changes", type=int, default=4)
    parser.add_argument("--maintain-batch", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    gate_root = ROOT / "results" / "gate_paper_faithful_train_eval"
    specs = [
        DynamicSpec("nsg_ep", "nsg_ep", "Single NSG entry point on dynamic no-rebuild graph."),
        DynamicSpec("gate_initial_hub", "gate_initial_hub", "GATE/HBKM-style initial hub set kept stale after stream drift."),
        DynamicSpec("gate_refreshed_hub", "gate_refreshed_hub", "GATE/HBKM-style hub set recomputed on final distribution; expensive refresh proxy."),
        DynamicSpec("online_hub_density_default", "online_hub_density", "Our online density+hubness entry maintenance with default hand-tuned score."),
        DynamicSpec(
            "ours_queryaware_marginal_q128",
            "online_hub_density",
            "Our best QPS policy so far: marginal-label density/hubness/freshness plus query-gain/load terms.",
            online_score_weights="0.000000,0.326441,0.580062,0.093497",
            online_query_gain_weight=0.3,
            online_query_load_weight=0.3,
            anchor_cost_query_limit=128,
        ),
        DynamicSpec(
            "ours_prior_rollout_learned",
            "online_hub_density",
            "Prior-regularized rollout-learned policy; learned-policy baseline, not QPS-best.",
            online_score_weights="0.021442,0.251929,0.302281,0.000000",
            online_query_gain_weight=0.19889411945510407,
            online_query_load_weight=0.22545385819820643,
            anchor_cost_query_limit=128,
        ),
        DynamicSpec(
            "paper_gate_initial_direct",
            "query_entries_paper_gate_initial_direct",
            "Paper-aligned GATE two-tower direct top-entry over stale initial hubs; query entries are precomputed, so selector overhead is excluded.",
            query_entries_ivecs=gate_root / "gate_initial_hub_selector" / "gate_paper_entries.ivecs",
        ),
        DynamicSpec(
            "paper_gate_initial_navigation",
            "query_entries_paper_gate_initial_navigation",
            "Paper-aligned GATE two-tower high-level navigation over stale initial hubs; query entries are precomputed.",
            query_entries_ivecs=gate_root / "gate_initial_hub_selector" / "gate_navigation_entries.ivecs",
        ),
        DynamicSpec(
            "paper_gate_refreshed_direct",
            "query_entries_paper_gate_refreshed_direct",
            "Paper-aligned GATE two-tower direct top-entry over refreshed final hubs; expensive refresh proxy.",
            query_entries_ivecs=gate_root / "gate_refreshed_hub_selector" / "gate_paper_entries.ivecs",
        ),
        DynamicSpec(
            "paper_gate_refreshed_navigation",
            "query_entries_paper_gate_refreshed_navigation",
            "Paper-aligned GATE two-tower high-level navigation over refreshed final hubs; expensive refresh proxy.",
            query_entries_ivecs=gate_root / "gate_refreshed_hub_selector" / "gate_navigation_entries.ivecs",
        ),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "manifest": str(args.manifest.resolve()),
        "initial_nsg": str(args.initial_nsg.resolve()),
        "search_l": args.search_l,
        "topk": args.topk,
        "entries": args.entries,
        "repeats": args.repeats,
        "specs": [spec.__dict__ | {"query_entries_ivecs": str(spec.query_entries_ivecs) if spec.query_entries_ivecs else None} for spec in specs],
        "note": "Dynamic rows use the same custom dynamic no-rebuild NSG runner. Official NSG row is final/rebuilt static reference and uses compact-id truth.",
    }
    (args.out_dir / "full_suite_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    for spec in specs:
        all_rows.extend(run_dynamic_spec(args, spec))
    all_rows.extend(run_official_nsg(args))

    raw_path = args.out_dir / "full_suite_raw.csv"
    agg_path = args.out_dir / "full_suite_aggregate.csv"
    write_csv(raw_path, all_rows)
    write_csv(agg_path, aggregate(all_rows))
    print(json.dumps({"raw": str(raw_path), "aggregate": str(agg_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
