from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.external import (
    probe_external_baselines,
    run_diskann_benchmark,
    run_nsg_build,
    run_nsg_search,
    run_sptag_build,
    run_sptag_search,
    write_diskann_benchmark_config,
)
from dynanchor.io_formats import load_manifest, prepare_external_dataset, recall_from_ivecs, recall_from_sptag_txt


def safe_print(text: str = "") -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def print_result(result) -> None:
    safe_print("$ " + " ".join(str(part) for part in result.command))
    safe_print(f"returncode={result.returncode}")
    if result.stdout:
        safe_print("stdout:")
        safe_print(result.stdout.rstrip())
    if result.stderr:
        safe_print("stderr:")
        safe_print(result.stderr.rstrip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and run external ANNS baselines.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="Check available external baseline tools and executables.")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "external_probe.json")

    p = sub.add_parser("prepare-data", help="Generate benchmark files for NSG/SPTAG/DiskANN.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "external_10k")
    p.add_argument("--n-base", type=int, default=10_000)
    p.add_argument("--n-query", type=int, default=1_000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--clusters", type=int, default=8)
    p.add_argument("--drift-fraction", type=float, default=0.20)
    p.add_argument("--gt-k", type=int, default=100)
    p.add_argument("--nsg-graph-k", type=int, default=100)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--batch-size", type=int, default=128)

    p = sub.add_parser("nsg-build", help="Build an NSG index using generated files.")
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "external_10k" / "manifest.json")
    p.add_argument("--index", type=Path, default=ROOT / "results" / "external" / "nsg" / "base.nsg")
    p.add_argument("--L", type=int, default=40)
    p.add_argument("--R", type=int, default=50)
    p.add_argument("--C", type=int, default=500)
    p.add_argument("--timeout", type=int, default=None)

    p = sub.add_parser("nsg-search", help="Search an NSG index.")
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "external_10k" / "manifest.json")
    p.add_argument("--index", type=Path, default=ROOT / "results" / "external" / "nsg" / "base.nsg")
    p.add_argument("--result", type=Path, default=ROOT / "results" / "external" / "nsg" / "result.ivecs")
    p.add_argument("--search-l", type=int, default=80)
    p.add_argument("--search-k", type=int, default=10)
    p.add_argument("--timeout", type=int, default=None)

    p = sub.add_parser("sptag-build", help="Build an SPTAG memory index.")
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "external_10k" / "manifest.json")
    p.add_argument("--index-dir", type=Path, default=ROOT / "results" / "external" / "sptag_index")
    p.add_argument("--algo", default="BKT")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--timeout", type=int, default=None)

    p = sub.add_parser("sptag-search", help="Search an SPTAG memory index.")
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "external_10k" / "manifest.json")
    p.add_argument("--index-dir", type=Path, default=ROOT / "results" / "external" / "sptag_index")
    p.add_argument("--result", type=Path, default=ROOT / "results" / "external" / "sptag_result.txt")
    p.add_argument("--maxcheck", type=int, default=1024)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--timeout", type=int, default=None)

    p = sub.add_parser("diskann-config", help="Write a DiskANN3 benchmark JSON file.")
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "external_10k" / "manifest.json")
    p.add_argument("--config", type=Path, default=ROOT / "results" / "external" / "diskann_benchmark.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "external" / "diskann")
    p.add_argument("--search-l", default="20,40,80,120")
    p.add_argument("--max-degree", type=int, default=32)
    p.add_argument("--l-build", type=int, default=80)
    p.add_argument("--alpha", type=float, default=1.2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--start-point-strategy", default="medoid")
    p.add_argument(
        "--entry-anchor-method",
        default=None,
        help="Use manifest external_anchor_files[method] and write a topk-entry-anchors DiskANN config.",
    )
    p.add_argument("--entry-anchor-file", type=Path, default=None, help="Explicit anchor id txt file for topk-entry-anchors.")
    p.add_argument("--entries", type=int, default=8, help="Number of query-nearest anchors to use as DiskANN entries.")
    p.add_argument("--load-path", type=Path, default=None, help="Load an existing DiskANN graph index instead of building.")
    p.add_argument("--save-path", default="diskann_graph_index", help="Build-mode DiskANN save_path.")

    p = sub.add_parser("diskann-run", help="Run DiskANN3 benchmark JSON with Cargo.")
    p.add_argument("--config", type=Path, default=ROOT / "results" / "external" / "diskann_benchmark.json")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "external" / "diskann_output.json")
    p.add_argument("--timeout", type=int, default=None)

    p = sub.add_parser("eval-ivecs", help="Evaluate an ivecs result file against exact truth.")
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--truth", type=Path, default=ROOT / "data" / "external_10k" / "truth.ivecs")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("eval-sptag", help="Evaluate an SPTAG text result file against exact truth.")
    p.add_argument("--result", type=Path, default=ROOT / "results" / "external" / "sptag_result.txt")
    p.add_argument("--truth", type=Path, default=ROOT / "data" / "external_10k" / "truth.ivecs")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "external" / "sptag_metrics.json")

    args = parser.parse_args()

    if args.cmd == "probe":
        probe = probe_external_baselines()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(probe, indent=2), encoding="utf-8")
        print(json.dumps(probe, indent=2))
        print(f"Wrote {args.out}")
        return

    if args.cmd == "prepare-data":
        manifest = prepare_external_dataset(
            out_dir=args.out_dir,
            n_base=args.n_base,
            n_query=args.n_query,
            dim=args.dim,
            clusters=args.clusters,
            drift_fraction=args.drift_fraction,
            gt_k=args.gt_k,
            nsg_graph_k=args.nsg_graph_k,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        print(json.dumps(manifest, indent=2))
        return

    if args.cmd == "diskann-config":
        manifest = load_manifest(args.manifest)
        search_l = [int(x.strip()) for x in args.search_l.split(",") if x.strip()]
        entry_anchor_file = args.entry_anchor_file
        if args.entry_anchor_method:
            try:
                entry_anchor_file = Path(manifest["external_anchor_files"][args.entry_anchor_method])
            except KeyError as exc:
                available = ", ".join(sorted(manifest.get("external_anchor_files", {}).keys()))
                raise KeyError(
                    f"manifest has no external_anchor_files entry for {args.entry_anchor_method!r}; "
                    f"available: {available}"
                ) from exc
        payload = write_diskann_benchmark_config(
            manifest=manifest,
            config_path=args.config,
            output_dir=args.output_dir,
            search_l=search_l,
            max_degree=args.max_degree,
            l_build=args.l_build,
            alpha=args.alpha,
            threads=args.threads,
            reps=args.reps,
            topk=args.topk,
            start_point_strategy=args.start_point_strategy,
            entry_anchor_file=entry_anchor_file,
            entries=args.entries,
            load_path=args.load_path,
            save_path=args.save_path,
        )
        print(json.dumps(payload, indent=2))
        print(f"Wrote {args.config}")
        return

    if args.cmd == "diskann-run":
        print_result(run_diskann_benchmark(args.config, args.output, timeout=args.timeout))
        return

    if args.cmd == "eval-ivecs":
        summary = recall_from_ivecs(args.result, args.truth, args.k)
        text = json.dumps(summary, indent=2)
        safe_print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            safe_print(f"Wrote {args.out}")
        return

    if args.cmd == "eval-sptag":
        summary = recall_from_sptag_txt(args.result, args.truth, args.k)
        text = json.dumps(summary, indent=2)
        safe_print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            safe_print(f"Wrote {args.out}")
        return

    manifest = load_manifest(args.manifest)
    if args.cmd == "nsg-build":
        print_result(run_nsg_build(manifest, args.index, args.L, args.R, args.C, timeout=args.timeout))
    elif args.cmd == "nsg-search":
        print_result(run_nsg_search(manifest, args.index, args.result, args.search_l, args.search_k, timeout=args.timeout))
    elif args.cmd == "sptag-build":
        print_result(run_sptag_build(manifest, args.index_dir, args.algo, args.threads, timeout=args.timeout))
    elif args.cmd == "sptag-search":
        print_result(run_sptag_search(manifest, args.index_dir, args.result, args.maxcheck, args.k, args.threads, timeout=args.timeout))


if __name__ == "__main__":
    main()
