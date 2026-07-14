from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.external import CommandResult, run_command, run_external_binary, to_wsl_path
from dynanchor.io_formats import load_manifest


def has_wsl_avx512() -> bool:
    if shutil.which("wsl") is None:
        return False
    result = run_command(["wsl", "bash", "-lc", "grep -m1 -qi avx512f /proc/cpuinfo"], timeout=10)
    return result.returncode == 0


def log_result(path: Path, result: CommandResult) -> None:
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


def run_wsl_shell(shell: str, timeout: int | None) -> CommandResult:
    result = run_command(["wsl", "bash", "-lc", shell], timeout=timeout)
    return CommandResult(command=["wsl", "bash", "-lc", shell], returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def build_ngfix(repo: Path, build_dir: Path, portable_distance: bool, jobs: int, timeout: int | None) -> CommandResult:
    repo_wsl = to_wsl_path(str(repo.resolve()))
    build_wsl = to_wsl_path(str(build_dir.resolve()))
    portable = "ON" if portable_distance else "OFF"
    shell = (
        f"set -e; rm -rf {build_wsl}; mkdir -p {build_wsl}; cd {build_wsl}; "
        f"cmake -DNGFIX_PORTABLE_DISTANCE={portable} {repo_wsl}; "
        f"make -j{jobs}"
    )
    return run_wsl_shell(shell, timeout=timeout)


def parse_ngfix_result(result_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not result_csv.exists():
        return rows
    with result_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            if not row.get("efs"):
                continue
            rows.append(
                {
                    "ef_search": int(float(row["efs"])),
                    "recall_at_10": float(row["recall"]),
                    "avg_distance_computations": float(row["ndc"]),
                    "avg_latency_ms": float(row["latency"]),
                    "relative_distance_error": float(row["rderr"]),
                }
            )
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
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


def run_step(name: str, command: list[str], log_dir: Path, timeout: int | None) -> CommandResult:
    result = run_external_binary(command, timeout=timeout)
    log_result(log_dir / f"{name}.log.json", result)
    return result


def failure_summary(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    status: str,
    note: str,
    build_mode: str,
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": manifest.get("dataset", args.manifest.parent.name),
            "backend": "ngfix_official_artifact",
            "method": "NGFix/RFix official artifact",
            "graph_backend": "HNSW-NGFix",
            "dynamic_graph_regime": "author_artifact_static_repair",
            "comparison_status": status,
            "build_mode": build_mode,
            "note": note,
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official NGFix/RFix artifact as a close hard-region repair baseline.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift100k_dynamic" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "ngfix_official_sift100k")
    parser.add_argument("--ngfix-repo", type=Path, default=ROOT / "third_party" / "NGFix")
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--portable-distance", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-bottom", action="store_true")
    parser.add_argument("--skip-ngfix", action="store_true")
    parser.add_argument("--metric", default="l2_float")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--mex", type=int, default=48)
    parser.add_argument("--efc", type=int, default=200)
    parser.add_argument("--efc-aknn", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    avx512 = has_wsl_avx512()
    if args.portable_distance == "auto":
        portable_distance = not avx512
    else:
        portable_distance = args.portable_distance == "on"
    build_mode = "portable_distance_compat" if portable_distance else "official_avx512"

    build_dir = args.build_dir
    if build_dir is None:
        build_dir = args.ngfix_repo / ("build_portable" if portable_distance else "build")
    build_dir = build_dir.resolve()

    meta = {
        "manifest": str(args.manifest.resolve()),
        "ngfix_repo": str(args.ngfix_repo.resolve()),
        "build_dir": str(build_dir),
        "cpu_has_avx512f": avx512,
        "build_mode": build_mode,
        "note": "portable_distance_compat keeps NGFix/RFix graph logic but replaces AVX-512 distance kernels with scalar kernels because this WSL CPU does not expose avx512f.",
    }
    (args.out_dir / "ngfix_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    build_exe = build_dir / "test" / "build_hnsw_bottom"
    ngfix_exe = build_dir / "test" / "build_hnsw_ngfix_aknn"
    search_exe = build_dir / "test" / "search_hnsw_ngfix"
    if not args.skip_build and (args.force_rebuild or not (build_exe.exists() and ngfix_exe.exists() and search_exe.exists())):
        build_result = build_ngfix(args.ngfix_repo, build_dir, portable_distance, args.jobs, args.timeout)
        log_result(log_dir / "build.log.json", build_result)
        if build_result.returncode != 0:
            rows = failure_summary(args, manifest, "build_failed", "NGFix artifact build failed; see logs/build.log.json.", build_mode)
            write_summary(args.out_dir / "ngfix_official_summary.csv", rows)
            raise RuntimeError("NGFix build failed; see logs/build.log.json")

    missing = [path for path in [build_exe, ngfix_exe, search_exe] if not path.exists()]
    if missing:
        rows = failure_summary(args, manifest, "missing_binaries", f"Missing NGFix binaries: {', '.join(str(p) for p in missing)}", build_mode)
        write_summary(args.out_dir / "ngfix_official_summary.csv", rows)
        raise FileNotFoundError(rows[0]["note"])

    bottom_index = args.out_dir / "hnsw_bottom.index"
    ngfix_index = args.out_dir / "hnsw_ngfix.index"
    result_csv = args.out_dir / "ngfix_search.csv"

    if not args.skip_bottom and (args.force_rebuild or not bottom_index.exists()):
        result = run_step(
            "build_hnsw_bottom",
            [
                str(build_exe),
                "--base_data_path",
                manifest["paths"]["base_fbin"],
                "--metric",
                args.metric,
                "--result_hnsw_index_path",
                str(bottom_index.resolve()),
                "--efC",
                str(args.efc),
                "--M",
                str(args.m),
                "--MEX",
                str(args.mex),
            ],
            log_dir,
            args.timeout,
        )
        if result.returncode != 0:
            status = "runtime_failed_missing_avx512" if "Illegal instruction" in result.stderr + result.stdout else "runtime_failed"
            rows = failure_summary(args, manifest, status, "build_hnsw_bottom failed; see logs/build_hnsw_bottom.log.json.", build_mode)
            write_summary(args.out_dir / "ngfix_official_summary.csv", rows)
            raise RuntimeError(rows[0]["note"])

    if not args.skip_ngfix and (args.force_rebuild or not ngfix_index.exists()):
        result = run_step(
            "build_hnsw_ngfix_aknn",
            [
                str(ngfix_exe),
                "--train_query_path",
                manifest["paths"]["query_fbin"],
                "--base_graph_path",
                str(bottom_index.resolve()),
                "--metric",
                args.metric,
                "--result_index_path",
                str(ngfix_index.resolve()),
                "--efC_AKNN",
                str(args.efc_aknn),
            ],
            log_dir,
            args.timeout,
        )
        if result.returncode != 0:
            status = "runtime_failed_missing_avx512" if "Illegal instruction" in result.stderr + result.stdout else "runtime_failed"
            rows = failure_summary(args, manifest, status, "build_hnsw_ngfix_aknn failed; see logs/build_hnsw_ngfix_aknn.log.json.", build_mode)
            write_summary(args.out_dir / "ngfix_official_summary.csv", rows)
            raise RuntimeError(rows[0]["note"])

    result = run_step(
        "search_hnsw_ngfix",
        [
            str(search_exe),
            "--test_query_path",
            manifest["paths"]["query_fbin"],
            "--test_gt_path",
            manifest["paths"]["truth_bin"],
            "--metric",
            args.metric,
            "--index_path",
            str(ngfix_index.resolve()),
            "--result_path",
            str(result_csv.resolve()),
            "--K",
            str(args.k),
        ],
        log_dir,
        args.timeout,
    )
    if result.returncode != 0:
        status = "runtime_failed_missing_avx512" if "Illegal instruction" in result.stderr + result.stdout else "runtime_failed"
        rows = failure_summary(args, manifest, status, "search_hnsw_ngfix failed; see logs/search_hnsw_ngfix.log.json.", build_mode)
        write_summary(args.out_dir / "ngfix_official_summary.csv", rows)
        raise RuntimeError(rows[0]["note"])

    rows = []
    for row in parse_ngfix_result(result_csv):
        row.update(
            {
                "dataset": manifest.get("dataset", args.manifest.parent.name),
                "backend": "ngfix_official_artifact",
                "method": "NGFix/RFix official artifact",
                "graph_backend": "HNSW-NGFix",
                "dynamic_graph_regime": "author_artifact_static_repair",
                "comparison_status": "executed_official_artifact",
                "build_mode": build_mode,
                "note": "Official NGFix/RFix artifact. Direct backend differs from our NSG/Vamana entry-ablation backend; use as hard-region repair baseline, not same-graph ablation.",
            }
        )
        rows.append(row)
    summary_path = args.out_dir / "ngfix_official_summary.csv"
    write_summary(summary_path, rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
