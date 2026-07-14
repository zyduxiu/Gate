from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "third_party"


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(command: list[str], cwd: Path | None = None, timeout: int | None = None) -> CommandResult:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        message = f"command timed out after {timeout} seconds"
        stderr = f"{stderr}\n{message}".strip()
        return CommandResult(command=command, returncode=124, stdout=stdout, stderr=stderr)
    except OSError as exc:
        return CommandResult(command=command, returncode=127, stdout="", stderr=str(exc))
    return CommandResult(command=command, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def is_windows_path(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[2] in ("\\", "/")


def to_wsl_path(value: str) -> str:
    if not is_windows_path(value):
        return value
    drive = value[0].lower()
    rest = value[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def run_external_binary(command: list[str], timeout: int | None = None) -> CommandResult:
    exe = command[0]
    if os.name == "nt" and not exe.lower().endswith(".exe"):
        converted = [to_wsl_path(part) for part in command]
        lib_dir = str(Path(converted[0]).parent)
        shell_command = (
            f"export LD_LIBRARY_PATH={shlex.quote(lib_dir)}:${{LD_LIBRARY_PATH:-}}; "
            + " ".join(shlex.quote(part) for part in converted)
        )
        wsl_command = ["wsl", "bash", "-lc", shell_command]
        result = run_command(wsl_command, timeout=timeout)
        return CommandResult(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    return run_command(command, timeout=timeout)


def find_first(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def nsg_paths() -> dict[str, str | None]:
    repo = THIRD_PARTY / "nsg"
    build = find_first(
        [
            repo / "build" / "tests" / "test_nsg_index",
            repo / "build" / "tests" / "Release" / "test_nsg_index.exe",
            repo / "build" / "tests" / "test_nsg_index.exe",
        ]
    )
    search = find_first(
        [
            repo / "build" / "tests" / "test_nsg_optimized_search",
            repo / "build" / "tests" / "Release" / "test_nsg_optimized_search.exe",
            repo / "build" / "tests" / "test_nsg_optimized_search.exe",
        ]
    )
    return {
        "repo": str(repo.resolve()),
        "build_exe": str(build) if build else None,
        "search_exe": str(search) if search else None,
    }


def sptag_paths() -> dict[str, str | None]:
    repo = THIRD_PARTY / "SPTAG"
    builder = find_first(
        [
            repo / "Release" / "IndexBuilder",
            repo / "build" / "IndexBuilder",
            repo / "Release" / "IndexBuilder.exe",
            repo / "build" / "Release" / "IndexBuilder.exe",
            repo / "build" / "AnnService" / "src" / "IndexBuilder" / "Release" / "IndexBuilder.exe",
        ]
    )
    searcher = find_first(
        [
            repo / "Release" / "IndexSearcher",
            repo / "build" / "IndexSearcher",
            repo / "Release" / "IndexSearcher.exe",
            repo / "build" / "Release" / "IndexSearcher.exe",
            repo / "build" / "AnnService" / "src" / "IndexSearcher" / "Release" / "IndexSearcher.exe",
        ]
    )
    return {
        "repo": str(repo.resolve()),
        "builder_exe": str(builder) if builder else None,
        "searcher_exe": str(searcher) if searcher else None,
    }


def diskann_paths() -> dict[str, str | None]:
    repo = THIRD_PARTY / "DiskANN"
    cargo = shutil.which("cargo")
    cargo_wsl = None
    if os.name == "nt" and shutil.which("wsl"):
        probe = run_command(
            ["wsl", "bash", "-lc", ". \"$HOME/.cargo/env\" 2>/dev/null || true; command -v cargo || true"],
            timeout=10,
        )
        cargo_wsl = probe.stdout.strip() or None
    return {
        "repo": str(repo.resolve()),
        "cargo": cargo,
        "cargo_wsl": cargo_wsl,
    }


def gate_paths() -> dict[str, str | None]:
    repo = THIRD_PARTY / "GATE" / "jacksondca-gate-78334b4"
    build = repo / "build" / "tests"
    nsg_index = find_first(
        [
            build / "test_nsg_index",
            build / "test_nsg_index.exe",
        ]
    )
    gate_search = find_first(
        [
            build / "test_gate_search",
            build / "test_gate_search.exe",
        ]
    )
    gate_optimized_search = find_first(
        [
            build / "test_gate_optimized_search",
            build / "test_gate_optimized_search.exe",
        ]
    )
    cos_navigate = find_first(
        [
            build / "test_gate_cos_navigate",
            build / "test_gate_cos_navigate.exe",
        ]
    )
    return {
        "repo": str(repo.resolve()),
        "nsg_index_exe": str(nsg_index) if nsg_index else None,
        "gate_search_exe": str(gate_search) if gate_search else None,
        "gate_optimized_search_exe": str(gate_optimized_search) if gate_optimized_search else None,
        "cos_navigate_exe": str(cos_navigate) if cos_navigate else None,
    }


def probe_external_baselines() -> dict[str, Any]:
    return {
        "nsg": nsg_paths(),
        "sptag": sptag_paths(),
        "diskann": diskann_paths(),
        "gate": gate_paths(),
        "tools": {
            "cmake": shutil.which("cmake"),
            "cargo": shutil.which("cargo"),
            "cl": shutil.which("cl"),
            "msbuild": shutil.which("msbuild"),
            "nmake": shutil.which("nmake"),
        },
    }


def run_nsg_build(manifest: dict[str, Any], index_path: Path, L: int, R: int, C: int, timeout: int | None) -> CommandResult:
    paths = nsg_paths()
    exe = paths["build_exe"]
    if exe is None:
        raise FileNotFoundError("NSG build executable not found. Run scripts/build_external_baselines.ps1 -Baseline nsg")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    return run_external_binary(
        [
            exe,
            manifest["paths"]["base_fvecs"],
            manifest["paths"]["nsg_knn_graph"],
            str(L),
            str(R),
            str(C),
            str(index_path.resolve()),
        ],
        timeout=timeout,
    )


def run_nsg_search(
    manifest: dict[str, Any],
    index_path: Path,
    result_path: Path,
    search_l: int,
    search_k: int,
    timeout: int | None,
) -> CommandResult:
    paths = nsg_paths()
    exe = paths["search_exe"]
    if exe is None:
        raise FileNotFoundError("NSG search executable not found. Run scripts/build_external_baselines.ps1 -Baseline nsg")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    return run_external_binary(
        [
            exe,
            manifest["paths"]["base_fvecs"],
            manifest["paths"]["query_fvecs"],
            str(index_path.resolve()),
            str(search_l),
            str(search_k),
            str(result_path.resolve()),
        ],
        timeout=timeout,
    )


def run_sptag_build(
    manifest: dict[str, Any],
    index_dir: Path,
    algo: str,
    threads: int,
    timeout: int | None,
) -> CommandResult:
    paths = sptag_paths()
    exe = paths["builder_exe"]
    if exe is None:
        raise FileNotFoundError("SPTAG IndexBuilder not found. Run scripts/build_external_baselines.ps1 -Baseline sptag")
    index_dir.mkdir(parents=True, exist_ok=True)
    return run_external_binary(
        [
            exe,
            "-d",
            str(manifest["dim"]),
            "-v",
            "Float",
            "-f",
            "XVEC",
            "-i",
            manifest["paths"]["base_fvecs"],
            "-o",
            str(index_dir.resolve()),
            "-a",
            algo,
            "-t",
            str(threads),
            "Index.DistCalcMethod=L2",
        ],
        timeout=timeout,
    )


def run_sptag_search(
    manifest: dict[str, Any],
    index_dir: Path,
    result_path: Path,
    maxcheck: int,
    k: int,
    threads: int,
    timeout: int | None,
) -> CommandResult:
    paths = sptag_paths()
    exe = paths["searcher_exe"]
    if exe is None:
        raise FileNotFoundError("SPTAG IndexSearcher not found. Run scripts/build_external_baselines.ps1 -Baseline sptag")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    return run_external_binary(
        [
            exe,
            "-d",
            str(manifest["dim"]),
            "-v",
            "Float",
            "-f",
            "XVEC",
            "-i",
            manifest["paths"]["query_fvecs"],
            "-x",
            str(index_dir.resolve()),
            "-r",
            manifest["paths"]["truth_bin"],
            "-o",
            str(result_path.resolve()),
            "-m",
            str(maxcheck),
            "-k",
            str(k),
            "-tk",
            str(manifest["gt_k"]),
            "-t",
            str(threads),
        ],
        timeout=timeout,
    )


def write_diskann_benchmark_config(
    manifest: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    search_l: list[int],
    max_degree: int,
    l_build: int,
    alpha: float,
    threads: int,
    reps: int,
    topk: int,
    start_point_strategy: str,
    entry_anchor_file: Path | str | None = None,
    entries: int = 8,
    load_path: Path | str | None = None,
    save_path: str = "diskann_graph_index",
) -> dict[str, Any]:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if start_point_strategy == "medoid":
        strategy: str | dict[str, Any] = "medoid"
    elif start_point_strategy.startswith("latin:"):
        samples = int(start_point_strategy.split(":", 1)[1])
        strategy = {"latin_hyper_cube": {"nsamples": samples, "seed": 17}}
    else:
        raise ValueError("start_point_strategy must be 'medoid' or 'latin:<samples>'")

    if entry_anchor_file is None:
        search_phase = {
            "search-type": "topk",
            "queries": manifest["paths"]["query_fbin"],
            "groundtruth": manifest["paths"]["truth_bin"],
            "reps": reps,
            "num_threads": [threads],
            "runs": [
                {
                    "search_n": topk,
                    "search_l": search_l,
                    "recall_k": topk,
                }
            ],
        }
    else:
        search_phase = {
            "search-type": "topk-entry-anchors",
            "queries": manifest["paths"]["query_fbin"],
            "groundtruth": manifest["paths"]["truth_bin"],
            "anchors": str(Path(entry_anchor_file).resolve()),
            "entries": entries,
            "reps": reps,
            "num_threads": [threads],
            "runs": [
                {
                    "search_n": topk,
                    "search_l": search_l,
                    "recall_k": topk,
                }
            ],
        }

    if load_path is None:
        source = {
            "index-source": "Build",
            "data_type": "float32",
            "data": manifest["paths"]["base_fbin"],
            "distance": "squared_l2",
            "max_degree": max_degree,
            "l_build": l_build,
            "insert_retry": None,
            "start_point_strategy": strategy,
            "alpha": alpha,
            "backedge_ratio": 1.0,
            "num_threads": threads,
            "multi_insert": {
                "batch_parallelism": max(1, threads),
                "batch_size": 128,
                "intra_batch_candidates": "none",
            },
            "save_path": save_path,
        }
    else:
        source = {
            "index-source": "Load",
            "data_type": "float32",
            "distance": "squared_l2",
            "load_path": str(Path(load_path).resolve()),
        }

    payload = {
        "search_directories": [str(Path(manifest["manifest"]).parent.resolve())],
        "output_directory": str(output_dir.resolve()),
        "jobs": [
            {
                "type": "graph-index-build",
                "content": {
                    "source": source,
                    "search_phase": search_phase,
                },
            }
        ],
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _convert_paths_for_wsl(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _convert_paths_for_wsl(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_convert_paths_for_wsl(item) for item in value]
    if isinstance(value, str) and is_windows_path(value):
        return to_wsl_path(value)
    return value


def write_wsl_diskann_config(config_path: Path) -> Path:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    converted = _convert_paths_for_wsl(payload)
    wsl_config = config_path.with_name(config_path.stem + ".wsl.json")
    wsl_config.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    return wsl_config


def run_diskann_benchmark(config_path: Path, output_path: Path, timeout: int | None) -> CommandResult:
    paths = diskann_paths()
    cargo = paths["cargo"]
    if cargo is None and paths.get("cargo_wsl") is None:
        raise FileNotFoundError("Cargo not found; DiskANN3 benchmark requires Rust/Cargo.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cargo is None:
        wsl_config_path = write_wsl_diskann_config(config_path)
        repo = to_wsl_path(paths["repo"])
        config = to_wsl_path(str(wsl_config_path.resolve()))
        output = to_wsl_path(str(output_path.resolve()))
        shell_command = (
            f"cd {shlex.quote(repo)} && "
            ". \"$HOME/.cargo/env\" && "
            "export RUSTFLAGS='-Ctarget-cpu=x86-64-v3' && "
            "cargo run --release --package diskann-benchmark -- --quiet run "
            f"--input-file {shlex.quote(config)} --output-file {shlex.quote(output)}"
        )
        result = run_command(["wsl", "bash", "-lc", shell_command], timeout=timeout)
        return CommandResult(
            command=["cargo", "run", "--release", "--package", "diskann-benchmark", "--", "--quiet", "run"],
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    command = [
        cargo,
        "run",
        "--release",
        "--package",
        "diskann-benchmark",
        "--",
        "--quiet",
        "run",
        "--input-file",
        str(config_path.resolve()),
        "--output-file",
        str(output_path.resolve()),
    ]
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-Ctarget-cpu=x86-64-v3")
    proc = subprocess.run(
        command,
        cwd=paths["repo"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    return CommandResult(command=command, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
