from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.external import diskann_paths, gate_paths, nsg_paths, sptag_paths


def exists(value: str | None) -> bool:
    return bool(value) and Path(value).exists()


def git_commit(repo: Path) -> str | None:
    if not repo.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def read_gate_adapter_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "search_l": int(row["search_l"]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "qps": float(row["qps"]) if row["qps"] else None,
                    "note": row["note"],
                }
            )
        return rows


def read_ngfix_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "ef_search": int(float(row["ef_search"])),
                    "recall_at_10": float(row["recall_at_10"]),
                    "avg_distance_computations": float(row["avg_distance_computations"]),
                    "avg_latency_ms": float(row["avg_latency_ms"]),
                    "build_mode": row.get("build_mode", ""),
                    "comparison_status": row.get("comparison_status", ""),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Report status of close related baselines.")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "close_baselines" / "status.json")
    args = parser.parse_args()

    gate = gate_paths()
    gate_adapter_csv = ROOT / "results" / "gate_adapter_10k" / "gate_adapter_results.csv"
    gate_two_tower_meta = ROOT / "results" / "entry_topology_backend_matrix" / "gate_two_tower" / "gate_two_tower_meta.json"
    ngfix_repo = ROOT / "third_party" / "NGFix"
    ngfix_summary = ROOT / "results" / "ngfix_official_sift100k" / "ngfix_official_summary.csv"
    ngfix_meta = ROOT / "results" / "ngfix_official_sift100k" / "ngfix_run_meta.json"
    steiner_repo = ROOT / "third_party" / "Steiner-hardness"
    payload: dict[str, Any] = {
        "gate": {
            "paper": "Empowering Graph-based Approximate Nearest Neighbor Search with Adaptive Awareness Capabilities",
            "artifact": "https://zenodo.org/records/15523071",
            "local_repo": gate["repo"],
            "status": "buildable_with_static_adapter",
            "executables": {
                "test_nsg_index": gate["nsg_index_exe"],
                "test_gate_search": gate["gate_search_exe"],
                "test_gate_cos_navigate": gate["cos_navigate_exe"],
            },
            "binary_ready": all(exists(gate[key]) for key in ["nsg_index_exe", "gate_search_exe"]),
            "paper_faithful_ready": False,
            "static_adapter": {
                "script": str(ROOT / "experiments" / "run_gate_adapter.py"),
                "results_csv": str(gate_adapter_csv),
                "ready": gate_adapter_csv.exists(),
                "rows": read_gate_adapter_rows(gate_adapter_csv),
                "caveat": "Uses generated two-level centroid hubs with GATE C++ search; it does not train GATE's two-tower model.",
            },
            "two_tower_proxy": {
                "script": str(ROOT / "experiments" / "train_gate_two_tower.py"),
                "meta_json": str(gate_two_tower_meta),
                "ready": gate_two_tower_meta.exists(),
                "caveat": "Local PyTorch query/hub selector with graph features; closer than ridge, but not the original GATE Graph2Vec artifact pipeline.",
            },
            "blocked_on": [
                "README pipeline requires hub extraction and learned hub embeddings.",
                "Zenodo snapshot's Python kmeans/train scripts contain hard-coded globals and CUDA assumptions.",
                "A paper-faithful run needs generated HUB_PATH and EMB_PATH for our dynamic dataset.",
            ],
            "comparison_role": "closest entry-overlay baseline; current adapter measures static hub entry, paper-faithful comparison still needs learned hub embeddings",
        },
        "diskann_plus_plus": {
            "paper": "DiskANN++: Efficient Page-based Search over Isomorphic Mapped Graph Index using Query-sensitivity Entry Vertex",
            "paper_url": "https://arxiv.org/abs/2310.00402",
            "status": "paper_only_no_public_code_found",
            "blocked_on": [
                "No official code link found on arXiv page.",
                "Requires implementing query-sensitive entry vertex and page-search layout or obtaining authors' artifact.",
            ],
            "comparison_role": "query-sensitive entry vertex for DiskANN; must be discussed and approximated if code remains unavailable",
        },
        "hardness_rfix_ngfix": {
            "paper": "Dynamically Detect and Fix Hardness for Efficient Approximate Nearest Neighbor Search",
            "paper_url": "https://arxiv.org/abs/2510.22316",
            "artifact": "https://github.com/yuhuifishash/NGFix",
            "local_repo": str(ngfix_repo),
            "commit": git_commit(ngfix_repo),
            "status": "executed_official_artifact" if ngfix_summary.exists() else "artifact_cloned_not_run",
            "official_artifact": {
                "script": str(ROOT / "experiments" / "run_ngfix_baseline.py"),
                "summary_csv": str(ngfix_summary),
                "meta_json": str(ngfix_meta),
                "ready": ngfix_summary.exists(),
                "rows": read_ngfix_rows(ngfix_summary),
                "caveat": "Current run uses portable scalar distance kernels because this WSL CPU lacks avx512f; backend is HNSW-NGFix, not same-graph NSG/Vamana.",
            },
            "blocked_on": [
                "Final performance-faithful numbers should be rerun on AVX-512 hardware with --portable-distance off.",
                "Same-graph NSG/Vamana hard-region repair remains a proxy unless RFix/NGFix is ported into those backends.",
            ],
            "comparison_role": "dynamic hard-region graph repair; closest conceptual competitor to entry-to-vicinity failure analysis",
        },
        "steiner_hardness": {
            "paper": "Steiner-hardness: A Query Hardness Measure for Graph-based ANN Indexes",
            "repo_url": "https://github.com/CaucherWang/Steiner-hardness",
            "local_repo": str(steiner_repo),
            "commit": git_commit(steiner_repo),
            "status": "cloned_diagnostic_workload_generator",
            "is_rfix_ngfix_implementation": False,
            "blocked_on": [
                "Pipeline depends on efanna_graph plus MRNG/reverse-graph preprocessing.",
                "Default config is hard-coded to the authors' local paths and SIFT-scale settings.",
                "It measures/generates hard queries; it does not implement dynamic graph repair.",
            ],
            "comparison_role": "hard-query diagnostic and unbiased workload generator for stress-testing entry maintenance",
        },
        "already_executable": {
            "nsg": nsg_paths(),
            "sptag": sptag_paths(),
            "diskann": diskann_paths(),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    for name, item in payload.items():
        if isinstance(item, dict) and "status" in item:
            print(f"{name}: {item['status']}")


if __name__ == "__main__":
    main()
