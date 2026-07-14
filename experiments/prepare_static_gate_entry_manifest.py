from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.io_formats import read_fvecs, write_diskann_fbin, write_ivecs, write_u32_list


def parse_gate_eps(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    iter1, iter2, _dim = [int(x) for x in lines[0].split()[:3]]
    eps_lines = lines[1 + iter1 : 1 + iter1 + iter1]
    eps: list[int] = []
    for line in eps_lines:
        values = [int(x) for x in line.split()]
        if len(values) != iter2:
            raise ValueError(f"expected {iter2} eps per row in {path}")
        eps.extend(values)
    if len(eps) != iter1 * iter2:
        raise ValueError(f"bad eps count in {path}")
    return np.asarray(eps, dtype=np.int32)


def cosine_top_entries(queries: np.ndarray, embeddings: np.ndarray, eps: np.ndarray, top: int) -> np.ndarray:
    queries = np.asarray(queries, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    q_norm = np.linalg.norm(queries, axis=1, keepdims=True)
    e_norm = np.linalg.norm(embeddings, axis=1, keepdims=True).T
    scores = queries @ embeddings.T
    scores = scores / np.maximum(q_norm * e_norm, 1e-12)
    top_idx = np.argpartition(-scores, kth=top - 1, axis=1)[:, :top]
    row = np.arange(top_idx.shape[0])[:, None]
    top_idx = top_idx[row, np.argsort(-scores[row, top_idx], axis=1)]
    return eps[top_idx].astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a pure-static manifest and GATE-style per-query entry files.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_drift" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "static_fair_gate_vs_nsg")
    parser.add_argument("--index-path", type=Path, default=ROOT / "results" / "gate_official_selector_sift_stress_rerun" / "gate_final_rebuilt.nsg")
    parser.add_argument("--top-entries", type=int, default=3)
    parser.add_argument(
        "--embeddings",
        default=(
            "gate_centroid_refreshed:data/sift_stress_drift/gate_hubs/gate_refreshed_hub_embeddings.fvecs,"
            "gate_paper_costlabel:results/gate_paper_costlabel_sift_stress/selectors/gate_refreshed_hub/gate_paper_hub_embeddings.fvecs"
        ),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zero_insert = args.out_dir / "zero_insert.fbin"
    empty_delete = args.out_dir / "empty_delete.u32"
    write_diskann_fbin(zero_insert, np.empty((0, int(manifest["dim"])), dtype=np.float32))
    write_u32_list(empty_delete, np.asarray([], dtype=np.uint32))

    static_manifest = dict(manifest)
    static_manifest["dataset"] = manifest.get("dataset", "dataset") + "_static_fair_entry"
    static_manifest["n_initial"] = int(manifest["n_base"])
    static_manifest["n_insert"] = 0
    static_manifest["n_delete"] = 0
    static_manifest["paths"] = dict(paths)
    static_manifest["paths"]["initial_fbin"] = paths["base_fbin"]
    static_manifest["paths"]["insert_fbin"] = str(zero_insert.resolve())
    static_manifest["paths"]["delete_u32"] = str(empty_delete.resolve())
    static_manifest["paths"]["truth_dynamic_bin"] = paths["truth_bin"]
    static_manifest["paths"]["initial_nsg"] = str(args.index_path.resolve())
    static_manifest["paths"]["all_nsg_knn_graph"] = paths["nsg_knn_graph"]

    queries = read_fvecs(Path(paths["query_fvecs"]))
    eps = parse_gate_eps(Path(static_manifest["gate_hub_files"]["gate_refreshed_hub_file"]))
    query_entries: dict[str, str] = {}
    for spec in [item.strip() for item in args.embeddings.split(",") if item.strip()]:
        label, _, raw_embedding = spec.partition(":")
        if not label or not raw_embedding:
            raise ValueError(f"bad embedding spec: {spec}")
        embedding_path = Path(raw_embedding)
        if not embedding_path.is_absolute():
            embedding_path = ROOT / embedding_path
        embeddings = read_fvecs(embedding_path)
        entries = cosine_top_entries(queries, embeddings, eps, args.top_entries)
        out = args.out_dir / f"{label}_entries_top{args.top_entries}.ivecs"
        write_ivecs(out, entries)
        query_entries[label] = str(out.resolve())

    static_manifest["query_entry_files"] = query_entries
    static_manifest_path = args.out_dir / "static_manifest.json"
    static_manifest["manifest"] = str(static_manifest_path.resolve())
    static_manifest_path.write_text(json.dumps(static_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(static_manifest_path), "query_entry_files": query_entries}, indent=2))


if __name__ == "__main__":
    main()
