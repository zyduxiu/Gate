from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.io_formats import read_fvecs, write_diskann_fbin, write_fvecs, write_u32_list


def squared_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return (
        np.sum(points * points, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1).reshape(1, -1)
        - 2.0 * points @ centers.T
    )


def balanced_assign(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = squared_distances(points, centers)
    order = np.argsort(distances, axis=1)
    n, k = distances.shape
    capacity = int(math.ceil(n / k))
    labels = np.full(n, -1, dtype=np.int32)
    counts = np.zeros(k, dtype=np.int32)
    if k > 1:
        margin = distances[np.arange(n), order[:, 1]] - distances[np.arange(n), order[:, 0]]
    else:
        margin = np.zeros(n, dtype=np.float32)
    point_order = np.argsort(-margin)
    for idx in point_order:
        for cluster in order[idx]:
            c = int(cluster)
            if counts[c] < capacity:
                labels[idx] = c
                counts[c] += 1
                break
        if labels[idx] < 0:
            c = int(np.argmin(counts))
            labels[idx] = c
            counts[c] += 1
    return labels


def balanced_kmeans(points: np.ndarray, k: int, iterations: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]
    k = max(1, min(int(k), n))
    centers = points[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max(1, int(iterations))):
        labels = balanced_assign(points, centers)
        for cluster in range(k):
            members = labels == cluster
            if np.any(members):
                centers[cluster] = np.mean(points[members], axis=0)
            else:
                centers[cluster] = points[int(rng.integers(0, n))]
    labels = balanced_assign(points, centers)
    return labels, centers.astype(np.float32)


def medoid_id(points: np.ndarray, ids: np.ndarray, center: np.ndarray) -> int:
    distances = np.sum((points - center.reshape(1, -1)) ** 2, axis=1)
    return int(ids[int(np.argmin(distances))])


def build_hbkm_hubs(
    base: np.ndarray,
    hub_path: Path,
    embedding_path: Path,
    anchor_path: Path,
    stage1: int,
    stage2: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    base = np.asarray(base, dtype=np.float32)
    labels1, centers1 = balanced_kmeans(base, stage1, iterations, rng)
    dim = base.shape[1]
    global_ids = np.arange(base.shape[0], dtype=np.int32)
    eps_by_stage1: list[list[int]] = []
    embeddings: list[np.ndarray] = []
    anchors: list[int] = []

    for cluster_id in range(centers1.shape[0]):
        ids1 = global_ids[labels1 == cluster_id]
        if ids1.size == 0:
            ids1 = np.asarray([medoid_id(base, global_ids, centers1[cluster_id])], dtype=np.int32)
        points1 = base[ids1]
        sub_k = min(stage2, points1.shape[0])
        labels2, centers2 = balanced_kmeans(points1, sub_k, iterations, rng)
        row_eps: list[int] = []
        for sub_id in range(stage2):
            if sub_id < sub_k:
                ids2 = ids1[labels2 == sub_id]
                if ids2.size == 0:
                    ep = medoid_id(points1, ids1, centers2[sub_id])
                    centroid = base[ep]
                else:
                    centroid = np.mean(base[ids2], axis=0).astype(np.float32)
                    ep = medoid_id(base[ids2], ids2, centroid)
            else:
                ep = row_eps[-1] if row_eps else int(ids1[0])
                centroid = base[ep]
            row_eps.append(int(ep))
            anchors.append(int(ep))
            embeddings.append(np.asarray(centroid, dtype=np.float32))
        eps_by_stage1.append(row_eps)

    hub_path.parent.mkdir(parents=True, exist_ok=True)
    with hub_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{centers1.shape[0]} {stage2} {dim}\n")
        for center in centers1:
            handle.write(" ".join(f"{float(x):.9g}" for x in center) + "\n")
        for eps in eps_by_stage1:
            handle.write(" ".join(str(x) for x in eps) + "\n")
    write_fvecs(embedding_path, np.vstack(embeddings).astype(np.float32))
    anchor_path.write_text("\n".join(str(x) for x in anchors) + "\n", encoding="utf-8")
    return {
        "stage1": int(centers1.shape[0]),
        "stage2": int(stage2),
        "hub_count": int(len(anchors)),
        "dim": int(dim),
        "hub_path": str(hub_path.resolve()),
        "embedding_path": str(embedding_path.resolve()),
        "anchor_path": str(anchor_path.resolve()),
        "balanced": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a paper-aligned static GATE manifest with 512 HBKM-style hubs.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "sift_stress_drift" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "gate_paper_aligned_512_static")
    parser.add_argument("--index-path", type=Path, default=ROOT / "results" / "gate_official_selector_sift_stress_rerun" / "gate_final_rebuilt.nsg")
    parser.add_argument("--stage1", type=int, default=64)
    parser.add_argument("--stage2", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = manifest["paths"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gate_dir = args.out_dir / "gate_hubs"
    anchors_dir = args.out_dir / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)

    base = read_fvecs(Path(paths["base_fvecs"]))
    hub_meta = build_hbkm_hubs(
        base=base,
        hub_path=gate_dir / "gate512_refreshed_hubs.txt",
        embedding_path=gate_dir / "gate512_centroid_embeddings.fvecs",
        anchor_path=anchors_dir / "gate512_refreshed_hub.txt",
        stage1=args.stage1,
        stage2=args.stage2,
        iterations=args.iterations,
        seed=args.seed,
    )

    zero_insert = args.out_dir / "zero_insert.fbin"
    empty_delete = args.out_dir / "empty_delete.u32"
    write_diskann_fbin(zero_insert, np.empty((0, int(manifest["dim"])), dtype=np.float32))
    write_u32_list(empty_delete, np.asarray([], dtype=np.uint32))

    static_manifest = dict(manifest)
    static_manifest["dataset"] = manifest.get("dataset", "dataset") + "_gate512_static"
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
    static_manifest["anchor_files"] = dict(manifest.get("anchor_files", {}))
    static_manifest["anchor_files"]["gate512_refreshed_hub"] = hub_meta["anchor_path"]
    static_manifest["gate_hub_files"] = dict(manifest.get("gate_hub_files", {}))
    static_manifest["gate_hub_files"]["gate_refreshed_hub_file"] = hub_meta["hub_path"]
    static_manifest["gate_hub_files"]["gate512_refreshed_hub_file"] = hub_meta["hub_path"]
    static_manifest["gate_hub_files"]["gate512_centroid_embeddings"] = hub_meta["embedding_path"]
    static_manifest["query_entry_files"] = {}
    static_manifest["gate512_hub_meta"] = hub_meta
    static_manifest_path = args.out_dir / "static_manifest_base.json"
    static_manifest["manifest"] = str(static_manifest_path.resolve())
    static_manifest_path.write_text(json.dumps(static_manifest, indent=2), encoding="utf-8")
    (args.out_dir / "prepare_meta.json").write_text(json.dumps({"hub_meta": hub_meta, "manifest": str(static_manifest_path.resolve())}, indent=2), encoding="utf-8")
    print(json.dumps({"hub_meta": hub_meta, "manifest": str(static_manifest_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
