from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .metrics import squared_l2_to_matrix


@dataclass(frozen=True)
class SearchResult:
    ids: list[int]
    distances: list[float]
    visited: int
    max_depth: int
    latency_ms: float
    entry_points: tuple[int, ...]


class DynamicKNNGraph:
    """Small research backend for graph-search experiments.

    This is intentionally simple and inspectable. It is not a production ANN
    index. The point is to expose graph-search telemetry that is hard to get
    from optimized libraries: visited nodes, entry points, and traversal depth.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        k: int = 16,
        max_degree: int | None = None,
        seed: int = 13,
    ) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        self.k = int(k)
        self.max_degree = int(max_degree or max(k * 2, k + 4))
        self.rng = np.random.default_rng(seed)
        self.alive = np.ones(self.vectors.shape[0], dtype=bool)
        self.adj: list[set[int]] = [set() for _ in range(self.vectors.shape[0])]
        self.build_full()

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def alive_count(self) -> int:
        return int(np.count_nonzero(self.alive))

    def alive_ids(self) -> np.ndarray:
        return np.flatnonzero(self.alive)

    def build_full(self) -> None:
        ids = self.alive_ids()
        self.adj = [set() for _ in range(self.size)]
        if ids.size <= 1:
            return
        k = min(self.k, ids.size - 1)
        for node_id in ids:
            nearest = self.nearest_ids(self.vectors[node_id], k=k, exclude={int(node_id)})
            for neighbor in nearest:
                self._connect(int(node_id), int(neighbor))
        for node_id in ids:
            self._prune_node(int(node_id))

    def nearest_ids(
        self,
        query: np.ndarray,
        k: int,
        exclude: Iterable[int] | None = None,
        candidates: Sequence[int] | np.ndarray | None = None,
    ) -> list[int]:
        if k <= 0:
            return []
        exclude_set = set(int(x) for x in (exclude or ()))
        ids = np.asarray(candidates, dtype=np.int64) if candidates is not None else self.alive_ids()
        if ids.size == 0:
            return []
        if exclude_set:
            ids = np.asarray([idx for idx in ids if int(idx) not in exclude_set], dtype=np.int64)
        if ids.size == 0:
            return []
        distances = squared_l2_to_matrix(query, self.vectors[ids])
        take = min(int(k), ids.size)
        part = np.argpartition(distances, take - 1)[:take]
        ordered = part[np.argsort(distances[part])]
        return [int(ids[pos]) for pos in ordered]

    def exact_search(self, query: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        ids = self.alive_ids()
        if ids.size == 0:
            return [], []
        distances = squared_l2_to_matrix(query, self.vectors[ids])
        take = min(int(k), ids.size)
        part = np.argpartition(distances, take - 1)[:take]
        ordered = part[np.argsort(distances[part])]
        result_ids = [int(ids[pos]) for pos in ordered]
        return result_ids, [float(distances[pos]) for pos in ordered]

    def medoid_id(self) -> int:
        ids = self.alive_ids()
        if ids.size == 0:
            raise ValueError("cannot compute medoid of an empty graph")
        center = np.mean(self.vectors[ids], axis=0)
        return self.nearest_ids(center, k=1)[0]

    def insert(self, batch: np.ndarray) -> list[int]:
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        if batch.ndim != 2 or batch.shape[1] != self.vectors.shape[1]:
            raise ValueError("inserted vectors must match graph dimensionality")

        start = self.size
        self.vectors = np.vstack([self.vectors, batch])
        self.alive = np.concatenate([self.alive, np.ones(batch.shape[0], dtype=bool)])
        self.adj.extend(set() for _ in range(batch.shape[0]))

        new_ids = list(range(start, start + batch.shape[0]))
        for node_id in new_ids:
            neighbors = self.nearest_ids(
                self.vectors[node_id],
                k=min(self.k, max(1, self.alive_count - 1)),
                exclude={node_id},
            )
            for neighbor in neighbors:
                self._connect(node_id, neighbor)
            self._prune_node(node_id)
            for neighbor in neighbors:
                self._prune_node(neighbor)
        return new_ids

    def delete(self, ids: Iterable[int]) -> None:
        for node_id in ids:
            if 0 <= int(node_id) < self.size:
                self.alive[int(node_id)] = False

    def repair_nodes(
        self,
        nodes: Iterable[int],
        target_degree: int | None = None,
    ) -> dict[str, int | float]:
        """Locally restore graph degree around selected alive nodes.

        This is deliberately not a rebuild. It only touches the requested nodes
        and any neighbors that receive new back-edges.
        """

        target = int(target_degree or self.k)
        target = max(1, min(target, self.max_degree))
        alive_total = self.alive_count
        if alive_total <= 1:
            return {"repaired_nodes": 0, "edges_added": 0, "edges_removed": 0, "target_degree": target}

        touched: set[int] = set()
        repaired = 0
        edges_added = 0
        edges_removed = 0

        for raw_node_id in nodes:
            node_id = int(raw_node_id)
            if not (0 <= node_id < self.size) or not self.alive[node_id]:
                continue

            before = set(self.adj[node_id])
            alive_neighbors = {n for n in before if 0 <= n < self.size and self.alive[n] and n != node_id}
            edges_removed += len(before) - len(alive_neighbors)
            self.adj[node_id] = alive_neighbors

            nearest = self.nearest_ids(
                self.vectors[node_id],
                k=min(target, alive_total - 1),
                exclude={node_id},
            )
            for neighbor in nearest:
                if neighbor not in self.adj[node_id]:
                    edges_added += 1
                self._connect(node_id, neighbor)
                touched.add(int(neighbor))

            touched.add(node_id)
            repaired += 1

        for node_id in sorted(touched):
            if 0 <= node_id < self.size and self.alive[node_id]:
                self._prune_node(node_id)

        return {
            "repaired_nodes": repaired,
            "touched_nodes": len(touched),
            "edges_added": edges_added,
            "edges_removed": edges_removed,
            "target_degree": target,
        }

    def repair_after_delete(
        self,
        deleted_ids: Iterable[int],
        inserted_ids: Iterable[int] = (),
        target_degree: int | None = None,
    ) -> dict[str, int | float]:
        """Repair nodes most likely affected by dynamic updates."""

        deleted = {int(node_id) for node_id in deleted_ids if 0 <= int(node_id) < self.size}
        affected: set[int] = set()
        for node_id in deleted:
            affected.update(n for n in self.adj[node_id] if 0 <= n < self.size and self.alive[n])
            self.adj[node_id].clear()

        inserted = {int(node_id) for node_id in inserted_ids if 0 <= int(node_id) < self.size and self.alive[int(node_id)]}
        repair_set = affected | inserted
        report = self.repair_nodes(repair_set, target_degree=target_degree)
        report["deleted_nodes"] = len(deleted)
        report["delete_affected_nodes"] = len(affected)
        report["inserted_nodes"] = len(inserted)
        return report

    def search(
        self,
        query: np.ndarray,
        entry_points: Sequence[int],
        topk: int = 10,
        ef: int = 80,
        early_stop: bool = True,
    ) -> SearchResult:
        start = time.perf_counter()
        entries = tuple(int(x) for x in entry_points if 0 <= int(x) < self.size and self.alive[int(x)])
        if not entries:
            entries = (self.medoid_id(),)

        candidate_heap: list[tuple[float, int, int]] = []
        enqueued: set[int] = set()
        visited: set[int] = set()
        top_heap: list[tuple[float, int]] = []
        max_depth = 0

        for node_id in entries:
            dist = float(np.sum((self.vectors[node_id] - query) ** 2))
            heapq.heappush(candidate_heap, (dist, node_id, 0))
            enqueued.add(node_id)

        while candidate_heap and len(visited) < ef:
            dist, node_id, depth = heapq.heappop(candidate_heap)
            if node_id in visited or not self.alive[node_id]:
                continue
            if early_stop and len(top_heap) >= topk and dist > -top_heap[0][0]:
                break
            visited.add(node_id)
            max_depth = max(max_depth, depth)

            heapq.heappush(top_heap, (-dist, node_id))
            if len(top_heap) > topk:
                heapq.heappop(top_heap)

            for neighbor in self.adj[node_id]:
                if neighbor in enqueued or neighbor in visited or not self.alive[neighbor]:
                    continue
                ndist = float(np.sum((self.vectors[neighbor] - query) ** 2))
                heapq.heappush(candidate_heap, (ndist, neighbor, depth + 1))
                enqueued.add(neighbor)

        ordered = sorted([(-dist, node_id) for dist, node_id in top_heap])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SearchResult(
            ids=[int(node_id) for _, node_id in ordered],
            distances=[float(dist) for dist, _ in ordered],
            visited=len(visited),
            max_depth=max_depth,
            latency_ms=elapsed_ms,
            entry_points=entries,
        )

    def _connect(self, left: int, right: int) -> None:
        if left == right:
            return
        self.adj[left].add(right)
        self.adj[right].add(left)

    def _prune_node(self, node_id: int) -> None:
        neighbors = [n for n in self.adj[node_id] if self.alive[n]]
        if len(neighbors) <= self.max_degree:
            self.adj[node_id] = set(neighbors)
            return
        distances = squared_l2_to_matrix(self.vectors[node_id], self.vectors[neighbors])
        keep_pos = np.argpartition(distances, self.max_degree - 1)[: self.max_degree]
        keep = {int(neighbors[pos]) for pos in keep_pos}
        removed = self.adj[node_id] - keep
        self.adj[node_id] = keep
        for other in removed:
            self.adj[other].discard(node_id)
