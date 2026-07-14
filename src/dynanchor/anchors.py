from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .graph import DynamicKNNGraph, SearchResult
from .metrics import gini, squared_l2_to_matrix


@dataclass
class AnchorStats:
    queries: int = 0
    visited_ema: float = 0.0
    success_ema: float = 1.0


class BaseRouter:
    name = "base"

    def fit(self, graph: DynamicKNNGraph) -> None:
        raise NotImplementedError

    def select(self, query: np.ndarray, graph: DynamicKNNGraph, s: int | None = None) -> tuple[int, ...]:
        raise NotImplementedError

    def after_query(
        self,
        query: np.ndarray,
        graph: DynamicKNNGraph,
        result: SearchResult,
        recall: float,
    ) -> None:
        return None

    def maintain(self, graph: DynamicKNNGraph) -> dict[str, int | float]:
        return {"added": 0, "removed": 0}

    def anchor_ids(self) -> list[int]:
        return []

    def load_gini(self) -> float:
        return 0.0


class MedoidRouter(BaseRouter):
    name = "fixed_medoid"

    def __init__(self) -> None:
        self.entry: int | None = None
        self.queries = 0

    def fit(self, graph: DynamicKNNGraph) -> None:
        self.entry = graph.medoid_id()

    def select(self, query: np.ndarray, graph: DynamicKNNGraph, s: int | None = None) -> tuple[int, ...]:
        if self.entry is None or not graph.alive[self.entry]:
            return (graph.medoid_id(),)
        return (self.entry,)

    def after_query(
        self,
        query: np.ndarray,
        graph: DynamicKNNGraph,
        result: SearchResult,
        recall: float,
    ) -> None:
        self.queries += 1

    def anchor_ids(self) -> list[int]:
        return [] if self.entry is None else [self.entry]


class RandomAnchorRouter(BaseRouter):
    name = "random_multi"

    def __init__(self, m: int = 32, s: int = 4, seed: int = 19) -> None:
        self.m = int(m)
        self.s = int(s)
        self.rng = np.random.default_rng(seed)
        self.anchors: list[int] = []
        self.counts: dict[int, int] = {}

    def fit(self, graph: DynamicKNNGraph) -> None:
        ids = graph.alive_ids()
        take = min(self.m, ids.size)
        self.anchors = [int(x) for x in self.rng.choice(ids, size=take, replace=False)]
        self.counts = {node_id: 0 for node_id in self.anchors}

    def select(self, query: np.ndarray, graph: DynamicKNNGraph, s: int | None = None) -> tuple[int, ...]:
        alive = [a for a in self.anchors if 0 <= a < graph.size and graph.alive[a]]
        if not alive:
            return (graph.medoid_id(),)
        vectors = graph.vectors[alive]
        distances = squared_l2_to_matrix(query, vectors)
        take = min(int(s or self.s), len(alive))
        order = np.argsort(distances)[:take]
        return tuple(int(alive[pos]) for pos in order)

    def after_query(
        self,
        query: np.ndarray,
        graph: DynamicKNNGraph,
        result: SearchResult,
        recall: float,
    ) -> None:
        for entry in result.entry_points:
            self.counts[entry] = self.counts.get(entry, 0) + 1

    def anchor_ids(self) -> list[int]:
        return list(self.anchors)

    def load_gini(self) -> float:
        return gini(self.counts.values())


class DensityAnchorRouter(BaseRouter):
    """Density-aware routing anchors that do not own storage partitions."""

    def __init__(
        self,
        m: int = 32,
        s: int = 4,
        dynamic: bool = False,
        min_anchors: int | None = None,
        max_anchors: int | None = None,
        density_knn: int = 10,
        coverage_weight: float = 0.65,
        density_weight: float = 0.25,
        degree_weight: float = 0.10,
        load_weight: float = 0.05,
        visited_weight: float = 0.002,
        split_factor: float = 1.8,
        merge_factor: float = 0.20,
        max_changes_per_maintain: int = 6,
    ) -> None:
        self.m = int(m)
        self.s = int(s)
        self.dynamic = bool(dynamic)
        self.min_anchors = int(min_anchors or max(4, m // 2))
        self.max_anchors = int(max_anchors or max(m * 2, m + 4))
        self.density_knn = int(density_knn)
        self.coverage_weight = float(coverage_weight)
        self.density_weight = float(density_weight)
        self.degree_weight = float(degree_weight)
        self.load_weight = float(load_weight)
        self.visited_weight = float(visited_weight)
        self.split_factor = float(split_factor)
        self.merge_factor = float(merge_factor)
        self.max_changes_per_maintain = int(max_changes_per_maintain)
        self.anchors: list[int] = []
        self.stats: dict[int, AnchorStats] = {}
        self._name = "dynamic_density" if self.dynamic else "static_density"

    @property
    def name(self) -> str:
        return self._name

    def fit(self, graph: DynamicKNNGraph) -> None:
        self.anchors = self._select_density_anchors(graph, self.m)
        self.stats = {node_id: AnchorStats() for node_id in self.anchors}

    def select(self, query: np.ndarray, graph: DynamicKNNGraph, s: int | None = None) -> tuple[int, ...]:
        alive = self._alive_anchors(graph)
        if not alive:
            return (graph.medoid_id(),)
        distances = squared_l2_to_matrix(query, graph.vectors[alive])
        max_queries = max((self.stats.get(a, AnchorStats()).queries for a in alive), default=1)
        costs = []
        for pos, anchor in enumerate(alive):
            stat = self.stats.get(anchor, AnchorStats())
            load_penalty = self.load_weight * (stat.queries / max(1, max_queries))
            visited_penalty = self.visited_weight * stat.visited_ema
            success_bonus = 0.05 * stat.success_ema
            costs.append(float(distances[pos]) + load_penalty + visited_penalty - success_bonus)
        take = min(int(s or self.s), len(alive))
        order = np.argsort(np.asarray(costs))[:take]
        return tuple(int(alive[pos]) for pos in order)

    def after_query(
        self,
        query: np.ndarray,
        graph: DynamicKNNGraph,
        result: SearchResult,
        recall: float,
    ) -> None:
        for entry in result.entry_points:
            stat = self.stats.setdefault(entry, AnchorStats())
            stat.queries += 1
            if stat.visited_ema == 0.0:
                stat.visited_ema = float(result.visited)
            else:
                stat.visited_ema = 0.85 * stat.visited_ema + 0.15 * float(result.visited)
            stat.success_ema = 0.90 * stat.success_ema + 0.10 * float(recall >= 0.99)

    def maintain(self, graph: DynamicKNNGraph) -> dict[str, int | float]:
        if not self.dynamic:
            return {"added": 0, "removed": 0, "anchors": len(self._alive_anchors(graph))}

        self.anchors = self._alive_anchors(graph)
        if len(self.anchors) < self.min_anchors:
            needed = min(self.m - len(self.anchors), self.max_anchors - len(self.anchors))
            for node_id in self._select_density_anchors(graph, max(0, needed), exclude=set(self.anchors)):
                self._add_anchor(node_id)

        if not self.anchors:
            self.fit(graph)
            return {"added": len(self.anchors), "removed": 0, "anchors": len(self.anchors)}

        report = self.coverage_report(graph)
        target = max(1.0, graph.alive_count / max(1, len(self.anchors)))
        added = 0
        removed = 0

        overloaded = sorted(
            report["per_anchor"],
            key=lambda item: (item["count"] / target, item["mean_distance"]),
            reverse=True,
        )
        for item in overloaded:
            if added >= self.max_changes_per_maintain or len(self.anchors) >= self.max_anchors:
                break
            if item["count"] < self.split_factor * target and item["mean_distance"] <= report["mean_distance_p75"]:
                continue
            candidate = self._split_candidate(graph, item["members"], item["anchor"])
            if candidate is not None and candidate not in self.anchors:
                self._add_anchor(candidate)
                added += 1

        report = self.coverage_report(graph)
        target = max(1.0, graph.alive_count / max(1, len(self.anchors)))
        underloaded = sorted(report["per_anchor"], key=lambda item: (item["count"], item["mean_distance"]))
        for item in underloaded:
            if removed >= self.max_changes_per_maintain or len(self.anchors) <= self.min_anchors:
                break
            stat = self.stats.get(item["anchor"], AnchorStats())
            if item["count"] <= self.merge_factor * target and stat.queries <= 1:
                self._remove_anchor(item["anchor"])
                removed += 1

        return {
            "added": added,
            "removed": removed,
            "anchors": len(self.anchors),
            "coverage_mean": report["mean_distance"],
            "coverage_p75": report["mean_distance_p75"],
        }

    def coverage_report(self, graph: DynamicKNNGraph) -> dict[str, object]:
        alive_anchors = self._alive_anchors(graph)
        ids = graph.alive_ids()
        if not alive_anchors or ids.size == 0:
            return {"per_anchor": [], "mean_distance": 0.0, "mean_distance_p75": 0.0}

        anchor_vectors = graph.vectors[alive_anchors]
        assignments: dict[int, list[int]] = {anchor: [] for anchor in alive_anchors}
        distances_by_anchor: dict[int, list[float]] = {anchor: [] for anchor in alive_anchors}

        for node_id in ids:
            distances = squared_l2_to_matrix(graph.vectors[node_id], anchor_vectors)
            pos = int(np.argmin(distances))
            anchor = int(alive_anchors[pos])
            assignments[anchor].append(int(node_id))
            distances_by_anchor[anchor].append(float(distances[pos]))

        per_anchor = []
        means = []
        for anchor in alive_anchors:
            dists = distances_by_anchor[anchor]
            mean_distance = float(np.mean(dists)) if dists else 0.0
            means.append(mean_distance)
            per_anchor.append(
                {
                    "anchor": int(anchor),
                    "count": len(assignments[anchor]),
                    "mean_distance": mean_distance,
                    "members": assignments[anchor],
                }
            )
        return {
            "per_anchor": per_anchor,
            "mean_distance": float(np.mean(means)) if means else 0.0,
            "mean_distance_p75": float(np.percentile(means, 75)) if means else 0.0,
        }

    def anchor_ids(self) -> list[int]:
        return list(self.anchors)

    def load_gini(self) -> float:
        return gini(stat.queries for stat in self.stats.values())

    def _alive_anchors(self, graph: DynamicKNNGraph) -> list[int]:
        seen = set()
        alive = []
        for anchor in self.anchors:
            if anchor in seen:
                continue
            if 0 <= anchor < graph.size and graph.alive[anchor]:
                alive.append(int(anchor))
                seen.add(int(anchor))
        return alive

    def _add_anchor(self, node_id: int) -> None:
        self.anchors.append(int(node_id))
        self.stats.setdefault(int(node_id), AnchorStats())

    def _remove_anchor(self, node_id: int) -> None:
        self.anchors = [anchor for anchor in self.anchors if anchor != int(node_id)]
        self.stats.pop(int(node_id), None)

    def _select_density_anchors(
        self,
        graph: DynamicKNNGraph,
        count: int,
        exclude: set[int] | None = None,
    ) -> list[int]:
        exclude = exclude or set()
        ids = np.asarray([idx for idx in graph.alive_ids() if int(idx) not in exclude], dtype=np.int64)
        if ids.size == 0 or count <= 0:
            return []
        count = min(int(count), ids.size)
        density = self._local_density(graph, ids)
        degree = np.asarray([len(graph.adj[int(idx)]) for idx in ids], dtype=np.float64)
        degree = _normalize(degree)
        density_norm = _normalize(density)

        selected: list[int] = []
        first = int(ids[int(np.argmax(density_norm + self.degree_weight * degree))])
        selected.append(first)
        while len(selected) < count:
            selected_vectors = graph.vectors[selected]
            min_dist = np.empty(ids.size, dtype=np.float64)
            for pos, node_id in enumerate(ids):
                if int(node_id) in selected:
                    min_dist[pos] = -1.0
                    continue
                distances = squared_l2_to_matrix(graph.vectors[int(node_id)], selected_vectors)
                min_dist[pos] = float(np.min(distances))
            coverage = _normalize(min_dist)
            score = (
                self.coverage_weight * coverage
                + self.density_weight * density_norm
                + self.degree_weight * degree
            )
            score[[np.where(ids == node_id)[0][0] for node_id in selected]] = -1.0
            selected.append(int(ids[int(np.argmax(score))]))
        return selected

    def _split_candidate(
        self,
        graph: DynamicKNNGraph,
        members: Sequence[int],
        anchor: int,
    ) -> int | None:
        candidates = [int(x) for x in members if int(x) != int(anchor) and graph.alive[int(x)]]
        if not candidates:
            return None
        density = self._local_density(graph, np.asarray(candidates, dtype=np.int64))
        dist_to_anchor = squared_l2_to_matrix(graph.vectors[anchor], graph.vectors[candidates])
        score = 0.55 * _normalize(dist_to_anchor) + 0.45 * _normalize(density)
        return int(candidates[int(np.argmax(score))])

    def _local_density(self, graph: DynamicKNNGraph, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size <= 2:
            return np.ones(ids.size, dtype=np.float64)
        vectors = graph.vectors[ids]
        density = np.empty(ids.size, dtype=np.float64)
        k = min(self.density_knn, ids.size - 1)
        for pos, node_id in enumerate(ids):
            distances = squared_l2_to_matrix(graph.vectors[int(node_id)], vectors)
            distances[pos] = np.inf
            nearest = np.partition(distances, k - 1)[:k]
            density[pos] = 1.0 / (float(np.mean(np.sqrt(nearest))) + 1e-6)
        return density


def _normalize(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)
