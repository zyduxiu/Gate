from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .anchors import BaseRouter
from .graph import DynamicKNNGraph
from .metrics import percentile, recall_at_k, safe_mean


@dataclass(frozen=True)
class EvalSummary:
    router: str
    phase: str
    recall_at_k: float
    avg_visited: float
    p95_visited: float
    avg_depth: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    load_gini: float
    anchors: int
    queries: int

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def evaluate_router(
    graph: DynamicKNNGraph,
    router: BaseRouter,
    queries: Iterable[np.ndarray],
    phase: str,
    k: int = 10,
    ef: int = 80,
    s: int | None = None,
    early_stop: bool = True,
) -> EvalSummary:
    recalls: list[float] = []
    visited: list[float] = []
    depths: list[float] = []
    latencies: list[float] = []
    query_count = 0

    for query in queries:
        query_count += 1
        truth, _ = graph.exact_search(query, k=k)
        entries = router.select(query, graph, s=s)
        result = graph.search(query, entries, topk=k, ef=ef, early_stop=early_stop)
        recall = recall_at_k(result.ids, truth, k)
        router.after_query(query, graph, result, recall)
        recalls.append(recall)
        visited.append(float(result.visited))
        depths.append(float(result.max_depth))
        latencies.append(float(result.latency_ms))

    return EvalSummary(
        router=router.name,
        phase=phase,
        recall_at_k=safe_mean(recalls),
        avg_visited=safe_mean(visited),
        p95_visited=percentile(visited, 95),
        avg_depth=safe_mean(depths),
        avg_latency_ms=safe_mean(latencies),
        p95_latency_ms=percentile(latencies, 95),
        p99_latency_ms=percentile(latencies, 99),
        load_gini=router.load_gini(),
        anchors=len(router.anchor_ids()),
        queries=query_count,
    )
