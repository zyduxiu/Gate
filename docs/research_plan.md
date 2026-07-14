# Research Plan

## One-Sentence Claim

Dynamic graph-based ANNS should maintain entry quality as a first-class object:
under streaming base-data distribution drift, a density-aware routing-anchor
layer can reduce high-recall search cost without rebuilding the graph or
repartitioning storage.

## Non-Claim

This project does not claim that multi-entry search is new. SPTAG, DiskANN
variants, GATE, and adaptive entry-point work already make that unsafe.

This project also does not claim that dynamic ANNS updates are new. FreshDiskANN,
IP-DiskANN, SPFresh, and related systems already cover dynamic insertion and
deletion from different angles.

## Core Distinction

```text
Dynamic graph update: repair edges.
Dynamic partition update: repair centroids/postings.
Dynamic entry routing: repair where graph traversal starts.
```

The proposed layer is the third object.

## Algorithm Sketch

Build phase:

1. Build a base proximity graph.
2. Estimate local density from kNN distances.
3. Select routing anchors by coverage gain, density, and graph centrality.

Query phase:

1. Score anchors by distance to query, recent load, recent visited-node cost,
   and success rate.
2. Start graph search from the best `s` anchors.
3. Merge candidates from the shared graph traversal.

Maintenance phase:

1. Monitor anchor coverage with probe assignments. These assignments are
   telemetry only; they are not storage ownership.
2. Split an anchor when its monitored region has high density growth, high
   coverage radius, high query load, or high traversal cost.
3. Retire an anchor when it has low coverage responsibility, low query utility,
   and high overlap with neighbors.

Learned maintenance variant:

1. Treat stale-entry search cost on a small probe workload as maintenance
   telemetry, not as the final evaluation workload.
2. Learn a lightweight scorer for candidate anchors from density, distance to
   stale static anchors, inserted/deleted-region indicators, drift projection,
   and medoid distance.
3. Select anchors by combining the learned utility score with coverage, so the
   entry set remains spatially spread instead of collapsing onto only the
   hardest observed queries.

This is not the same claim as GATE-style query-time learned entry selection:
the learned object is the maintained entry set, not only the per-query choice
among a fixed hub set.

## Key Metrics

- Recall@k at high recall operating points.
- Average and p95/p99 visited nodes.
- Entry-to-result graph depth.
- p95/p99 latency.
- Anchor load skew, e.g. Gini coefficient.
- Maintenance cost per inserted/deleted vector.
- Number of anchors and memory overhead.

## Workloads

The first prototype uses synthetic clustered drift because it makes failure
modes visible.

Paper-grade workloads should include:

- SIFT1M / DEEP1M / GIST1M style static benchmarks.
- Clustered insert bursts.
- Clustered delete bursts.
- OOD query distributions.
- Hot-query skew.
- Alternating drift, where the dense region moves over time.

## Reviewer Defense

Against GATE:

> GATE learns to select entry points from candidate hubs on a mostly static
> graph. We study maintenance of the entry set itself under base-data drift.

Against SPFresh:

> SPFresh maintains storage partitions and nearest partition assignment. Our
> anchors are routing-only and do not own vectors or posting lists.

Against Dynamically Detect and Fix Hardness:

> Hardness-fix methods repair defective graph regions by adding edges. Our
> primary mechanism is to keep traversal from starting in stale regions.

Against FreshDiskANN/IP-DiskANN:

> These methods maintain dynamic graph/index correctness and quality. We isolate
> the entry-routing layer and measure the cost of entry degradation separately.
