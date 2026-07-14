# GATE-Aligned Evaluation Protocol

This note translates the GATE experimental setup into the dynamic entry-set
maintenance setting used by this project. The goal is to make the comparison
reviewer-defensible without pretending that our problem is identical to GATE's
static entry-selection problem.

## What To Align With GATE

GATE reports recall-QPS trade-off curves on real datasets, with single-threaded
query execution and repeated runs to reduce system noise. We should use the same
surface:

- Main plot: `Recall@10` versus `QPS`; top-right is better.
- Main table: high-recall operating points, especially `Recall@10 >= 0.99`.
- Repeat policy: run each query benchmark 10 times; report mean, median, min,
  max, and standard deviation for QPS and p99 latency.
- Query execution: single-threaded query search when comparing QPS.
- Search work: report distance computations and expanded nodes as supporting
  diagnostics.
- Update overhead: report online entry-maintenance time and maintenance distance
  computations separately from query QPS.

GATE also reports routing hops. Our current NSG dynamic runner reports
`avg_expanded`, not exact hop-depth. In paper text, call this a search-work/path
proxy unless the runner is extended to track true graph-hop depth.

## Datasets

GATE uses `Gist1M`, `Tiny5M`, `Sift10M`, `Laion3M`, and `Text2Image10M`.
For our paper, the clean target split is:

| Role | Datasets |
|---|---|
| Same-family sanity | SIFT dynamic stress, SIFT larger subset / Sift10M if storage permits |
| Modality coverage | GIST, GloVe/Text2Image-like embeddings, LAION/CLIP-like embeddings |
| Current local evidence | SIFT stress, GloVe100k DiskANN adapter, existing multidataset online-maintenance runs |

If we cannot run GATE's full dataset scale locally, we should still match the
protocol: multiple real datasets, high-recall curves, single-thread query QPS,
and 10-run repeats.

## Baseline Groups

Use two tables. The first table is a GATE-style static ANNS table. The second is
our dynamic no-rebuild table.

### Static / Rebuilt ANNS Baselines

These answer: if every method is allowed to build or rebuild a good index, how
competitive is the search layer?

- `HNSW`
- `NSG`
- `NSSG` if buildable
- `DiskANN/Vamana`
- `SPTAG`
- `GATE` official artifact when reproducible
- `GATE-style two-tower` reimplementation when official artifact cannot run
- `LSH-APG` / adaptive-entry baseline if code can be obtained

This table is not where our main novelty should win; HNSW and refreshed static
methods are very strong.

### Dynamic No-Rebuild Entry Baselines

These answer our actual claim: after insert/delete drift, does the query-facing
entry set become stale, and can we maintain it cheaply?

- `nsg_ep`: original NSG/root entry.
- `fixed_medoid`: static center entry.
- `static_density`: density anchors selected before drift.
- `dynamic_density`: final-refresh proxy; use as an upper/proxy, not online.
- `gate_initial_hub`: GATE-style hub set selected before drift.
- `gate_refreshed_hub`: offline refreshed hub upper bound after drift.
- `query_entries_gate_initial`: GATE-style selector over stale hubs.
- `query_entries_gate_refreshed`: GATE-style selector over refreshed hubs.
- `online_density`: our online density/coverage maintenance.
- `online_hub_density`: our online density + hubness maintenance.
- `query_entries_gate_online_E`: GATE-style selector over our maintained entry set.
- `hybrid_gate_online_E`: fused nearest-online anchors plus GATE-predicted entries.

This table should be run on both `no repair` and `local graph repair` backends
when possible to show that entry maintenance is complementary to graph repair.

## Metrics To Report

| Metric | Role | Notes |
|---|---|---|
| `Recall@10` | Accuracy | Primary x-axis / operating point filter |
| `QPS` | Main efficiency metric | GATE's primary search-efficiency metric |
| `avg_latency_us`, `p95_latency_us`, `p99_latency_us` | Tail latency | Especially important for high-recall dynamic workloads |
| `avg_distance_computations` | Hardware-independent search work | More stable than wall-clock QPS on this workstation |
| `avg_expanded` | Search path/work proxy | Do not call it routing hops unless hop-depth is implemented |
| `maintenance_seconds` | Update overhead | Our online cost; refreshed baselines should include refresh/retrain cost if measured |
| `maintenance_distance_computations` | Update work | Useful for comparing with refresh/retrain |

Helper script:

```powershell
python experiments\summarize_gate_aligned_results.py `
  --inputs results\sift_stress_insert80_qps_repeats\median_with_hybrid_summary.csv `
  --dataset sift1m_stress_streaming_drift `
  --workload insert80 `
  --default-search-l 240 `
  --default-entries 4 `
  --out results\gate_aligned_insert80_summary.csv

python experiments\summarize_gate_aligned_results.py `
  --inputs results\stress_qps_repeats\median_summary.csv `
  --dataset sift1m_stress_streaming_drift `
  --workload full_repeat_median `
  --default-search-l 240 `
  --default-entries 4 `
  --out results\gate_aligned_full_repeat_summary.csv
```

Current generated summaries:

- `results/gate_aligned_insert80_summary.csv`
- `results/gate_aligned_full_repeat_summary.csv`

## Fairness Rules

- Keep the backend graph fixed within each dynamic no-rebuild comparison.
- Vary only the entry initialization strategy unless explicitly testing graph
  repair composition.
- Do not compare official GATE artifact QPS directly against our O3 runner QPS
  unless the binaries and hardware path are matched.
- Label `gate_refreshed_hub` and `dynamic_density` as offline refresh proxies or
  upper bounds.
- Separate query-time QPS from entry-maintenance/update cost.
- Use high-recall regions (`0.99+`) and inserted-region-heavy/hard-query bins,
  because entry staleness is diluted in easy full-query averages.

## Current Best Paper-Safe Claim

The current strongest result is the inserted-region-heavy SIFT stress workload:

- Same no-rebuild NSG backend.
- Stale GATE hubs are selected before drift.
- Our entry set is maintained online without final refresh.
- A hybrid GATE selector over our maintained `E` reaches the best median QPS in
  the current table.

Safe wording:

> GATE improves query-sensitive selection from a candidate hub set. Our method
> maintains that candidate entry set under insert/delete drift. On inserted-heavy
> dynamic queries, maintaining `E` improves QPS and p99 latency over stale GATE
> hubs, and GATE-style selectors can be composed on top of the maintained `E`.

Unsafe wording to avoid:

> We beat GATE in general.

GATE and our method are partly overlapping at the entry layer, but the central
problem is different: GATE chooses entries from an available candidate set,
whereas we maintain that set as the data distribution changes.
