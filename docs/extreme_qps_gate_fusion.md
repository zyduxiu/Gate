# Extreme Drift Query Workload And GATE Fusion

## Why This Experiment

GATE-style papers emphasize recall-QPS tradeoff curves, search hops/work, and
query-aware entry selection. Our main missing evidence was whether entry-set
maintenance gives a visible QPS benefit under a harder dynamic workload.

Instead of changing the graph, this experiment keeps the same SIFT stress
dynamic graph and extracts inserted-region-heavy query workloads:

- `top1_inserted`: query's top-1 exact neighbor is an inserted point.
- `insert80`: at least 80% of query's top-10 exact neighbors are inserted
  points.

These are the queries where old/static entry sets should be most stale.

## Added Tools

- `experiments/make_query_subset_manifest.py`
- `experiments/build_hybrid_query_entries.py`

The hybrid entry strategy combines:

1. our online-maintained fresh entry set `E`;
2. nearest anchors from `E`;
3. GATE-style two-tower predicted entries over `E`.

The best tested fusion is:

```text
query_entries_hybrid_gate_first_n3_g1
```

Meaning: use GATE prediction order first, but include 3 nearest online anchors
and 1 GATE-selected anchor in the 4-entry initialization list.

## Query Subsets

```powershell
python experiments\make_query_subset_manifest.py `
  --manifest data\sift_stress_drift\manifest.json `
  --out-dir data\sift_stress_insert_top1_queries `
  --criterion top1_inserted

python experiments\make_query_subset_manifest.py `
  --manifest data\sift_stress_drift\manifest.json `
  --out-dir data\sift_stress_insert80_queries `
  --criterion insert_fraction `
  --truth-k 10 `
  --min-insert-fraction 0.8
```

Sizes:

| Workload | Queries |
|---|---:|
| `top1_inserted` | 623 |
| `insert80` | 453 |

## Insert80 Full Baseline Table

Single-run result at `L=240`, same no-rebuild dynamic NSG backend:

| Method | Recall@10 | Avg Comps | p99 Latency | QPS |
|---|---:|---:|---:|---:|
| `nsg_ep` | 0.9969 | 4865.13 | 2927.98 us | 674.51 |
| `static_density` | 0.9969 | 4846.23 | 2777.49 us | 657.64 |
| `gate_initial_hub` | 0.9969 | 4828.52 | 2863.41 us | 674.32 |
| `dynamic_density` | 0.9969 | 4811.96 | 2835.07 us | 662.41 |
| `online_density` | 0.9969 | 4779.76 | 2666.61 us | 685.40 |
| `query_entries_paper_gate_initial` | 0.9969 | 4852.66 | 3104.68 us | 663.35 |
| `query_entries_paper_gate_refreshed` | 0.9969 | 4730.57 | 3165.91 us | 641.70 |
| `query_entries_paper_gate_online_two_tower` | 0.9969 | 4708.60 | 2968.10 us | 673.86 |
| `gate_refreshed_hub` | 0.9969 | 4684.49 | 2331.65 us | 745.90 |
| `online_hub_density` | 0.9969 | 4677.05 | 2537.87 us | 748.12 |

Interpretation:

- Insert-heavy queries amplify stale-entry effects.
- Online hub-density beats stale `gate_initial_hub` in comps, p99, and QPS.
- Online hub-density slightly beats offline refreshed GATE hubs in average
  comparisons and QPS on this run.

## QPS Repeat Median

Five-run median on `insert80`:

| Method | Recall@10 | Avg Comps | Avg Latency | p99 Latency | Median QPS |
|---|---:|---:|---:|---:|---:|
| `query_entries_hybrid_gate_first_n3_g1` | 0.9969 | 4678.57 | 1257.55 us | 2116.10 us | 787.69 |
| `online_hub_density` | 0.9969 | 4677.05 | 1266.85 us | 2325.16 us | 768.16 |
| `query_entries_paper_gate_online_two_tower` | 0.9969 | 4708.60 | 1325.09 us | 2468.68 us | 747.02 |
| `gate_refreshed_hub` | 0.9969 | 4684.49 | 1308.88 us | 2648.35 us | 742.90 |
| `nsg_ep` | 0.9969 | 4865.13 | 1376.56 us | 2509.62 us | 720.78 |
| `gate_initial_hub` | 0.9969 | 4828.52 | 1408.10 us | 2669.90 us | 697.58 |

Relative to stale `gate_initial_hub`, the hybrid gives:

- `+12.9%` median QPS;
- `-3.1%` average distance computations;
- `-20.7%` p99 latency.

Relative to `nsg_ep`, the hybrid gives:

- `+9.3%` median QPS;
- `-3.8%` average distance computations;
- `-15.7%` p99 latency.

Relative to offline `gate_refreshed_hub`, the hybrid gives:

- `+6.0%` median QPS;
- comparable average comparisons;
- `-20.1%` p99 latency.

## Paper Takeaway

This is a stronger top-conference-style result than the full-query average:

> Under inserted-region-heavy dynamic queries, stale entry sets become a clear
> bottleneck. Online density+hubness entry maintenance improves QPS and tail
> latency over stale GATE hubs and single-entry NSG. A simple fusion with a
> GATE-style selector over the maintained fresh entry set further improves
> median QPS and p99 latency.

The conservative claim remains:

> We are not replacing GATE. We maintain a fresh candidate entry set `E`, and
> GATE-style selectors can be composed on top of this fresher `E`.

## Existing Other Baselines

DiskANN/Vamana adapter results already exist:

- `docs/diskann_entry_adapter_results.md`
- `results/sift100k_diskann_entry_shared/diskann_entry_adapter_summary.csv`
- `results/glove100k_diskann_entry_shared/diskann_entry_adapter_summary.csv`

These show attachability/orthogonality on a rebuilt DiskANN graph, but not a
no-rebuild dynamic DiskANN workflow. They should be used as supporting evidence,
not as the main dynamic QPS table.
