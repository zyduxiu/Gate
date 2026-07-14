# Online Entry Maintenance And Graph Repair Ablation

## What Was Added

The real NSG dynamic runner now has an `online_density` mode:

- Starts from the pre-drift `static_density` anchor set.
- Deletes dead anchors when their backing vector is deleted.
- Processes inserted vectors in batches.
- In each batch, samples candidate inserted vectors and scores them by:
  - distance from the current anchor set, and
  - local density estimated from the KNN graph.
- Adds high-value candidates and retires redundant anchors to keep the entry-set size fixed.

This is different from the previous `dynamic_density` result:

- `dynamic_density` is a batch refresh proxy using the final alive distribution.
- `online_density` is maintained incrementally from the stale entry set.

The runner also has a `--repair-degree` switch for a local graph-repair ablation. Repair is
restricted to inserted nodes and nodes whose adjacency included deleted nodes. It is not a full
rebuild.

## Commands

No graph repair:

```powershell
python experiments\run_nsg_dynamic_entry.py `
  --manifest data\sift100k_dynamic\manifest.json `
  --out-dir results\sift100k_online_nsg_entry_no_repair `
  --methods nsg_ep,static_density,dynamic_density,online_density `
  --search-l 80,240 --entries 8 --repair-degree 0 `
  --maintain-batch 1000 --maintain-changes 4 --maintain-sample 256 `
  --per-query-l 240 --timeout 1800
```

Local repair:

```powershell
python experiments\run_nsg_dynamic_entry.py `
  --manifest data\sift100k_dynamic\manifest.json `
  --out-dir results\sift100k_online_nsg_entry_local_repair32 `
  --methods nsg_ep,static_density,dynamic_density,online_density `
  --search-l 240 --entries 8 --repair-degree 32 `
  --maintain-batch 1000 --maintain-changes 4 --maintain-sample 256 `
  --per-query-l 240 --timeout 1800
```

## Main Results

SIFT100k dynamic no-rebuild, `search_l=240`:

| topology | entry method | Recall@10 | avg distance comps | p99 latency |
|---|---:|---:|---:|---:|
| no repair | `nsg_ep` | 0.9997 | 3877.51 | 3611 us |
| no repair | `online_density` | 0.9997 | 3824.03 | 3087 us |
| no repair | `dynamic_density` batch proxy | 0.9997 | 3829.21 | 6766 us |
| local repair32 | `nsg_ep` | 0.9999 | 4011.50 | 3422 us |
| local repair32 | `online_density` | 0.9999 | 3952.03 | 3803 us |
| local repair32 | `dynamic_density` batch proxy | 0.9999 | 3954.98 | 3569 us |

Online maintenance cost:

- `20` maintenance rounds for `20,000` inserted vectors.
- `100` anchors added, `80` retired, `20` dead anchors removed.
- `4,943,200` maintenance distance computations.
- About `165` distance computations per update if amortized over `20,000` inserts + `10,000` deletes.

## Hard-Query Result

Hardness bins use stale `nsg_ep` distance computations at `search_l=240`.

No repair, hardest quartile `Q4`:

| method | Recall@10 | avg distance comps | p99 latency |
|---|---:|---:|---:|
| `nsg_ep` | 0.9996 | 5065.54 | 3754 us |
| `static_density` | 0.9996 | 5034.89 | 7890 us |
| `dynamic_density` batch proxy | 0.9996 | 5011.63 | 7500 us |
| `online_density` | 0.9996 | 5008.12 | 3258 us |

Local repair32, hardest quartile `Q4`:

| method | Recall@10 | avg distance comps | p99 latency |
|---|---:|---:|---:|
| `nsg_ep` | 0.9996 | 5175.22 | 3800 us |
| `static_density` | 0.9996 | 5137.06 | 3975 us |
| `dynamic_density` batch proxy | 0.9996 | 5118.29 | 3752 us |
| `online_density` | 0.9996 | 5115.14 | 4350 us |

## Interpretation

This closes two reviewer holes:

1. The real NSG experiment now has a true online entry-set maintenance path, not only a final
   batch refresh proxy.
2. Entry maintenance and graph repair are not the same mechanism. Local graph repair changes
   edge quality; online density anchors change where traversal enters the same repaired graph.

The current result should be stated conservatively:

- Online entry maintenance consistently reduces average distance computations versus stale
  `nsg_ep` at high recall.
- It is lightweight relative to graph repair/rebuild: the entry layer keeps only `128` anchors
  and pays about `165` distance computations per update in this run.
- It composes with local graph repair in average search cost: under repair32, `online_density`
  reduces average distance computations from `4011.50` to `3952.03`.
- Tail latency is implementation-sensitive in this unoptimized C++ runner. The strongest p99
  result is the no-repair hard-query regime, where `online_density` reduces Q4 p99 from
  `3754 us` to `3258 us`.

Raw artifacts:

- `results/sift100k_online_nsg_entry_no_repair/summary.csv`
- `results/sift100k_online_nsg_entry_no_repair/hard_bins_l240.csv`
- `results/sift100k_online_nsg_entry_local_repair32/summary.csv`
- `results/sift100k_online_nsg_entry_local_repair32/hard_bins_l240.csv`
- `results/sift100k_online_entry_topology_2x2.csv`
