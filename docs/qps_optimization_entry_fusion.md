# QPS Optimization And Entry Fusion Notes

## Runner Fix

`tools/nsg_dynamic_entry_runner.cpp` now reuses per-query marker buffers through
timestamp workspaces instead of allocating and clearing large `seen` arrays for
every query.

The p99 latency timer also starts before entry selection and pool
initialization. This makes latency consistent with single-thread QPS, which is
closer to the GATE-style evaluation protocol.

## Main Workload

All results here use the real SIFT insert-heavy stress workload:

- Manifest: `data/sift_stress_insert80_queries/manifest.json`
- Backend: same no-rebuild dynamic NSG graph
- Initial graph: `results/sift_stress_drift_dynamic_nsg/initial.nsg`
- Queries: 453 queries whose top-10 exact neighbors are at least 80% inserted
  points
- Metric emphasized here: Recall@100 vs QPS

## Three-Run Result After Runner Fix

Combined 3-run aggregate:

`results/stress_insert80_workspace_opt_3repeats_combined/aggregate_recall1_recall100_3runs_with_hybrid.csv`

Plots:

- `results/stress_insert80_workspace_opt_3repeats_combined/recall1_recall100_mean_pareto_with_hybrid.png`
- `results/stress_insert80_workspace_opt_3repeats_combined/recall1_recall100_median_pareto_with_hybrid.png`

At Recall@100:

| Recall region | Best observed method | Takeaway |
|---|---|---|
| ~0.991 | `online_hub_density` / hybrid | Maintained entries reduce stale-entry work. |
| ~0.9946 | `online_hub_density` | Strongest mean/median QPS in the 3-run table. |
| ~0.996 | hybrid and `online_hub_density` are close | Fusion can help, but system noise is still visible. |
| >=0.998 | `online_hub_density` / hybrid remain competitive | Gains are modest; needs 10-run confirmation. |

## Hybrid Selector Sweep

Sweep output:

`results/stress_insert80_workspace_opt_hybrid_selector_sweep/summary.csv`

Top single-run configurations:

| L | Best hybrid | Recall@100 | QPS | p99 us |
|---:|---|---:|---:|---:|
| 240 | `query_entries_hybrid_gate_first_n2_g2` | 0.994592 | 643.907 | 2756.90 |
| 280 | `query_entries_hybrid_gate_first_n1_g3` | 0.996026 | 582.794 | 2920.07 |
| 320 | `query_entries_hybrid_gate_first_n2_g2` | 0.997020 | 501.922 | 3369.61 |

Interpretation:

`gate_first_n1_g3` and `gate_first_n2_g2` are better candidates than the older
`gate_first_n3_g1`. The stronger story is not "use only our maintained anchors";
it is "use GATE-style query selection first, then add fresh maintained anchors
to cover drifted regions."

## Current Paper Claim

Conservative claim:

Dynamic entry-set maintenance is complementary to GATE-style query-aware entry
selection. On inserted-region-heavy dynamic queries, maintaining a fresh
density+hubness-aware candidate entry set reduces stale-entry search work and
can improve QPS/tail latency. GATE-style selectors can be layered over this
maintained entry set.

What is still needed:

- 10-run confirmation for `gate_first_n1_g3` and `gate_first_n2_g2`.
- Paper-faithful GATE entries regenerated for the insert80 query subset.
- Same trend on at least one non-SIFT dataset.
- DiskANN/Vamana no-rebuild dynamic backend, not only attachability evidence.
