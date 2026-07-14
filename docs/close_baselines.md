# Close Baseline Status

Date: 2026-07-11

This file tracks baselines that directly challenge the entry-layer claim:
GATE, DiskANN++, and Dynamically Detect and Fix Hardness.

## Summary

| Baseline | Status | What is included now | Remaining blocker |
|---|---|---|---|
| GATE | Buildable with static adapter + local two-tower proxy | Zenodo artifact downloaded to `third_party/GATE`; C++ `test_nsg_index`, `test_gate_search`, and `test_gate_cos_navigate` build in WSL; `experiments/run_gate_adapter.py` runs static centroid hubs; `experiments/train_gate_two_tower.py` trains a GATE-like query/hub selector | Still not the original artifact's Graph2Vec + two-tower pipeline |
| DiskANN++ | Paper-only | Paper is tracked as required comparison | No public code link found on arXiv page; needs author artifact or reimplementation |
| Hardness / RFix / NGFix | Official artifact executed | `third_party/NGFix` cloned and built; `experiments/run_ngfix_baseline.py` runs HNSW bottom build, NGFix-AKNN repair, and search on SIFT100k dynamic final base | Current machine lacks AVX-512, so run uses a scalar distance compatibility build; same-graph NSG/Vamana RFix remains a proxy |
| Steiner-hardness | Cloned diagnostic tool | `third_party/Steiner-hardness` can provide query-hardness measurement and unbiased hard workloads | Not an RFix/NGFix implementation; needs efanna/MRNG pipeline adaptation before use on our dynamic data |

## Reviewer Taxonomy

The clean paper position is not that entry selection is new. The new object is
online maintenance of the entry candidate set under base-data drift. The closest
baselines should therefore be separated by what they maintain.

| Work group | Main maintained/optimized object | Directness to our claim | How to compare fairly |
|---|---|---|---|
| GATE | Query-time choice among candidate hub entries; learned hub/query embeddings | Close, but not identical | Compare as a static or learned entry-selector baseline, then show that its hub set can become stale under insert/delete drift |
| DiskANN++ | Query-sensitive entry vertex and SSD page layout for DiskANN-style search | Close, but mostly static/query-time | Compare or reimplement query-sensitive entry choice; emphasize that our layer maintains the candidate entry set online |
| Adaptive entry-point selection / multi-entry methods | Better query initialization from fixed candidate entries | Conceptual predecessor | Use as static entry-choice baselines; do not claim multi-entry as novelty |
| Hardness / RFix / NGFix | Graph reachability and local graph defects around hard query regions | Partly overlapping diagnosis, mostly orthogonal mechanism | Use hard-query bins and 2x2 ablation: fixed/dynamic entry crossed with no repair/local repair/RFix-like repair |
| Steiner-hardness | Query hardness measurement and workload generation | Diagnostic, not a competitor | Use to generate unbiased hard-query workloads and report per-bin gains |
| FreshDiskANN / IP-DiskANN / dynamic graph repair | Graph/index update under insert/delete | Orthogonal system baseline | Show entry maintenance adds value with the same repaired or unrepaired graph |
| SPANN / SPFresh | Coarse partitions, posting lists, and incremental rebalancing | Orthogonal storage/partition baseline | Compare end-to-end dynamic update cost; clarify anchors do not own storage or move vectors |
| SPTAG | Tree/partition-seeded graph search | Important industrial seed baseline | Treat as static multi-seed routing; our contribution is online drift maintenance of routing anchors |

Paper-safe wording:

> Prior work improves either query-time entry selection from a mostly fixed
> candidate set, graph/region repair after a bad route is observed, or dynamic
> maintenance of graph/partition structures. We isolate a different failure
> mode: under insert/delete drift, the entry candidate set itself becomes stale.
> We maintain this entry overlay online and show it composes with graph repair.

Generate machine-readable status:

```powershell
python experiments\close_baseline_status.py
```

Output:

```text
results/close_baselines/status.json
```

## GATE

Source:

```text
https://zenodo.org/records/15523071
```

Local path:

```text
third_party/GATE/jacksondca-gate-78334b4
```

Build:

```powershell
.\scripts\build_external_baselines.ps1 -Baseline gate -UseWsl
```

Built executables:

```text
third_party/GATE/jacksondca-gate-78334b4/build/tests/test_nsg_index
third_party/GATE/jacksondca-gate-78334b4/build/tests/test_gate_search
third_party/GATE/jacksondca-gate-78334b4/build/tests/test_gate_cos_navigate
```

Local portability patches:

- `tests/test_gate_search.cpp`: fixed argument count check from `argc != 7`
  to `argc != 9`; the program uses `argv[8]`.
- `tests/CMakeLists.txt`: made `tcmalloc` optional.
- `tests/CMakeLists.txt`: skipped `test_nsg_search.cpp` because this Zenodo
  snapshot has undefined training-helper variables in that file.

Why this baseline matters:

- GATE is a lightweight module on top of graph indexes.
- It extracts hub nodes as candidate entry points.
- It trains a two-tower model for query-aware entry selection.

Difference from our claim:

- GATE learns to choose entries from a mostly static candidate/hub set.
- Our claim is dynamic entry-set maintenance under base-data insert/delete
  drift: anchors split/merge/reweight as density and load change.
- Therefore, GATE is the closest entry-overlay baseline, but it does not by
  itself answer entry staleness under dynamic data updates.

Paper-faithful run blocker:

- GATE search needs `HUB_PATH` and `EMB_PATH`.
- The artifact README pipeline requires k-means hub extraction, subgraph
  generation, graph embedding, two-tower training, and inference.
- The artifact's Python scripts contain hard-coded globals and CUDA-only
  assumptions, so we need a small adapter before we can run it on
  `data/fair_10k`.

GATE-like two-tower proxy now included:

```powershell
python experiments\train_gate_two_tower.py --manifest data\sift100k_dynamic\manifest.json --anchor-file results\entry_topology_backend_matrix\anchors\gate_initial_hub.txt --out-dir results\entry_topology_backend_matrix\gate_two_tower --epochs 120 --batch-size 128
python experiments\run_nsg_dynamic_entry.py --manifest results\entry_topology_backend_matrix\manifest_with_gate_variants.json --out-dir results\entry_topology_backend_matrix\nsg\gate_two_tower_no_repair --methods nsg_ep,gate_initial_hub,query_entries,online_density --query-entries-ivecs results\entry_topology_backend_matrix\gate_two_tower\gate_two_tower_entries.ivecs --search-l 240 --entries 8 --repair-degree 0 --timeout 2400 --per-query-l 240
```

Result at `search_l=240`:

| Method | Recall@10 | Avg distance comps | p99 latency |
|---|---:|---:|---:|
| `nsg_ep` | 0.9997 | 3877.51 | 4000 us |
| `gate_initial_hub` | 0.9998 | 3774.31 | 2981 us |
| `query_entries` two-tower proxy | 0.9997 | 3812.39 | 3315 us |
| `online_density` | 0.9997 | 3824.03 | 3435 us |

Interpretation:

- GATE-style hubs are a strong close baseline.
- The local two-tower selector is closer to GATE than ridge regression, but it
  is still a proxy because it does not reproduce the artifact's Graph2Vec
  subgraph embedding pipeline.
- This strengthens the paper position: our claim should be online maintenance
  of the entry candidate set under drift, not universal superiority over
  query-aware GATE-style entry selection.

Static adapter now included:

```powershell
python experiments\run_gate_adapter.py --manifest data\fair_10k\manifest.json --search-l 80,120,240 --timeout 1200
python experiments\summarize_fair_results.py --fair-dir results\fair_10k --hnsw-dir results\hnsw_anchor_10k --gate-dir results\gate_adapter_10k
```

Observed on the dynamic-drift `fair_10k` dataset:

```text
method                     search_l  Recall@10  QPS
static_centroid_hub_entry  80        0.4653     389.812
static_centroid_hub_entry  120       0.4509     524.521
static_centroid_hub_entry  240       0.7349     367.680
```

This is an executable GATE-style baseline, not a paper-faithful GATE run. It
uses generated two-level centroid hubs and GATE's C++ search path, but it does
not train GATE's two-tower model. Its role is to test a static hub-entry
overlay under dynamic data drift.

## DiskANN++

Source:

```text
https://arxiv.org/abs/2310.00402
```

Why it matters:

- It explicitly targets DiskANN's long route from a static graph-central entry
  vertex to the query neighborhood.
- It proposes query-sensitive entry vertex selection and an SSD page-search
  layout.

Current status:

- No public code link was found on the arXiv page.
- It remains a paper-only baseline until we either obtain the authors' artifact
  or implement its query-sensitive entry vertex strategy.

Difference from our claim:

- DiskANN++ optimizes query-sensitive entry and disk page layout for static
  DiskANN-style search.
- We focus on online maintenance of the entry set itself under dynamic
  insert/delete drift.

## Dynamically Detect and Fix Hardness

Source:

```text
https://arxiv.org/abs/2510.22316
```

Why it matters:

- It divides graph search into two stages: entry point to query vicinity, and
  search inside the vicinity.
- RFix repairs reachability from entry to vicinity; NGFix repairs local graph
  defects in dense query regions.

Current status:

- Official repository is cloned at `third_party/NGFix`.
- The full-precision artifact builds in WSL after local compatibility patches
  that disable RaBitQ targets.
- This CPU/WSL environment does not expose `avx512f`, so the executed run uses
  `NGFIX_PORTABLE_DISTANCE=ON`, which keeps the NGFix/RFix graph logic but
  replaces AVX-512 distance kernels with scalar kernels.

Run:

```powershell
python experiments\run_ngfix_baseline.py --skip-build --timeout 3600
```

Outputs:

```text
results/ngfix_official_sift100k/ngfix_official_summary.csv
results/ngfix_official_sift100k/logs/build_hnsw_bottom.log.json
results/ngfix_official_sift100k/logs/build_hnsw_ngfix_aknn.log.json
results/ngfix_official_sift100k/logs/search_hnsw_ngfix.log.json
```

Observed on SIFT100k dynamic final base:

| Backend | Build mode | ef | Recall@10 | Avg distance comps | Avg latency |
|---|---|---:|---:|---:|---:|
| HNSW-NGFix official artifact | portable distance | 10 | 1.0000 | 326.73 | 49.6 us |
| HNSW-NGFix official artifact | portable distance | 100 | 1.0000 | 1501.26 | 361.9 us |
| HNSW-NGFix official artifact | portable distance | 300 | 1.0000 | 3405.57 | 908.6 us |

Difference from our claim:

- Hardness/RFix/NGFix repairs graph regions and navigability.
- Our method maintains a density-aware entry overlay: it changes where the
  query enters, rather than repairing the graph path after entry.
- Our 2x2 ablation explicitly tests this distinction: topology repair alone
  does not rescue stale fixed entries, while dynamic anchors remain the main
  effect and compose with topology repair.

Paper-safe caveat:

- These NGFix numbers are an official-artifact system baseline, not a
  same-graph NSG/Vamana ablation.
- The portable distance build is for hardware compatibility on this machine;
  for final paper numbers, rerun the same wrapper on an AVX-512 machine with
  `--portable-distance off`.

## Steiner-hardness

Source:

```text
https://github.com/CaucherWang/Steiner-hardness
```

Local path:

```text
third_party/Steiner-hardness
```

Commit:

```text
823737a085bb6501847f486066e0b23acf725617
```

Why it matters:

- It estimates query cost and measures query hardness for graph-based ANN
  indexes such as HNSW, NSG, and KGraph.
- It can generate less biased hard-query workloads for stress-testing indexes.
- It gives us a good diagnostic axis: entry maintenance should help not only
  average queries, but especially queries whose route from entry to vicinity is
  expensive.

Why it is not the SIGMOD 2026 Hardness/RFix/NGFix baseline:

- Steiner-hardness measures hardness and builds workloads.
- RFix/NGFix dynamically repairs graph reachability/local neighborhoods.
- Therefore, Steiner-hardness is useful for workload construction and analysis,
  but it is not a graph-repair competitor.

Adapter status:

- Cloned and inspected.
- Needs `efanna_graph`, MRNG/reverse-graph preprocessing, and config-path
  cleanup before running on `data/fair_10k`.
- Best next use: generate or score hard query subsets, then compare fixed
  medoid, static density, GATE static hubs, and dynamic density anchors on the
  same hard-query bins.

## Experiment Matrix

Minimum final paper matrix:

```text
Entry baselines:
- fixed medoid
- static density anchors
- random/farthest anchors
- GATE static centroid hub adapter
- paper-faithful GATE, once learned HUB_PATH/EMB_PATH generation is complete
- DiskANN++ query-sensitive entry, if code/artifact or reimplementation is available

Topology baselines:
- no repair
- local repair
- NSG rebuilt final index
- DiskANN/Vamana rebuilt final index
- SPTAG tree-seeded graph search
- Hardness/RFix/NGFix, if code/artifact or reimplementation is available

Combination:
- fixed entry + no repair
- fixed entry + topology repair
- dynamic anchors + no repair
- dynamic anchors + topology repair
```
