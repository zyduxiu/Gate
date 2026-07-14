# Paper-Aligned GATE Status

This note separates three different GATE-related comparisons.

## What Was Added

New scripts:

- `experiments/train_gate_paper_selector.py`
- `experiments/run_gate_paper_comparison.py`
- `experiments/run_gate_official_selector.py`

The paper-aligned selector follows the GATE pipeline more closely than the
earlier proxy:

1. use a fixed candidate hub/entry set;
2. sample each hub's local graph neighborhood as a topological feature;
3. generate positive/negative query samples from actual per-anchor search cost
   when `--anchor-costs-csv` is available;
4. train a fusion hub tower with triplet cosine loss;
5. optionally train a query tower and export per-query entries.

The C++ dynamic runner now supports:

```text
--out-anchor-costs-csv
--anchor-cost-query-limit
```

This lets us produce GATE-like query-aware labels from the exact same dynamic
NSG search path used in evaluation.

## Dynamic No-Rebuild Result

Command:

```powershell
python experiments\run_gate_paper_comparison.py `
  --manifest data\sift_stress_drift\manifest.json `
  --initial-nsg results\sift_stress_drift_dynamic_nsg\initial.nsg `
  --out-dir results\gate_paper_costlabel_sift_stress `
  --search-l 240 `
  --entries 4 `
  --epochs 40 `
  --train-query-fraction 0.6 `
  --timeout 7200
```

Main result at `L=240`, same dynamic NSG backend:

| Method | Recall@10 | Avg Distance Computations | p99 Latency | QPS |
|---|---:|---:|---:|---:|
| `nsg_ep` | 0.9964 | 4645.13 | 2459.75 us | 749.33 |
| `gate_initial_hub` | 0.9964 | 4600.35 | 2414.56 us | 882.13 |
| `gate_refreshed_hub` | 0.9963 | 4482.11 | 2323.84 us | 867.93 |
| `online_hub_density` | 0.9963 | 4484.38 | 2886.35 us | 774.44 |
| `query_entries_paper_gate_gate_initial_hub` | 0.9965 | 4629.56 | 6063.18 us | 629.60 |
| `query_entries_paper_gate_gate_refreshed_hub` | 0.9963 | 4523.25 | 2296.49 us | 822.13 |
| `query_entries_paper_gate_online_hub_density_E` | 0.9963 | 4519.41 | 2405.62 us | 682.30 |

Two-tower selector over the maintained online entry set:

```powershell
python experiments\train_gate_paper_selector.py `
  --manifest data\sift_stress_drift\manifest.json `
  --anchor-file results\gate_paper_costlabel_sift_stress\anchors\online_hub_density_maintained_E.txt `
  --anchor-costs-csv results\gate_paper_costlabel_sift_stress\anchor_costs\online_hub_density_E_costs.csv `
  --out-dir results\gate_paper_querytower_sift_stress\online_hub_density_E `
  --entries 4 `
  --epochs 80 `
  --train-query-fraction 0.6 `
  --query-tower
```

Result:

| Method | Recall@10 | Avg Distance Computations | p99 Latency | QPS |
|---|---:|---:|---:|---:|
| `query_entries_paper_gate_two_tower_online_E` | 0.9964 | 4512.77 | 2626.06 us | 613.26 |

Interpretation:

- Real search-cost labels improve the GATE-style selector versus the earlier
  hop-proxy labels.
- The learned selector over the maintained online set is close to the refreshed
  GATE upper bound, but it still does not beat direct nearest-anchor routing on
  this stress workload.
- This supports the paper positioning: the main contribution is maintaining a
  fresh candidate entry set `E`; GATE-style query selection is composable but is
  not the dominant bottleneck in this setting.

## Official GATE C++ Sanity Check

The official GATE C++ path can run when reusing an already built GATE index:

```powershell
python experiments\run_gate_official_selector.py `
  --manifest data\sift100k_dynamic\manifest.json `
  --out-dir results\gate_official_selector_sift100k_dynamic `
  --hub-path results\sift100k_gate_adapter\gate_hubs.txt `
  --embeddings centroid:results\sift100k_gate_adapter\gate_centroid_embeddings.fvecs `
  --index-path results\sift100k_gate_adapter\gate_final.nsg `
  --search-l 240 `
  --timeout 1800
```

Result:

| Backend | Method | Recall@10 | QPS |
|---|---|---:|---:|
| `official_gate_test_gate_search` | `centroid` | 0.9999 | 19.54 |

Caveat: this official artifact path is static/rebuilt-graph only. It does not
support our insert/delete/no-rebuild dynamic graph. Its QPS is also not directly
comparable with the O3 dynamic runner because the binaries and measurement paths
are different.

Attempting to build a fresh official GATE final index for the SIFT stress set
failed with `std::bad_alloc` in `test_nsg_index`, likely due to artifact/input
graph-format fragility. The dynamic comparison above is therefore the main fair
comparison because all methods share the same backend and only the entry
strategy changes.
