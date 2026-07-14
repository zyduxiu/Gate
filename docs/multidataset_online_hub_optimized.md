# Multi-Dataset Online Hub-Density Maintenance

This page reruns the dynamic no-rebuild NSG entry experiments with the current
`online_hub_density` method, including the drift-aware retire rule.

Command pattern:

```powershell
python experiments\run_nsg_dynamic_entry.py `
  --manifest <manifest> `
  --initial-nsg <initial.nsg> `
  --out-dir <out> `
  --methods nsg_ep,fixed_medoid,static_density,dynamic_density,learned_balanced,online_density,online_hub_density `
  --search-l <L> `
  --entries 4 `
  --repair-degree 0 `
  --maintain-batch 1000 `
  --maintain-changes 4 `
  --maintain-sample 512 `
  --timeout 3600
```

Merged result:

```text
results/multidataset_online_hub_optimized/summary_all.csv
```

## Main Datasets

| Dataset | L | Method | Recall@10 | Avg distance comps | p99 latency | QPS |
|---|---:|---|---:|---:|---:|---:|
| GloVe-100-100k | 480 | `nsg_ep` | 0.9927 | 12465.50 | 13292.40 us | 298.02 |
| GloVe-100-100k | 480 | `dynamic_density` | 0.9928 | 12432.30 | 4184.76 us | 369.89 |
| GloVe-100-100k | 480 | `online_density` | 0.9927 | 12437.00 | 5961.09 us | 291.04 |
| GloVe-100-100k | 480 | `online_hub_density` | 0.9928 | 12371.50 | 9299.69 us | 238.03 |
| Fashion-MNIST-55k | 240 | `nsg_ep` | 0.9992 | 1583.39 | 4210.84 us | 479.48 |
| Fashion-MNIST-55k | 240 | `dynamic_density` | 0.9992 | 1570.47 | 2256.46 us | 656.65 |
| Fashion-MNIST-55k | 240 | `online_density` | 0.9999 | 1567.30 | 3907.35 us | 583.30 |
| Fashion-MNIST-55k | 240 | `online_hub_density` | 0.9992 | 1516.25 | 2621.40 us | 632.37 |
| MNIST-55k | 120 | `nsg_ep` | 0.9994 | 1131.54 | 1787.48 us | 924.64 |
| MNIST-55k | 120 | `dynamic_density` | 0.9995 | 1106.32 | 1856.73 us | 824.76 |
| MNIST-55k | 120 | `online_density` | 0.9994 | 1106.71 | 5854.45 us | 582.67 |
| MNIST-55k | 120 | `online_hub_density` | 0.9994 | 1093.02 | 1661.30 us | 935.16 |

## Relative Effects

Compared with `nsg_ep`:

| Dataset | Avg comp reduction | p99 reduction | QPS change |
|---|---:|---:|---:|
| GloVe-100-100k | 0.75% | 30.04% | -20.13% |
| Fashion-MNIST-55k | 4.24% | 37.75% | +31.88% |
| MNIST-55k | 3.40% | 7.06% | +1.14% |

Compared with `online_density`:

| Dataset | Avg comp reduction | p99 reduction | QPS change |
|---|---:|---:|---:|
| GloVe-100-100k | 0.53% | -56.01% | -18.22% |
| Fashion-MNIST-55k | 3.26% | 32.91% | +8.41% |
| MNIST-55k | 1.24% | 71.62% | +60.50% |

## Diagnostic Negative Case

NYTimes is not a main high-recall result under the current NSG construction.
At `L=240`, Recall@10 is only about `0.20-0.22`, so it mostly diagnoses a
backend graph-quality problem:

| Method | Recall@10 | Avg distance comps | p99 latency | QPS |
|---|---:|---:|---:|---:|
| `nsg_ep` | 0.2023 | 840.53 | 1251.71 us | 2410.07 |
| `dynamic_density` | 0.2176 | 844.38 | 1012.17 us | 1781.80 |
| `online_density` | 0.2238 | 801.33 | 1428.50 us | 1549.25 |
| `online_hub_density` | 0.2164 | 765.23 | 1259.97 us | 1711.14 |

Interpretation: entry maintenance can reduce search work, but it cannot rescue
a low-quality backend graph into a top-recall regime by itself.

## What These Results Support

- The method is not SIFT-only. It improves stable search-work metrics on GloVe,
  Fashion-MNIST, and MNIST.
- The strongest non-SIFT wins are Fashion-MNIST and MNIST, where hubness-aware
  online maintenance improves both distance computations and p99/QPS.
- GloVe is mixed: it has the best average distance computations, but p99/QPS are
  worse in this run. Treat it as a search-work win, not a systems-latency win.
- NYTimes should be kept as a negative/diagnostic result unless the backend
  graph construction is improved.

## Core Innovation

The core claim should be:

> Dynamic graph-based ANNS has entry-set staleness in addition to graph-edge
> staleness. We introduce a small online entry overlay that maintains the entry
> candidate set under insert/delete drift using density, coverage, hubness, and
> drift-aware retirement.

More concretely:

1. **Problem formulation: entry-set staleness.**
   Existing dynamic ANNS work mostly repairs graph edges, partitions, hard
   regions, or query-to-entry selection. We isolate the separate failure mode
   where the candidate entry set `E` itself becomes stale after distribution
   drift.

2. **Online maintenance of `E`, not final refresh.**
   The method starts from pre-drift anchors and updates them during stream
   batches. It removes dead anchors, adds candidates in under-covered/new dense
   regions, and retires stale/redundant/weak-hub anchors.

3. **Density + hubness + coverage objective.**
   Density gives workload coverage, coverage distance prevents blind spots,
   graph hubness keeps entries navigationally useful, and drift-aware retirement
   lets the entry set move into new inserted regions without rebuilding the
   graph.

4. **Composable with GATE/DiskANN++/RFix-like work.**
   GATE/DiskANN++ answer "which entry should this query use?" and RFix/NGFix
   repair hard graph regions. Our layer answers "which candidate entries should
   remain available after dynamic drift?" These are not identical problems and
   can be stacked.

The paper should not claim to invent multi-entry search or beat every GATE-style
selector. The safer and stronger claim is:

> Maintaining a fresh candidate entry set is a lightweight, orthogonal layer
> that reduces high-recall search work under dynamic drift and can approach
> offline refreshed-entry upper bounds without full refresh or rebuild.
