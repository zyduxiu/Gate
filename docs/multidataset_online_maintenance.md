# Multi-Dataset Online Entry Maintenance

## Purpose

This experiment checks whether `online_density` is only a SIFT artifact.

Unlike `dynamic_density`, which refreshes anchors from the final alive distribution,
`online_density` starts from stale pre-drift anchors and maintains the entry set during
insert/delete updates:

- remove deleted anchors,
- sample inserted-vector candidates per update batch,
- add candidates that are far from current anchors and locally dense,
- retire redundant anchors to keep memory fixed.

## Summary

All rows use the no-rebuild dynamic NSG runner: initial NSG graph, local insert edges,
deleted-node masking, and no full graph rebuild.

| dataset | L | method | Recall@10 | avg comps | p99 latency | distance reduction | p99 reduction |
|---|---:|---|---:|---:|---:|---:|---:|
| SIFT1M-100k | 240 | `nsg_ep` | 0.9997 | 3877.51 | 3611 us | 0.00% | 0.00% |
| SIFT1M-100k | 240 | `online_density` | 0.9997 | 3824.03 | 3087 us | 1.38% | 14.50% |
| GloVe-100-100k | 480 | `nsg_ep` | 0.9927 | 12465.50 | 29079 us | 0.00% | 0.00% |
| GloVe-100-100k | 480 | `online_density` | 0.9927 | 12419.40 | 7278 us | 0.37% | 74.97% |
| Fashion-MNIST-55k | 240 | `nsg_ep` | 0.9992 | 1583.39 | 4554 us | 0.00% | 0.00% |
| Fashion-MNIST-55k | 240 | `online_density` | 0.9992 | 1567.81 | 3099 us | 0.98% | 31.95% |
| MNIST-55k | 120 | `nsg_ep` | 0.9994 | 1131.54 | 6000 us | 0.00% | 0.00% |
| MNIST-55k | 120 | `online_density` | 0.9994 | 1110.29 | 2141 us | 1.88% | 64.33% |

Raw table:

```text
results/multidataset_online_entry_summary.csv
```

## Maintenance Cost

| dataset | updates | anchors added | anchors retired | dead anchors removed | maintenance distance comps | comps/update |
|---|---:|---:|---:|---:|---:|---:|
| SIFT1M-100k | 30,000 | 100 | 80 | 20 | 4,943,200 | 164.77 |
| GloVe-100-100k | 30,000 | 91 | 80 | 11 | 4,648,450 | 154.95 |
| Fashion-MNIST-55k | 10,000 | 36 | 20 | 16 | 1,619,970 | 162.00 |
| MNIST-55k | 10,000 | 35 | 20 | 15 | 1,587,200 | 158.72 |

The entry layer keeps only `128` anchors. The online maintenance cost is small enough to
report as an update-side overhead instead of hiding it inside query latency.

## Hard-Query Evidence

GloVe, `L=480`, hardest quartile under stale `nsg_ep`:

| method | Recall@10 | avg comps | p99 latency |
|---|---:|---:|---:|
| `nsg_ep` | 0.9808 | 16038.14 | 30247 us |
| `dynamic_density` batch proxy | 0.9804 | 16015.06 | 12425 us |
| `online_density` | 0.9808 | 16023.06 | 7684 us |
| `learned_balanced` | 0.9808 | 16010.49 | 14428 us |

Fashion-MNIST, `L=240`, hardest quartile:

| method | Recall@10 | avg comps | p99 latency |
|---|---:|---:|---:|
| `nsg_ep` | 0.9972 | 2051.76 | 5214 us |
| `dynamic_density` batch proxy | 0.9972 | 2032.93 | 2823 us |
| `online_density` | 0.9972 | 2032.72 | 3312 us |

MNIST, `L=120`, hardest quartile:

| method | Recall@10 | avg comps | p99 latency |
|---|---:|---:|---:|
| `nsg_ep` | 0.9984 | 1492.71 | 14381 us |
| `dynamic_density` batch proxy | 0.9992 | 1467.71 | 4802 us |
| `online_density` | 0.9984 | 1469.07 | 2529 us |

Hard-bin artifacts:

```text
results/glove100k_dynamic_nsg_entry/hard_bins_l480_with_online.csv
results/fashion55k_dynamic_nsg_entry/hard_bins_l240_with_online.csv
results/mnist55k_dynamic_nsg_entry/hard_bins_l120_with_online.csv
results/mnist55k_dynamic_nsg_entry/hard_bins_l240_with_online.csv
```

## Paper Claim This Supports

The safer top-conference claim is:

> Entry-set staleness is a separate dynamic-ANNS failure mode. Maintaining a small
> density-aware entry overlay online can reduce high-recall search work and tail latency
> across multiple real datasets without rebuilding the graph.

What this still does not claim:

- It does not claim multi-entry search is new.
- It does not claim graph repair is unnecessary.
- It does not claim every p99 result improves under every topology repair setting.

The strongest phrasing is that online entry maintenance is lightweight, graph-agnostic,
and complementary to graph repair.
