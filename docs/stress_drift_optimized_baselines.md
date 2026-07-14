# Stress Drift Optimized Baselines

Dataset:

```text
data/sift_stress_drift/manifest.json
```

Backend:

```text
results/sift_stress_drift_dynamic_nsg/initial.nsg
```

This benchmark is intentionally biased toward entry-set staleness:

- initial graph is built before drift;
- inserted vectors form a new dense region;
- old dense/high-hub regions are deleted;
- no final graph rebuild is allowed for online methods.

## Parameter Sweep

Sweep summary:

```text
results/online_hub_density_stress_sweep/aggregate.csv
```

Best high-recall point:

| Config | Recall@10 | Avg distance comps | p99 latency | QPS | Maintenance |
|---|---:|---:|---:|---:|---:|
| `L240_e4_c4_s512` | 0.9963 | 4484.38 | 1700.61 us | 853.58 | 1.33 s |
| `L240_e4_c4_s1000` | 0.9963 | 4497.57 | 1770.90 us | 791.31 | 3.10 s |
| `L240_e4_c2_s1000` | 0.9963 | 4498.93 | 1847.79 us | 843.24 | 1.51 s |
| `L200_e4_c4_s1000` | 0.9958 | 3937.95 | 1836.47 us | 877.96 | 3.69 s |

Takeaway:

- `maintain_sample=512` is better than `1000/2000` here: cheaper maintenance,
  slightly fewer search distance computations, and better QPS.
- `entries=4` is the best current default. `entries=2` is cheaper but loses
  search quality; `entries=8` increases overhead and lowers QPS.
- `L=200` is a useful speed/recall trade-off, but `L=240` is better for the
  paper's high-recall story.

## Full Baseline Table

Full summary:

```text
results/stress_full_baselines_optimized_entry/summary.csv
```

At `L=240`, `entries=4`, `maintain_sample=512`:

| Method | Recall@10 | Avg distance comps | p99 latency | QPS | Maintenance |
|---|---:|---:|---:|---:|---:|
| `nsg_ep` | 0.9964 | 4645.13 | 2786.58 us | 697.12 | 0 s |
| `fixed_medoid` | 0.9964 | 4646.11 | 3993.33 us | 600.47 | 0 s |
| `static_density` | 0.9964 | 4626.76 | 2517.23 us | 715.10 | 0 s |
| `dynamic_density` final refresh proxy | 0.9963 | 4599.55 | 2791.26 us | 642.50 | 0 s in runner |
| `gate_initial_hub` stale | 0.9964 | 4600.35 | 1667.28 us | 852.49 | 0 s |
| `gate_refreshed_hub` upper bound | 0.9963 | 4482.11 | 1757.68 us | 866.51 | 0 s in runner |
| `online_density` | 0.9963 | 4575.99 | 2376.28 us | 754.74 | 1.41 s |
| `online_hub_density` | 0.9963 | 4484.38 | 1686.65 us | 869.50 | 1.34 s |

`online_hub_density` is now essentially tied with the final-refresh
`gate_refreshed_hub` upper bound on search work, without using a final refresh.

## QPS Repeats

Repeat summary:

```text
results/stress_qps_repeats/median_summary.csv
```

Three runs at `L=240`, `entries=4`, `maintain_sample=512`:

| Method | Recall@10 median | Avg comps median | p99 median | QPS median | QPS range |
|---|---:|---:|---:|---:|---:|
| `nsg_ep` | 0.9964 | 4645.13 | 2971.28 us | 631.46 | 427.89-680.32 |
| `gate_initial_hub` | 0.9964 | 4600.35 | 2399.84 us | 652.60 | 427.48-743.14 |
| `gate_refreshed_hub` | 0.9963 | 4482.11 | 2246.17 us | 717.91 | 618.47-816.16 |
| `online_density` | 0.9963 | 4575.99 | 1890.03 us | 787.60 | 770.65-848.72 |
| `online_hub_density` | 0.9963 | 4484.38 | 2119.32 us | 762.18 | 739.54-790.66 |

Takeaway:

- Distance computations are stable and should be the primary paper metric.
- QPS is noisy on this workstation, but `online_hub_density` remains above
  stale `gate_initial_hub` and close to/refreshed-GATE in median QPS.
- `online_density` has strong QPS but worse search work; this is a useful
  ablation showing hubness is what closes the gap to refreshed GATE.

## Paper-Safe Claim

In this stress drift setting, online hubness-aware entry maintenance:

- reduces search work versus stale GATE initial hubs;
- closes almost all of the gap to offline refreshed GATE hubs;
- keeps online maintenance cost low, about 1.3-1.6 seconds over 50 stream
  batches in this setup;
- improves the entry set without a final refresh or full graph rebuild.
