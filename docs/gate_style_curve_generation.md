# GATE-Style Recall-QPS Curves

## Where The Curves Come From

GATE's Figure 5 is a recall-QPS trade-off plot. Each point is produced by
running one method with a different search parameter setting, measuring:

- recall at a target `k` such as `Recall@1` or `Recall@100`;
- query throughput in QPS.

The curve connects the parameter sweep points for the same method. The top
right is better: higher recall and higher QPS.

For our dynamic-entry setting, the equivalent control parameter is mainly
`search_l`. Larger `search_l` usually increases recall but lowers QPS.

## Generated Figures

Dense `Recall@10` curve on full insert/delete drift queries:

```text
results/figures/stress_gate_style_recall10_qps_dense.png
```

Dense `Recall@10` curve on inserted-region-heavy queries:

```text
results/figures/stress_insert80_gate_style_recall10_qps_dense.png
```

GATE-style two-row curve on inserted-region-heavy queries:

```text
results/figures/stress_insert80_gate_style_recall1_recall100_qps.png
results/figures/stress_insert80_gate_style_recall1_recall100_qps_pareto.png
```

The Pareto version removes dominated points where another point has both higher
recall and higher QPS. This is closer to the "grid search best trade-off" style
used in many ANNS papers.

## Commands

`Recall@10` dense curve:

```powershell
python experiments\run_nsg_dynamic_entry.py `
  --manifest data\sift_stress_insert80_queries\manifest.json `
  --initial-nsg results\sift_stress_drift_dynamic_nsg\initial.nsg `
  --out-dir results\stress_insert80_gate_style_curve `
  --methods nsg_ep,gate_initial_hub,gate_refreshed_hub,online_hub_density,online_density,static_density,fixed_medoid `
  --search-l 80,100,120,140,160,180,200,220,240,280,320 `
  --entries 4 `
  --maintain-sample 512 `
  --maintain-changes 4 `
  --maintain-batch 1000 `
  --timeout 7200

python experiments\summarize_nsg_entry_results.py `
  --dir results\stress_insert80_gate_style_curve `
  --manifest data\sift_stress_insert80_queries\manifest.json `
  --out results\stress_insert80_gate_style_curve\summary.csv

python experiments\plot_gate_style_curves.py `
  --inputs results\stress_insert80_gate_style_curve\summary.csv `
  --out results\figures\stress_insert80_gate_style_recall10_qps_dense.png `
  --recall-col recall_at_10 `
  --qps-col qps `
  --dataset-col dataset `
  --methods nsg_ep,gate_initial_hub,gate_refreshed_hub,online_density,online_hub_density,static_density,fixed_medoid `
  --title "SIFT insert-heavy dynamic no-rebuild: Recall@10-QPS" `
  --xlabel "Recall@10" `
  --scale-y-thousands `
  --min-recall 0.983
```

`Recall@1` and `Recall@100` GATE-style curve:

```powershell
python experiments\run_nsg_dynamic_entry.py `
  --manifest data\sift_stress_insert80_queries\manifest.json `
  --initial-nsg results\sift_stress_drift_dynamic_nsg\initial.nsg `
  --out-dir results\stress_insert80_gate_style_curve_recall1 `
  --methods nsg_ep,gate_initial_hub,gate_refreshed_hub,online_hub_density,online_density,static_density `
  --search-l 20,40,60,80,100,120,160,200,240 `
  --entries 4 `
  --topk 1 `
  --maintain-sample 512 `
  --maintain-changes 4 `
  --maintain-batch 1000 `
  --timeout 7200

python experiments\run_nsg_dynamic_entry.py `
  --manifest data\sift_stress_insert80_queries\manifest.json `
  --initial-nsg results\sift_stress_drift_dynamic_nsg\initial.nsg `
  --out-dir results\stress_insert80_gate_style_curve_recall100 `
  --methods nsg_ep,gate_initial_hub,gate_refreshed_hub,online_hub_density,online_density,static_density `
  --search-l 100,120,160,200,240,280,320,400 `
  --entries 4 `
  --topk 100 `
  --maintain-sample 512 `
  --maintain-changes 4 `
  --maintain-batch 1000 `
  --timeout 7200
```

Then summarize and plot:

```powershell
python experiments\summarize_nsg_entry_results.py `
  --dir results\stress_insert80_gate_style_curve_recall1 `
  --manifest data\sift_stress_insert80_queries\manifest.json `
  --out results\stress_insert80_gate_style_curve_recall1\summary.csv

python experiments\summarize_nsg_entry_results.py `
  --dir results\stress_insert80_gate_style_curve_recall100 `
  --manifest data\sift_stress_insert80_queries\manifest.json `
  --out results\stress_insert80_gate_style_curve_recall100\summary.csv

python experiments\plot_gate_style_curves.py `
  --inputs `
    results\stress_insert80_gate_style_curve_recall1\summary.csv `
    results\stress_insert80_gate_style_curve_recall100\summary.csv `
  --out results\figures\stress_insert80_gate_style_recall1_recall100_qps_pareto.png `
  --recall-col recall_at_k `
  --qps-col qps `
  --dataset-col dataset `
  --panel-col recall_metric `
  --methods nsg_ep,gate_initial_hub,gate_refreshed_hub,online_density,online_hub_density,static_density `
  --title "SIFT insert-heavy dynamic no-rebuild: GATE-style QPS curves" `
  --xlabel "Recall" `
  --scale-y-thousands `
  --min-recall 0.90 `
  --pareto-frontier
```

## Current Caveat

These figures are single-run curves on this workstation. For a paper-grade plot,
repeat each point 10 times, report mean/median QPS, and preferably plot the
median Pareto frontier with error bands or a separate table for variance.

## 10-Run GATE-Style Curves

The paper-grade repeat runner is:

```powershell
python experiments\run_gate_style_qps_repeats.py `
  --skip-existing `
  --plot
```

It runs only `Recall@1` and `Recall@100`, with 10 repeated single-thread query
runs for each method and `search_l` point.

Outputs:

```text
results/stress_insert80_gate_style_10repeats/recall1/aggregate_10runs.csv
results/stress_insert80_gate_style_10repeats/recall100/aggregate_10runs.csv
results/stress_insert80_gate_style_10repeats/aggregate_recall1_recall100_10runs.csv
results/stress_insert80_gate_style_10repeats/gate_style_recall1_recall100_qps_median_pareto.png
results/stress_insert80_gate_style_10repeats/gate_style_recall1_recall100_qps_mean_pareto.png
```

Validation:

- `Recall@1`: 54 aggregate points, each with `runs=10`.
- `Recall@100`: 48 aggregate points, each with `runs=10`.

Representative high-recall point on inserted-heavy SIFT, `Recall@100`, `L=320`:

| Method | Median Recall@100 | Median QPS | Median p99 latency | Median distance comps |
|---|---:|---:|---:|---:|
| `gate_initial_hub` | 0.996998 | 567.45 | 3317.23 us | 5937.09 |
| `gate_refreshed_hub` | 0.996998 | 648.56 | 2451.92 us | 5781.56 |
| `nsg_ep` | 0.997108 | 552.80 | 3411.68 us | 5979.68 |
| `online_hub_density` | 0.996998 | 650.23 | 2391.74 us | 5775.31 |

Interpretation:

- The 10-run result is more conservative than the single-run curve.
- At high `Recall@100`, `online_hub_density` is essentially tied with or
  slightly above `gate_refreshed_hub` in median QPS, while using an online
  maintained entry set rather than an offline refreshed hub set.
- The advantage over stale `gate_initial_hub` is clearer in p99 latency and
  distance computations than in raw recall.
