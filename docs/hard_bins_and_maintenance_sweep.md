# Hard Query Bins And Maintenance Sweep

## Hard-Query Bins

Command:

```powershell
python experiments\run_nsg_dynamic_entry.py `
  --manifest data\sift_stress_drift\manifest.json `
  --initial-nsg results\sift_stress_drift_dynamic_nsg\initial.nsg `
  --out-dir results\stress_hard_bins_perq `
  --methods nsg_ep,gate_initial_hub,gate_refreshed_hub,online_hub_density `
  --search-l 240 `
  --entries 4 `
  --repair-degree 0 `
  --maintain-batch 1000 `
  --maintain-changes 4 `
  --maintain-sample 512 `
  --timeout 3600 `
  --per-query-l 240

python experiments\analyze_hard_query_bins.py `
  --dir results\stress_hard_bins_perq `
  --baseline gate_initial_hub `
  --methods nsg_ep,gate_initial_hub,gate_refreshed_hub,online_hub_density `
  --search-l 240 `
  --out results\stress_hard_bins_perq\hard_query_bins_gate_initial.csv
```

Key result versus stale `gate_initial_hub`:

| Bin | Method | Avg Distance Computations | Delta | Avg Latency Delta |
|---|---|---:|---:|---:|
| all | `online_hub_density` | 4484.38 | -2.52% | -23.13% |
| top 20% hard | `online_hub_density` | 5544.58 | -2.44% | -24.86% |
| top 10% hard | `online_hub_density` | 5729.95 | -2.50% | -25.99% |
| top 5% hard | `online_hub_density` | 5874.90 | -2.51% | -22.76% |

Interpretation:

- The current stress workload does not yet produce a 10%+ search-work win on
  hard queries.
- It does show a strong tail-latency reduction in the same hard bins.
- For a top-tier paper, we need a longer/more adversarial stream where stale
  entry sets degrade more severely.

## Maintenance Sweep

Command:

```powershell
$out='results\maintenance_sweep_sift_stress'
foreach ($changes in 1,2,4,8,12) {
  foreach ($sample in 256,512,1000,2000) {
    $dir="$out\c${changes}_s${sample}"
    python experiments\run_nsg_dynamic_entry.py `
      --manifest data\sift_stress_drift\manifest.json `
      --initial-nsg results\sift_stress_drift_dynamic_nsg\initial.nsg `
      --out-dir $dir `
      --methods online_hub_density `
      --search-l 240 `
      --entries 4 `
      --repair-degree 0 `
      --maintain-batch 1000 `
      --maintain-changes $changes `
      --maintain-sample $sample `
      --timeout 1800
    python experiments\summarize_nsg_entry_results.py `
      --dir $dir `
      --manifest data\sift_stress_drift\manifest.json `
      --out "$dir\summary.csv"
  }
}
```

Top configurations:

| maintain_changes | maintain_sample | Avg Distance Computations | Recall@10 | Maintenance Seconds |
|---:|---:|---:|---:|---:|
| 4 | 512 | 4484.38 | 0.9963 | 2.63 |
| 12 | 512 | 4494.73 | 0.9963 | 5.60 |
| 8 | 1000 | 4496.91 | 0.9963 | 5.99 |
| 4 | 1000 | 4497.57 | 0.9963 | 3.35 |

Interpretation:

- The current default, `maintain_changes=4, maintain_sample=512`, is still the
  best average-work setting on this stress workload.
- More aggressive maintenance increases churn and maintenance cost, but does
  not improve search work.
- The next evidence gain should come from a stronger drift benchmark, not from
  minor parameter tuning on this dataset.

## Next Stronger Experiment

Prepare a longer SIFT stream, then rebuild dynamic assets:

```powershell
python experiments\prepare_sift_stress_drift.py `
  --out-dir data\sift_stress_long_drift `
  --pool-size 400000 `
  --n-initial 100000 `
  --n-insert 80000 `
  --n-delete 50000 `
  --insert-high-pool 200000 `
  --density-sample 8000 `
  --anchor-candidates 40000 `
  --seed 123

python experiments\real_datasets.py build-dynamic-nsg-assets `
  --manifest data\sift_stress_long_drift\manifest.json `
  --index results\sift_stress_long_drift_dynamic_nsg\initial.nsg `
  --timeout 7200
```

This should amplify entry-set staleness more than the current 50k insert /
30k delete stress set.
