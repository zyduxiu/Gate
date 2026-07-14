# Online Entry Maintenance + GATE-Style Selector

This experiment tests the compositional claim:

> Online density/hubness-aware maintenance keeps the candidate entry set `E`
> fresh under drift; GATE-like query selectors can then operate on this fresher
> `E`.

Dataset:

```text
data/sift_stress_drift/manifest.json
```

Backend:

```text
results/sift_stress_drift_dynamic_nsg/initial.nsg
```

Command:

```powershell
python experiments\run_online_gate_composition.py `
  --manifest data\sift_stress_drift\manifest.json `
  --initial-nsg results\sift_stress_drift_dynamic_nsg\initial.nsg `
  --out-dir results\online_gate_composition_sift_stress_drift_retire `
  --search-l 240 `
  --entries 4 `
  --epochs 80 `
  --timeout 3600
```

## Main Result

Summary:

```text
results/online_gate_composition_sift_stress_drift_retire/summary.csv
```

At `L=240`, `entries=4`:

| Method | Recall@10 | Avg distance comps | p99 latency | QPS |
|---|---:|---:|---:|---:|
| `nsg_ep` | 0.9964 | 4645.13 | 4790.89 us | 424.06 |
| `gate_initial_hub` | 0.9964 | 4600.35 | 3958.23 us | 476.49 |
| `gate_refreshed_hub` | 0.9963 | 4482.11 | 4057.47 us | 451.57 |
| `online_hub_density` | 0.9963 | 4497.57 | 2920.64 us | 565.95 |
| `static_online_hub_density_E` | 0.9963 | 4497.57 | 2948.20 us | 552.27 |
| `query_entries_gate_initial_hub` | 0.9964 | 4634.03 | 13336.50 us | 444.78 |
| `query_entries_gate_refreshed_hub` | 0.9963 | 4553.36 | 2739.59 us | 676.15 |
| `query_entries_online_hub_density_E` | 0.9963 | 4554.62 | 13015.90 us | 411.81 |

## Entry-Set Drift

The drift-aware retire rule substantially changes the maintained entry set:

| Entry set | Anchors from inserted region |
|---|---:|
| Initial `static_density` | 0 / 128 |
| `gate_initial_hub` | 0 / 128 |
| `gate_refreshed_hub` | 74 / 128 |
| Old `online_hub_density_E` | 24 / 128 |
| Drift-retire `online_hub_density_E` | 92 / 128 |

## Interpretation

- The maintained entry set now moves with the stream instead of staying mostly
  tied to the initial distribution.
- `online_hub_density` beats stale `gate_initial_hub` on average search work
  (`4600.35 -> 4497.57`) without a final refresh.
- It nearly reaches the final-refresh upper bound `gate_refreshed_hub`
  (`4482.11`), while paying about `4.23s` online maintenance time over 50
  stream batches.
- The current two-tower proxy does not improve average search work over
  nearest-entry routing. Treat it as a baseline/negative result, not as the
  main contribution.

Paper-safe claim:

> Online entry-set maintenance is complementary to GATE-style selection, but
> the current evidence supports maintaining a fresh `E` more strongly than
> replacing nearest-anchor routing with the present two-tower proxy.
