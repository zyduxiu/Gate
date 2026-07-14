# DiskANN/Vamana Entry Adapter Results

## What This Tests

This experiment keeps the DiskANN/Vamana graph fixed and changes only the search-time entry
initialization:

- `builtin_topk`: DiskANN's normal search on the saved medoid-start graph.
- `fixed_medoid`: the entry adapter with one medoid-like data anchor.
- `static_density`: query-nearest entries from density anchors selected before drift.
- `dynamic_density`: query-nearest entries from density anchors refreshed after drift.
- `learned_balanced`: learned anchor scoring exported into the same entry adapter.

The runner first builds one shared medoid graph, saves it, then loads the same graph for all
methods:

```powershell
python experiments\run_diskann_entry_adapter.py `
  --manifest data\sift100k_dynamic\manifest.json `
  --out-dir results\sift100k_diskann_entry_shared `
  --methods fixed_medoid,static_density,dynamic_density,learned_balanced `
  --search-l 20,40,80,120,240 --threads 4 --reps 1 --entries 8 --timeout 900
```

GloVe:

```powershell
python experiments\run_diskann_entry_adapter.py `
  --manifest data\glove100k_dynamic\manifest.json `
  --out-dir results\glove100k_diskann_entry_shared `
  --methods fixed_medoid,static_density,dynamic_density,learned_balanced `
  --search-l 20,40,80,120,240 --threads 4 --reps 1 --entries 8 --timeout 900
```

## Current Takeaway

On rebuilt/strong DiskANN graphs, the entry overlay is not a free reduction in distance
computations. Multi-entry anchors improve coverage and recall, especially at low `L`, but
they also seed more candidates. This is expected and useful: it shows the mechanism is
orthogonal to the graph topology, while the main paper win should still be evaluated under
dynamic drift, stale entry sets, and hard-query bins.

SIFT100k:

- `dynamic_density`, `L=20`: Recall@10 improves from `0.9497` to `0.9684`; p99 drops from
  `232us` to `212us`; mean comparisons rise by `12.4%`.
- `learned_balanced`, `L=40`: Recall@10 improves from `0.9852` to `0.9887`; p99 drops from
  `261us` to `208us`; mean comparisons rise by `6.8%`.

GloVe100k:

- `dynamic_density`, `L=20`: Recall@10 improves from `0.7309` to `0.7664`; p99 drops from
  `289us` to `258us`; mean comparisons rise by `14.8%`.
- `dynamic_density`, `L=80`: Recall@10 improves from `0.8829` to `0.8896`; p99 drops from
  `861us` to `446us`; mean comparisons rise by `4.9%`.
- `learned_balanced`, `L=40`: Recall@10 improves from `0.8156` to `0.8337`; p99 drops from
  `346us` to `257us`; mean comparisons rise by `8.9%`.

Raw summaries:

- `results/sift100k_diskann_entry_shared/diskann_entry_adapter_summary.csv`
- `results/glove100k_diskann_entry_shared/diskann_entry_adapter_summary.csv`

## Paper Positioning

This supports the conservative claim:

> Dynamic anchors are a search-time entry overlay that can be attached to Vamana/DiskANN
> without changing graph construction. They improve query coverage under the same graph,
> and their strongest expected benefit is in dynamic/stale-entry regimes rather than on a
> freshly rebuilt strong graph.

The next stronger experiment is to combine this adapter with a no-full-rebuild dynamic
DiskANN/Vamana workflow or hard-query bins, where entry staleness should be amplified.
