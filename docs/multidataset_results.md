# Multi-Dataset Dynamic Entry Results

## Scope

These runs extend the no-rebuild dynamic NSG-style entry experiment beyond
SIFT. Every row uses the same dynamic graph per dataset:

- Build NSG on the initial base vectors.
- Insert drifted vectors with local KNN/back edges.
- Mask deleted old vectors.
- Do not rebuild the global NSG graph.
- Change only the entry initialization.

The added datasets come from ANN-Benchmarks HDF5 files:

- `glove-100-angular`: 100,000 initial + 20,000 inserted - 10,000 deleted.
- `fashion-mnist-784-euclidean`: 50,000 initial + 5,000 inserted - 5,000 deleted.
- `mnist-784-euclidean`: 50,000 initial + 5,000 inserted - 5,000 deleted.

For GloVe and NYTimes, vectors are L2-normalized and then searched with L2
distance. NYTimes was also prepared and measured as a stress diagnostic, but is
not included in the main positive table because this NSG configuration does not
reach a useful high-recall regime on that dataset.

## Main Cross-Dataset Table

Artifact:

```text
results/multidataset_entry_summary.csv
```

```text
dataset             L    method            Recall@10  avg comps  p99 us   p99 reduction vs nsg_ep
SIFT1M-100k         240  nsg_ep            0.9997     3877.51    4976.96  0.0%
SIFT1M-100k         240  static_density    0.9997     3848.10    3144.39  36.8%
SIFT1M-100k         240  dynamic_density   0.9997     3829.21    2948.30  40.8%
SIFT1M-100k         240  learned_balanced  0.9998     3820.14    1958.52  60.6%

GloVe-100-100k      480  nsg_ep            0.9927     12465.50   29079.20 0.0%
GloVe-100-100k      480  static_density    0.9928     12423.90   16892.80 41.9%
GloVe-100-100k      480  dynamic_density   0.9928     12415.50   11080.10 61.9%
GloVe-100-100k      480  learned_balanced  0.9927     12414.30   9939.50  65.8%

Fashion-MNIST-55k   240  nsg_ep            0.9992     1583.39    4554.24  0.0%
Fashion-MNIST-55k   240  static_density    0.9992     1575.68    3167.14  30.5%
Fashion-MNIST-55k   240  dynamic_density   0.9992     1567.73    2601.79  42.9%
Fashion-MNIST-55k   240  learned_balanced  0.9992     1565.52    4281.79  6.0%

MNIST-55k           120  nsg_ep            0.9994     1131.54    6000.17  0.0%
MNIST-55k           120  static_density    0.9994     1110.53    5329.84  11.2%
MNIST-55k           120  dynamic_density   0.9996     1109.36    4340.72  27.7%
MNIST-55k           120  learned_balanced  0.9995     1107.96    3831.21  36.1%
```

## Interpretation

- `dynamic_density` is the most robust main method. It improves p99 latency
  versus stale `nsg_ep` on all four main datasets while preserving recall.
- `learned_balanced` is best on SIFT and high-budget GloVe, but not robust on
  Fashion-MNIST. On MNIST it helps at `L=120`, but hurts at the saturated
  `L=240` point. This should be framed as an optional workload-aware extension,
  not as the core contribution.
- The paper's core claim should remain entry-set maintenance under dynamic
  drift. The strongest general result is density-aware maintenance; the learned
  extension is useful when its coverage/load regularization is well matched to
  the dataset.
- NYTimes is a useful negative/stress result: with the current NSG construction
  parameters, recall stays near 0.25 even at `L=1920`. That is a graph-quality
  failure mode, not an entry-layer win/loss. It should motivate better NSG
  tuning or a Vamana/DiskANN backend before being used in the main paper table.

## Dataset-Specific Notes

SIFT:

- Best result is `learned_balanced` at `L=240`.
- It improves p99 latency from `4976.96 us` to `1958.52 us` at the same high
  recall regime.

GloVe:

- `L=240` is not yet a high-recall operating point, so the cross-dataset table
  uses `L=480`.
- At `L=480`, `dynamic_density` and `learned_balanced` both reduce p99 latency
  substantially. `learned_balanced` gives the best p99 but only a small
  distance-computation reduction.

Fashion-MNIST:

- `dynamic_density` is the best paper-grade method.
- `learned_balanced` slightly reduces average distance computations but adds
  overhead and has worse p99 latency. This supports an ablation claim: learned
  maintenance needs dataset-aware regularization and should not replace the
  density-aware core.

MNIST:

- At `L=120`, `dynamic_density` improves Recall@10 from `0.9994` to `0.9996`
  while reducing average distance computations from `1131.54` to `1109.36`.
- The same run reduces p99 latency from `6000.17 us` to `4340.72 us`.
- At saturated `L=240`, distance computations still drop, but latency can be
  dominated by anchor routing overhead. This is the clearest evidence that the
  production version needs indexed anchor routing instead of scanning all
  anchors per query.

NYTimes:

- Prepared `nytimes-256-angular` as `100,000 initial + 20,000 inserted -
  10,000 deleted`.
- At `L=240`, `dynamic_density` improves Recall@10 from `0.2023` to `0.2169`
  and reduces p99 latency from `1515.32 us` to `1322.04 us`, but recall is far
  below the high-recall regime.
- At `L=1920`, recall remains around `0.25` for both `nsg_ep` and
  `dynamic_density`. This should be treated as a backend graph construction
  issue. Do not use this as a main positive result until the NSG/Vamana/DiskANN
  backend is tuned for this dataset.

## Artifacts

```text
data/glove100k_dynamic/manifest.json
results/glove100k_dynamic_nsg_entry/dynamic_nsg_entry_summary.csv
results/glove100k_dynamic_nsg_entry/hard_bins_l240.csv
results/glove100k_dynamic_nsg_entry/hard_bins_l480.csv

data/fashion55k_dynamic/manifest.json
results/fashion55k_dynamic_nsg_entry/dynamic_nsg_entry_summary.csv
results/fashion55k_dynamic_nsg_entry/hard_bins_l240.csv

data/mnist55k_dynamic/manifest.json
results/mnist55k_dynamic_nsg_entry/dynamic_nsg_entry_summary.csv
results/mnist55k_dynamic_nsg_entry/hard_bins_l120.csv
results/mnist55k_dynamic_nsg_entry/hard_bins_l240.csv

data/nytimes100k_dynamic/manifest.json
results/nytimes100k_dynamic_nsg_entry/dynamic_nsg_entry_summary.csv
results/nytimes100k_dynamic_nsg_entry/hard_bins_l1920.csv

results/multidataset_entry_summary.csv
```

## Recommended Paper Position

Use `dynamic_density` as the main method:

> Dynamic data drift makes entry sets stale. Maintaining density-aware entry
> anchors is a lightweight, graph-agnostic overlay that reduces high-recall
> visited work and tail latency without rebuilding the graph.

Use `learned_balanced` as an extension:

> Workload-aware learned maintenance can further improve tail latency, but only
> when constrained by density coverage and insert-region quotas. Without these
> constraints, it can overfit hard-query regions and hurt tail latency.

For the next top-conference step, prioritize:

- Repeat each p99 point 3-5 times and report median p99 plus distance
  computations.
- Add an indexed anchor router so entry selection does not linearly scan all
  anchors in high-dimensional datasets.
- Move the entry overlay into Vamana/DiskANN or a tuned NSG backend, then rerun
  NYTimes and other text datasets in a true high-recall regime.
