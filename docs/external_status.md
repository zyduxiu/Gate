# External Baseline Status

Date: 2026-07-09

## Dataset

Generated with:

```powershell
python experiments\external_baseline.py prepare-data --n-base 10000 --n-query 1000 --dim 64 --gt-k 100 --nsg-graph-k 100 --batch-size 128
```

Manifest:

```text
data/external_10k/manifest.json
```

This is intentionally larger than a smoke test while still cheap enough for
wrapper iteration.

## NSG

Build path:

```powershell
.\scripts\build_external_baselines.ps1 -Baseline nsg -UseWsl
```

Local portability note:

- WSL lacked `libgoogle-perftools-dev`.
- `third_party/nsg/tests/CMakeLists.txt` was patched to link `tcmalloc` only when found.
- This changes allocator choice only; NSG algorithm and parameters are unchanged.

Run:

```powershell
python experiments\external_baseline.py nsg-build --timeout 600
python experiments\external_baseline.py nsg-search --search-l 80 --search-k 10 --timeout 600
python experiments\external_baseline.py eval-ivecs --result results\external\nsg\result.ivecs --k 10 --out results\external\nsg\metrics.json
```

Observed:

```text
indexing time: 1.60842 s
search time: 0.026423 s
Recall@10: 0.9991
```

## SPTAG

Build path:

```powershell
.\scripts\build_external_baselines.ps1 -Baseline sptag -UseWsl
```

Local setup note:

- The shallow clone did not materialize the `ThirdParty/zstd` submodule.
- `ThirdParty/zstd` was manually cloned from `https://github.com/facebook/zstd.git`.
- Full `make all` failed only in Python wrapper copy targets, but `Release/indexbuilder`
  and `Release/indexsearcher` were built successfully.

Run:

```powershell
python experiments\external_baseline.py sptag-build --threads 4 --timeout 600
python experiments\external_baseline.py sptag-search --maxcheck 1024 --k 10 --threads 1 --timeout 600
python experiments\external_baseline.py eval-sptag --k 10
```

Important wrapper details:

- SPTAG defaults to `Cosine`; wrapper passes `Index.DistCalcMethod=L2`.
- SPTAG treats truth as binary only when the filename contains `bin`; wrapper uses
  `data/external_10k/truth.bin`.

Observed:

```text
BuildGraph time: 18 s
QPS: 19550.3418
Recall@10: 0.9957
```

SPTAG's own printed `recall=0.1000` uses `truthK=100` as denominator, so the
separate `eval-sptag` command reports standard Recall@10.

## DiskANN

Wrapper/config status:

```powershell
python experiments\external_baseline.py diskann-config --search-l 20,40,80,120 --threads 4 --start-point-strategy medoid
```

Generated:

```text
results/external/diskann_benchmark.json
```

Status:

- Rust/Cargo was installed in WSL through user-level `rustup`.
- `diskann-benchmark` built successfully with `RUSTFLAGS="-Ctarget-cpu=x86-64-v3"`.
- The wrapper writes `diskann_benchmark.wsl.json` with `/mnt/c/...` paths before running DiskANN in WSL.

Run:

```powershell
.\scripts\build_external_baselines.ps1 -Baseline diskann -UseWsl
python experiments\external_baseline.py diskann-run --timeout 1200
```

Observed:

```text
Index Build Time: 0.214593 s
Vectors Inserted: 10000

search_l  Recall@10  Avg cmps  Avg hops  best p99 latency  best QPS
20        0.9226     577.84    24.17     49 us             154918.67
40        0.9762     821.17    43.66     54 us             116959.06
80        0.9890     1096.63   83.38     89 us             86296.17
120       0.9903     1244.07   123.29    70 us             84559.45
```

Artifacts:

```text
results/external/diskann_benchmark.json
results/external/diskann_benchmark.wsl.json
results/external/diskann_output.json
results/external/diskann/
```

## Dynamic Anchor Prototype

Run at the same order of magnitude as the external baseline data, not as a
small smoke test:

```powershell
python experiments\run_streaming_demo.py --n-initial 10000 --n-insert 2000 --n-delete 1000 --n-queries 1000 --dim 64 --anchors 64 --entries 4 --ef 120 --graph-k 16 --out-dir results\own_10k
python experiments\run_streaming_demo.py --n-initial 10000 --n-insert 2000 --n-delete 1000 --n-queries 1000 --dim 64 --anchors 128 --entries 8 --ef 240 --graph-k 32 --out-dir results\own_10k_k32
```

Observed, stronger `graph-k=32 / anchors=128 / entries=8` setting:

```text
phase         router           Recall@10  avg visited  p99 latency  anchors
before drift  fixed_medoid     0.2091     12.886       6.514 ms     1
before drift  random_multi     0.7943     12.181       6.753 ms     128
before drift  static_density   0.7969     11.655       3.172 ms     128
before drift  dynamic_density  0.7969     11.655       3.149 ms     128
after drift   fixed_medoid     0.0954     12.123       2.972 ms     1
after drift   random_multi     0.3927     12.476       3.159 ms     128
after drift   static_density   0.3939     12.827       3.106 ms     128
after drift   dynamic_density  0.8133     11.727       3.207 ms     120
```

Interpretation:

- The current Python backend is a research prototype, not a production graph index.
- The dynamic routing layer still shows the intended effect: after clustered drift,
  dynamic anchors recover recall from about `0.394` to `0.813`.
- The next engineering step is to move this anchor layer onto HNSW/Vamana-style
  backends so the absolute recall/latency can be compared fairly with DiskANN,
  SPTAG, and NSG.

## Conservative Fair Comparison

Run:

```powershell
python experiments\run_fair_comparison.py --n-initial 10000 --n-insert 2000 --n-delete 1000 --n-queries 1000 --dim 64 --graph-k 16 --ef 240 --anchors 128 --entries 8 --diskann-search-l 20,40,80,120,240 --timeout 1200
python experiments\run_own_fair_sweep.py --efs 20,40,80,120,240 --n-initial 10000 --n-insert 2000 --n-delete 1000 --n-queries 1000 --dim 64 --graph-k 16 --anchors 128 --entries 8 --out-dir results\fair_10k
python experiments\summarize_fair_results.py --fair-dir results\fair_10k
```

Artifacts:

```text
results/fair_10k/fair_summary.csv
results/fair_10k/own_ef_sweep.csv
results/fair_10k/external_baselines.csv
results/fair_10k/fair_recall_cost.csv
results/fair_10k/fair_recall_cost.json
```

Protocol:

- Initial index contains 10,000 vectors.
- Stream inserts 2,000 drifted vectors and deletes 1,000 old vectors.
- The dynamic-anchor prototype updates without graph rebuild.
- NSG, SPTAG, and DiskANN are rebuilt on the final alive dataset. This is
  intentionally favorable to the external baselines.

After-drift own Python graph sweep:

```text
method           budget  Recall@10  avg visited  p99 latency
fixed_medoid     ef=240  0.1219     240.0        10.91 ms
static_density   ef=240  0.4986     240.0        11.05 ms
dynamic_density  ef=20   0.7424     20.0         2.60 ms
dynamic_density  ef=40   0.9007     40.0         4.02 ms
dynamic_density  ef=80   0.9720     80.0         6.24 ms
dynamic_density  ef=120  0.9891     120.0        7.93 ms
dynamic_density  ef=240  0.9984     240.0        10.86 ms
```

Rebuilt external baselines on the same final alive dataset:

```text
backend  budget            Recall@10  avg cmps  avg hops  best p99  best QPS
NSG      search_l=120      0.9977     n/a       n/a       n/a       n/a
SPTAG    maxcheck=1024     0.9864     n/a       n/a       n/a       n/a
DiskANN  search_l=20       0.8844     675.58    25.74     65 us     135611.61
DiskANN  search_l=40       0.9707     991.28    45.36     79 us     80173.17
DiskANN  search_l=80       0.9948     1385.59   84.93     106 us    66684.45
DiskANN  search_l=120      0.9987     1621.16   124.75    90 us     67535.63
DiskANN  search_l=240      1.0000     1955.49   244.59    143 us    47723.58
```

Interpretation:

- The entry-layer hypothesis survives the conservative comparison: under drift,
  dynamic density anchors recover high recall without rebuilding the graph,
  while fixed-medoid and static-density entry sets collapse.
- The current latency numbers are not a production speed comparison. The own
  backend is a Python research graph, while DiskANN/NSG/SPTAG are optimized
  Rust/C++ implementations.
- The next fair systems step is to attach the same dynamic anchor router to a
  compiled HNSW/Vamana-style backend, or to expose multi-start anchor entry
  points inside DiskANN/Vamana directly.

## GATE Static Hub Adapter

Implementation:

```text
experiments/run_gate_adapter.py
```

This adapter uses the GATE C++ search executable with generated two-level
centroid hubs:

- `HUB_PATH`: text hub file generated from k-means centroids.
- `EMB_PATH`: fvecs centroid embeddings.
- Search: GATE's `test_gate_search` binary.

It does not train GATE's two-tower model, so this is a GATE-style static
hub-entry baseline rather than a paper-faithful GATE result.

Run:

```powershell
.\scripts\build_external_baselines.ps1 -Baseline gate -UseWsl
python experiments\run_gate_adapter.py --manifest data\fair_10k\manifest.json --search-l 80,120,240 --timeout 1200
python experiments\summarize_fair_results.py --fair-dir results\fair_10k --hnsw-dir results\hnsw_anchor_10k --gate-dir results\gate_adapter_10k
```

Artifacts:

```text
results/gate_adapter_10k/gate_adapter_results.csv
results/gate_adapter_10k/gate_adapter_meta.json
results/gate_adapter_10k/gate_hubs.txt
results/gate_adapter_10k/gate_centroid_embeddings.fvecs
results/gate_adapter_10k/gate_final.nsg
```

Observed on the same dynamic-drift `fair_10k` data:

```text
backend       method                    search_l  Recall@10  QPS
gate_adapter  static_centroid_hub_entry  80       0.4653     389.812
gate_adapter  static_centroid_hub_entry  120      0.4509     524.521
gate_adapter  static_centroid_hub_entry  240      0.7349     367.680
```

Interpretation:

- This is the closest currently executable entry-overlay baseline.
- It is intentionally static: hubs are generated from the base data and are not
  split, retired, or reweighted after drift.
- In the current setup, dynamic density anchors outperform this static hub
  overlay at the same broad search-budget scale, but the final paper must still
  either run paper-faithful GATE or clearly label this as a static adapter.

## Compiled HNSW Anchor Probe

Implementation:

```text
tools/hnsw_anchor_runner.cpp
experiments/run_hnsw_anchor_fair.py
```

The C++ runner uses local `third_party/hnswlib` directly. It builds a normal
HNSW index on the initial 10,000 vectors, inserts 2,000 drifted vectors, marks
1,000 deleted vectors, and then searches the final after-drift queries without
rebuilding the graph.

Two search modes are measured:

- `builtin_hnsw_entry`: standard hnswlib search through HNSW's own hierarchy.
- `*_density` / `fixed_medoid`: custom layer-0 search from selected anchor
  entry points via `searchBaseLayerST`.

Run:

```powershell
python experiments\run_hnsw_anchor_fair.py --efs 20,40,80,120,240 --timeout 1200
```

Artifacts:

```text
tools/hnsw_anchor_runner.cpp
build/hnsw_anchor_runner
data/hnsw_anchor_10k/
results/hnsw_anchor_10k/hnsw_anchor_fair.csv
results/hnsw_anchor_10k/maintenance.json
```

Observed:

```text
backend                method              ef   Recall@10  avg cmps  avg hops  p99 latency  QPS
hnswlib builtin        builtin_hnsw_entry   20  0.7925     57.98     8.59      45.06 us     91943.2
hnswlib builtin        builtin_hnsw_entry   80  0.9833     57.98     8.59      79.92 us     29332.5
hnswlib builtin        builtin_hnsw_entry  240  0.9993     57.98     8.59      138.26 us    12960.9

hnswlib layer-0 anchor fixed_medoid         80  0.6132     1802.71   81.81     94.91 us     26165.5
hnswlib layer-0 anchor static_density       80  0.8174     3277.24   134.44    82.04 us     19400.7
hnswlib layer-0 anchor dynamic_density      80  0.8473     2891.84   117.33    76.29 us     22979.5

hnswlib layer-0 anchor fixed_medoid        240  0.6216     5192.04   241.14    150.50 us    11264.1
hnswlib layer-0 anchor static_density      240  0.9085     7130.75   295.90    217.18 us    7960.2
hnswlib layer-0 anchor dynamic_density     240  0.9180     6754.06   279.19    224.98 us    8882.0
```

Interpretation:

- Within the same anchor-start layer-0 HNSW mode, dynamic density anchors beat
  static density and fixed medoid after drift. At `ef=240`, dynamic improves
  Recall@10 from `0.9085` to `0.9180` over static density and from `0.6216`
  over fixed medoid.
- Standard HNSW's built-in hierarchical entry remains much stronger. This is
  expected: HNSW already has an upper routing layer, so replacing it with a
  flat layer-0 anchor start is not the best target for the paper claim.
- The result sharpens the positioning: dynamic density anchors are most
  relevant to single-layer graph systems such as NSG/Vamana/DiskANN-style graph
  search, where entry quality is not already protected by an HNSW hierarchy.

## Topology + Entry 2x2 Ablation

Implementation:

```text
DynamicKNNGraph.repair_after_delete(...)
experiments/run_topology_entry_ablation.py
```

The local topology repair is intentionally not a full rebuild. After dynamic
insert/delete, it only repairs nodes affected by deleted neighbors plus newly
inserted nodes, restoring local degree with nearest alive nodes.

Run:

```powershell
python experiments\run_topology_entry_ablation.py --efs 240
python experiments\run_topology_entry_ablation.py --efs 80 --out-dir results\topology_entry_10k_ef80
```

Artifacts:

```text
results/topology_entry_10k/topology_entry_ablation.csv
results/topology_entry_10k/topology_entry_ablation.json
results/topology_entry_10k_ef80/topology_entry_ablation.csv
results/topology_entry_10k_ef80/topology_entry_ablation.json
```

After-drift results:

```text
ef   topology      router           Recall@10  avg visited  avg depth  anchors
80   no_repair     fixed_medoid     0.1191     80.0         14.088     1
80   no_repair     static_density   0.4821     80.0         11.819     128
80   no_repair     dynamic_density  0.9720     80.0         10.098     121
80   local_repair  fixed_medoid     0.1191     80.0         14.088     1
80   local_repair  static_density   0.4872     80.0         11.596     128
80   local_repair  dynamic_density  0.9796     80.0         9.820      121

240  no_repair     fixed_medoid     0.1219     240.0        15.756     1
240  no_repair     static_density   0.4986     240.0        13.367     128
240  no_repair     dynamic_density  0.9984     240.0        11.333     121
240  local_repair  fixed_medoid     0.1219     240.0        15.756     1
240  local_repair  static_density   0.4997     240.0        13.089     128
240  local_repair  dynamic_density  0.9997     240.0        10.949     121
```

Interpretation:

- Topology repair alone does not rescue a bad fixed entry: fixed-medoid recall
  stays at `0.1191` for `ef=80` and `0.1219` for `ef=240`.
- Dynamic entry maintenance alone is the main effect under drift: without graph
  repair, dynamic anchors raise Recall@10 from static's `0.4821` to `0.9720`
  at `ef=80`, and from `0.4986` to `0.9984` at `ef=240`.
- Combining local topology repair with dynamic anchors is best: at `ef=80`,
  Recall@10 improves from `0.9720` to `0.9796`; at `ef=240`, it improves from
  `0.9984` to `0.9997`, with lower average search depth.
- This supports the positioning that graph topology maintenance and entry-layer
  maintenance are complementary: repairing edges makes the graph easier to
  traverse, but dynamic anchors determine whether the search enters the right
  region in the first place.
