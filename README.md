# Dynamic Anchor ANNS

Research prototype for **density-aware dynamic entry routing** in graph-based
approximate nearest neighbor search.

The project hypothesis is narrow on purpose:

> Dynamic ANNS has two different maintenance problems: maintaining graph edges
> and maintaining entry quality. Most dynamic graph indexes repair edges or
> storage partitions; this project maintains a lightweight routing-anchor layer
> above a proximity graph.

The current code is a compact Python prototype. It is meant to validate the
idea and produce early figures before integrating heavier baselines such as
DiskANN, SPTAG, or NSG.

## Layout

```text
dynamic-anchor-anns/
  docs/
    baselines.md              Baseline map and commit pins.
    research_plan.md          Problem framing and experiment plan.
  experiments/
    run_streaming_demo.py     One-command streaming drift demo.
  src/dynanchor/
    graph.py                  Inspectable dynamic kNN graph backend.
    anchors.py                Medoid, random, static density, dynamic density routers.
    data.py                   Synthetic clustered drift workload.
    evaluation.py             Recall, visited-node, latency, load metrics.
  third_party/
    hnswlib/                  Lightweight HNSW baseline source.
    nsg/                      NSG baseline source.
    DiskANN/                  Microsoft DiskANN/Vamana baseline source.
    SPTAG/                    Microsoft SPTAG baseline source.
```

## Quick Start

```powershell
cd C:\Users\zhangyue\Desktop\compression\dynamic-anchor-anns
python experiments\run_streaming_demo.py
```

For a faster smoke run:

```powershell
python experiments\run_streaming_demo.py --n-initial 300 --n-insert 80 --n-delete 30 --n-queries 80 --dim 12 --anchors 16 --entries 4 --ef 60 --graph-k 10
```

To generate recall-cost curves across graph-search budgets:

```powershell
python experiments\run_ef_sweep.py
```

The script writes:

```text
results/streaming_demo_results.json
results/streaming_demo_results.csv
results/ef_sweep_results.json
results/ef_sweep_results.csv
```

## Implemented Baselines

The first prototype includes in-process routers over the same graph backend:

- `fixed_medoid`: fixed single entry point.
- `random_multi`: static random multi-entry anchors.
- `static_density`: density-aware anchors selected once, no maintenance.
- `dynamic_density`: density-aware anchors with split/retire maintenance.

The important comparison is:

```text
static_density vs dynamic_density after clustered insert/delete drift
```

That isolates whether maintaining entry coverage helps beyond simply having
good initial anchors.

## What The Dynamic Anchor Layer Does

Anchors are routing objects, not storage partitions.

During queries:

```text
query -> select s anchors -> graph search from anchors -> merge top-k
```

During maintenance, the layer monitors coverage and traversal telemetry:

- local density growth,
- coverage holes,
- overloaded anchors,
- high visited-node cost,
- low-utility anchors.

It can add or retire anchors without moving vectors between posting lists. This
is the intended separation from SPANN/SPFresh-style partition maintenance.

## Next Integration Steps

1. Replace the toy `DynamicKNNGraph` backend with adapters for `hnswlib`.
2. Build and run external wrappers for NSG, DiskANN/Vamana, and SPTAG.
3. Add paper baselines for GATE, Adaptive Entry Point Selection, DiskANN++, and
   Dynamically Detect and Fix Hardness when code is available.
4. Move from synthetic drift to SIFT/DEEP/GIST-style workloads with clustered
   inserts, clustered deletes, OOD queries, and hot-query skew.

## External Baseline Wrappers

Prepare a non-smoke benchmark set:

```powershell
python experiments\external_baseline.py prepare-data --n-base 10000 --n-query 1000 --dim 64
```

Probe whether external executables are available:

```powershell
python experiments\external_baseline.py probe
```

Build external projects when compiler toolchains are installed:

```powershell
.\scripts\build_external_baselines.ps1 -Baseline nsg
.\scripts\build_external_baselines.ps1 -Baseline sptag
.\scripts\build_external_baselines.ps1 -Baseline diskann
```

On this Windows workspace, NSG/SPTAG usually build more cleanly through WSL:

```powershell
.\scripts\build_external_baselines.ps1 -Baseline nsg -UseWsl
.\scripts\build_external_baselines.ps1 -Baseline sptag -UseWsl
.\scripts\build_external_baselines.ps1 -Baseline diskann -UseWsl
```

Run wrappers:

```powershell
python experiments\external_baseline.py nsg-build
python experiments\external_baseline.py nsg-search
python experiments\external_baseline.py sptag-build
python experiments\external_baseline.py sptag-search
python experiments\external_baseline.py diskann-config
python experiments\external_baseline.py diskann-run
```
