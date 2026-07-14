# Baseline Map

This document separates executable source baselines from paper baselines that
must be implemented, wrapped, or cited.

## Pulled GitHub Baselines

| Baseline | Local path | Commit | Role |
|---|---|---:|---|
| hnswlib | `third_party/hnswlib` | `d9b3608c83d83b46c96e25088cb1d729b29dcfe9` | Lightweight dynamic graph baseline and first adapter target. |
| NSG | `third_party/nsg` | `5ec8fadf6af71b6b48a486bbaa28c8b7cae34275` | Single-layer graph / fixed navigation baseline. |
| DiskANN | `third_party/DiskANN` | `603d5967108a03db9980cf23099169dc94e856e7` | Vamana/DiskANN high-recall graph baseline. |
| SPTAG | `third_party/SPTAG` | `87054512780223872e4e779242c01d54c5e9f91a` | Tree-seeded graph search baseline. |
| GATE | `third_party/GATE/jacksondca-gate-78334b4` | Zenodo `15523071` | Closest entry-overlay baseline; static centroid-hub adapter runs, paper-faithful run needs learned hub embeddings. |
| Steiner-hardness | `third_party/Steiner-hardness` | `823737a085bb6501847f486066e0b23acf725617` | Hard-query measurement and workload generation, not an RFix/NGFix repair baseline. |

## Executable Wrappers

The project now includes wrappers in `experiments/external_baseline.py`.
Current run status is tracked in `docs/external_status.md`.

Dataset preparation:

```powershell
python experiments\external_baseline.py prepare-data --n-base 10000 --n-query 1000 --dim 64
```

Build/probe:

```powershell
python experiments\external_baseline.py probe
.\scripts\build_external_baselines.ps1 -Baseline nsg
.\scripts\build_external_baselines.ps1 -Baseline sptag
.\scripts\build_external_baselines.ps1 -Baseline diskann
```

WSL/Linux build path:

```powershell
.\scripts\build_external_baselines.ps1 -Baseline nsg -UseWsl
.\scripts\build_external_baselines.ps1 -Baseline sptag -UseWsl
.\scripts\build_external_baselines.ps1 -Baseline diskann -UseWsl
.\scripts\build_external_baselines.ps1 -Baseline gate -UseWsl
```

Close baseline status is tracked separately:

```powershell
python experiments\close_baseline_status.py
```

See `docs/close_baselines.md`.

Run:

```powershell
python experiments\external_baseline.py nsg-build
python experiments\external_baseline.py nsg-search
python experiments\external_baseline.py sptag-build
python experiments\external_baseline.py sptag-search
python experiments\external_baseline.py diskann-config
python experiments\external_baseline.py diskann-run
python experiments\run_gate_adapter.py --manifest data\fair_10k\manifest.json --search-l 80,120,240
```

## Must-Compare Research Baselines

These are the papers the proposal must defend against.

| Work | Why it matters | Difference we must preserve |
|---|---|---|
| GATE | Hub/candidate entry points plus query-aware entry selection. | GATE optimizes entry recommendation on a mostly static graph; we maintain entry coverage under base-data insert/delete drift. |
| Steiner-hardness | Query hardness metric and unbiased hard-query workload generation. | It is an evaluation/diagnostic tool; it does not repair hard regions or maintain entries. |
| Adaptive Entry Point Selection | Shows fixed central entry is weak and adaptive entry matters. | We focus on dynamic maintenance of the entry set, not only selecting from a static entry set. |
| DiskANN++ | Query-sensitive entry vertex for disk ANN. | We maintain a dynamic routing-anchor layer and measure update/drift behavior. |
| SPANN / SPFresh | Dynamic centroid/posting-list partition maintenance. | Their centroids own storage partitions; our anchors do not move data or define posting lists. |
| FreshDiskANN / IP-DiskANN | Dynamic graph updates for insert/delete. | They repair graph/index structure; we isolate entry-quality maintenance. |
| Dynamically Detect and Fix Hardness | Detects hard query regions and repairs graph edges. | It fixes roads by adding/repairing edges; we change and maintain entry routing anchors. |
| CDMG | Robust OOD vector search via cross-distribution monotonic graph. | It changes graph navigability; our layer is a lightweight entry-routing overlay. |

## Experimental Baseline Groups

Minimum paper-grade baseline groups:

- Fixed entry: medoid, NSG-style root.
- Static multi-entry: random anchors, Latin hypercube/farthest anchors, static density anchors.
- Adaptive entry: GATE-style entry selection and Adaptive Entry Point Selection.
- Dynamic graph: HNSW updates, FreshDiskANN/IP-DiskANN, Greator/SPatch if reproducible.
- Partition systems: SPANN/SPFresh.
- Industrial seed systems: SPTAG tree seeds plus graph search.

## Required Ablations

- No dynamic maintenance: static density anchors.
- No density term: farthest-only anchors.
- No load term: dynamic anchors without load balancing.
- No split: retire-only maintenance.
- No retire: split-only maintenance.
- Query-aware only: re-rank anchors by recent success without changing anchor set.
- Data-drift only: maintain anchors from data statistics without query telemetry.
