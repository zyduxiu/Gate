# NGFix/RFix Official Artifact Baseline

Date: 2026-07-11

This is the close hard-region repair baseline for:

```text
Dynamically Detect and Fix Hardness for Efficient ANNS, SIGMOD 2026
```

## Status

The official artifact is cloned at:

```text
third_party/NGFix
```

We added a small compatibility switch:

```text
NGFIX_PORTABLE_DISTANCE
```

Default/off keeps the artifact's AVX-512 distance kernels. On this machine WSL
does not expose `avx512f`, so the executed run uses scalar distance kernels.
This keeps the NGFix/RFix graph-repair logic but should be reported as a
portable compatibility build, not as the authors' AVX-512 performance setup.

## Command

```powershell
python experiments\run_ngfix_baseline.py --force-rebuild --jobs 2 --timeout 3600
```

If the portable build already exists:

```powershell
python experiments\run_ngfix_baseline.py --skip-build --timeout 3600
```

## Outputs

```text
results/ngfix_official_sift100k/ngfix_official_summary.csv
results/ngfix_official_sift100k/ngfix_run_meta.json
results/ngfix_official_sift100k/logs/*.log.json
```

## Result Snapshot

Dataset: SIFT1M-derived dynamic subset, final alive base of 110k vectors,
1k queries, Recall@10.

| ef | Recall@10 | Avg distance comps | Avg latency |
|---:|---:|---:|---:|
| 10 | 1.0000 | 326.73 | 49.6 us |
| 100 | 1.0000 | 1501.26 | 361.9 us |
| 300 | 1.0000 | 3405.57 | 908.6 us |

## Interpretation

NGFix/RFix is very strong here, but it is not the same experimental axis as our
entry overlay. It repairs graph reachability/local hard regions on an
HNSW-NGFix backend. Our controlled NSG/Vamana experiments keep the graph fixed
and change only the entry layer.

Paper-safe statement:

> NGFix/RFix addresses hard-region graph repair; our method addresses online
> maintenance of the entry candidate set under insert/delete drift. The two are
> complementary and should be compared both as separate system baselines and,
> where possible, in a crossed entry x repair ablation.
