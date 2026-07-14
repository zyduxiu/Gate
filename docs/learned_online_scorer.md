# Learned Online Entry-Set Scorer

## Goal

The online entry maintainer chooses new candidate anchors from each inserted batch.  The original scorer is linear over four normalized features:

- `coverage_far`: distance to the nearest current anchor.
- `density`: local neighbor density.
- `hubness`: graph indegree / navigational centrality.
- `inserted_bonus`: whether the candidate is from the inserted stream.

This note records a lightweight MLP variant for the same online maintenance path.  The main paper claim should still be entry-set maintenance, not the MLP itself.

## Implementation

`tools/nsg_dynamic_entry_runner.cpp` now accepts:

```text
--online-mlp-model path/to/model.txt
```

The model is a small ReLU MLP over `[coverage_far, density, hubness, inserted_bonus]`.  The text format is:

```text
hidden
w1 row-major, hidden * 4 numbers
b1, hidden numbers
w2, hidden numbers
b2
```

`experiments/sweep_online_mlp_scorer.py` generates several scorer variants and runs them through the same `online_hub_density` maintenance path.

## SIFT Stress Results

Workload: `data/sift_stress_insert80_queries/manifest.json`

Backend: same no-rebuild dynamic NSG graph, `topk=100`, `entries=4`, `maintain_sample=512`, `maintain_changes=4`, `maintain_batch=1000`.

3-repeat aggregate:

| scorer | L | Recall@100 mean | QPS mean | QPS median | p99 us median | distance comps median |
|---|---:|---:|---:|---:|---:|---:|
| mlp_conservative_hub | 240 | 0.994658 | 764.53 | 796.11 | 2008.88 | 4670.16 |
| mlp_linear_query_fresh | 240 | 0.994636 | 743.09 | 797.79 | 2205.82 | 4674.17 |
| mlp_conservative_hub | 280 | 0.996049 | 695.24 | 694.47 | 2336.03 | 5232.82 |
| mlp_linear_query_fresh | 280 | 0.996026 | 629.23 | 683.33 | 2386.17 | 5237.88 |
| mlp_conservative_hub | 320 | 0.997042 | 625.09 | 622.23 | 2416.57 | 5768.80 |
| mlp_linear_query_fresh | 320 | 0.997020 | 592.88 | 618.95 | 2653.05 | 5774.24 |

Output files:

- `results/online_mlp_scorer_sweep_insert80/aggregate.csv`
- `results/online_mlp_scorer_sweep_insert80_top3_3runs/aggregate_3runs.csv`

## Interpretation

The MLP scorer is useful, but it is not the main novelty.

`mlp_conservative_hub` is the best current MLP variant: it keeps hubness strong and only promotes new inserted-region candidates when coverage is clearly stale.  It slightly improves Recall@100 and distance computations over the MLP-linear proxy, with more stable QPS at `L=280` and `L=320`.

Compared with the earlier linear scorer sweep, the MLP gains are modest.  The strongest linear policies (`query_fresh_proxy`, `density_heavy`) are already competitive.  For a paper, this should be framed as an optional learned policy ablation:

> The entry-set maintenance framework supports both hand-designed and learned scorers.  A lightweight MLP can capture feature interactions such as uncovered-and-hub-like candidate regions, but the larger contribution is maintaining a fresh entry set under stream drift.

## Next Step

A fully reviewer-safe learned scorer should train on a separate query split or earlier stream window and evaluate on held-out queries / later stream windows.  Do not tune the MLP directly on the final reported test queries.

## Held-Out Train/Eval Selection

Implemented:

- `experiments/split_query_manifest.py`
- `experiments/train_eval_online_mlp_scorer.py`

Run:

```text
python experiments/train_eval_online_mlp_scorer.py --out-dir results/online_mlp_train_eval_insert80 --search-l 240,280,320 --select-l 280 --topk 100 --entries 4 --maintain-sample 512 --maintain-changes 4 --maintain-batch 1000 --random-configs 8 --eval-repeats 3 --timeout 7200
```

Split:

- Train queries: 226
- Eval queries: 227
- Source workload: `data/sift_stress_insert80_queries/manifest.json`

The train split selected `mlp_random_03` at `L=280` when using the rule "highest QPS among candidates within recall slack".

Held-out eval aggregate:

| method | L | Recall@100 mean | QPS mean | QPS median | p99 us median | distance comps median |
|---|---:|---:|---:|---:|---:|---:|
| online_default_linear | 240 | 0.994009 | 868.39 | 846.38 | 1828.91 | 4672.31 |
| selected_mlp_random_03 | 240 | 0.994097 | 854.69 | 853.25 | 1810.12 | 4667.84 |
| gate_refreshed_hub | 240 | 0.994009 | 841.02 | 843.80 | 1932.56 | 4680.21 |
| nsg_ep | 240 | 0.994141 | 815.46 | 815.97 | 1899.12 | 4863.32 |
| online_default_linear | 280 | 0.995463 | 731.98 | 732.68 | 2031.77 | 5233.32 |
| selected_mlp_random_03 | 280 | 0.995507 | 705.36 | 707.82 | 2098.60 | 5229.86 |
| gate_refreshed_hub | 280 | 0.995507 | 661.22 | 730.95 | 2258.95 | 5240.79 |
| nsg_ep | 280 | 0.995639 | 723.47 | 717.28 | 2154.70 | 5435.89 |
| online_default_linear | 320 | 0.996564 | 682.19 | 683.95 | 2205.36 | 5766.13 |
| selected_mlp_random_03 | 320 | 0.996608 | 656.42 | 654.05 | 2379.15 | 5763.89 |
| gate_refreshed_hub | 320 | 0.996564 | 657.20 | 673.71 | 2429.03 | 5773.73 |
| nsg_ep | 320 | 0.996784 | 648.17 | 652.83 | 2296.43 | 5972.84 |

Files:

- `results/online_mlp_train_eval_insert80/train_aggregate.csv`
- `results/online_mlp_train_eval_insert80/selected_config.json`
- `results/online_mlp_train_eval_insert80/eval_aggregate.csv`
- `results/online_mlp_train_eval_insert80/eval_aggregate_with_conservative.csv`

Takeaway:

The held-out result is more conservative than the full-query sweep.  The train-selected MLP improves Recall@100 slightly and reduces distance computations, but it does not consistently beat the default linear scorer on QPS.  This is useful evidence: QPS is too noisy to be the only training-selection objective on a 226-query train split.  Future learned scorer selection should use deterministic work metrics, such as distance computations and p99, or use more train repeats with randomized method order.

## Sanity Check And Diagnosis

A direct identity check compares:

- Linear scorer with `--online-score-weights 0.25,0.20,0.35,0.20`
- MLP scorer `mlp_linear_query_fresh`, whose hidden units are identity features and whose output weights are the same four weights

Result on eval split at `L=280`:

- Anchor diff count: `0`
- Linear Recall@100: `0.995463`
- MLP-linear Recall@100: `0.995463`
- Linear avg distance computations: `5232.58`
- MLP-linear avg distance computations: `5232.58`

Files:

- `results/mlp_sanity_identity_check/linear_query_fresh_anchors.txt`
- `results/mlp_sanity_identity_check/mlp_linear_query_fresh_anchors.txt`
- `results/mlp_sanity_identity_check/linear_query_fresh.json`
- `results/mlp_sanity_identity_check/mlp_linear_query_fresh.json`

Diagnosis:

The C++ MLP inference path is not the source of the mismatch.  The current learned scorer is weak because it is not a real supervised MLP.  It is an interpretable feature-gate template plus random validation selection.  The earlier train/eval run selected `mlp_random_03` because the selection objective was train QPS, but train QPS is noisy.  Reselecting by deterministic work metric chooses:

```text
python experiments/reselect_online_mlp_from_train.py --run-dir results/online_mlp_train_eval_insert80 --select-l 280 --selection-objective distance --recall-slack 0.00005
```

Selected config:

- `mlp_conservative_hub`
- Train Recall@100 at L=280: `0.996593`
- Train avg distance computations at L=280: `5236.75`

Practical conclusion:

For the current paper prototype, the safest learned-scorer claim is:

> MLP-style nonlinear candidate scoring is supported, but the current evidence favors stable linear or conservative hub-aware scoring.  A real learned policy needs supervised labels from held-out stream windows, and selection should optimize deterministic work metrics rather than single-run QPS.

## Supervised MLP From Actual Search-Cost Labels

Implemented:

- `experiments/train_supervised_online_mlp.py`

This script samples candidate anchors, probes each candidate with the C++ runner on train queries, and trains a small ReLU MLP from actual per-anchor search costs.  The input features remain the online-maintainable features:

- `coverage_far`
- `density`
- `hubness`
- `inserted_bonus`

The first supervised model used a utility label dominated by normalized recall.  This was a mistake because candidate Recall@100 varied only from `0.996354` to `0.996458`; normalizing such a tiny range amplified noise.  That model performed worse on eval.

The corrected model uses distance-only labels:

```text
python experiments/train_supervised_online_mlp.py --out-dir results/supervised_online_mlp_insert80_distance --candidate-count 512 --label-query-limit 96 --search-l 280 --topk 100 --hidden 8 --epochs 3000 --lr 0.02 --label-objective distance --timeout 1800
```

Training diagnostics:

- Candidate anchors: 512
- Train query labels: 96
- Prediction-label correlation: `0.602`
- Train MSE: `0.0593 -> 0.0223`

Held-out eval, 3 repeats:

| method | L | Recall@100 mean | QPS mean | QPS median | p99 us median | distance comps median |
|---|---:|---:|---:|---:|---:|---:|
| online_default_linear | 240 | 0.994009 | 868.39 | 846.38 | 1828.91 | 4672.31 |
| supervised_mlp_distance | 240 | 0.993965 | 601.15 | 679.70 | 2480.47 | 4663.70 |
| online_default_linear | 280 | 0.995463 | 731.98 | 732.68 | 2031.77 | 5233.32 |
| supervised_mlp_distance | 280 | 0.995463 | 516.09 | 552.81 | 3009.78 | 5225.89 |
| online_default_linear | 320 | 0.996564 | 682.19 | 683.95 | 2205.36 | 5766.13 |
| supervised_mlp_distance | 320 | 0.996564 | 482.15 | 444.42 | 4072.39 | 5758.81 |

Files:

- `results/supervised_online_mlp_insert80_distance/supervised_online_mlp.txt`
- `results/supervised_online_mlp_insert80_distance/supervised_online_mlp_meta.json`
- `results/supervised_online_mlp_insert80_distance/candidate_anchor_costs.csv`
- `results/supervised_online_mlp_insert80_distance_eval/aggregate_3runs.csv`

Interpretation:

The supervised MLP is now genuinely learned from search-cost labels and it does reduce distance computations on held-out queries.  However, the gain is small and does not currently translate into stable QPS or p99 improvements.  The likely reason is that four structural features are too weak to predict query-facing entry utility; many candidate anchors have almost identical high-recall behavior.  To make the learned scorer paper-worthy, the next version needs richer features, especially a historical-query affinity feature or labels generated from the full maintained entry set rather than single-anchor probes.
