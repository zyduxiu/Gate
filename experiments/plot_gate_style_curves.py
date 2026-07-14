from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STYLE = {
    "GATE": {"color": "#3f6fb5", "marker": "o", "linestyle": "-"},
    "gate_initial_hub": {"color": "#d97745", "marker": "x", "linestyle": "--"},
    "gate_refreshed_hub": {"color": "#3f6fb5", "marker": "o", "linestyle": "-"},
    "query_entries_paper_gate_online_two_tower": {"color": "#7c5ab6", "marker": "D", "linestyle": "-."},
    "query_entries_hybrid_gate_first_n1_g3": {"color": "#0f8b8d", "marker": "P", "linestyle": "-"},
    "query_entries_hybrid_gate_first_n2_g2": {"color": "#2f9e44", "marker": "X", "linestyle": "-"},
    "query_entries_hybrid_gate_first_n3_g1": {"color": "#2f9e44", "marker": "P", "linestyle": "-"},
    "online_hub_density": {"color": "#2f9e44", "marker": "s", "linestyle": ":"},
    "online_density": {"color": "#13a8a8", "marker": "^", "linestyle": "-."},
    "nsg_ep": {"color": "#7c5ab6", "marker": "D", "linestyle": "-."},
    "fixed_medoid": {"color": "#8c6d55", "marker": "*", "linestyle": ":"},
    "static_density": {"color": "#b23a48", "marker": "+", "linestyle": "--"},
    "dynamic_density": {"color": "#17a2b8", "marker": "v", "linestyle": "-."},
}


DISPLAY = {
    "gate_initial_hub": "GATE-stale",
    "gate_refreshed_hub": "GATE-refresh",
    "query_entries_paper_gate_online_two_tower": "GATE over our E",
    "query_entries_hybrid_gate_first_n1_g3": "Hybrid G1+O3",
    "query_entries_hybrid_gate_first_n2_g2": "Hybrid G2+O2",
    "query_entries_hybrid_gate_first_n3_g1": "Hybrid",
    "online_hub_density": "Ours",
    "online_density": "Ours-density",
    "nsg_ep": "NSG-EP",
    "fixed_medoid": "Medoid",
    "static_density": "Static-density",
    "dynamic_density": "Density-refresh",
}


def read_inputs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("no input CSV files")
    return pd.concat(frames, ignore_index=True)


def pick_column(frame: pd.DataFrame, requested: str, fallbacks: list[str]) -> str:
    if requested in frame.columns:
        return requested
    for name in fallbacks:
        if name in frame.columns:
            return name
    raise KeyError(f"missing column {requested!r}; available columns: {list(frame.columns)}")


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GATE-style Recall-QPS curves from result CSV files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/figures/gate_style_recall_qps.png"))
    parser.add_argument("--recall-col", default="recall_at_10")
    parser.add_argument("--qps-col", default="qps")
    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--panel-col", default="")
    parser.add_argument("--method-col", default="method")
    parser.add_argument("--methods", default="")
    parser.add_argument("--title", default="Dynamic Entry Maintenance")
    parser.add_argument("--xlabel", default="")
    parser.add_argument("--ylabel", default="QPS")
    parser.add_argument("--scale-y-thousands", action="store_true")
    parser.add_argument("--annotate-search-l", action="store_true")
    parser.add_argument("--pareto-frontier", action="store_true")
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--max-recall", type=float, default=None)
    args = parser.parse_args()

    frame = read_inputs(args.inputs)
    recall_col = pick_column(frame, args.recall_col, ["recall_at_10", "recall_median", "recall_mean", "recall_med"])
    qps_col = pick_column(frame, args.qps_col, ["qps", "qps_median", "qps_mean", "qps_med"])
    method_col = pick_column(frame, args.method_col, ["method"])

    if args.dataset_col not in frame.columns:
        frame[args.dataset_col] = "dataset"
    if args.panel_col and args.panel_col not in frame.columns:
        frame[args.panel_col] = "panel"

    frame["_recall"] = numeric(frame, recall_col)
    frame["_qps"] = numeric(frame, qps_col)
    frame = frame.dropna(subset=["_recall", "_qps", method_col])
    if args.min_recall is not None:
        frame = frame[frame["_recall"] >= args.min_recall]
    if args.max_recall is not None:
        frame = frame[frame["_recall"] <= args.max_recall]
    if args.methods:
        wanted = [item.strip() for item in args.methods.split(",") if item.strip()]
        frame = frame[frame[method_col].isin(wanted)]

    if frame.empty:
        raise SystemExit("No rows left after filtering.")

    panel_col = args.panel_col or "__single_panel"
    if panel_col == "__single_panel":
        frame[panel_col] = args.title

    datasets = list(dict.fromkeys(frame[args.dataset_col].astype(str)))
    panels = list(dict.fromkeys(frame[panel_col].astype(str)))
    fig, axes = plt.subplots(
        nrows=len(panels),
        ncols=len(datasets),
        figsize=(max(4.0, 3.2 * len(datasets)), max(3.2, 2.8 * len(panels))),
        squeeze=False,
        sharey=False,
    )

    handles = {}
    for row_idx, panel in enumerate(panels):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            sub = frame[(frame[panel_col].astype(str) == panel) & (frame[args.dataset_col].astype(str) == dataset)]
            for method, group in sub.groupby(method_col, sort=False):
                if args.pareto_frontier:
                    frontier_rows = []
                    best_qps = -math.inf
                    for _, point in group.sort_values(["_recall", "_qps"], ascending=[False, False]).iterrows():
                        if point["_qps"] > best_qps:
                            frontier_rows.append(point)
                            best_qps = point["_qps"]
                    group = pd.DataFrame(frontier_rows)
                group = group.sort_values("_recall")
                style = STYLE.get(str(method), {})
                y = group["_qps"] / 1000.0 if args.scale_y_thousands else group["_qps"]
                (line,) = ax.plot(
                    group["_recall"],
                    y,
                    label=DISPLAY.get(str(method), str(method)),
                    linewidth=1.8,
                    markersize=5.5,
                    **style,
                )
                handles[DISPLAY.get(str(method), str(method))] = line
                for _, point in group.iterrows():
                    if args.annotate_search_l and "search_l" in group.columns and not math.isnan(pd.to_numeric(point.get("search_l"), errors="coerce")):
                        ax.annotate(
                            str(int(float(point["search_l"]))),
                            (point["_recall"], (point["_qps"] / 1000.0 if args.scale_y_thousands else point["_qps"])),
                            textcoords="offset points",
                            xytext=(3, 3),
                            fontsize=7,
                            alpha=0.75,
                        )
            ax.grid(True, alpha=0.28, linewidth=0.8)
            ax.set_title(dataset if len(panels) == 1 else f"{panel} / {dataset}", fontsize=10)
            ax.set_xlabel(args.xlabel or recall_col.replace("_", "@").replace("recall@at", "Recall@"))
            if col_idx == 0:
                ax.set_ylabel("QPS (x1000)" if args.scale_y_thousands else args.ylabel)

    if handles:
        fig.legend(
            handles.values(),
            handles.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(5, len(handles)),
            frameon=True,
            fontsize=9,
        )
    fig.suptitle(args.title, y=1.08 if handles else 1.0, fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    if args.out.suffix.lower() != ".pdf":
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
