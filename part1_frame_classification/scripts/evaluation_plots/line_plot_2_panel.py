import os
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wide_to_long_topk(
    df: pd.DataFrame,
    method_col: str = "name",
    fold_col: str = "fold",
    top_map: Optional[Dict] = None,
    acc_col: str = "acc",
) -> pd.DataFrame:
    if top_map is None:
        top_map = {
            "accuracy": 1,
            "accuracy_top3": 3,
            "accuracy_top5": 5,
        }

    required = {method_col, fold_col, *top_map.keys()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    long = df[[method_col, fold_col] + list(top_map.keys())].melt(
        id_vars=[method_col, fold_col],
        value_vars=list(top_map.keys()),
        var_name="metric",
        value_name=acc_col,
    )
    long["topk"] = long["metric"].map(top_map).astype(int)
    long = long.drop(columns=["metric"])
    return long


def plot_topk_grouped_bars_mean_sd(
    df: pd.DataFrame,
    dataset_order=("optimal", "suboptimal"),
    k_values=(1, 3, 5),
    dataset_col="dataset",
    method_col="name",
    topk_col="topk",
    acc_col="acc",
    sharey=True,
    results_folder: Optional[str] = None,
    out_name: str = "topk_bars.png",
):
    agg = (
        df[df[topk_col].isin(k_values)]
        .groupby([dataset_col, method_col, topk_col])[acc_col]
        .agg(mean="mean", sd="std")
        .reset_index()
    )

    methods = sorted(df[method_col].dropna().unique())
    k_values = list(k_values)

    fig, axes = plt.subplots(1, len(dataset_order), figsize=(15, 6), sharey=sharey)

    fig.subplots_adjust(top=0.9)

    if len(dataset_order) == 1:
        axes = [axes]

    # Professional, muted palette (colorblind-friendly)
    colors = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # green
        "#D55E00",  # vermillion
        "#CC79A7",  # purple
        "#56B4E9",  # extra
        "#F0E442",  # extra
        "#000000",  # extra
    ]

    x = np.arange(len(k_values))  # group centers
    n_methods = len(methods)
    group_width = 0.8
    bar_width = group_width / max(n_methods, 1)

    for ax, ds in zip(axes, dataset_order):
        sub = agg[agg[dataset_col] == ds]

        for i, method in enumerate(methods):
            m = sub[sub[method_col] == method].copy()
            if m.empty:
                continue

            m = m.set_index(topk_col).reindex(k_values)

            y = m["mean"].to_numpy()
            yerr = m["sd"].fillna(0.0).to_numpy()

            offsets = (i - (n_methods - 1) / 2) * bar_width
            xpos = x + offsets

            ax.bar(
                xpos,
                y,
                width=bar_width,
                yerr=yerr,
                capsize=4,
                color=colors[i % len(colors)],
                edgecolor="white",
                linewidth=0.6,
                label=method,
            )

        ax.set_title(ds.capitalize(), fontsize=16)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Top-{k}" for k in k_values], fontsize=13)

        ax.set_xlabel("")  # remove x-axis title
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylim(0, 1.17)

        # Legend INSIDE each subplot (works for both optimal & suboptimal)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.9,
            fontsize=11,
        )

    axes[0].set_ylabel("Accuracy", fontsize=14)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout()

    if results_folder is not None:
        os.makedirs(results_folder, exist_ok=True)
        plt.savefig(os.path.join(results_folder, out_name), dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    results_folder = "/home/bella/Academy/results/Opt_cases"
    excel_path = os.path.join(results_folder, "overall_summary.xlsx")
    sheet_opt = "top1_5_folds"
    sheet_sub = "top_folds_subopt"

    df_opt = pd.read_excel(excel_path, sheet_name=sheet_opt)
    df_sub = pd.read_excel(excel_path, sheet_name=sheet_sub)

    df_opt_long = wide_to_long_topk(df_opt, method_col="name", fold_col="fold")
    df_opt_long["dataset"] = "optimal"

    df_sub_long = wide_to_long_topk(df_sub, method_col="name", fold_col="fold")
    df_sub_long["dataset"] = "suboptimal"

    df_all = pd.concat([df_opt_long, df_sub_long], ignore_index=True)

    plot_topk_grouped_bars_mean_sd(
        df_all,
        results_folder=results_folder,
        out_name="topk_bars.png",
    )
