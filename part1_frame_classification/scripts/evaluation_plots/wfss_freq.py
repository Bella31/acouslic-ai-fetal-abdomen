from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import colorsys


@dataclass(frozen=True)
class WFSSPlotConfig:
    excel_path: str
    wfss_cols: Dict[str, str]
    algorithm_col: str = "Algorithm"
    round_decimals: Optional[int] = 3
    figsize: Tuple[float, float] = (7.6, 8.8)
    colormap: str = "tab10"


def load_all_sheets(cfg: WFSSPlotConfig) -> pd.DataFrame:
    sheets = pd.read_excel(cfg.excel_path, sheet_name=None)
    if not sheets:
        raise ValueError(f"No sheets found in {cfg.excel_path}")

    dfs = []
    for name, df in sheets.items():
        df = df.copy()
        df[cfg.algorithm_col] = " ".join(name.split()[:-1])  # algorithm name from sheet name
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    for col in cfg.wfss_cols.values():
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if cfg.round_decimals is not None:
            data[col] = data[col].round(cfg.round_decimals)

    return data


def wfss_order(data: pd.DataFrame, cols: List[str]) -> List[float]:
    vals: List[float] = []
    for c in cols:
        vals.extend(data[c].dropna().tolist())
    return sorted(set(vals))


def pivot_counts(data: pd.DataFrame, wfss_col: str, alg_col: str) -> pd.DataFrame:
    return (
        data.groupby([wfss_col, alg_col])
        .size()
        .reset_index(name="Count")
        .pivot(index=wfss_col, columns=alg_col, values="Count")
        .fillna(0)
    )


def journal_tone(color, sat_factor: float = 0.7, light_factor: float = 1):
    """
    Make colors a bit more journal-friendly:
    - slightly reduce saturation (less flashy)
    - slightly increase lightness (cleaner)
    """
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    s = max(0.0, min(1.0, s * sat_factor))
    l = max(0.0, min(1.0, l * light_factor))

    return colorsys.hls_to_rgb(h, l, s)


def build_algorithm_color_map(algorithms_present: List[str], cfg: WFSSPlotConfig) -> Dict[str, tuple]:
    """
    Match your main figure mapping but with a journal-friendly tone.
      One-step            -> tab10[0] blue
      Opt-only            -> tab10[1] orange
      Three-class         -> tab10[2] green
      two-step (ours)     -> tab10[3] red
      two-step (relevant) -> tab10[6] pink
    """
    preferred_order = [
        "One-step",
        "Opt-only",
        "Three-class",
        "two-step (ours)",
        "two-step (relevant)",
    ]

    algorithms = [a for a in preferred_order if a in algorithms_present] + \
                 [a for a in sorted(algorithms_present) if a not in preferred_order]

    base = mpl.colormaps[cfg.colormap].colors  # tab10

    # Force the first 5 to match the main-figure look (pink = tab10[6])
    first5 = [base[0], base[1], base[2], base[3], base[6]]

    # If extras exist, continue with remaining tab10 colors not already used
    used = {0, 1, 2, 3, 6}
    remaining = [base[i] for i in range(len(base)) if i not in used]
    full_base = first5 + remaining

    palette = [journal_tone(full_base[i % len(full_base)]) for i in range(len(algorithms))]
    return {alg: palette[i] for i, alg in enumerate(algorithms)}


def plot_wfss_three_panels(cfg: WFSSPlotConfig, savepath: str | None = None) -> None:
    data = load_all_sheets(cfg)

    titles = list(cfg.wfss_cols.keys())
    cols = list(cfg.wfss_cols.values())
    order = wfss_order(data, cols)

    algorithms_present = list(data[cfg.algorithm_col].unique())
    colors = build_algorithm_color_map(algorithms_present, cfg)

    # Journal-ish styling
    mpl.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 0.8,
    })

    # --- Layout: dedicate a top row for the legend so it cannot be cut ---
    fig = plt.figure(figsize=cfg.figsize)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[0.18, 1, 1, 1], hspace=0.35)

    ax_leg = fig.add_subplot(gs[0])
    ax_leg.axis("off")

    axes = [
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[2], sharex=None, sharey=None),
        fig.add_subplot(gs[3], sharex=None, sharey=None),
    ]

    # share x/y across the three panels
    axes[1].sharex(axes[0]); axes[2].sharex(axes[0])
    axes[1].sharey(axes[0]); axes[2].sharey(axes[0])

    preferred_order = [
        "One-step",
        "Opt-only",
        "Three-class",
        "two-step (ours)",
        "two-step (relevant)",
    ]

    legend_handles, legend_labels = None, None

    for ax, title, col in zip(axes, titles, cols):
        pivot = pivot_counts(data, col, cfg.algorithm_col)
        pivot = pivot.reindex(order).fillna(0)

        # Ensure algorithm order matches main figure
        col_order = [a for a in preferred_order if a in pivot.columns] + \
                    [a for a in pivot.columns if a not in preferred_order]
        pivot = pivot[col_order]

        pivot.plot(
            kind="bar",
            ax=ax,
            rot=0,
            color=[colors[a] for a in pivot.columns],
            width=0.84,
            legend=False,
            edgecolor="0.30",
            linewidth=0.35,
        )

        ax.set_title(title, pad=6)
        ax.set_ylabel("Count")

        # Lighter grid, journal-friendly
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
        ax.margins(x=0.01)

        # keep spines clean
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[-1].set_xlabel("WFSS Score")

    # Put legend in dedicated legend axis (never clipped)
    ax_leg.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=min(5, len(legend_labels)),
        frameon=False,
        columnspacing=1.4,
        handlelength=1.6,
        handletextpad=0.6,
    )

    if savepath is not None:
        fig.savefig(savepath, dpi=400, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    eval_path = "/home/bella/Academy/results/Opt_cases/unified_opt.xlsx"
    cfg = WFSSPlotConfig(
        excel_path=eval_path,
        wfss_cols={
            "Top-1": "wfss",
            "Top-3": "wfss_top3",
            "Top-5": "wfss_top5",
        },
        round_decimals=3,
        colormap="tab10",
        figsize=(7.6, 8.8),
    )

    plot_wfss_three_panels(cfg)
