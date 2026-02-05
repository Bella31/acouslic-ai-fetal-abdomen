from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors


@dataclass(frozen=True)
class WFSSPlotConfig:
    excel_path: str
    wfss_cols: Dict[str, str]               # {"Top-1": "wfss", "Top-3": "wfss_top3", "Top-5": "wfss_top5"}
    algorithm_col: str = "Algorithm"
    round_decimals: Optional[int] = 3
    figsize: Tuple[int, int] = (14, 4)
    colormap: str = "tab10"                 # nice qualitative palette


def load_all_sheets(cfg: WFSSPlotConfig) -> pd.DataFrame:
    sheets = pd.read_excel(cfg.excel_path, sheet_name=None)
    if not sheets:
        raise ValueError(f"No sheets found in {cfg.excel_path}")

    dfs = []
    for name, df in sheets.items():
        df = df.copy()
        df[cfg.algorithm_col] = name.split()[0]
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    for col in cfg.wfss_cols.values():
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if cfg.round_decimals is not None:
            data[col] = data[col].round(cfg.round_decimals)

    return data


def wfss_order(data: pd.DataFrame, cols: List[str]) -> List[float]:
    vals = []
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


def plot_wfss_three_panels(cfg: WFSSPlotConfig) -> None:
    data = load_all_sheets(cfg)

    titles = list(cfg.wfss_cols.keys())
    cols = list(cfg.wfss_cols.values())
    order = wfss_order(data, cols)

    algorithms = sorted(data[cfg.algorithm_col].unique())
    import colorsys
    import matplotlib.colors as mcolors

    palette = mpl.colormaps[cfg.colormap].colors

    def increase_saturation_and_darkness(color, sat_factor=1.4, light_factor=0.8):
        r, g, b = mcolors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(r, g, b)

        s = min(1.0, s * sat_factor)  # increase saturation
        l = max(0.0, l * light_factor)  # reduce lightness → darker

        return colorsys.hls_to_rgb(h, l, s)

    palette = [increase_saturation_and_darkness(c) for c in palette]

    colors = {
        alg: palette[i % len(palette)]
        for i, alg in enumerate(algorithms)
    }

    fig, axes = plt.subplots(1, 3, figsize=cfg.figsize, sharey=True)

    for ax, title, col in zip(axes, titles, cols):
        pivot = pivot_counts(data, col, cfg.algorithm_col)
        pivot = pivot.reindex(order).fillna(0)

        pivot.plot(
            kind="bar",
            ax=ax,
            rot=0,
            color=[colors[a] for a in pivot.columns],
            legend=True,
        )

        ax.set_title(title)
        ax.set_xlabel("WFSS Score")
        ax.set_ylabel("Count")
        ax.yaxis.label.set_visible(True)
        ax.tick_params(axis="y", labelleft=True)
        ax.yaxis.set_visible(True)
        ax.legend(frameon=False)

    plt.tight_layout()
    for ax in axes:
        ax.tick_params(axis="y", which="both", labelleft=True)
        for t in ax.get_yticklabels():
            t.set_visible(True)

        ax.set_ylabel("Count")
        ax.yaxis.label.set_visible(True)
    plt.show()

if __name__ == "__main__":
    eval_path = '/home/bella/Academy/results/Opt_cases/unified_opt.xlsx'
    cfg = WFSSPlotConfig(
        excel_path=eval_path,
        wfss_cols={
            "Top-1": "wfss",
            "Top-3": "wfss_top3",
            "Top-5": "wfss_top5",
        },
        round_decimals=3,
        figsize=(14, 4),
        colormap="Pastel1",

    )

    plot_wfss_three_panels(cfg)