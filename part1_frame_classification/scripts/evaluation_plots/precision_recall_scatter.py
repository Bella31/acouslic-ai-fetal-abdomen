import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def _label_points(
    ax,
    xs,
    ys,
    labels,
    *,
    color=None,
    fontsize=11,
    connect=False,
    lw=0.7,
    # label placement controls
    dx=0.015,                 # horizontal offset in data units
    dy_base=0.00,             # base vertical offset
    dy_step=0.018,            # vertical separation step for collisions
    x_right_threshold=0.92,   # if x > threshold, place label to LEFT
    collision_eps=(0.018, 0.018),  # (eps_x, eps_y) for "too close"
    per_label_dy=None,        # dict: {"label": dy_adjust}
):
    """
    Deterministic near-point labeling:
      - if point is near right edge: label to the LEFT (keeps text inside)
      - if labels collide: add small vertical offsets
      - optional short connector line (color-matched if color is provided)
      - optional per-label vertical tweak via per_label_dy
    """
    placed = []  # store used label anchor positions (x_label, y_label)
    eps_x, eps_y = collision_eps
    per_label_dy = per_label_dy or {}

    for x, y, lab in zip(xs, ys, labels):
        if np.isnan(x) or np.isnan(y):
            continue

        lab_str = str(lab).strip()
        dy_extra = per_label_dy.get(lab_str, 0.0)

        # Decide label side based on x position
        place_left = x > x_right_threshold
        x_label = x - dx if place_left else x + dx
        ha = "right" if place_left else "left"

        # Start near the point
        y_label = y + dy_base + dy_extra

        # If this label position would collide, nudge vertically
        attempts = 0
        while any(abs(x_label - xp) < eps_x and abs(y_label - yp) < eps_y for xp, yp in placed):
            attempts += 1
            direction = 1 if attempts % 2 == 1 else -1
            y_label = y + dy_base + dy_extra + direction * dy_step * ((attempts + 1) // 2)

        # Draw text
        ax.text(
            x_label, y_label, lab_str,
            fontsize=fontsize,
            ha=ha,
            va="center",
            color=color,
            clip_on=False,
            zorder=4
        )

        # Optional connector (short)
        if connect:
            ax.plot(
                [x, x_label], [y, y_label],
                linewidth=lw,
                alpha=0.75,
                color=color,   # match connector to label/point color
                zorder=2
            )

        placed.append((x_label, y_label))


def _to_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _set_limits_from_data(ax, x_all, y_all, pad=0.05, hard=(0.0, 1.0)):
    x_all = np.asarray(x_all, dtype=float)
    y_all = np.asarray(y_all, dtype=float)
    x_all = x_all[~np.isnan(x_all)]
    y_all = y_all[~np.isnan(y_all)]

    if len(x_all) == 0 or len(y_all) == 0:
        ax.set_xlim(hard[0], hard[1])
        ax.set_ylim(hard[0], hard[1])
        return

    xmin, xmax = x_all.min(), x_all.max()
    ymin, ymax = y_all.min(), y_all.max()

    xmin = max(hard[0], xmin - pad)
    xmax = min(hard[1] + 0.05, xmax + pad)  # allow a bit >1 for labels
    ymin = max(hard[0], ymin - pad)
    ymax = min(hard[1] + 0.05, ymax + pad)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def plot_side_by_side(excel_path):
    df_rel = pd.read_excel(excel_path, sheet_name="rel")
    df_opt = pd.read_excel(excel_path, sheet_name="opt")
    plot_pr(df_rel, df_opt)
    plt.show()


def plot_pr(df_rel, df_opt):
    """Draw the figure from the two tables (columns as in the rel and opt sheets); returns the figure."""
    df_rel = df_rel.copy()
    df_opt = df_opt.copy()
    df_rel.columns = df_rel.columns.str.strip()
    df_opt.columns = df_opt.columns.str.strip()
    df_rel["method"] = df_rel["method"].astype(str).str.strip()
    df_opt["method"] = df_opt["method"].astype(str).str.strip()

    # Coerce numeric (prevents plotting nothing if Excel stored numbers as text)
    df_opt = _to_numeric(df_opt, ["precision_opt_only", "recall_opt_only"])
    df_rel = _to_numeric(df_rel, [
        "precision_rel_Opt_data",
        "recall_rel_opt_data",
        "precision_rel_subopt_data",
        "recall_rel_subopt_data"
    ])

    plt.rcParams.update({
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # ---------------- LEFT: OPT ----------------
    xL = df_opt["recall_opt_only"].to_numpy()
    yL = df_opt["precision_opt_only"].to_numpy()

    ax_left.scatter(xL, yL, s=55, zorder=3)

    _set_limits_from_data(ax_left, xL, yL, pad=0.05, hard=(0.0, 1.0))
 #   xmin, xmax = ax_right.get_xlim()
  #  ax_right.set_xlim(xmin, xmax + 0.08)

    ax_left.set_xlabel("Recall")
    ax_left.set_ylabel("Precision")
    ax_left.set_title("Optimal Classification", fontsize=16)
    ax_left.grid(alpha=0.2)

    _label_points(
        ax_left,
        xL,
        yL,
        df_opt["method"].to_numpy(),
        fontsize=14,
        connect=False,
        dx=0.020,
        dy_step=0.020,
        x_right_threshold=0.90,
        collision_eps=(0.030, 0.025),
        color="black"
    )

    # ---------------- RIGHT: REL ----------------
    xO = df_rel["recall_rel_opt_data"].to_numpy()
    yO = df_rel["precision_rel_Opt_data"].to_numpy()
    xS = df_rel["recall_rel_subopt_data"].to_numpy()
    yS = df_rel["precision_rel_subopt_data"].to_numpy()

    ax_right.scatter(xO, yO, s=55, color="tab:blue", label="Optimal", zorder=3)
    ax_right.scatter(xS, yS, s=55, color="tab:orange", label="Suboptimal", zorder=3)

    _set_limits_from_data(ax_right, np.r_[xO, xS], np.r_[yO, yS], pad=0.06, hard=(0.0, 1.0))

    ax_right.set_xlabel("Recall")
    ax_right.set_ylabel("Precision")
    ax_right.set_title("Relevant Classification", fontsize=16)
    ax_right.grid(alpha=0.2)
    ax_right.legend(loc="upper left")

    # Blue labels (Optimal)
    _label_points(
        ax_right,
        xO,
        yO,
        df_rel["method"].to_numpy(),
        fontsize=14,
        connect=True,
        dx=0.018,
        dy_step=0.022,
        x_right_threshold=0.90,
        collision_eps=(0.030, 0.028),
        color="black"
    )

    # Orange labels (Suboptimal) — move "Three-class" down so the dot doesn't occlude it
    _label_points(
        ax_right,
        xS,
        yS,
        df_rel["method"].to_numpy(),
        fontsize=14,
        connect=True,
        dx=0.018,
        dy_step=0.022,
        x_right_threshold=0.90,
        collision_eps=(0.030, 0.028),
        color="black",
        per_label_dy={"Three-class": -0.03}  # adjust to -0.04 if you want it lower
    )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    excel_file = "/home/bella/Academy/results/Opt_cases/results_pr.xlsx"
    plot_side_by_side(excel_file)
