import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def load_confusion_matrices(excel_path, sheet_name="conf"):
    """
    Parse confusion matrices from Excel sheet.
    Expected repeating structure:
        row: method name (col0), NaN, NaN
        row+1: class labels (first 3 columns)
        row+2..row+4: 3x3 matrix
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    methods = []
    matrices = []
    class_labels = None

    for i in range(len(df)):
        v0, v1, v2 = df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2]

        if isinstance(v0, str) and pd.isna(v1) and pd.isna(v2):
            method = v0.strip()
            labels = [str(x).strip() for x in df.iloc[i + 1, 0:3].tolist()]
            mat = df.iloc[i + 2:i + 5, 0:3].to_numpy(dtype=float)

            methods.append(method)
            matrices.append(mat)
            class_labels = labels

    return methods, matrices, class_labels


def plot_selected_confusions(methods, matrices, class_labels,
                             selected_methods,
                             out_basename="conf_mats_selected"):

    # Keep only selected methods (preserve order)
    lookup = {m: mat for m, mat in zip(methods, matrices)}
    filtered = [(m, lookup[m]) for m in selected_methods if m in lookup]

    if len(filtered) != len(selected_methods):
        missing = [m for m in selected_methods if m not in lookup]
        raise ValueError(f"Missing methods in sheet: {missing}")

    methods_f = [m for m, _ in filtered]
    matrices_f = [mat for _, mat in filtered]

    vmax = max(mat.max() for mat in matrices_f)

    # Softer upper contrast (less saturated dark blue)
    norm = Normalize(vmin=0, vmax=vmax * 0.9)

    # Layout: 2 plots + 1 colorbar column
    fig = plt.figure(figsize=(12.5, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    axes = [ax1, ax2]

    im = None
    for ax, method, mat in zip(axes, methods_f, matrices_f):

        im = ax.imshow(mat, cmap="Blues", norm=norm, aspect="equal")

        ticks = np.arange(len(class_labels))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticklabels(class_labels)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(method, fontsize=14)

        # Automatic annotation color
        threshold = vmax * 0.6
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                text_color = "white" if value > threshold else "black"
                ax.text(j, i, f"{int(value)}",
                        ha="center", va="center",
                        fontsize=11,
                        color=text_color)

    # Dedicated colorbar axis (no overlap)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Count")

    fig.suptitle("Confusion Matrices for 3-Class Methods", fontsize=18)

    fig.savefig(f"{out_basename}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_basename}.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":

    excel_file = "/home/bella/Academy/results/Opt_cases/overall_summary.xlsx"

    methods, matrices, class_labels = load_confusion_matrices(excel_file)

    selected = ["Three-class", "two-step (relevant)"]

    plot_selected_confusions(methods, matrices, class_labels, selected)
