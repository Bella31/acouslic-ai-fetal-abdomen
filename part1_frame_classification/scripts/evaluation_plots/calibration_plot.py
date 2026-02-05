import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calibration_bins(scores, labels, n_bins=10, strategy="uniform"):
    """
    scores: array-like in [0, 1]
    labels: array-like in {0, 1}
    strategy: "uniform" (fixed-width bins) or "quantile" (equal-count bins)
    Returns a dataframe with per-bin stats.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)

    # Keep only finite values
    m = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[m], labels[m]

    # Clip scores if they're *supposed* to be probabilities
    scores = np.clip(scores, 0.0, 1.0)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(scores, edges, right=True) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    elif strategy == "quantile":
        # Quantile bins can repeat edges if many identical scores; handle gracefully.
        edges = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        # Make edges strictly non-decreasing; digitize still works.
        bin_ids = np.digitize(scores, edges, right=True) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    rows = []
    for b in range(n_bins):
        idx = bin_ids == b
        if not np.any(idx):
            continue
        bin_scores = scores[idx]
        bin_labels = labels[idx]
        rows.append({
            "bin": b,
            "count": int(idx.sum()),
            "score_mean": float(bin_scores.mean()),
            "acc_mean": float(bin_labels.mean()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # ECE (Expected Calibration Error)
    total = out["count"].sum()
    out["abs_gap"] = (out["acc_mean"] - out["score_mean"]).abs()
    ece = float((out["count"] / total * out["abs_gap"]).sum())
    out.attrs["ece"] = ece
    return out


def plot_calibration_from_excel(
    excel_path,
    score_col="score",
    acc_col="accuracy",
    n_bins=10,
    strategy="uniform",   # "uniform" or "quantile"
    combine=True          # True: all algorithms on one figure, False: one figure per algorithm
):
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names

    if combine:
        plt.figure()
        plt.plot([0, 1], [0, 1])  # perfect calibration line

    results = {}

    for sheet in sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet)

        if score_col not in df.columns or acc_col not in df.columns:
            print(f"Skipping sheet '{sheet}': missing '{score_col}' or '{acc_col}'")
            continue

        # If accuracy is True/False, convert to 1/0
        scores = df[score_col].astype(float).to_numpy()
        acc = df[acc_col].astype(float).to_numpy()

        bins_df = calibration_bins(scores, acc, n_bins=n_bins, strategy=strategy)
        if bins_df.empty:
            print(f"Skipping sheet '{sheet}': no valid data after filtering")
            continue

        ece = bins_df.attrs.get("ece", np.nan)
        results[sheet] = {"bins": bins_df, "ece": ece, "n": int(bins_df["count"].sum())}

        if not combine:
            plt.figure()
            plt.plot([0, 1], [0, 1])
            plt.plot(bins_df["score_mean"], bins_df["acc_mean"], marker="o")
            plt.xlabel("Mean predicted score (confidence)")
            plt.ylabel("Empirical accuracy")
            plt.title(f"Calibration: {sheet} | ECE={ece:.4f} | N={results[sheet]['n']}")
            plt.ylim(0, 1)
            plt.xlim(0, 1)
            plt.grid(True)
            plt.show()
        else:
            plt.plot(bins_df["score_mean"], bins_df["acc_mean"], marker="o", label=f"{sheet} (ECE={ece:.3f})")

    if combine:
        plt.xlabel("Mean predicted score (confidence)")
        plt.ylabel("Empirical accuracy")
        plt.title(f"Calibration curves ({strategy}, {n_bins} bins)")
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()

    return results


if __name__ == "__main__":
    eval_path = '/home/bella/Academy/results/Opt_cases/unified_opt.xlsx'
    results = plot_calibration_from_excel(
        excel_path=eval_path,
        score_col="score",
        acc_col="accuracy",
        n_bins=4,
        strategy="uniform",
        combine=True
    )