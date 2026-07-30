import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

# ==================================================
# 設定
# ==================================================

BASE_DIR = Path("./plots_test")

# プロトコル
PROTOCOLS = {
    "bennet": "Bennet result_1.csv",
    "deutsch": "Deutsch result_1.csv",
    "filter": "Filter result_1.csv",
    "protect": "Protect_summary.csv",
    "standard": "Teleportation result_1.csv"
}

LABEL = {
    "bennet": "Non-breeding",
    "deutsch": "QPA",
    "filter": "Filter",
    "protect": "WMFR",
    "standard": "Standard"
}

# ノイズ
NOISES = [
    "amplitude",
    "depolar",
    "phase"
]

# 評価指標
PARAMETER = {
    "fidelity": "Fidelity",
    "pairs": "Pairs",
    "probability": "Probability",
    "time": "Time [ns]"
}

SAVE_DIR = BASE_DIR / "comparison"

# ==================================================
# 関数
# ==================================================

def calc_statistics(df, noise, protocol):
    """node_distanceごとの平均と標準誤差を計算"""

    if noise == "depolar":
        if protocol == "filter":
            summary = (
                df.groupby(["depolar_rate", "epsilon"])
                .agg(
                    fidelity=("fidelity", "mean"),
                    pairs=("pairs", "mean"),
                    probability=("probability", "mean"),
                    time=("time", "mean")
                )
                .reset_index()
            )
            best_rows = summary.loc[summary.groupby("depolar_rate")["fidelity"].idxmax()]
            best_rows = best_rows.sort_values("depolar_rate")
            return best_rows
        elif protocol == "standard":
            return (
                df.groupby("depolar_rate")
                .agg(
                    fidelity=("fidelity", "mean"),
                    time=("time", "mean")
                )
                .reset_index()
            )
        else: 
            return (
                df.groupby("depolar_rate")
                .agg(
                    fidelity=("fidelity", "mean"),
                    pairs=("pairs", "mean"),
                    probability=("probability", "mean"),
                    time=("time", "mean")
                )
                .reset_index()
            )
    else:
        if protocol == "filter":
            summary = (
                df.groupby(["damp_rate", "epsilon"])
                .agg(
                    fidelity=("fidelity", "mean"),
                    pairs=("pairs", "mean"),
                    probability=("probability", "mean"),
                    time=("time", "mean")
                )
                .reset_index()
            )
            best_rows = summary.loc[summary.groupby("damp_rate")["fidelity"].idxmax()]
            best_rows = best_rows.sort_values("damp_rate")
            return best_rows
        elif protocol == "standard":
            return (
                df.groupby("damp_rate")
                .agg(
                    fidelity=("fidelity", "mean"),
                    time=("time", "mean")
                )
                .reset_index()
            )
        else: 
            return (
                df.groupby("damp_rate")
                .agg(
                    fidelity=("fidelity", "mean"),
                    pairs=("pairs", "mean"),
                    probability=("probability", "mean"),
                    time=("time", "mean")
                )
                .reset_index()
            )


def load_csv(protocol, filename, noise):
    """CSVを読み込む"""

    path = (
        BASE_DIR
        / protocol
        / "noise"
        / noise
        / filename
    )

    return pd.read_csv(path)


# ==================================================
# グラフ作成
# ==================================================

for noise in NOISES:

    plt.figure(figsize=(8, 6))

    for column, ylabel in PARAMETER.items():
        for protocol, filename in PROTOCOLS.items():
            if protocol == "standard" and column in ("pairs", "probability"):
                continue
            df = load_csv(protocol, filename, noise)
            data = calc_statistics(df, noise, protocol)

            if noise == "depolar":
                plt.errorbar(
                    data["depolar_rate"],
                    data[column],
                    marker="o",
                    capsize=3,
                    linewidth=2,
                    label=LABEL[protocol]
                )
            else:
                plt.errorbar(
                    data["damp_rate"],
                    data[column],
                    marker="o",
                    capsize=3,
                    linewidth=2,
                    label=LABEL[protocol]
                )

        plt.xlabel("Noise rate")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} comparison\n{noise}")
        plt.grid(True)
        plt.legend()

        save_path = SAVE_DIR / f"noise/{noise}/{noise}_{column}.png"

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved : {save_path}")

print("Finished.")