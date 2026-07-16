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
    "bennet": "fidelity summary_1.csv",
    "deutsch": "fidelity summary_1.csv",
    "filter": "fidelity summary_1.csv",
    "protect": "Protect_summary.csv",
    "standard": "Teleportation summary_1.csv"
}

# ノイズ
NOISES = [
    "amplitude",
    "depolar",
    "phase"
]

SAVE_DIR = BASE_DIR / "comparison"

# ==================================================
# 関数
# ==================================================

def calc_statistics(df, noise):
    """node_distanceごとの平均と標準誤差を計算"""

    if noise == "depolar":
        return (
            df.groupby(["depolar_rate", "epsilon"])["fidelity"]
            .mean()
            .reset_index()
        )
    else:
        return (
            df.groupby(["damp_rate", "epsilon"])["fidelity"]
            .mean()
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

    for protocol, filename in PROTOCOLS.items():
        df = load_csv(protocol, filename, noise)

        if protocol == "filter":
            data = calc_statistics(df, noise)
            if noise == "depolar":
                best_rows = data.loc[data.groupby("depolar_rate")["fidelity"].idxmax()]
                best_rows = best_rows.sort_values("depolar_rate")
                plt.errorbar(
                    best_rows["depolar_rate"],
                    best_rows["fidelity"],
                    marker="o",
                    capsize=3,
                    linewidth=2,
                    label=protocol.capitalize()
                )
            else:
                best_rows = data.loc[data.groupby("damp_rate")["fidelity"].idxmax()]
                best_rows = best_rows.sort_values("damp_rate")
                plt.errorbar(
                    best_rows["damp_rate"],
                    best_rows["fidelity"],
                    marker="o",
                    capsize=3,
                    linewidth=2,
                    label=protocol.capitalize()
                )
            
        else:
            if noise == "depolar":
                plt.errorbar(
                    df["depolar_rate"],
                    df["fidelity"],
                    marker="o",
                    capsize=3,
                    linewidth=2,
                    label=protocol.capitalize()
                )
            else:
                plt.errorbar(
                    df["damp_rate"],
                    df["fidelity"],
                    marker="o",
                    capsize=3,
                    linewidth=2,
                    label=protocol.capitalize()
                )

    plt.xlabel("Noise rate")
    plt.ylabel("Fidelity")
    plt.title(f"Fidelity of the teleported quantum state\n{noise}")
    plt.grid(True)
    plt.legend()

    save_path = SAVE_DIR / f"noise/{noise}/{noise}_fidelity.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved : {save_path}")

print("Finished.")