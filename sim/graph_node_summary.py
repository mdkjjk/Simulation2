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

# ノイズごとのサブフォルダ
SUB_DIR = {
    "amplitude": "500",
    "depolar": None,
    "phase": None
}

SAVE_DIR = BASE_DIR / "comparison"

# ==================================================
# 関数
# ==================================================

def calc_statistics(df, noise):
    """node_distanceごとの平均と標準誤差を計算"""

    return (
        df.groupby("node_distance")["fidelity"]
        .mean()
        .reset_index()
    )


def load_csv(protocol, filename, noise):
    """CSVを読み込む"""

    if SUB_DIR[noise] is None:
        path = (
            BASE_DIR
            / protocol
            / "node_distance"
            / noise
            / filename
        )
    else:
        path = (
            BASE_DIR
            / protocol
            / "node_distance"
            / noise
            / SUB_DIR[noise]
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
            best_rows = data.loc[data.groupby("node_distance")["fidelity"].idxmax()]
            best_rows = best_rows.sort_values("node_distance")
            plt.errorbar(
                best_rows["node_distance"],
                best_rows["fidelity"],
                marker="o",
                capsize=3,
                linewidth=2,
                label=LABEL[protocol]
            )
            
        else:
            plt.errorbar(
                df["node_distance"],
                df["fidelity"],
                marker="o",
                capsize=3,
                linewidth=2,
                label=LABEL[protocol]
            )

    plt.xlabel("Node distance")
    plt.ylabel("Fidelity")
    plt.title(f"Fidelity of the teleported quantum state\n{noise}")
    plt.grid(True)
    plt.legend()

    save_path = SAVE_DIR / f"node_distance/{noise}/{noise}_fidelity.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved : {save_path}")

print("Finished.")