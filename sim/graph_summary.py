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
    "filter": "optimal summary_1.csv",
    "protect": "Protect_summary.csv",
    "standard": "Teleportation summary1.csv"
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

def calc_statistics(df):
    """node_distanceごとの平均と標準誤差を計算"""

    return (
        df.groupby("node_distance")["fidelity"]
        .agg(mean="mean", sem="sem")
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
    plt.ylabel("Teleportation fidelity")
    plt.title(f"Fidelity of the teleported quantum state\n{noise}")
    plt.grid(True)
    plt.legend()

    save_path = SAVE_DIR / f"{noise}.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved : {save_path}")

print("Finished.")