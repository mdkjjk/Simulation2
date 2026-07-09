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
    "bennet": "Bennet fidelity_2.csv",
    "deutsch": "Deutsch fidelity_2.csv",
    "filter": "Filter fidelity_2.csv",
    "protect": "Protect fidelity_2.csv",
    "standard": "Teleportation fidelity_2.csv"
}

# ノイズ
NOISES = [
    "amplitude",
    "depolar",
    "phase"
]

SAVE_DIR = BASE_DIR / "comparison"
SAVE_DIR.mkdir(exist_ok=True)

# ==================================================
# 関数
# ==================================================

def calc_statistics(df):
    """node_distanceごとの平均と標準誤差を計算"""

    return (
        df.groupby("node_distance")["F2"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )


def load_csv(protocol, filename, noise):
    """CSVを読み込む"""

    path = (
        BASE_DIR
        / protocol
        / "node_distance"
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

        data = calc_statistics(df)

        plt.errorbar(
            data["node_distance"],
            data["mean"],
            yerr=data["sem"],
            marker="o",
            capsize=3,
            linewidth=2,
            label=protocol.capitalize()
        )

    plt.xlabel("Node distance")
    plt.ylabel("Teleportation fidelity")
    plt.title(f"{noise}")
    plt.grid(True)
    plt.legend()

    save_path = SAVE_DIR / f"{noise}.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved : {save_path}")

print("Finished.")