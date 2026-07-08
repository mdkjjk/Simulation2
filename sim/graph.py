import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

# ============================================
# フォルダ設定
# ============================================

BENNET_DIR = Path("./plots_test/bennet/node_distance")
STANDARD_DIR = Path("./plots_test/standard/node_distance")

NOISES = [
    "amplitude",
    "depolar",
    "phase"
]

# ============================================
# ノイズごとにグラフを作成
# ============================================

for noise in NOISES:

    # -------------------------------
    # CSV読み込み
    # -------------------------------

    bennet = pd.read_csv(
        BENNET_DIR / noise / "Bennet result_1.csv"
    )

    standard = pd.read_csv(
        STANDARD_DIR / noise / "Teleportation result_1.csv"
    )

    # -------------------------------
    # 平均と標準誤差
    # -------------------------------

    bennet_data = (
        bennet.groupby("node_distance")["fidelity"]
              .agg(fidelity="mean", sem="sem")
              .reset_index()
    )

    standard_data = (
        standard.groupby("node_distance")["F2"]
                .agg(fidelity="mean", sem="sem")
                .reset_index()
    )

    # -------------------------------
    # グラフ
    # -------------------------------

    plt.figure(figsize=(8,6))

    plt.errorbar(
        bennet_data["node_distance"],
        bennet_data["fidelity"],
        yerr=bennet_data["sem"],
        marker="o",
        capsize=3,
        label="Bennet"
    )

    plt.errorbar(
        standard_data["node_distance"],
        standard_data["fidelity"],
        yerr=standard_data["sem"],
        marker="s",
        capsize=3,
        label="Standard"
    )

    plt.xlabel("Node distance")
    plt.ylabel("Teleportation fidelity")
    plt.title(f"{noise}")
    plt.grid(True)
    plt.legend()

    plt.savefig(f"./plots_test/bennet/node_distance/{noise}/{noise}_comparison.png", dpi=300)
    plt.close()

    print(f"Saved : {noise}_comparison.png")