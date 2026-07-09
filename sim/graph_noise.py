import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

# ============================================
# フォルダ設定
# ============================================

BENNET_DIR = Path("./plots_test/bennet/noise")
STANDARD_DIR = Path("./plots_test/standard/noise")

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

    if noise == "depolar":
        bennet_data = (
            bennet.groupby("depolar_rate")["fidelity"]
                .agg(fidelity="mean", sem="sem")
                .reset_index()
        )

        standard_data = (
            standard.groupby("depolar_rate")["fidelity"]
                    .agg(fidelity="mean", sem="sem")
                    .reset_index()
        )
    else:
        bennet_data = (
            bennet.groupby("damp_rate")["fidelity"]
                .agg(fidelity="mean", sem="sem")
                .reset_index()
        )

        standard_data = (
            standard.groupby("damp_rate")["fidelity"]
                    .agg(fidelity="mean", sem="sem")
                    .reset_index()
        )

    # -------------------------------
    # グラフ
    # -------------------------------

    plt.figure(figsize=(8,6))

    if noise == "depolar":
        plt.errorbar(
            bennet_data["depolar_rate"],
            bennet_data["fidelity"],
            yerr=bennet_data["sem"],
            marker="o",
            capsize=3,
            label="Bennet"
        )

        plt.errorbar(
            standard_data["depolar_rate"],
            standard_data["fidelity"],
            yerr=standard_data["sem"],
            marker="s",
            capsize=3,
            label="Standard"
        )
    else:
        plt.errorbar(
        bennet_data["damp_rate"],
        bennet_data["fidelity"],
        yerr=bennet_data["sem"],
        marker="o",
        capsize=3,
        label="Bennet"
        )

        plt.errorbar(
            standard_data["damp_rate"],
            standard_data["fidelity"],
            yerr=standard_data["sem"],
            marker="s",
            capsize=3,
            label="Standard"
        )

    plt.xlabel("Noise rate")
    plt.ylabel("Teleportation fidelity")
    plt.title(f"{noise}")
    plt.grid(True)
    plt.legend()

    plt.savefig(f"./plots_test/bennet/noise/{noise}/{noise}_comparison.png", dpi=300)
    plt.close()

    print(f"Saved : {noise}_comparison.png")