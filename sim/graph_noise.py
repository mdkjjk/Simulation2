import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

# ============================================
# フォルダ設定
# ============================================

PROTECT_DIR = Path("./plots_test/protect/noise")
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

    protect = pd.read_csv(
        PROTECT_DIR / noise / "Protect_summary.csv"
    )

    standard = pd.read_csv(
        STANDARD_DIR / noise / "Teleportation result_1.csv"
    )

    # -------------------------------
    # 平均と標準誤差
    # -------------------------------

    if noise == "depolar":
        protect_data = (
            protect.groupby("depolar_rate")["fidelity"]
                .agg(fidelity="mean", sem="sem")
                .reset_index()
        )

        standard_data = (
            standard.groupby("depolar_rate")["fidelity"]
                    .agg(fidelity="mean", sem="sem")
                    .reset_index()
        )
    else:
        protect_data = (
            protect.groupby("damp_rate")["fidelity"]
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
            protect_data["depolar_rate"],
            protect_data["fidelity"],
            yerr=protect_data["sem"],
            marker="o",
            capsize=3,
            label="Protect"
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
            protect_data["damp_rate"],
            protect_data["fidelity"],
            yerr=protect_data["sem"],
            marker="o",
            capsize=3,
            label="Protect"
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
    plt.title(f"Fidelity of the teleported quantum state\n{noise}")
    plt.grid(True)
    plt.legend()

    plt.savefig(f"./plots_test/protect/noise/{noise}/{noise}_comparison.png", dpi=300)
    plt.close()

    print(f"Saved : {noise}_comparison.png")