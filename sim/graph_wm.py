import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

# ========================================
# 設定
# ========================================

BASE_DIR = Path("./plots_test/protect/noise")

NOISES = [
    "amplitude",
    "depolar",
    "phase"
]

NOISE_RATES = range(100, 1500, 100)

# ========================================
# ノイズごとに処理
# ========================================

for noise in NOISES:

    results = []

    for rate in NOISE_RATES:

        csv_path = (
            BASE_DIR
            / str(rate)
            / noise
            / "fidelity summary_1.csv"
        )

        df = pd.read_csv(csv_path)

        # fidelityが最大の行を取得
        best_row = df.loc[df["fidelity"].idxmax()]

        # その行を辞書として保存
        result = best_row.to_dict()
        result["noise_rate"] = rate

        results.append(result)

    # DataFrame化
    result_df = pd.DataFrame(results)

    # noise_rateを先頭列にする
    cols = ["noise_rate"] + [c for c in result_df.columns if c != "noise_rate"]
    result_df = result_df[cols]

    # CSV保存
    save_csv = BASE_DIR / f"{noise}_max_fidelity.csv"
    result_df.to_csv(save_csv, index=False)

    print(f"Saved : {save_csv}")

    # -----------------------
    # グラフ
    # -----------------------

    plt.figure(figsize=(8,6))

    plt.plot(
        result_df["noise_rate"],
        result_df["fidelity"],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Noise rate")
    plt.ylabel("Maximum fidelity")
    plt.title(f"Protect ({noise})")
    plt.grid(True)

    plt.savefig(BASE_DIR / f"{noise}_max_fidelity.png", dpi=300)
    plt.close()