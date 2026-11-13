import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

# ==================================================
# 設定
# ==================================================

BASE_DIR = Path("./plots_test/protect/noise")

NOISES = [
    "amplitude",
    "depolar",
    "phase"
]

NOISE_RATES = range(100, 1500, 100)

CSV_NAME = "Protect result_1.csv"


# ==================================================
# 関数
# ==================================================

def summarize_result(df):
    """
    omega, thetaごとに平均を計算
    """

    summary = (
        df.groupby(["omega", "theta"])
        .agg(
            fidelity=("fidelity", "mean"),
            pairs=("pairs", "mean"),
            probability=("probability", "mean"),
            time=("time", "mean"),
        )
        .reset_index()
    )

    return summary


def get_best_result(noise):
    """
    各ノイズレートについて
    fidelity最大となるデータを取得
    """

    results = []

    for rate in NOISE_RATES:

        csv_path = (
            BASE_DIR
            / str(rate)
            / noise
            / CSV_NAME
        )

        df = pd.read_csv(csv_path)

        # omega, thetaごとに平均
        summary = summarize_result(df)

        # fidelity最大
        best = summary.loc[summary["fidelity"].idxmax()].copy()

        if noise == "depolar":
            best["depolar_rate"] = rate
        else:
            best["damp_rate"] = rate

        results.append(best)

    result_df = pd.DataFrame(results)

    if noise == "depolar":
        columns = ["depolar_rate"] + [
            c for c in result_df.columns
            if c != "depolar_rate"
        ]
    else:
        columns = ["damp_rate"] + [
            c for c in result_df.columns
            if c != "damp_rate"
        ]

    return result_df[columns]


def save_summary(df, noise):

    save_path = BASE_DIR / f"{noise}/Protect_summary.csv"

    df.to_csv(save_path, index=False)

    print(f"Saved : {save_path}")


def save_graph(df, y, ylabel, noise):

    plt.figure(figsize=(8,6))

    if noise == "depolar":
        plt.plot(
            df["depolar_rate"],
            df[y],
            marker="o",
            linewidth=2
        )
    else:
        plt.plot(
            df["damp_rate"],
            df[y],
            marker="o",
            linewidth=2
        )

    plt.xlabel("Noise rate")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} ({df.name})")
    plt.grid(True)

    save_path = BASE_DIR / f"{noise}/{df.name}_{y}.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved : {save_path}")


# ==================================================
# メイン処理
# ==================================================

for noise in NOISES:

    print(f"\n===== {noise.upper()} =====")

    # 最大fidelityを取得
    result = get_best_result(noise)

    # 名前を保存（グラフ保存用）
    result.name = noise

    # CSV保存
    save_summary(result, noise)

    # グラフ保存
    save_graph(
        result,
        "fidelity",
        "Maximum Fidelity",
        noise
    )

    save_graph(
        result,
        "pairs",
        "Pairs",
        noise
    )

    save_graph(
        result,
        "probability",
        "Probability",
        noise
    )

    save_graph(
        result,
        "time",
        "Time [ns]",
        noise
    )

print("\nFinished.")