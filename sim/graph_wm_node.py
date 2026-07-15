import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

# ==================================================
# 設定
# ==================================================

BASE_DIR = Path("./plots_test/protect/node_distance")

NOISES = [
    "amplitude",
    "depolar",
    "phase"
]

# 10, 60, 110, ..., 960
NODE_DISTANCES = range(10, 961, 50)

CSV_NAME = "Protect result_1.csv"

# ノイズごとのサブフォルダ
SUB_DIR = {
    "amplitude": "500",
    "depolar": "500",
    "phase": None
}


# ==================================================
# 関数
# ==================================================

def load_result_csv(node_distance, noise):
    """
    Protect result_1.csv を読み込む
    """

    sub_dir = SUB_DIR[noise]

    if sub_dir is None:
        csv_path = (
            BASE_DIR
            / str(node_distance)
            / noise
            / CSV_NAME
        )
    else:
        csv_path = (
            BASE_DIR
            / str(node_distance)
            / noise
            / sub_dir
            / CSV_NAME
        )

    return pd.read_csv(csv_path), csv_path.parent


def summarize_result(df):
    """
    omega・thetaごとに平均を計算
    """

    summary = (
        df.groupby(["omega", "theta"])
        .agg(
            fidelity=("fidelity", "mean"),
            pairs=("pairs", "mean"),
            probability=("probability", "mean"),
            time=("time", "mean")
        )
        .reset_index()
    )

    return summary


def get_best_results(noise):
    """
    各node_distanceについて
    fidelity最大となるデータを取得
    """

    results = []

    for distance in NODE_DISTANCES:

        try:
            df, save_dir = load_result_csv(distance, noise)
        except FileNotFoundError:
            print(f"Skip : {distance} ({noise})")
            continue

        # omega・thetaごとに平均
        summary = summarize_result(df)

        # fidelity最大
        best = summary.loc[summary["fidelity"].idxmax()].copy()

        best["node_distance"] = distance

        results.append(best)

    if len(results) == 0:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    columns = ["node_distance"] + [
        c for c in result_df.columns
        if c != "node_distance"
    ]

    return result_df[columns]


def save_summary(df, noise):
    """
    最大値一覧をCSV保存
    """

    save_path = BASE_DIR / f"{noise}/Protect_summary.csv"

    df.to_csv(save_path, index=False)

    print(f"Saved : {save_path}")


def save_graph(df, noise, y_column, ylabel):
    """
    グラフ保存
    """

    plt.figure(figsize=(8, 6))

    plt.plot(
        df["node_distance"],
        df[y_column],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Node distance")
    plt.ylabel(ylabel)
    plt.title(f"{noise} : {ylabel}")

    plt.grid(True)

    save_path = BASE_DIR / f"{noise}/{noise}_{y_column}.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved : {save_path}")


# ==================================================
# メイン処理
# ==================================================

for noise in NOISES:

    print(f"\n===== {noise.upper()} =====")

    summary_df = get_best_results(noise)

    if summary_df.empty:
        print("No data.")
        continue

    # CSV保存
    save_summary(summary_df, noise)

    # グラフ保存
    save_graph(
        summary_df,
        noise,
        "fidelity",
        "Maximum Fidelity"
    )

    save_graph(
        summary_df,
        noise,
        "pairs",
        "Pairs"
    )

    save_graph(
        summary_df,
        noise,
        "probability",
        "Probability"
    )

    save_graph(
        summary_df,
        noise,
        "time",
        "Time [ns]"
    )

print("\nFinished.")