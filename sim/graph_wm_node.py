import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

# ==================================================
# 設定
# ==================================================

BASE_DIR = Path("./combination/plots")

PROTOCOLS = {
    "filter": "Filter result_1.csv",
    "filter-bennet": "Filter-Bennet result_1.csv",
    "filter-deutsch": "Filter-Deutsch result_1.csv"
}

NOISES = [
    "amplitude",
    "depolar",
    "phase"
]

CSV_NAMES = {
    "filter": "Filter summary.csv",
    "filter-bennet": "Filter-Bennet summary.csv",
    "filter-deutsch": "Filter-Deutsch summary.csv"    
}

# 10, 60, 110, ..., 960
NODE_DISTANCES = range(10, 961, 50)

# サブフォルダ
# SUB_DIR = {
#     "amplitude": None,
#     "depolar": None,
#     "phase": None
# }


# ==================================================
# 関数
# ==================================================

def load_result_csv(protocol, filename, noise):
    """
    csvファイルを読み込む
    """

    csv_path = (
        BASE_DIR
        / protocol
        / "node_distance"
        / noise
        / filename
    )

    return pd.read_csv(csv_path), csv_path.parent


def summarize_result(df):
    """
    パラメータごとに平均を計算
    """

    summary = (
        df.groupby(["epsilon"])
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
            df, save_dir = load_result_csv(protocol, filename, noise)
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

    return result_df[columns], save_dir


def save_summary(save_dir, df, filename):
    """
    最大値一覧をCSV保存
    """
    save_path = save_dir / filename

    df.to_csv(save_path, index=False)

    print(f"Saved : {save_path}")


def save_graph(df, noise, y_column, ylabel, save_dir):
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

    plt.xlabel("Node distance [km]")
    plt.ylabel(ylabel)
    if noise == "depolar":
        plt.title(f"Fidelity of the teleported quantum state with filtering\n{noise}_rate=200 Hz")
    elif noise == "amplitude":
        plt.title(f"Fidelity of the teleported quantum state with filtering\n{noise}_damp_rate=200 Hz")
    else:
        plt.title(f"Fidelity of the teleported quantum state with filtering\n{noise}_damp_rate=100 Hz")

    plt.grid(True)

    save_path = save_dir / f"Optimal {y_column} result.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved : {save_path}")


# ==================================================
# メイン処理
# ==================================================

for noise in NOISES:

    print(f"\n===== {noise.upper()} =====")

    for protocol, filename in PROTOCOLS.items():
        summary_df, save_dir = get_best_results(noise)

    if summary_df.empty:
        print("No data.")
        continue

    # CSV保存
    save_summary(save_dir,summary_df)

    # グラフ保存
    save_graph(
        summary_df,
        noise,
        "fidelity",
        "Maximum Fidelity"
    )

    # save_graph(
    #     summary_df,
    #     noise,
    #     "pairs",
    #     "Pairs"
    # )

    # save_graph(
    #     summary_df,
    #     noise,
    #     "probability",
    #     "Probability"
    # )

    # save_graph(
    #     summary_df,
    #     noise,
    #     "time",
    #     "Time [ns]"
    # )

print("\nFinished.")