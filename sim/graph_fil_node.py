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

# 保存するCSVファイル名
CSV_NAMES = {
    "filter": "Filter summary.csv",
    "filter-bennet": "Filter-Bennet summary.csv",
    "filter-deutsch": "Filter-Deutsch summary.csv"
}


# ==================================================
# CSV読み込み
# ==================================================

def load_result_csv(protocol, filename, noise):
    """
    CSVファイルを読み込む
    """

    csv_path = (
        BASE_DIR
        / protocol
        / "node_distance"
        / noise
        / filename
    )

    return pd.read_csv(csv_path, index_col=0), csv_path.parent


# ==================================================
# epsilon・node_distanceごとの平均値を計算
# ==================================================

def summarize_result(df):
    """
    node_distanceとepsilonの組み合わせごとに
    数値列の平均値を計算する
    """

    summary = (
        df.groupby(["node_distance", "epsilon"])
        .mean(numeric_only=True)
        .reset_index()
    )

    return summary


# ==================================================
# 各node_distanceでfidelityが最大のデータを取得
# ==================================================

def get_best_results(df):
    """
    各node_distanceについて
    fidelityが最大となる行を取得する
    """

    # node_distance・epsilonごとの平均値
    summary = summarize_result(df)

    # 各node_distanceでfidelityが最大となる行のindex
    best_indices = (
        summary.groupby("node_distance")["fidelity"].idxmax()
    )

    # 該当する行を取得
    result_df = (
        summary.loc[best_indices]
        .sort_values("node_distance")
        .reset_index(drop=True)
    )

    return result_df


# ==================================================
# CSV保存
# ==================================================

def save_summary(save_dir, df, filename):
    """
    最適なパラメータの一覧をCSV保存
    """

    save_path = save_dir / filename

    df.to_csv(save_path, index=False)

    print(f"Saved CSV : {save_path}")


# ==================================================
# グラフ保存
# ==================================================

def save_graph(df, y_column, protocol, noise, save_dir):
    """
    node_distanceを横軸、
    指定した列を縦軸としてグラフを保存する
    """

    plt.figure(figsize=(8, 6))

    plt.plot(
        df["node_distance"],
        df[y_column],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Node distance [km]")
    plt.ylabel(y_column)
    plt.title(
        f"{y_column} of {protocol}\nNoise: {noise}"
    )

    plt.grid(True)
    plt.tight_layout()

    save_path = (
        save_dir / f"{protocol} {y_column} summary.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved Graph : {save_path}")


# ==================================================
# メイン処理
# ==================================================

for noise in NOISES:

    print(f"\n===== {noise.upper()} =====")

    for protocol, filename in PROTOCOLS.items():

        print(f"\n--- {protocol} ---")

        try:
            df, save_dir = load_result_csv(
                protocol,
                filename,
                noise
            )

        except FileNotFoundError:
            print(f"CSV not found : {protocol} ({noise})")
            continue

        # 各node_distanceでfidelityが最大の行を取得
        summary_df = get_best_results(df)

        if summary_df.empty:
            print("No data.")
            continue

        # CSV保存
        save_summary(
            save_dir,
            summary_df,
            CSV_NAMES[protocol]
        )

        # epsilonとnode_distance以外の列をグラフ化
        y_columns = [
            "fidelity"
            # column
            # for column in summary_df.columns
            # if column not in ["node_distance", "epsilon"]
        ]

        for y_column in y_columns:

            save_graph(
                summary_df,
                y_column,
                protocol,
                noise,
                save_dir
            )


print("\nFinished.")
