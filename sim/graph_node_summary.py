import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

# ==================================================
# 設定
# ==================================================

BASE_DIR = Path("./combination/plots")

# プロトコルと読み込むCSVファイル名
PROTOCOLS = {
    "bennet": "Bennet result_1.csv",
    "deutsch": "Deutsch result_1.csv",
    "filter": "Filter summary.csv",
    "filter-bennet": "Filter-Bennet summary.csv",
    "filter-deutsch": "Filter-Deutsch summary.csv",
    "original": "Teleportation result_1.csv",
    "protect": "protect summary.csv",
    "protect-bennet": "protect-bennet summary.csv",
    "protect-deutsch": "protect-deutsch summary.csv"
}

# グラフの凡例に表示する名前
LABEL = {
    "bennet": "Recurrence",
    "deutsch": "QPA",
    "filter": "Filter",
    "filter-bennet": "Filter-Bennet",
    "filter-deutsch": "Filter-QPA",
    "original": "Srandard",
    "protect": "WMFR",
    "protect-bennet": "WMFR-Recurrence",
    "protect-deutsch": "WMFR-QPA"
}

# ノイズ
NOISES = [
    "amplitude",
    "depolar",
    "phase"
]

# node_distanceごとに平均値を計算するプロトコル
AVERAGE_PROTOCOLS = [
    "bennet",
    "deutsch",
    "original"
]

# グラフの保存先
SAVE_DIR = BASE_DIR / "comparison"


# ==================================================
# CSV読み込み
# ==================================================

def load_csv(protocol, filename, noise):
    """
    CSVファイルを読み込む
    """

    path = (
        BASE_DIR
        / protocol
        / "node_distance"
        / noise
        / filename
    )

    df = pd.read_csv(path)

    # CSV保存時に追加された行番号の列を削除
    unnamed_columns = [
        column for column in df.columns
        if str(column).startswith("Unnamed:")
    ]

    df = df.drop(columns=unnamed_columns)
    
    return df


# ==================================================
# 統計処理
# ==================================================

def calc_statistics(df, protocol):
    """
    node_distanceごとの平均値を計算する

    result_1.csv：
        node_distanceごとに数値列の平均値を計算

    summary.csv：
        すでに集計されたデータをそのまま使用
    """

    if protocol in AVERAGE_PROTOCOLS:

        summary = (
            df.groupby("node_distance")
            .mean(numeric_only=True)
            .reset_index()
        )

    else:
        summary = (
            df.sort_values("node_distance")
            .reset_index(drop=True)
        )

    return summary


# ==================================================
# グラフ作成
# ==================================================

for noise in NOISES:

    print(f"\n===== {noise.upper()} =====")

    # プロトコルごとの集計データを格納
    data_dict = {}

    # ----------------------------------------------
    # 各プロトコルのCSV読み込み・集計
    # ----------------------------------------------

    for protocol, filename in PROTOCOLS.items():

        print(f"\n--- {protocol} ---")

        try:
            df = load_csv(
                protocol,
                filename,
                noise
            )

        except FileNotFoundError:
            print(f"CSV not found : {protocol} ({noise})")
            continue

        # node_distanceごとの統計処理
        data = calc_statistics(df, protocol)

        data_dict[protocol] = data

    if len(data_dict) == 0:
        print("No data.")
        continue

    # ----------------------------------------------
    # グラフ化する列を取得
    # ----------------------------------------------

    # 全プロトコルに存在する数値列を収集
    parameters = set()

    for data in data_dict.values():

        for column in data.columns:

            if column == "node_distance":
                continue

            if pd.api.types.is_numeric_dtype(data[column]):
                parameters.add(column)

    # ----------------------------------------------
    # 列ごとにグラフを作成
    # ----------------------------------------------

    for column in sorted(parameters):

        plt.figure(figsize=(8, 6))

        plotted = False

        for protocol, data in data_dict.items():

            # 対象の列が存在しない場合はスキップ
            if column not in data.columns:
                continue

            # 欠損値を除外
            plot_data = data[
                ["node_distance", column]
            ].dropna()

            if plot_data.empty:
                continue

            plt.plot(
                plot_data["node_distance"],
                plot_data[column],
                marker="o",
                linewidth=2,
                label=LABEL[protocol]
            )

            plotted = True

        if not plotted:
            plt.close()
            continue

        plt.xlabel("Node distance [km]")
        plt.ylabel(column)

        plt.title(
            f"{column} comparison\n{noise}"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # 保存先ディレクトリを作成
        save_dir = (
            SAVE_DIR
            / "node_distance"
            / noise
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        save_path = (
            save_dir / f"{column}.png"
        )

        plt.savefig(
            save_path,
            dpi=300
        )

        plt.close()

        print("\n--------------------")
        print(f"Saved : {save_path}")


print("\nFinished.")
