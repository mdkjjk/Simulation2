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
        / "noise"
        / noise
        / filename
    )

    # 先頭列の行番号をインデックスとして読み込む
    df = pd.read_csv(csv_path, index_col=0)

    return df, csv_path.parent


# ==================================================
# 減衰率の列名を取得
# ==================================================

def get_rate_column(noise):
    """
    ノイズの種類に応じて減衰率の列名を取得する
    """

    if noise == "depolar":
        return "depolar_rate"
    else:
        return "damp_rate"


# ==================================================
# epsilon・減衰率ごとの平均値を計算
# ==================================================

def summarize_result(df, rate_column):
    """
    epsilonと減衰率の組み合わせごとに
    その他の数値列の平均値を計算する
    """

    summary = (
        df.groupby([rate_column, "epsilon"])
        .mean(numeric_only=True)
        .reset_index()
    )

    return summary


# ==================================================
# 各減衰率でfidelityが最大のデータを取得
# ==================================================

def get_best_results(df, rate_column):
    """
    各減衰率について
    fidelityが最大となる行を取得する
    """

    # epsilon・減衰率ごとの平均値
    summary = summarize_result(df, rate_column)

    # 各減衰率でfidelityが最大となる行のindex
    best_indices = (
        summary.groupby(rate_column)["fidelity"].idxmax()
    )

    # 該当する行を取得
    result_df = (
        summary.loc[best_indices]
        .sort_values(rate_column)
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

def save_graph(df, rate_column, y_column, protocol, noise, save_dir):
    """
    減衰率を横軸、指定した列を縦軸としてグラフを保存
    """

    plt.figure(figsize=(8, 6))

    plt.plot(
        df[rate_column],
        df[y_column],
        marker="o",
        linewidth=2
    )

    if noise == "depolar":
        xlabel = "Depolarization rate [Hz]"
    elif noise == "amplitude":
        xlabel = "Amplitude damping rate [Hz]"
    else:
        xlabel = "Phase damping rate [Hz]"

    plt.xlabel(xlabel)
    plt.ylabel(y_column)

    plt.title(
        f"{y_column} of {protocol}\nNoise: {noise}"
    )

    plt.grid(True)
    plt.tight_layout()

    save_path = (
        save_dir
        / f"{protocol} {y_column} summary.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved Graph : {save_path}")


# ==================================================
# メイン処理
# ==================================================

for noise in NOISES:

    print(f"\n===== {noise.upper()} =====")

    # ノイズの種類に応じた減衰率の列名
    rate_column = get_rate_column(noise)

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

        # 各減衰率でfidelityが最大の行を取得
        summary_df = get_best_results(
            df,
            rate_column
        )

        if summary_df.empty:
            print("No data.")
            continue

        # CSV保存
        save_summary(
            save_dir,
            summary_df,
            CSV_NAMES[protocol]
        )

        # 減衰率とepsilon以外の列をグラフ化
        y_columns = [
            "fidelity"
            # column
            # for column in summary_df.columns
            # if column not in [rate_column, "epsilon"]
        ]

        for y_column in y_columns:

            save_graph(
                summary_df,
                rate_column,
                y_column,
                protocol,
                noise,
                save_dir
            )


print("\nFinished.")
