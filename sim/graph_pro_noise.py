
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
    "Protect": "protect",
    "Protect-Bennet": "protect-bennet",
    "Protect-Deutsch": "protect-deutsch"
}

NOISES = [
    "amplitude",
    "depolar",
    "phase"
]

# ノード間距離：10～960 km、50 km間隔
NOISE_RATES = range(0, 1401, 100)

# 保存するCSVファイル名
CSV_NAMES = {
    "Protect": "protect summary.csv",
    "Protect-Bennet": "protect-bennet summary.csv",
    "Protect-Deutsch": "protect-deutsch summary.csv"
}


# ==================================================
# CSV読み込み
# ==================================================

def load_result_csv(directory_name, protocol, noise_rate, noise):
    """
    ノード間距離に対応するCSVファイルを読み込む
    """

    filename = f"{protocol} result_{noise_rate}Hz.csv"

    csv_path = (
        BASE_DIR
        / directory_name
        / "noise"
        / noise
        / filename
    )
    # print(csv_path)
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
# omega・thetaごとの平均値を計算
# ==================================================

def summarize_result(df):
    """
    omegaとthetaの組み合わせごとに
    その他の数値列の平均値を計算する
    """

    summary = (
        df.groupby(["omega", "theta"])
        .mean(numeric_only=True)
        .reset_index()
    )

    return summary


# ==================================================
# 各noise_rateでfidelityが最大のデータを取得
# ==================================================

def get_best_results(df, rate_column, noise_rate):
    """
    各減衰率について
    fidelityが最大となる行を取得する
    """

    # omega・thetaごとの平均値
    summary = summarize_result(df)

    if summary.empty:
        return pd.DataFrame()

    # fidelityが最大となる行のindex
    best_index = summary["fidelity"].idxmax()

    # 該当する行を取得
    best = summary.loc[best_index].copy()

    # ノード間距離を追加
    best[rate_column] = noise_rate

    return best


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
    減衰率を横軸、
    指定した列を縦軸としてグラフを保存する
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
        save_dir / f"{protocol} {y_column} summary.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved Graph : {save_path}")


# ==================================================
# メイン処理
# ==================================================

for protocol, directory_name in PROTOCOLS.items():

    print(f"\n===== {protocol.upper()} =====")

    for noise in NOISES:

        print(f"\n===== {noise.upper()} =====")

        # ノイズの種類に応じた減衰率の列名
        rate_column = get_rate_column(noise)

        results = []
        save_dir = BASE_DIR / directory_name / "noise" / noise

        # ----------------------------------------------
        # 各ノード間距離のデータを読み込む
        # ----------------------------------------------

        for noise_rate in NOISE_RATES:

            try:
                df, save_dir = load_result_csv(
                    directory_name,
                    protocol,
                    noise_rate,
                    noise
                )

            except FileNotFoundError:
                print(f"CSV not found : {protocol} ({noise_rate} km)")
                continue

            # 各ノード間距離でfidelityが最大の行を取得
            best = get_best_results(df, rate_column, noise_rate)

            if best.empty:
                print(f"No data : {noise_rate} km")
                continue

            results.append(best)

        # ----------------------------------------------
        # 結果を結合
        # ----------------------------------------------

        if len(results) == 0:
            print("No data.")
            continue

        summary_df = pd.DataFrame(results)

        # node_distance順に並べ替え
        summary_df = (
            summary_df
            .sort_values(rate_column)
            .reset_index(drop=True)
        )

        # ----------------------------------------------
        # CSV保存
        # ----------------------------------------------

        save_summary(
            save_dir,
            summary_df,
            CSV_NAMES[protocol]
        )

        # ----------------------------------------------
        # グラフ保存
        # ----------------------------------------------

        # node_distance・omega・theta以外の列をグラフ化
        y_columns = [
            "fidelity"
            # column
            # for column in summary_df.columns
            # if column not in [
            #     "node_distance",
            #     "omega",
            #     "theta"
            # ]
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
