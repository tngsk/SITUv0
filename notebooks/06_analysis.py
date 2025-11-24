import json
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --- 設定 ---
BASE_DIR = "."
# パスは環境に合わせて確認してください
EXP_DATA_PATH = os.path.join(
    BASE_DIR, "experiments", "prototype", "experiment_data.csv"
)
ACOUSTIC_DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "acoustic_features.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 処理 ---
if not os.path.exists(EXP_DATA_PATH):
    print(f"エラー: 実験データが見つかりません -> {EXP_DATA_PATH}")
elif not os.path.exists(ACOUSTIC_DATA_PATH):
    print("エラー: 音響特徴量データが見つかりません。")
else:
    # 1. データの読み込み
    df_exp = pd.read_csv(EXP_DATA_PATH)
    df_sound = pd.read_csv(ACOUSTIC_DATA_PATH)

    print(f"実験データ読み込み: {len(df_exp)} 行")

    parsed_responses = []

    # 2. 全行を走査して、「評価行」とその「直前の行（音声行）」をペアにする
    # iterrowsを使わず、インデックスでアクセスするほうが確実です
    for i in range(len(df_exp)):
        row = df_exp.iloc[i]

        # 評価画面 (survey-likert) の行を見つける
        if row["trial_type"] == "survey-likert":
            try:
                # --- A. ファイル名の取得 (Look-back戦略) ---
                filename = None

                # 戦略1: まず自分の行に 'filename' 列があるか確認 (jsPsychのバージョンによる)
                if "filename" in row and pd.notna(row["filename"]):
                    filename = row["filename"]

                # 戦略2: なければ「1つ前の行」の 'stimulus' を確認する (これが最強の手段)
                elif i > 0:
                    prev_row = df_exp.iloc[i - 1]
                    # 前の行が音声再生(audio-keyboard-response)か確認
                    if prev_row["trial_type"] == "audio-keyboard-response":
                        raw_stimulus = prev_row["stimulus"]
                        if isinstance(raw_stimulus, str):
                            filename = os.path.basename(raw_stimulus)

                # 戦略3: それでもダメなら自分の行の 'stimulus' を見る (一応)
                if filename is None and isinstance(row["stimulus"], str):
                    filename = os.path.basename(row["stimulus"])

                # ファイル名が取れなかったらスキップ
                if filename is None:
                    # print(f"Warning: 行 {i} のファイル名が特定できませんでした。スキップします。")
                    continue

                # --- B. 回答の取得 ---
                # response列をJSONパース
                if isinstance(row["response"], str):
                    resp_dict = json.loads(row["response"])

                    parsed_responses.append(
                        {
                            "filename": filename,
                            "pleasantness": resp_dict.get(
                                "pleasantness", None
                            ),  # 設定したname属性と一致させる
                            "annoyance": resp_dict.get("annoyance", None),
                            "rt": row["rt"],
                        }
                    )

            except Exception as e:
                print(f"行 {i} でエラー: {e}")
                continue

    # 3. 結合と保存
    if not parsed_responses:
        print("\n【エラー】有効なデータが抽出できませんでした。")
        print("CSVのカラム名:", df_exp.columns.tolist())
    else:
        df_parsed = pd.DataFrame(parsed_responses)
        print(f"\n抽出成功: {len(df_parsed)} 件の回答データ")

        # 特徴量と結合
        df_merged = pd.merge(df_parsed, df_sound, on="filename", how="left")

        # 保存
        output_csv = os.path.join(OUTPUT_DIR, "merged_analysis_data.csv")
        df_merged.to_csv(output_csv, index=False)
        print(f"結合データ保存完了: {output_csv}")

        # 4. グラフ作成
        if len(df_merged) >= 3:
            plt.figure(figsize=(12, 5))

            # グラフA: 鋭さ vs 煩わしさ
            plt.subplot(1, 2, 1)
            # データ型を数値に変換(念のため)
            df_merged["annoyance"] = pd.to_numeric(
                df_merged["annoyance"], errors="coerce"
            )
            sns.regplot(data=df_merged, x="spectral_centroid", y="annoyance")
            plt.title("Sharpness vs Annoyance")

            # グラフB: アタック vs 心地よさ
            plt.subplot(1, 2, 2)
            df_merged["pleasantness"] = pd.to_numeric(
                df_merged["pleasantness"], errors="coerce"
            )
            sns.regplot(data=df_merged, x="zero_crossing_rate", y="pleasantness")
            plt.title("Roughness vs Pleasantness")

            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, "preliminary_correlation.png")
            plt.savefig(plot_path)
            print(f"グラフ保存完了: {plot_path}")

            # 相関行列
            cols = [
                "annoyance",
                "pleasantness",
                "spectral_centroid",
                "zero_crossing_rate",
            ]
            print("\n--- 相関係数 ---")
            print(df_merged[cols].corr().iloc[:2, 2:])
