import json
import os

import pandas as pd

# --- 設定 ---
BASE_DIR = "."
INPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "selected_stimuli_set.csv")

# 実験用フォルダの作成
EXP_DIR = os.path.join(BASE_DIR, "experiments", "prototype")
os.makedirs(EXP_DIR, exist_ok=True)

# 出力ファイル名
OUTPUT_JS = os.path.join(EXP_DIR, "stimuli.js")

# --- 変換処理 ---
if not os.path.exists(INPUT_CSV):
    print(f"エラー: {INPUT_CSV} が見つかりません。自動選抜を先に実行してください。")
else:
    df = pd.read_csv(INPUT_CSV)

    # jsPsychで使いやすい形式のリストを作成
    stimuli_list = []

    for _, row in df.iterrows():
        # 重要: HTMLファイルから見た音声ファイルへの相対パスを作成
        # HTMLは /experiments/prototype/ にあるので、
        # 音声データの /data/raw_audio/ までは 2階層上がる(../../)必要があります。
        # ※ row['path'] には "./data/raw_audio/..." が入っている想定

        # パスの調整 (OSごとの区切り文字を / に統一)
        original_path = row["path"].replace("\\", "/")
        if original_path.startswith("./"):
            original_path = original_path[2:]

        # HTMLからの相対パス
        relative_path = "../../" + original_path

        stimuli_list.append(
            {
                "stimulus": relative_path,
                "data": {
                    "filename": row["filename"],
                    "category": row["category"],
                    "cluster_id": row["cluster_id"],
                    # 念のため特徴量も持たせておく（分析時に便利）
                    "spectral_centroid": row["spectral_centroid"],
                },
            }
        )

    # JavaScriptファイルとして書き出し
    # "const stimuli = [...]" という形式のテキストファイルを作ります
    js_content = (
        f"const stimuli = {json.dumps(stimuli_list, indent=4, ensure_ascii=False)};"
    )

    with open(OUTPUT_JS, "w", encoding="utf-8") as f:
        f.write(js_content)

    print(f"完了: 実験用データを作成しました -> {OUTPUT_JS}")
    print(f"データ数: {len(stimuli_list)}")
