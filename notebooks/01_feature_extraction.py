import glob
import os

import librosa
import librosa.display

# バックエンド設定: pyplotをimportする前に設定する
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # GUIを使わない
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 設定 ---
BASE_DIR = "."
AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw_audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
IMG_DIR = os.path.join(BASE_DIR, "data", "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


# --- 分析関数 ---
def extract_features(file_path):
    """
    1つの音声ファイルから基本的な音響特徴量を抽出する
    """
    try:
        y, sr = librosa.load(file_path, sr=None)

        # 1. 時間的特徴
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))

        rms_series = librosa.feature.rms(y=y)[0]
        max_rms = np.max(rms_series) if len(rms_series) > 0 else 0
        max_rms_frame = np.argmax(rms_series) if len(rms_series) > 0 else 0
        max_rms_time = max_rms_frame * (512 / sr)

        # 2. 周波数的特徴
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

        # ピッチ推定
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        f0_mean = np.nanmean(f0) if np.any(f0) else 0

        # カテゴリ取得
        rel_path = os.path.relpath(file_path, AUDIO_DIR)
        category = os.path.split(rel_path)[0]
        if category == "":
            category = "uncategorized"

        return (
            {
                "filename": os.path.basename(file_path),
                "category": category,
                "duration_sec": duration,
                "rms_mean": rms,
                "rms_max": max_rms,
                "attack_peak_sec": max_rms_time,
                "spectral_centroid": cent,
                "spectral_rolloff": rolloff,
                "zero_crossing_rate": zcr,
                "pitch_f0_mean": f0_mean,
                "path": file_path,
            },
            y,
            sr,
        )

    except Exception as e:
        tqdm.write(f"Error analyzing {file_path}: {e}")
        return None, None, None


# メイン処理
search_pattern = os.path.join(AUDIO_DIR, "**", "*.wav")
audio_files = glob.glob(search_pattern, recursive=True)

results = []

print(f"検索ディレクトリ: {AUDIO_DIR}")
print(f"検出ファイル数: {len(audio_files)}")

if not audio_files:
    print("エラー: wavファイルが見つかりません。パスを確認してください。")
else:
    for i, file_path in enumerate(tqdm(audio_files, desc="Processing Audio")):
        features, y, sr = extract_features(file_path)

        if features:
            results.append(features)

            # 最初の1ファイルだけ画像を保存して確認
            if i == 0:
                tqdm.write(f"Saving preview image for: {features['filename']}")

                plt.figure(figsize=(10, 6))

                # 波形
                plt.subplot(2, 1, 1)
                librosa.display.waveshow(y, sr=sr)
                plt.title(f"Waveform: {features['filename']}")

                # スペクトログラム
                plt.subplot(2, 1, 2)
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
                plt.colorbar(format="%+2.0f dB")
                plt.title("Spectrogram")

                plt.tight_layout()

                # ファイル保存
                save_path = os.path.join(IMG_DIR, "preview_first_file.png")
                plt.savefig(save_path)
                plt.close()

                tqdm.write(f"Preview saved to: {save_path}")

    # 結果
    if results:
        df = pd.DataFrame(results)
        cols = [
            "category",
            "filename",
            "duration_sec",
            "rms_mean",
            "rms_max",
            "attack_peak_sec",
            "spectral_centroid",
            "spectral_rolloff",
            "zero_crossing_rate",
            "pitch_f0_mean",
            "path",
        ]
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]

        csv_path = os.path.join(OUTPUT_DIR, "acoustic_features.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n完了: {csv_path} に保存しました。")
