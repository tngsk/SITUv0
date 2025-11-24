import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # 画像保存モード
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

# --- 設定 (Configuration) ---
# 1. フィルタリング設定
# 特定のカテゴリだけ選ぶ場合: ["alarm"] や ["ui", "notification"] のようにリストで指定
# 全てを対象にする場合: None または []
TARGET_CATEGORIES = None
# TARGET_CATEGORIES = ["ui"] # 例: UI音だけで選抜したい場合

# 2. 選抜数
N_SAMPLES = 20

# 3. クラスタリングに使用する特徴量
FEATURES = ["spectral_centroid", "zero_crossing_rate"]

# 4. 再現性のための乱数シード
RANDOM_SEED = 42

# パス設定
BASE_DIR = "."
INPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "acoustic_features.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "selected_stimuli_set.csv")
OUTPUT_IMG = os.path.join(BASE_DIR, "data", "plots", "selected_stimuli_map.png")

# --- 処理 ---
if not os.path.exists(INPUT_CSV):
    print("エラー: 入力CSVが見つかりません。")
else:
    df = pd.read_csv(INPUT_CSV)

    # --- 【追加】カテゴリフィルタリング ---
    if TARGET_CATEGORIES:
        print(f"フィルタ適用中: {TARGET_CATEGORIES} のみ抽出します")
        df_filtered = df[df["category"].isin(TARGET_CATEGORIES)].copy()

        # フィルタ後のチェック
        if len(df_filtered) == 0:
            print(
                f"エラー: 指定されたカテゴリ {TARGET_CATEGORIES} のデータがありません。"
            )
            # 処理を中断しないよう、空でなければ続行する制御が必要ですが、
            # ここでは単純に元データに戻すか、エラー終了するかの選択になります。
            # 今回はエラーとして停止します。
            exit()
    else:
        print("フィルタなし: 全カテゴリを対象にします")
        df_filtered = df.copy()

    print(f"対象データ数: {len(df_filtered)} 件")

    # --- 安全策: データ数が要求数より少ない場合の調整 ---
    n_clusters = min(N_SAMPLES, len(df_filtered))
    if n_clusters < N_SAMPLES:
        print(
            f"警告: データ数({len(df_filtered)})が要求数({N_SAMPLES})より少ないため、全データを採用します。"
        )

    # 1. データの前処理 (標準化)
    scaler = StandardScaler()
    X = df_filtered[FEATURES].values

    # データ数が1つ以上ある場合のみ実行
    if len(X) > 0:
        X_scaled = scaler.fit_transform(X)

        # 2. K-Means クラスタリング
        print(f"クラスタリング実行中 (Target: {n_clusters} files)...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        kmeans.fit(X_scaled)

        # 3. 各クラスタの中心に最も近いデータを探す
        closest_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, X_scaled
        )

        # 選ばれたデータを抽出
        selected_df = df_filtered.iloc[closest_indices].copy()
        selected_df["cluster_id"] = range(n_clusters)

        # 4. 結果の保存
        selected_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n完了: 選抜リストを保存しました -> {OUTPUT_CSV}")
        print("--- 選抜されたファイル (Top 5) ---")
        print(selected_df[["filename", "category"]].head())

        # 5. 可視化
        plt.figure(figsize=(12, 8))

        # 背景: 全体データ (薄いグレー) - フィルタ前の全体分布を見せる
        sns.scatterplot(
            data=df,
            x="spectral_centroid",
            y="zero_crossing_rate",
            color="lightgray",
            alpha=0.3,
            s=30,
            label="All Data (Background)",
        )

        # 対象データ: フィルタリングされたデータ (少し濃いグレー)
        if TARGET_CATEGORIES:
            sns.scatterplot(
                data=df_filtered,
                x="spectral_centroid",
                y="zero_crossing_rate",
                color="gray",
                alpha=0.5,
                s=50,
                label="Filtered Candidates",
            )

        # 選抜データ: 赤色で強調
        sns.scatterplot(
            data=selected_df,
            x="spectral_centroid",
            y="zero_crossing_rate",
            color="red",
            s=150,
            marker="*",
            label="Selected Stimuli",
            edgecolor="black",
        )

        # ID付与
        for _, row in selected_df.iterrows():
            plt.text(
                row["spectral_centroid"],
                row["zero_crossing_rate"],
                str(row["cluster_id"]),
                color="black",
                fontsize=9,
                ha="right",
                va="bottom",
            )

        title_text = f"Selection ({n_clusters} samples)"
        if TARGET_CATEGORIES:
            title_text += f" [Filter: {','.join(TARGET_CATEGORIES)}]"

        plt.title(title_text, fontsize=16)
        plt.xlabel("Spectral Centroid (Sharpness)", fontsize=12)
        plt.ylabel("ZCR (Roughness)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(OUTPUT_IMG)
        print(f"確認用マップを保存しました -> {OUTPUT_IMG}")
        plt.close()
    else:
        print("処理対象データが存在しませんでした。")
