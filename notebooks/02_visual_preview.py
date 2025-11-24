import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 日本語フォント設定
plt.rcParams["font.family"] = "Hiragino Sans"

# 1. データの読み込み
csv_path = "./data/processed/acoustic_features.csv"
df = pd.read_csv(csv_path)

# 2. 散布図の作成
plt.figure(figsize=(12, 8))

# カテゴリごとに色分けしてプロット
# カラムから日本語のラベルマッピングを作成
label_mapping = {
    "zero_crossing_rate": "Roughness",
    "spectral_centroid": "Sharpness",
}

# X軸とY軸のカラム名
x_col = "spectral_centroid"
y_col = "zero_crossing_rate"


sns.scatterplot(
    data=df,
    x=x_col,
    y=y_col,
    hue="category",
    style="category",
    s=100,  # 点のサイズ
    alpha=0.7,  # 透明度
)

# グラフの装飾
plt.title(
    f"Acoustic Feature Distribution Map ({label_mapping[x_col]} - {label_mapping[y_col]})",
    fontsize=16,
)
plt.xlabel(label_mapping[x_col], fontsize=12)
plt.ylabel(label_mapping[y_col], fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

# 3. 画像として保存
output_img = "./data/plots/sound_distribution_map.png"
plt.tight_layout()
plt.savefig(output_img)
print(f"Distribution map saved: {output_img}")
