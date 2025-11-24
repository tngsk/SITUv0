import matplotlib
import pandas as pd

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import seaborn as sns

# --- Settings ---
# File paths
csv_path = "./data/processed/acoustic_features.csv"
output_dir = "./data/plots"
os.makedirs(output_dir, exist_ok=True)

# --- Load Data ---
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found -> {csv_path}")
else:
    df = pd.read_csv(csv_path)
    print(f"Data loaded: {len(df)} files")

    # ==========================================
    # Plot 1: Sharpness (Centroid) vs Attack Time
    # ==========================================
    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=df,
        x="spectral_centroid",
        y="attack_peak_sec",
        hue="category",
        style="category",
        s=120,
        alpha=0.7,
    )

    # English Titles and Labels
    plt.title("Notification Sound Map (Sharpness - Attack)", fontsize=16)
    plt.xlabel("Spectral Centroid (Hz) -> Sharper/Brighter", fontsize=12)
    plt.ylabel("Attack Time (s) -> Slower/Gentler", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

    output_img1 = os.path.join(output_dir, "map_centroid_attack.png")
    plt.tight_layout()
    plt.savefig(output_img1)
    print(f"Saved Plot 1: {output_img1}")
    plt.close()

    # ==========================================
    # Plot 2: Feature Pairplot (Scatter Matrix)
    # ==========================================
    target_cols = [
        "spectral_centroid",
        "attack_peak_sec",
        "rms_max",
        "zero_crossing_rate",
    ]

    # Rename columns to English for the plot
    df_pair = df[target_cols + ["category"]].copy()
    df_pair.columns = [
        "Sharpness (Centroid)",
        "Attack Time",
        "Max Amplitude",
        "Roughness (ZCR)",
        "category",
    ]

    print("Generating pairplot... (this may take a moment)")
    pair_plot = sns.pairplot(
        df_pair, hue="category", diag_kind="hist", plot_kws={"alpha": 0.6, "s": 40}
    )
    pair_plot.fig.suptitle("Feature Pairplot Matrix", y=1.02, fontsize=16)

    output_img2 = os.path.join(output_dir, "feature_pairplot.png")
    plt.savefig(output_img2)
    print(f"Saved Plot 2: {output_img2}")
    plt.close()
