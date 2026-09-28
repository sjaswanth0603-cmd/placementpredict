"""
Principal Component Analysis (PCA) Module for Placement Prediction.

Performs 2-component PCA on standardized numeric student features,
computes explained variance and component loadings, generates a 2D scatter
plot colored by PlacementStatus, and saves the variance report and datasets.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from unsupervised_preprocessing import (
    ensure_output_directories,
    prepare_unsupervised_data,
    OUTPUTS_DIR,
    PLOTS_DIR,
    DATA_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    ID_COLUMN,
)

CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "charts")


def run_pca_analysis(filepath: str = None, sample_size: int = None) -> dict:
    """Execute complete 2-component PCA workflow.

    Args:
        filepath: Optional path to dataset CSV.
        sample_size: Optional sample size for faster processing.

    Returns:
        dict: Summary containing variance statistics, paths to plots, data, and report.
    """
    ensure_output_directories()
    os.makedirs(CHARTS_DIR, exist_ok=True)

    print("\n========== PCA MODULE STARTED ==========")
    X_scaled, df_imputed, df_meta, scaler, feature_names = prepare_unsupervised_data(
        filepath=filepath, sample_size=sample_size
    )

    # Fit 2-Component PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Explained Variance Metrics
    ev_pc1 = pca.explained_variance_[0]
    ev_pc2 = pca.explained_variance_[1]
    ev_ratio_pc1 = pca.explained_variance_ratio_[0]
    ev_ratio_pc2 = pca.explained_variance_ratio_[1]
    total_ev_ratio = ev_ratio_pc1 + ev_ratio_pc2

    print(f"PC1 Explained Variance: {ev_pc1:.4f} ({ev_ratio_pc1 * 100:.2f}%)")
    print(f"PC2 Explained Variance: {ev_pc2:.4f} ({ev_ratio_pc2 * 100:.2f}%)")
    print(f"Total Explained Variance (PC1 + PC2): {total_ev_ratio * 100:.2f}%")

    # Build Transformed DataFrame
    df_transformed = pd.DataFrame(
        X_pca, columns=["PC1", "PC2"], index=df_meta.index
    )
    if ID_COLUMN in df_meta.columns:
        df_transformed[ID_COLUMN] = df_meta[ID_COLUMN]
    df_transformed[TARGET_COLUMN] = df_meta[TARGET_COLUMN]

    # Save Transformed Data
    transformed_csv_path = os.path.join(DATA_DIR, "pca_transformed_features.csv")
    df_transformed.to_csv(transformed_csv_path, index=False)
    print(f"Saved transformed PCA data to: {transformed_csv_path}")

    # Component Loadings (Feature Contributions)
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=["PC1_Loading", "PC2_Loading"],
        index=feature_names,
    )
    loadings_df["PC1_Abs"] = loadings_df["PC1_Loading"].abs()
    loadings_df["PC2_Abs"] = loadings_df["PC2_Loading"].abs()
    loadings_sorted_pc1 = loadings_df.sort_values(by="PC1_Abs", ascending=False)

    # Generate Detailed Text Report
    report_txt_path = os.path.join(REPORTS_DIR, "pca_variance_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("      PRINCIPAL COMPONENT ANALYSIS (PCA) VARIANCE REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Total Samples Analyzed: {len(X_scaled):,}\n")
        f.write(f"Original Feature Dimension: {len(feature_names)}\n")
        f.write(f"Reduced Dimension: 2 Principal Components\n\n")
        f.write("-" * 65 + "\n")
        f.write("1. EXPLAINED VARIANCE SUMMARY\n")
        f.write("-" * 65 + "\n")
        f.write(f"Principal Component 1 (PC1):\n")
        f.write(f"  - Eigenvalue / Explained Variance: {ev_pc1:.4f}\n")
        f.write(f"  - Explained Variance Ratio:        {ev_ratio_pc1:.4f} ({ev_ratio_pc1 * 100:.2f}%)\n\n")
        f.write(f"Principal Component 2 (PC2):\n")
        f.write(f"  - Eigenvalue / Explained Variance: {ev_pc2:.4f}\n")
        f.write(f"  - Explained Variance Ratio:        {ev_ratio_pc2:.4f} ({ev_ratio_pc2 * 100:.2f}%)\n\n")
        f.write(f"Cumulative Variance (PC1 + PC2):     {total_ev_ratio:.4f} ({total_ev_ratio * 100:.2f}%)\n\n")
        f.write("-" * 65 + "\n")
        f.write("2. TOP CONTRIBUTING FEATURES TO PC1\n")
        f.write("-" * 65 + "\n")
        for rank, (feat, row) in enumerate(loadings_sorted_pc1.head(8).iterrows(), start=1):
            f.write(f"  {rank}. {feat:<22} Loading: {row['PC1_Loading']:+.4f} (Mag: {row['PC1_Abs']:.4f})\n")
        f.write("\n" + "-" * 65 + "\n")
        f.write("3. TOP CONTRIBUTING FEATURES TO PC2\n")
        f.write("-" * 65 + "\n")
        loadings_sorted_pc2 = loadings_df.sort_values(by="PC2_Abs", ascending=False)
        for rank, (feat, row) in enumerate(loadings_sorted_pc2.head(8).iterrows(), start=1):
            f.write(f"  {rank}. {feat:<22} Loading: {row['PC2_Loading']:+.4f} (Mag: {row['PC2_Abs']:.4f})\n")
        f.write("\n" + "=" * 65 + "\n")

    # Save Variance Metrics CSV
    variance_csv_path = os.path.join(REPORTS_DIR, "pca_variance_report.csv")
    var_summary_df = pd.DataFrame(
        [
            {
                "Component": "PC1",
                "Explained_Variance": ev_pc1,
                "Explained_Variance_Ratio": ev_ratio_pc1,
                "Explained_Variance_Percent": ev_ratio_pc1 * 100,
                "Cumulative_Variance_Percent": ev_ratio_pc1 * 100,
            },
            {
                "Component": "PC2",
                "Explained_Variance": ev_pc2,
                "Explained_Variance_Ratio": ev_ratio_pc2,
                "Explained_Variance_Percent": ev_ratio_pc2 * 100,
                "Cumulative_Variance_Percent": total_ev_ratio * 100,
            },
        ]
    )
    var_summary_df.to_csv(variance_csv_path, index=False)
    print(f"Saved variance reports to: {report_txt_path} and {variance_csv_path}")

    # Generate 2D Scatter Plot
    plt.figure(figsize=(10, 7.5), dpi=150)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # For clear visualization of large datasets, plot sample or adjust alpha
    plot_df = df_transformed
    if len(plot_df) > 10000:
        plot_sample = plot_df.sample(n=10000, random_state=42)
    else:
        plot_sample = plot_df

    palette = {0: "#e74c3c", 1: "#2980b9"}  # Red for Not Placed, Blue for Placed
    labels = {0: "Not Placed (0)", 1: "Placed (1)"}

    for status_val in [0, 1]:
        subset = plot_sample[plot_sample[TARGET_COLUMN] == status_val]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            c=palette[status_val],
            label=labels[status_val],
            alpha=0.55,
            edgecolors="none",
            s=22,
        )

    plt.title(
        f"2D PCA Projection of Student Placement Features\n"
        f"(PC1: {ev_ratio_pc1 * 100:.2f}% | PC2: {ev_ratio_pc2 * 100:.2f}% | Total: {total_ev_ratio * 100:.2f}%)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel(f"Principal Component 1 ({ev_ratio_pc1 * 100:.2f}% Variance)", fontsize=12, labelpad=10)
    plt.ylabel(f"Principal Component 2 ({ev_ratio_pc2 * 100:.2f}% Variance)", fontsize=12, labelpad=10)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.legend(title="Placement Status", loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()

    plot_file_path = os.path.join(PLOTS_DIR, "pca_2d_scatter.png")
    plt.savefig(plot_file_path, dpi=150, bbox_inches="tight")
    # Also save to static/charts for web compatibility
    static_plot_path = os.path.join(CHARTS_DIR, "pca_2d_scatter.png")
    plt.savefig(static_plot_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved PCA 2D scatter visualization to: {plot_file_path}")

    return {
        "status": "success",
        "ev_pc1": ev_pc1,
        "ev_pc2": ev_pc2,
        "ev_ratio_pc1": ev_ratio_pc1,
        "ev_ratio_pc2": ev_ratio_pc2,
        "total_ev_ratio": total_ev_ratio,
        "plot_path": plot_file_path,
        "csv_path": transformed_csv_path,
        "report_txt": report_txt_path,
        "report_csv": variance_csv_path,
    }


if __name__ == "__main__":
    results = run_pca_analysis()
    print("========== PCA MODULE COMPLETED ==========")
