"""
K-Means Clustering Module for Placement Prediction.

Evaluates K values from 2 to 10 using Inertia (Elbow Method) and Silhouette Scores.
Identifies the optimal K (maximum silhouette score), trains the final model,
assigns cluster labels, calculates placement rates per cluster, and generates
elbow, silhouette, and 2D PCA cluster visualizations.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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


def run_kmeans_clustering(
    filepath: str = None,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> dict:
    """Execute complete K-Means clustering workflow.

    Args:
        filepath: Optional path to dataset CSV.
        k_range: Range of cluster counts to evaluate (default 2 to 10).
        random_state: Random state for reproducibility.

    Returns:
        dict: Detailed results including optimal K, placement rates, and file paths.
    """
    ensure_output_directories()
    print("\n========== K-MEANS CLUSTERING STARTED ==========")

    X_scaled, df_imputed, df_meta, scaler, feature_names = prepare_unsupervised_data(
        filepath=filepath
    )
    total_samples = len(X_scaled)
    print(f"Loaded {total_samples:,} samples with {len(feature_names)} features.")

    # 1. Evaluate K from 2 to 10
    inertia_list = []
    silhouette_list = []
    k_values = list(k_range)

    print("\nEvaluating K from 2 to 10:")
    for k in k_values:
        kmeans_temp = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_temp = kmeans_temp.fit_predict(X_scaled)
        inertia_val = kmeans_temp.inertia_
        inertia_list.append(inertia_val)

        # Silhouette score computed using 10,000 representative samples for speed & precision
        sil_val = silhouette_score(
            X_scaled,
            labels_temp,
            sample_size=min(10000, total_samples),
            random_state=random_state,
        )
        silhouette_list.append(sil_val)
        print(f"  K = {k:2d} | Inertia: {inertia_val:12.2f} | Silhouette Score: {sil_val:.4f}")

    # 2. Select Optimal K (Maximum Silhouette Score)
    best_idx = int(np.argmax(silhouette_list))
    optimal_k = k_values[best_idx]
    best_sil_score = silhouette_list[best_idx]
    print(f"\nOptimal K selected based on max silhouette score: K = {optimal_k} (Score: {best_sil_score:.4f})")

    # 3. Train Final Model with Optimal K
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    final_clusters = final_kmeans.fit_predict(X_scaled)

    # 4. Assemble Clustered Dataset
    clustered_df = df_imputed.copy()
    if ID_COLUMN in df_meta.columns:
        clustered_df.insert(0, ID_COLUMN, df_meta[ID_COLUMN])
    clustered_df["Cluster"] = final_clusters
    clustered_df[TARGET_COLUMN] = df_meta[TARGET_COLUMN]

    clustered_csv_path = os.path.join(DATA_DIR, "kmeans_clustered_students.csv")
    clustered_df.to_csv(clustered_csv_path, index=False)
    print(f"Saved clustered dataset to: {clustered_csv_path}")

    # 5. Calculate Placement Rates Per Cluster
    cluster_stats = []
    for c_id in range(optimal_k):
        c_mask = clustered_df["Cluster"] == c_id
        c_total = int(c_mask.sum())
        c_placed = int((clustered_df.loc[c_mask, TARGET_COLUMN] == 1).sum())
        c_not_placed = c_total - c_placed
        c_rate = (c_placed / c_total * 100) if c_total > 0 else 0.0

        # Mean profile features
        avg_cgpa = float(clustered_df.loc[c_mask, "CGPA"].mean()) if "CGPA" in clustered_df else 0.0
        avg_coding = (
            float(clustered_df.loc[c_mask, "CodingTestScore"].mean())
            if "CodingTestScore" in clustered_df
            else 0.0
        )
        avg_aptitude = (
            float(clustered_df.loc[c_mask, "AptitudeTestScore"].mean())
            if "AptitudeTestScore" in clustered_df
            else 0.0
        )

        cluster_stats.append(
            {
                "Cluster": c_id,
                "Total_Students": c_total,
                "Percentage_of_Total": (c_total / total_samples) * 100,
                "Placed_Count": c_placed,
                "Not_Placed_Count": c_not_placed,
                "Placement_Rate_Pct": c_rate,
                "Mean_CGPA": avg_cgpa,
                "Mean_CodingTestScore": avg_coding,
                "Mean_AptitudeTestScore": avg_aptitude,
            }
        )

    stats_df = pd.DataFrame(cluster_stats)
    print("\nCluster Placement Summary:")
    print(stats_df[["Cluster", "Total_Students", "Placed_Count", "Placement_Rate_Pct", "Mean_CGPA"]])

    # 6. Generate Evaluation Metrics CSV
    metrics_df = pd.DataFrame(
        {
            "K": k_values,
            "Inertia_WCSS": inertia_list,
            "Silhouette_Score": silhouette_list,
        }
    )
    metrics_csv_path = os.path.join(REPORTS_DIR, "kmeans_evaluation_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # 7. Generate Text Report
    report_txt_path = os.path.join(REPORTS_DIR, "kmeans_clustering_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("                 K-MEANS CLUSTERING REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Dataset Instances: {total_samples:,}\n")
        f.write(f"Evaluated K Range:        2 to 10\n")
        f.write(f"Optimal Cluster Count K:  {optimal_k}\n")
        f.write(f"Optimal Silhouette Score: {best_sil_score:.4f}\n\n")
        f.write("-" * 70 + "\n")
        f.write("1. K EVALUATION METRICS (INERTIA & SILHOUETTE)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'K':<5} | {'Inertia (WCSS)':<18} | {'Silhouette Score':<18} | {'Status'}\n")
        f.write("-" * 70 + "\n")
        for k_val, in_val, sil_val in zip(k_values, inertia_list, silhouette_list):
            mark = " <-- OPTIMAL (Max Silhouette)" if k_val == optimal_k else ""
            f.write(f"{k_val:<5} | {in_val:<18.2f} | {sil_val:<18.4f} |{mark}\n")
        f.write("\n" + "-" * 70 + "\n")
        f.write("2. CLUSTER PLACEMENT RATES & PROFILES\n")
        f.write("-" * 70 + "\n")
        for row in cluster_stats:
            f.write(f"Cluster {row['Cluster']}:\n")
            f.write(f"  - Size:             {row['Total_Students']:,} students ({row['Percentage_of_Total']:.2f}% of cohort)\n")
            f.write(f"  - Placed Count:     {row['Placed_Count']:,}\n")
            f.write(f"  - Not Placed Count: {row['Not_Placed_Count']:,}\n")
            f.write(f"  - Placement Rate:   {row['Placement_Rate_Pct']:.2f}%\n")
            f.write(f"  - Avg CGPA:         {row['Mean_CGPA']:.2f}\n")
            f.write(f"  - Avg Coding Score: {row['Mean_CodingTestScore']:.2f}\n")
            f.write(f"  - Avg Aptitude:     {row['Mean_AptitudeTestScore']:.2f}\n\n")
        f.write("=" * 70 + "\n")
    print(f"Saved reports to: {report_txt_path} and {metrics_csv_path}")

    # 8. Visualizations
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # Plot 1: Elbow Curve
    plt.figure(figsize=(8.5, 5.5), dpi=150)
    plt.plot(k_values, inertia_list, marker="o", color="#2980b9", linewidth=2.5, markersize=8)
    plt.axvline(optimal_k, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Optimal K = {optimal_k}")
    plt.title("Elbow Method for Optimal K (Inertia / WCSS)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    plt.xticks(k_values)
    plt.legend(frameon=True)
    plt.tight_layout()
    elbow_path = os.path.join(PLOTS_DIR, "kmeans_elbow_curve.png")
    plt.savefig(elbow_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Plot 2: Silhouette Scores Curve
    plt.figure(figsize=(8.5, 5.5), dpi=150)
    plt.plot(k_values, silhouette_list, marker="s", color="#27ae60", linewidth=2.5, markersize=8)
    plt.axvline(optimal_k, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Optimal K = {optimal_k} (Max Score)")
    plt.scatter([optimal_k], [best_sil_score], color="#e74c3c", s=130, zorder=5)
    plt.title("Silhouette Score vs. Number of Clusters (K)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Silhouette Score", fontsize=11)
    plt.xticks(k_values)
    plt.legend(frameon=True)
    plt.tight_layout()
    sil_plot_path = os.path.join(PLOTS_DIR, "kmeans_silhouette_scores.png")
    plt.savefig(sil_plot_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Plot 3: 2-Panel Combined Evaluation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)
    ax1.plot(k_values, inertia_list, marker="o", color="#2980b9", linewidth=2, markersize=7)
    ax1.axvline(optimal_k, color="#e74c3c", linestyle="--", label=f"Optimal K = {optimal_k}")
    ax1.set_title("Elbow Method (Inertia)", fontweight="bold")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.set_xticks(k_values)
    ax1.legend()

    ax2.plot(k_values, silhouette_list, marker="s", color="#27ae60", linewidth=2, markersize=7)
    ax2.axvline(optimal_k, color="#e74c3c", linestyle="--", label=f"Optimal K = {optimal_k}")
    ax2.scatter([optimal_k], [best_sil_score], color="#e74c3c", s=100, zorder=5)
    ax2.set_title("Silhouette Scores", fontweight="bold")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_xticks(k_values)
    ax2.legend()
    plt.tight_layout()
    comb_eval_path = os.path.join(PLOTS_DIR, "kmeans_evaluation_metrics.png")
    plt.savefig(comb_eval_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Plot 4: 2D Cluster Visualization via PCA
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(final_kmeans.cluster_centers_)

    plt.figure(figsize=(10.5, 7.5), dpi=150)
    # Downsample for crisp plotting if large
    if total_samples > 10000:
        sample_indices = np.random.RandomState(random_state).choice(total_samples, size=10000, replace=False)
        plot_x = X_pca[sample_indices, 0]
        plot_y = X_pca[sample_indices, 1]
        plot_labels = final_clusters[sample_indices]
    else:
        plot_x = X_pca[:, 0]
        plot_y = X_pca[:, 1]
        plot_labels = final_clusters

    scatter = plt.scatter(
        plot_x,
        plot_y,
        c=plot_labels,
        cmap="tab10",
        alpha=0.5,
        s=22,
        edgecolors="none",
    )
    plt.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        c="black",
        marker="X",
        s=250,
        linewidths=2,
        edgecolors="white",
        label="Cluster Centroids",
        zorder=10,
    )
    plt.title(f"K-Means Student Clusters (K={optimal_k}) Projected in 2D PCA Space", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Principal Component 1", fontsize=11)
    plt.ylabel("Principal Component 2", fontsize=11)
    plt.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
    plt.tight_layout()
    cluster_plot_path = os.path.join(PLOTS_DIR, "kmeans_clusters_pca.png")
    plt.savefig(cluster_plot_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    print(f"Generated plots:\n  {elbow_path}\n  {sil_plot_path}\n  {comb_eval_path}\n  {cluster_plot_path}")

    return {
        "status": "success",
        "optimal_k": optimal_k,
        "silhouette_score": best_sil_score,
        "cluster_stats": cluster_stats,
        "clustered_csv": clustered_csv_path,
        "elbow_plot": elbow_path,
        "silhouette_plot": sil_plot_path,
        "cluster_pca_plot": cluster_plot_path,
        "report_txt": report_txt_path,
        "metrics_csv": metrics_csv_path,
    }


if __name__ == "__main__":
    run_kmeans_clustering()
    print("========== K-MEANS MODULE COMPLETED ==========")
