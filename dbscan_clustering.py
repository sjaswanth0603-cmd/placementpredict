"""
DBSCAN Clustering Module for Placement Prediction.

Evaluates a 5,000 student sample with min_samples=40.
Generates K-distance plot to estimate eps knee point, conducts comprehensive
eps sensitivity analysis, identifies core, border, and noise points,
visualizes clusters in 2D PCA space, and calculates placement rates for
clustered vs. noise points and per cluster.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

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


def run_dbscan_clustering(
    filepath: str = None,
    sample_size: int = 5000,
    min_samples: int = 40,
    eps_candidates: list = None,
    random_state: int = 42,
) -> dict:
    """Execute complete DBSCAN clustering workflow.

    Args:
        filepath: Optional path to dataset CSV.
        sample_size: Sample size (fixed to 5,000 per requirements).
        min_samples: Minimum neighborhood points (fixed to 40 per requirements).
        eps_candidates: Optional custom list of eps values for sensitivity analysis.
        random_state: Random state for reproducible sampling.

    Returns:
        dict: Detailed results including estimated eps, cluster counts, placement rates,
              and output file paths.
    """
    ensure_output_directories()
    print("\n========== DBSCAN CLUSTERING STARTED ==========")

    # 1. Sample 5,000 Students
    X_sample, df_sample, df_meta_sample, scaler, feature_names = prepare_unsupervised_data(
        filepath=filepath, sample_size=sample_size, random_state=random_state
    )
    print(f"Sampled {len(X_sample):,} students with {X_sample.shape[1]} numeric features.")
    print(f"Hyperparameters: min_samples = {min_samples}")

    # 2. Generate K-Distance Plot (k = min_samples = 40)
    print(f"\nComputing {min_samples}-Nearest Neighbors distances...")
    nbrs = NearestNeighbors(n_neighbors=min_samples, metric="euclidean")
    nbrs.fit(X_sample)
    distances, _ = nbrs.kneighbors(X_sample)

    # 40th neighbor distance sorted in ascending order
    k_distances = np.sort(distances[:, -1])

    # Estimate knee point using geometric maximum distance to chord
    # Line connecting first point (0, k_distances[0]) to last point (N-1, k_distances[-1])
    n_points = len(k_distances)
    indices = np.arange(n_points)
    start_pt = np.array([0, k_distances[0]])
    end_pt = np.array([n_points - 1, k_distances[-1]])
    line_vec = end_pt - start_pt
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    vec_to_start = np.column_stack([indices - start_pt[0], k_distances - start_pt[1]])
    proj_lengths = np.dot(vec_to_start, line_vec_norm)
    proj_points = np.outer(proj_lengths, line_vec_norm)
    perp_vecs = vec_to_start - proj_points
    perp_dists = np.linalg.norm(perp_vecs, axis=1)

    knee_idx = int(np.argmax(perp_dists))
    estimated_eps = float(k_distances[knee_idx])
    print(f"Automated Knee Point Detection: Index {knee_idx}, Estimated eps: {estimated_eps:.3f}")

    # Plot K-Distance Curve
    plt.figure(figsize=(9, 5.5), dpi=150)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.plot(indices, k_distances, color="#2980b9", linewidth=2.2, label=f"{min_samples}-NN Distance Curve")
    plt.axvline(knee_idx, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Elbow/Knee Point (Index {knee_idx})")
    plt.axhline(estimated_eps, color="#27ae60", linestyle=":", linewidth=1.5, label=f"Estimated eps ≈ {estimated_eps:.2f}")
    plt.scatter([knee_idx], [estimated_eps], color="#e74c3c", s=100, zorder=5)

    plt.title(f"DBSCAN K-Distance Plot (k = min_samples = {min_samples})\nSample Size N = {sample_size:,}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Points sorted by distance to 40th Nearest Neighbor", fontsize=11)
    plt.ylabel(f"{min_samples}-NN Distance (Euclidean)", fontsize=11)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    kdist_plot_path = os.path.join(PLOTS_DIR, "dbscan_k_distance_plot.png")
    plt.savefig(kdist_plot_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved K-distance plot to: {kdist_plot_path}")

    # 3. Perform EPS Sensitivity Analysis
    if eps_candidates is None:
        # Construct range around estimated eps
        base_eps = round(estimated_eps, 1)
        eps_candidates = sorted(list(set([
            max(1.0, round(base_eps - 1.0, 1)),
            max(1.5, round(base_eps - 0.5, 1)),
            max(1.8, round(base_eps - 0.2, 1)),
            base_eps,
            round(base_eps + 0.2, 1),
            round(base_eps + 0.5, 1),
            round(base_eps + 1.0, 1),
            round(base_eps + 1.5, 1),
        ])))

    print("\nPerforming EPS Sensitivity Analysis:")
    sensitivity_records = []

    for eps_val in eps_candidates:
        db = DBSCAN(eps=eps_val, min_samples=min_samples)
        labels = db.fit_predict(X_sample)

        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})
        n_noise = int((labels == -1).sum())
        noise_pct = (n_noise / sample_size) * 100
        n_core = len(db.core_sample_indices_)
        n_border = sample_size - n_noise - n_core

        # Silhouette score on non-noise points
        non_noise_mask = labels != -1
        if n_clusters > 1 and non_noise_mask.sum() > n_clusters:
            sil_val = silhouette_score(X_sample[non_noise_mask], labels[non_noise_mask])
        else:
            sil_val = -1.0

        rec = {
            "Eps": eps_val,
            "Min_Samples": min_samples,
            "Num_Clusters": n_clusters,
            "Core_Points": n_core,
            "Border_Points": n_border,
            "Noise_Points": n_noise,
            "Noise_Percentage": noise_pct,
            "Silhouette_Clustered": sil_val,
        }
        sensitivity_records.append(rec)
        print(
            f"  eps = {eps_val:4.2f} | Clusters: {n_clusters:2d} | Core: {n_core:4d} "
            f"| Border: {n_border:4d} | Noise: {n_noise:4d} ({noise_pct:5.1f}%) | Silhouette: {sil_val:6.4f}"
        )

    sensitivity_df = pd.DataFrame(sensitivity_records)
    sensitivity_csv_path = os.path.join(REPORTS_DIR, "dbscan_sensitivity_analysis.csv")
    sensitivity_df.to_csv(sensitivity_csv_path, index=False)
    print(f"Saved sensitivity analysis to: {sensitivity_csv_path}")

    # 4. Train Final Model with Optimal EPS
    # Choose eps that gives meaningful clusters (>= 2) with manageable noise (<30%)
    # If none gives >= 2 clusters, choose closest around estimated_eps
    valid_configs = sensitivity_df[
        (sensitivity_df["Num_Clusters"] >= 2) & (sensitivity_df["Noise_Percentage"] <= 40)
    ]
    if not valid_configs.empty:
        best_eps = float(valid_configs.sort_values(by="Silhouette_Clustered", ascending=False).iloc[0]["Eps"])
    else:
        # Fall back to estimated knee eps
        best_eps = estimated_eps

    print(f"\nFinal DBSCAN configuration chosen: eps = {best_eps:.2f}, min_samples = {min_samples}")
    final_dbscan = DBSCAN(eps=best_eps, min_samples=min_samples)
    final_labels = final_dbscan.fit_predict(X_sample)

    # Identify Core, Border, and Noise Points
    core_sample_indices = set(final_dbscan.core_sample_indices_)
    point_types = []
    for idx, lbl in enumerate(final_labels):
        if lbl == -1:
            point_types.append("Noise")
        elif idx in core_sample_indices:
            point_types.append("Core")
        else:
            point_types.append("Border")

    # 5. Assemble Clustered DataFrame
    df_clustered = df_sample.copy()
    if ID_COLUMN in df_meta_sample.columns:
        df_clustered.insert(0, ID_COLUMN, df_meta_sample[ID_COLUMN])
    df_clustered["DBSCAN_Cluster"] = final_labels
    df_clustered["Point_Type"] = point_types
    df_clustered[TARGET_COLUMN] = df_meta_sample[TARGET_COLUMN]

    clustered_csv_path = os.path.join(DATA_DIR, "dbscan_clustered_sample.csv")
    df_clustered.to_csv(clustered_csv_path, index=False)
    print(f"Saved DBSCAN clustered dataset to: {clustered_csv_path}")

    # 6. Placement Rate Analysis
    # Clustered Points vs. Noise Points
    clustered_mask = final_labels != -1
    noise_mask = final_labels == -1

    total_clustered = int(clustered_mask.sum())
    placed_clustered = int((df_clustered.loc[clustered_mask, TARGET_COLUMN] == 1).sum())
    rate_clustered = (placed_clustered / total_clustered * 100) if total_clustered > 0 else 0.0

    total_noise = int(noise_mask.sum())
    placed_noise = int((df_clustered.loc[noise_mask, TARGET_COLUMN] == 1).sum())
    rate_noise = (placed_noise / total_noise * 100) if total_noise > 0 else 0.0

    # Per-Cluster Placement Rates
    cluster_placement_summary = []
    unique_clusters = sorted([c for c in set(final_labels) if c != -1])
    for c_id in unique_clusters:
        c_m = final_labels == c_id
        c_tot = int(c_m.sum())
        c_plc = int((df_clustered.loc[c_m, TARGET_COLUMN] == 1).sum())
        c_rate = (c_plc / c_tot * 100) if c_tot > 0 else 0.0
        cluster_placement_summary.append(
            {
                "Cluster": f"Cluster {c_id}",
                "Total": c_tot,
                "Placed": c_plc,
                "Placement_Rate_Pct": c_rate,
            }
        )

    print("\nPlacement Rate Breakdown:")
    print(f"  Clustered Students: {placed_clustered:,} / {total_clustered:,} ({rate_clustered:.2f}%)")
    print(f"  Noise Students:     {placed_noise:,} / {total_noise:,} ({rate_noise:.2f}%)")
    for cp in cluster_placement_summary:
        print(f"  {cp['Cluster']}: {cp['Placed']:,} / {cp['Total']:,} ({cp['Placement_Rate_Pct']:.2f}%)")

    # 7. 2D Cluster Visualization via PCA
    print("\nVisualizing DBSCAN clusters in 2D PCA space...")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_sample)

    plt.figure(figsize=(10.5, 7.5), dpi=150)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # Plot Noise Points First
    if total_noise > 0:
        plt.scatter(
            X_pca[noise_mask, 0],
            X_pca[noise_mask, 1],
            color="#7f8c8d",
            marker="x",
            s=28,
            alpha=0.6,
            label=f"Noise (Label -1, N={total_noise})",
        )

    # Plot Clustered Points
    if total_clustered > 0:
        scatter = plt.scatter(
            X_pca[clustered_mask, 0],
            X_pca[clustered_mask, 1],
            c=final_labels[clustered_mask],
            cmap="tab10",
            s=30,
            alpha=0.65,
            edgecolors="none",
        )
        legend1 = plt.legend(*scatter.legend_elements(), title="Dense Clusters", loc="upper right")
        plt.gca().add_artist(legend1)

    if total_noise > 0:
        plt.legend(loc="lower right")

    plt.title(
        f"DBSCAN Placement Clusters Projected in 2D PCA Space\n(eps={best_eps:.2f}, min_samples={min_samples} | Sample N={sample_size})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Principal Component 1", fontsize=11)
    plt.ylabel("Principal Component 2", fontsize=11)
    plt.tight_layout()
    dbscan_pca_path = os.path.join(PLOTS_DIR, "dbscan_clusters_pca.png")
    plt.savefig(dbscan_pca_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved DBSCAN PCA visualization to: {dbscan_pca_path}")

    # 8. Generate Text Report
    report_path = os.path.join(REPORTS_DIR, "dbscan_clustering_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("                   DBSCAN CLUSTERING REPORT\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Sample Size:                  {sample_size:,} students\n")
        f.write(f"Min Samples Hyperparameter:   {min_samples}\n")
        f.write(f"Estimated Knee Eps:           {estimated_eps:.3f}\n")
        f.write(f"Selected Optimal Eps:         {best_eps:.2f}\n")
        f.write(f"Total Clusters Identified:    {len(unique_clusters)}\n\n")
        f.write("-" * 72 + "\n")
        f.write("1. POINT CLASSIFICATION BREAKDOWN\n")
        f.write("-" * 72 + "\n")
        core_cnt = len(core_sample_indices)
        border_cnt = total_clustered - core_cnt
        f.write(f"  - Core Points:    {core_cnt:,} ({core_cnt / sample_size * 100:.2f}%)\n")
        f.write(f"  - Border Points:  {border_cnt:,} ({border_cnt / sample_size * 100:.2f}%)\n")
        f.write(f"  - Noise Points:   {total_noise:,} ({total_noise / sample_size * 100:.2f}%)\n")
        f.write(f"  - Total Clustered: {total_clustered:,} ({total_clustered / sample_size * 100:.2f}%)\n\n")
        f.write("-" * 72 + "\n")
        f.write("2. PLACEMENT RATES (CLUSTERED VS. NOISE)\n")
        f.write("-" * 72 + "\n")
        f.write(f"  - Clustered Points Placement Rate: {rate_clustered:.2f}% ({placed_clustered:,}/{total_clustered:,})\n")
        f.write(f"  - Noise Points Placement Rate:     {rate_noise:.2f}% ({placed_noise:,}/{total_noise:,})\n\n")
        if cluster_placement_summary:
            f.write("Per-Cluster Placement Rates:\n")
            for cp in cluster_placement_summary:
                f.write(f"    * {cp['Cluster']}: {cp['Placement_Rate_Pct']:.2f}% ({cp['Placed']:,}/{cp['Total']:,})\n")
        f.write("\n" + "-" * 72 + "\n")
        f.write("3. EPS SENSITIVITY ANALYSIS\n")
        f.write("-" * 72 + "\n")
        f.write(f"{'Eps':<6} | {'Clusters':<9} | {'Core':<7} | {'Border':<7} | {'Noise (%)':<11} | {'Silhouette':<10}\n")
        f.write("-" * 72 + "\n")
        for _, r in sensitivity_df.iterrows():
            f.write(
                f"{r['Eps']:<6.2f} | {int(r['Num_Clusters']):<9d} | {int(r['Core_Points']):<7d} | "
                f"{int(r['Border_Points']):<7d} | {r['Noise_Percentage']:<10.2f}% | {r['Silhouette_Clustered']:<10.4f}\n"
            )
        f.write("=" * 72 + "\n")

    print(f"Saved DBSCAN report to: {report_path}")

    return {
        "status": "success",
        "estimated_eps": estimated_eps,
        "selected_eps": best_eps,
        "n_clusters": len(unique_clusters),
        "total_clustered": total_clustered,
        "total_noise": total_noise,
        "rate_clustered": rate_clustered,
        "rate_noise": rate_noise,
        "kdist_plot": kdist_plot_path,
        "pca_plot": dbscan_pca_path,
        "clustered_csv": clustered_csv_path,
        "sensitivity_csv": sensitivity_csv_path,
        "report_txt": report_path,
    }


if __name__ == "__main__":
    run_dbscan_clustering()
    print("========== DBSCAN MODULE COMPLETED ==========")
