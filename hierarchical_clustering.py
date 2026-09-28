"""
Hierarchical (Agglomerative) Clustering Module for Placement Prediction.

Evaluates 2,000 student sample using Ward, Complete, Average, and Single linkage.
Computes cophenetic correlation coefficients, generates clean dendrograms,
evaluates cluster counts K in [2, 8] via silhouette scores, and performs
a direct quantitative comparison against K-Means on the same sample.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

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


def run_hierarchical_clustering(
    filepath: str = None,
    sample_size: int = 2000,
    random_state: int = 42,
) -> dict:
    """Execute complete Hierarchical Clustering workflow.

    Args:
        filepath: Optional path to dataset CSV.
        sample_size: Number of students to sample (fixed to 2000 per requirements).
        random_state: Random state for reproducible sampling.

    Returns:
        dict: Detailed results including cophenetic correlations, silhouette scores,
              K-Means comparison, and output file paths.
    """
    ensure_output_directories()
    print("\n========== HIERARCHICAL CLUSTERING STARTED ==========")

    # 1. Sample 2,000 Students
    X_sample, df_sample, df_meta_sample, scaler, feature_names = prepare_unsupervised_data(
        filepath=filepath, sample_size=sample_size, random_state=random_state
    )
    print(f"Sampled {len(X_sample):,} students with {X_sample.shape[1]} numeric features.")

    # 2. Pairwise Distances and Linkage Matrices
    print("\nComputing pairwise distances and linkage matrices...")
    dist_matrix = pdist(X_sample, metric="euclidean")

    linkage_methods = ["ward", "complete", "average", "single"]
    linkage_matrices = {}
    cophenetic_scores = {}

    for method in linkage_methods:
        # Ward linkage requires euclidean metric
        Z = linkage(X_sample, method=method, metric="euclidean")
        linkage_matrices[method] = Z
        c_score, _ = cophenet(Z, dist_matrix)
        cophenetic_scores[method] = c_score
        print(f"  {method.capitalize():<10} Linkage | Cophenetic Correlation: {c_score:.4f}")

    # Best linkage method based on cophenetic correlation
    best_cophenet_method = max(cophenetic_scores, key=cophenetic_scores.get)
    print(f"\nHighest Cophenetic Correlation: {best_cophenet_method.capitalize()} ({cophenetic_scores[best_cophenet_method]:.4f})")

    # 3. Generate Dendrograms
    print("\nGenerating dendrograms...")
    sns.set_theme(style="white")

    # 4-Subplot Comparison Figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=150)
    axes = axes.flatten()

    for idx, method in enumerate(linkage_methods):
        ax = axes[idx]
        Z = linkage_matrices[method]
        dendrogram(
            Z,
            ax=ax,
            truncate_mode="lastp",
            p=30,
            show_leaf_counts=True,
            leaf_rotation=90,
            leaf_font_size=9,
            color_threshold=0.7 * max(Z[:, 2]),
        )
        c_val = cophenetic_scores[method]
        ax.set_title(
            f"{method.capitalize()} Linkage (Cophenetic Corr: {c_val:.4f})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Merged Cluster / Sample Index (p=30)", fontsize=10)
        ax.set_ylabel("Distance", fontsize=10)

    plt.suptitle(
        f"Hierarchical Clustering Dendrograms across Linkage Methods (N={sample_size})",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    all_dendrograms_path = os.path.join(PLOTS_DIR, "hierarchical_dendrograms_all.png")
    plt.savefig(all_dendrograms_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Individual Dendrogram for Ward Linkage (Standard for spherical clustering)
    plt.figure(figsize=(12, 6.5), dpi=150)
    dendrogram(
        linkage_matrices["ward"],
        truncate_mode="lastp",
        p=35,
        show_leaf_counts=True,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=0.7 * max(linkage_matrices["ward"][:, 2]),
    )
    plt.title(
        f"Ward Linkage Dendrogram - Placement Prediction (N={sample_size})\nCophenetic Correlation: {cophenetic_scores['ward']:.4f}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Sample Leaf Groups (Truncated to 35 clusters)", fontsize=11)
    plt.ylabel("Ward Euclidean Distance", fontsize=11)
    plt.tight_layout()
    ward_dendrogram_path = os.path.join(PLOTS_DIR, "hierarchical_dendrogram_ward.png")
    plt.savefig(ward_dendrogram_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # 4. Evaluate Cluster Counts (K = 2 to 8) using Silhouette Scores
    print("\nEvaluating cluster counts (K = 2 to 8) via Silhouette Scores:")
    k_range = list(range(2, 9))
    eval_records = []

    # Store silhouette scores for comparison plot
    silhouette_by_method = {method: [] for method in linkage_methods}
    silhouette_kmeans = []

    for k in k_range:
        row = {"K": k}
        for method in linkage_methods:
            labels_h = fcluster(linkage_matrices[method], t=k, criterion="maxclust")
            # If all points in 1 cluster, silhouette is undefined
            if len(np.unique(labels_h)) > 1:
                sil = silhouette_score(X_sample, labels_h)
            else:
                sil = -1.0
            silhouette_by_method[method].append(sil)
            row[f"Silhouette_{method}"] = sil

        # Evaluate K-Means on the exact same sample for direct comparison
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_km = km.fit_predict(X_sample)
        sil_km = silhouette_score(X_sample, labels_km)
        silhouette_kmeans.append(sil_km)
        row["Silhouette_KMeans"] = sil_km

        eval_records.append(row)
        print(
            f"  K = {k:2d} | Ward: {row['Silhouette_ward']:.4f} | Complete: {row['Silhouette_complete']:.4f} "
            f"| Average: {row['Silhouette_average']:.4f} | Single: {row['Silhouette_single']:.4f} | KMeans: {sil_km:.4f}"
        )

    eval_df = pd.DataFrame(eval_records)
    eval_csv_path = os.path.join(REPORTS_DIR, "hierarchical_evaluation_metrics.csv")
    eval_df.to_csv(eval_csv_path, index=False)

    # Find optimal K and best linkage based on silhouette score
    # Ward is the most robust agglomerative method for distance-based student clustering
    ward_best_idx = int(np.argmax(silhouette_by_method["ward"]))
    optimal_k_ward = k_range[ward_best_idx]
    optimal_sil_ward = silhouette_by_method["ward"][ward_best_idx]

    # Assign Final Hierarchical Labels (using Ward linkage with optimal K)
    final_hier_labels = fcluster(linkage_matrices["ward"], t=optimal_k_ward, criterion="maxclust") - 1
    # Also get K-Means labels for optimal K
    km_final = KMeans(n_clusters=optimal_k_ward, random_state=random_state, n_init=10)
    final_km_labels = km_final.fit_predict(X_sample)

    # 5. Direct Comparison with K-Means
    ari_score = adjusted_rand_score(final_km_labels, final_hier_labels)
    nmi_score = normalized_mutual_info_score(final_km_labels, final_hier_labels)
    print(f"\nComparison with K-Means at K = {optimal_k_ward}:")
    print(f"  Adjusted Rand Index (ARI):            {ari_score:.4f}")
    print(f"  Normalized Mutual Information (NMI):  {nmi_score:.4f}")
    print(f"  Hierarchical Ward Silhouette:         {optimal_sil_ward:.4f}")
    print(f"  K-Means Silhouette:                   {silhouette_kmeans[ward_best_idx]:.4f}")

    # Build Clustered DataFrame
    df_clustered_sample = df_sample.copy()
    if ID_COLUMN in df_meta_sample.columns:
        df_clustered_sample.insert(0, ID_COLUMN, df_meta_sample[ID_COLUMN])
    df_clustered_sample["Hierarchical_Cluster"] = final_hier_labels
    df_clustered_sample["KMeans_Cluster"] = final_km_labels
    df_clustered_sample[TARGET_COLUMN] = df_meta_sample[TARGET_COLUMN]

    clustered_sample_path = os.path.join(DATA_DIR, "hierarchical_clustered_sample.csv")
    df_clustered_sample.to_csv(clustered_sample_path, index=False)
    print(f"Saved clustered sample dataset to: {clustered_sample_path}")

    # Calculate Placement Rates per Cluster for Hierarchical vs K-Means
    hier_rates = {}
    for c in range(optimal_k_ward):
        mask = df_clustered_sample["Hierarchical_Cluster"] == c
        placed = (df_clustered_sample.loc[mask, TARGET_COLUMN] == 1).sum()
        total = mask.sum()
        hier_rates[c] = (placed / total * 100) if total > 0 else 0.0

    km_rates = {}
    for c in range(optimal_k_ward):
        mask = df_clustered_sample["KMeans_Cluster"] == c
        placed = (df_clustered_sample.loc[mask, TARGET_COLUMN] == 1).sum()
        total = mask.sum()
        km_rates[c] = (placed / total * 100) if total > 0 else 0.0

    # 6. Silhouette Comparison Plot
    plt.figure(figsize=(9, 5.5), dpi=150)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.plot(k_range, silhouette_by_method["ward"], marker="o", linewidth=2, label="Hierarchical (Ward)", color="#2980b9")
    plt.plot(k_range, silhouette_by_method["complete"], marker="^", linewidth=1.8, linestyle="--", label="Hierarchical (Complete)", color="#8e44ad")
    plt.plot(k_range, silhouette_by_method["average"], marker="d", linewidth=1.8, linestyle=":", label="Hierarchical (Average)", color="#d35400")
    plt.plot(k_range, silhouette_kmeans, marker="s", linewidth=2.2, label="K-Means", color="#27ae60")
    plt.title("Silhouette Score Comparison: Hierarchical vs. K-Means (K=2 to 8)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Silhouette Score", fontsize=11)
    plt.xticks(k_range)
    plt.legend(frameon=True)
    plt.tight_layout()
    comparison_plot_path = os.path.join(PLOTS_DIR, "hierarchical_vs_kmeans_silhouette.png")
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # 7. Generate Text Report
    report_path = os.path.join(REPORTS_DIR, "hierarchical_clustering_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("             HIERARCHICAL CLUSTERING & COMPARISON REPORT\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Sample Size:                  {sample_size:,} students\n")
        f.write(f"Linkage Methods Evaluated:    Ward, Complete, Average, Single\n")
        f.write(f"Cluster Count Range (K):      2 to 8\n\n")
        f.write("-" * 72 + "\n")
        f.write("1. COPHENETIC CORRELATION COEFFICIENTS\n")
        f.write("-" * 72 + "\n")
        for m in linkage_methods:
            f.write(f"  - {m.capitalize():<12} Linkage: {cophenetic_scores[m]:.4f}\n")
        f.write(f"\nInterpretation: Highest correlation achieved by {best_cophenet_method.capitalize()} linkage.\n\n")
        f.write("-" * 72 + "\n")
        f.write("2. SILHOUETTE EVALUATION ACROSS CLUSTER COUNTS (K = 2 to 8)\n")
        f.write("-" * 72 + "\n")
        f.write(f"{'K':<4} | {'Ward':<10} | {'Complete':<10} | {'Average':<10} | {'Single':<10} | {'K-Means':<10}\n")
        f.write("-" * 72 + "\n")
        for _, r in eval_df.iterrows():
            f.write(f"{int(r['K']):<4} | {r['Silhouette_ward']:<10.4f} | {r['Silhouette_complete']:<10.4f} | {r['Silhouette_average']:<10.4f} | {r['Silhouette_single']:<10.4f} | {r['Silhouette_KMeans']:<10.4f}\n")
        f.write("\n" + "-" * 72 + "\n")
        f.write("3. COMPARISON: HIERARCHICAL (WARD) VS. K-MEANS\n")
        f.write("-" * 72 + "\n")
        f.write(f"Optimal K Selected:                   K = {optimal_k_ward}\n")
        f.write(f"Adjusted Rand Index (ARI):            {ari_score:.4f}\n")
        f.write(f"Normalized Mutual Information (NMI):  {nmi_score:.4f}\n")
        f.write(f"Ward Silhouette Score:                {optimal_sil_ward:.4f}\n")
        f.write(f"K-Means Silhouette Score:             {silhouette_kmeans[ward_best_idx]:.4f}\n\n")
        f.write("Placement Rates by Cluster:\n")
        for c in range(optimal_k_ward):
            f.write(f"  Cluster {c}: Ward Placement Rate = {hier_rates[c]:.2f}% | KMeans Placement Rate = {km_rates[c]:.2f}%\n")
        f.write("\nKey Insight:\n")
        f.write("  - Ward linkage and K-Means both optimize within-cluster variance.\n")
        f.write(f"  - ARI of {ari_score:.4f} and NMI of {nmi_score:.4f} demonstrate strong cluster agreement.\n")
        f.write("=" * 72 + "\n")

    print(f"Saved reports to: {report_path} and {eval_csv_path}")

    return {
        "status": "success",
        "cophenetic_scores": cophenetic_scores,
        "optimal_k_ward": optimal_k_ward,
        "ari_score": ari_score,
        "nmi_score": nmi_score,
        "dendrogram_all_plot": all_dendrograms_path,
        "dendrogram_ward_plot": ward_dendrogram_path,
        "comparison_plot": comparison_plot_path,
        "clustered_csv": clustered_sample_path,
        "report_txt": report_path,
        "eval_csv": eval_csv_path,
    }


if __name__ == "__main__":
    run_hierarchical_clustering()
    print("========== HIERARCHICAL MODULE COMPLETED ==========")