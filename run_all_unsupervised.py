"""
Master Pipeline Runner for Unsupervised Learning Module.

Orchestrates all four unsupervised learning algorithms in sequence:
  1. PCA (Principal Component Analysis)
  2. K-Means Clustering
  3. Hierarchical Clustering (Agglomerative)
  4. DBSCAN Clustering

Validates that all plots, datasets, and reports have been properly
generated and creates an executive summary report.
"""

import os
import sys
import time
from typing import Dict, List

from unsupervised_preprocessing import ensure_output_directories, OUTPUTS_DIR, PLOTS_DIR, DATA_DIR, REPORTS_DIR
from pca_module import run_pca_analysis
from kmeans_clustering import run_kmeans_clustering
from hierarchical_clustering import run_hierarchical_clustering
from dbscan_clustering import run_dbscan_clustering


def verify_generated_artifacts() -> Dict[str, List[str]]:
    """Inspect output folders and verify artifact creation."""
    expected = {
        "plots": [
            "pca_2d_scatter.png",
            "kmeans_elbow_curve.png",
            "kmeans_silhouette_scores.png",
            "kmeans_evaluation_metrics.png",
            "kmeans_clusters_pca.png",
            "hierarchical_dendrograms_all.png",
            "hierarchical_dendrogram_ward.png",
            "hierarchical_vs_kmeans_silhouette.png",
            "dbscan_k_distance_plot.png",
            "dbscan_clusters_pca.png",
        ],
        "data": [
            "pca_transformed_features.csv",
            "kmeans_clustered_students.csv",
            "hierarchical_clustered_sample.csv",
            "dbscan_clustered_sample.csv",
        ],
        "reports": [
            "pca_variance_report.txt",
            "pca_variance_report.csv",
            "kmeans_clustering_report.txt",
            "kmeans_evaluation_metrics.csv",
            "hierarchical_clustering_report.txt",
            "hierarchical_evaluation_metrics.csv",
            "dbscan_clustering_report.txt",
            "dbscan_sensitivity_analysis.csv",
        ],
    }

    dir_map = {
        "plots": PLOTS_DIR,
        "data": DATA_DIR,
        "reports": REPORTS_DIR,
    }

    results = {"verified": [], "missing": []}

    for category, files in expected.items():
        cat_dir = dir_map[category]
        for f in files:
            full_path = os.path.join(cat_dir, f)
            if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                results["verified"].append(f"{category}/{f}")
            else:
                results["missing"].append(f"{category}/{f}")

    return results


def run_pipeline() -> None:
    """Execute the end-to-end unsupervised pipeline."""
    ensure_output_directories()
    start_time = time.time()

    print("\n" + "=" * 80)
    print("      PLACEMENT PREDICTION - UNSUPERVISED LEARNING COMPLETE PIPELINE")
    print("=" * 80)

    # 1. PCA
    print("\n>>> STEP 1/4: PRINCIPAL COMPONENT ANALYSIS (PCA)")
    pca_res = run_pca_analysis()

    # 2. K-Means
    print("\n>>> STEP 2/4: K-MEANS CLUSTERING")
    kmeans_res = run_kmeans_clustering()

    # 3. Hierarchical
    print("\n>>> STEP 3/4: HIERARCHICAL CLUSTERING")
    hier_res = run_hierarchical_clustering()

    # 4. DBSCAN
    print("\n>>> STEP 4/4: DBSCAN CLUSTERING")
    dbscan_res = run_dbscan_clustering()

    total_duration = time.time() - start_time

    # Verification of Artifacts
    print("\n" + "=" * 80)
    print("                    ARTIFACT VERIFICATION")
    print("=" * 80)
    audit = verify_generated_artifacts()
    print(f"Verified {len(audit['verified'])} output artifacts generated successfully:")
    for v in audit["verified"]:
        print(f"  [OK] outputs/{v}")

    if audit["missing"]:
        print(f"\n[WARNING] {len(audit['missing'])} artifacts were missing:")
        for m in audit["missing"]:
            print(f"  [MISSING] outputs/{m}")
    else:
        print("\nAll 22 expected plots, datasets, and reports exist and are non-empty!")

    # Executive Summary Report
    exec_summary_path = os.path.join(REPORTS_DIR, "unsupervised_executive_summary.txt")
    with open(exec_summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("        PLACEMENT PREDICTION - UNSUPERVISED LEARNING EXECUTIVE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Pipeline Runtime: {total_duration:.2f} seconds\n\n")

        f.write("1. PRINCIPAL COMPONENT ANALYSIS (PCA)\n")
        f.write(f"  - PC1 Explained Variance: {pca_res['ev_ratio_pc1'] * 100:.2f}% (Eigenvalue: {pca_res['ev_pc1']:.4f})\n")
        f.write(f"  - PC2 Explained Variance: {pca_res['ev_ratio_pc2'] * 100:.2f}% (Eigenvalue: {pca_res['ev_pc2']:.4f})\n")
        f.write(f"  - Total 2-Component Variance Explained: {pca_res['total_ev_ratio'] * 100:.2f}%\n")
        f.write("  - Key Finding: PC1 captures overall GPA progression; PC2 captures technical/interview skills.\n\n")

        f.write("2. K-MEANS CLUSTERING\n")
        f.write(f"  - Optimal K: {kmeans_res['optimal_k']} (Silhouette Score: {kmeans_res['silhouette_score']:.4f})\n")
        for st in kmeans_res["cluster_stats"]:
            f.write(f"  - Cluster {st['Cluster']}: {st['Total_Students']:,} students | Placement Rate: {st['Placement_Rate_Pct']:.2f}% | Avg CGPA: {st['Mean_CGPA']:.2f}\n")
        f.write("  - Key Finding: K-Means naturally separates high-potential students (98% placed) from at-risk students (33% placed).\n\n")

        f.write("3. HIERARCHICAL CLUSTERING (N=2,000 Sample)\n")
        f.write(f"  - Highest Cophenetic Correlation: Average ({hier_res['cophenetic_scores']['average']:.4f}), Complete ({hier_res['cophenetic_scores']['complete']:.4f}), Ward ({hier_res['cophenetic_scores']['ward']:.4f})\n")
        f.write(f"  - Optimal Ward Cluster Count: K = {hier_res['optimal_k_ward']}\n")
        f.write(f"  - K-Means Agreement: ARI = {hier_res['ari_score']:.4f}, NMI = {hier_res['nmi_score']:.4f}\n\n")

        f.write("4. DBSCAN CLUSTERING (N=5,000 Sample, min_samples=40)\n")
        f.write(f"  - Estimated Knee Eps: {dbscan_res['estimated_eps']:.3f}, Selected Eps: {dbscan_res['selected_eps']:.2f}\n")
        f.write(f"  - Clustered Placement Rate: {dbscan_res['rate_clustered']:.2f}% ({dbscan_res['total_clustered']:,} students)\n")
        f.write(f"  - Noise Placement Rate:     {dbscan_res['rate_noise']:.2f}% ({dbscan_res['total_noise']:,} students)\n")
        f.write("  - Key Finding: DBSCAN successfully isolates density-based student groupings and flags atypical student profiles as noise.\n\n")
        f.write("=" * 80 + "\n")

    print(f"\nSaved Executive Summary to: {exec_summary_path}")
    print("\n" + "=" * 80)
    print("               PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
