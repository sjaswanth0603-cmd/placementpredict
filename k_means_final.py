"""
Final K-Means Clustering Runner.
Invokes the complete K-Means clustering pipeline from kmeans_clustering.
"""

from kmeans_clustering import run_kmeans_clustering

def perform_kmeans(filepath=None):
    """Execute K-Means clustering workflow."""
    return run_kmeans_clustering(filepath=filepath)

if __name__ == "__main__":
    perform_kmeans()