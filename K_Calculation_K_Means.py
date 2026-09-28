import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Load the dataset
def load_data():

    data = pd.read_csv(
        "placement_predict_50k Dataset (3)(in).csv"
    )

    print("\nDataset loaded successfully!")
    print("Dataset shape:", data.shape)

    # Convert CGPA tiers into numbers
    tier_mapping = {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }

    data["CGPA_Tier"] = data["CGPA_Tier"].map(tier_mapping)

    # Convert aptitude score into numbers
    data["AptitudeTestScore"] = pd.to_numeric(
        data["AptitudeTestScore"],
        errors="coerce"
    )

    # Remove missing values
    data = data.dropna(
        subset=[
            "CGPA_Tier",
            "AptitudeTestScore"
        ]
    )

    # Select features
    X = data[
        [
            "CGPA_Tier",
            "AptitudeTestScore"
        ]
    ]

    return data, X


# Manual K selection
def manual_k():

    print("\n--- Manual K-Means ---")

    k = int(
        input("Enter the number of clusters: ")
    )

    while k < 2:

        print("\nK must be at least 2.")

        k = int(
            input("Enter the number of clusters: ")
        )

    data, X = load_data()

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    data["Cluster"] = kmeans.fit_predict(X)

    print("\nCluster Results:")

    print(
        data[
            [
                "CGPA_Tier",
                "AptitudeTestScore",
                "Cluster"
            ]
        ].head(10)
    )

    print("\nCluster Centers:")

    print(kmeans.cluster_centers_)

    print("\nNumber of Students in Each Cluster:")

    print(
        data["Cluster"]
        .value_counts()
        .sort_index()
    )

    # Calculate silhouette score
    score = silhouette_score(
        X,
        data["Cluster"]
    )

    print("\nSilhouette Score:")
    print(score)

    # Display clusters
    plt.figure(figsize=(8, 5))

    plt.scatter(
        data["CGPA_Tier"],
        data["AptitudeTestScore"],
        c=data["Cluster"]
    )

    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="X",
        s=200
    )

    plt.xlabel("CGPA Tier")
    plt.ylabel("Aptitude Test Score")
    plt.title("Manual K-Means Clustering")

    plt.show()


# Elbow Method
def elbow_method():

    data, X = load_data()

    print("\n--- Elbow Method ---")

    wcss = []

    k_values = range(1, 11)

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        kmeans.fit(X)

        wcss.append(
            kmeans.inertia_
        )

    print("\nWCSS Values:")

    for k, value in zip(k_values, wcss):

        print(
            "K =", k,
            "WCSS =", value
        )

    # Elbow graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        wcss,
        marker="o"
    )

    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")

    plt.show()

    # Ask user to select K
    k = int(
        input(
            "\nEnter the value of K from the Elbow Method: "
        )
    )

    while k < 2:

        print("\nK must be at least 2.")

        k = int(
            input(
                "Enter the value of K from the Elbow Method: "
            )
        )

    # Apply K-Means
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    data["Cluster"] = kmeans.fit_predict(X)

    print("\nCluster Results:")

    print(
        data[
            [
                "CGPA_Tier",
                "AptitudeTestScore",
                "Cluster"
            ]
        ].head(10)
    )

    print("\nCluster Centers:")

    print(kmeans.cluster_centers_)

    print("\nNumber of Students in Each Cluster:")

    print(
        data["Cluster"]
        .value_counts()
        .sort_index()
    )

    # Silhouette score
    score = silhouette_score(
        X,
        data["Cluster"]
    )

    print("\nSilhouette Score:")
    print(score)

    # Cluster graph
    plt.figure(figsize=(8, 5))

    plt.scatter(
        data["CGPA_Tier"],
        data["AptitudeTestScore"],
        c=data["Cluster"]
    )

    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="X",
        s=200
    )

    plt.xlabel("CGPA Tier")
    plt.ylabel("Aptitude Test Score")
    plt.title("K-Means Clustering")

    plt.show()


# Silhouette Method
def silhouette_method():

    data, X = load_data()

    print("\n--- Silhouette Method ---")

    k_values = range(2, 11)

    silhouette_scores = []

    # Calculate silhouette score for each K
    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(X)

        score = silhouette_score(
            X,
            labels
        )

        silhouette_scores.append(score)

        print(
            "\nK =", k
        )

        print(
            "Silhouette Score:",
            score
        )

    # Silhouette graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        silhouette_scores,
        marker="o"
    )

    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method")

    plt.show()

    # Find the best K
    best_k = k_values[
        silhouette_scores.index(
            max(silhouette_scores)
        )
    ]

    best_score = max(
        silhouette_scores
    )

    print(
        "\nBest K from Silhouette Method:",
        best_k
    )

    print(
        "Best Silhouette Score:",
        best_score
    )

    # Apply K-Means using best K
    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )

    data["Cluster"] = kmeans.fit_predict(X)

    print("\nCluster Centers:")

    print(
        kmeans.cluster_centers_
    )

    print("\nNumber of Students in Each Cluster:")

    print(
        data["Cluster"]
        .value_counts()
        .sort_index()
    )

    # Final clustering graph
    plt.figure(figsize=(8, 5))

    plt.scatter(
        data["CGPA_Tier"],
        data["AptitudeTestScore"],
        c=data["Cluster"]
    )

    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="X",
        s=200
    )

    plt.xlabel("CGPA Tier")
    plt.ylabel("Aptitude Test Score")
    plt.title(
        "K-Means Clustering using Best K"
    )

    plt.show()


# -----------------------------------------------------
# Run the methods
# -----------------------------------------------------

elbow_method()

silhouette_method()