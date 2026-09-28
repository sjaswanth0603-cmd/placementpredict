import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv("placement_predict_50k Dataset (3)(in).csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)

# Check the available columns
print("\nAvailable columns:")
print(data.columns.tolist())

# Check the values in CGPA_Tier
print("\nCGPA Tier values:")
print(data["CGPA_Tier"].value_counts())

# Convert CGPA tiers into numbers
tier_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

data["CGPA_Tier_Numeric"] = data["CGPA_Tier"].map(tier_mapping)

# Convert aptitude score into numeric values
data["AptitudeTestScore"] = pd.to_numeric(
    data["AptitudeTestScore"],
    errors="coerce"
)

# Select the features for clustering
X = data[
    [
        "CGPA_Tier_Numeric",
        "AptitudeTestScore"
    ]
].copy()

# Check for missing values
print("\nMissing values:")
print(X.isnull().sum())

# Remove rows with missing values
X = X.dropna()

print("\nData used for clustering:")
print(X.head())

print("\nTotal records used:", len(X))

# Create the K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Assign each student to a cluster
clusters = kmeans.fit_predict(X)

# Add cluster numbers to the dataset
data.loc[X.index, "Cluster"] = clusters

# Display the cluster results
print("\nCluster Results:")
print(
    data.loc[
        X.index,
        [
            "CGPA_Tier",
            "AptitudeTestScore",
            "Cluster"
        ]
    ].head(10)
)

# Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Count students in each cluster
print("\nNumber of Students in Each Cluster:")
print(
    data.loc[X.index, "Cluster"]
    .value_counts()
    .sort_index()
)

# Display the clusters using a scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(
    X["CGPA_Tier_Numeric"],
    X["AptitudeTestScore"],
    c=clusters
)

# Show the cluster centers
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

# Find WCSS values for different K values
wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)
    wcss.append(model.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 6))

plt.plot(
    range(1, 11),
    wcss,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method for K-Means")
plt.show()

# Save the final results
data.to_csv(
    "placement_kmeans_result.csv",
    index=False
)

print("\nK-Means clustering completed successfully!")
print("Result saved as: placement_kmeans_result.csv")