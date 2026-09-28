import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv("placement_predict_50k Dataset (3)(in).csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)

# Check CGPA tier values
print("\nCGPA Tier values:")
print(data["CGPA_Tier"].value_counts())

# Convert CGPA tiers into numbers
tier_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

data["CGPA_Tier_Numeric"] = data["CGPA_Tier"].map(tier_mapping)

# Convert aptitude score into numbers
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

# Check missing values
print("\nMissing values:")
print(X.isnull().sum())

# Remove missing values
X = X.dropna()

print("\nData before removing outliers:")
print(X.head())

print("\nTotal records:", len(X))

# -----------------------------------------------------
# Detect outliers using IQR
# -----------------------------------------------------

Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Find outliers
outliers = (
    (X < lower_limit) |
    (X > upper_limit)
).any(axis=1)

print("\nNumber of outliers:", outliers.sum())

# Separate normal data and outliers
X_clean = X[~outliers]
X_outliers = X[outliers]

print("Records after removing outliers:", len(X_clean))

# -----------------------------------------------------
# Apply K-Means
# -----------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_clean)

# Add cluster numbers to the original dataset
data.loc[X_clean.index, "Cluster"] = clusters

# Mark outliers
data.loc[X_clean.index, "Outlier"] = "No"
data.loc[X_outliers.index, "Outlier"] = "Yes"

# -----------------------------------------------------
# Display cluster results
# -----------------------------------------------------

print("\nCluster Results:")

print(
    data.loc[
        X_clean.index,
        [
            "CGPA_Tier",
            "AptitudeTestScore",
            "Cluster"
        ]
    ].head(10)
)

# -----------------------------------------------------
# Display cluster centers
# -----------------------------------------------------

print("\nCluster Centers:")

print(kmeans.cluster_centers_)

# -----------------------------------------------------
# Count students in each cluster
# -----------------------------------------------------

print("\nNumber of Students in Each Cluster:")

print(
    data.loc[X_clean.index, "Cluster"]
    .value_counts()
    .sort_index()
)

# -----------------------------------------------------
# K-Means graph with outliers
# -----------------------------------------------------

plt.figure(figsize=(8, 6))

# Normal clustered points
plt.scatter(
    X_clean["CGPA_Tier_Numeric"],
    X_clean["AptitudeTestScore"],
    c=clusters,
    label="Clusters"
)

# Outlier points
plt.scatter(
    X_outliers["CGPA_Tier_Numeric"],
    X_outliers["AptitudeTestScore"],
    marker="x",
    s=100,
    label="Outliers"
)

# Cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200,
    label="Cluster Centers"
)

plt.xlabel("CGPA Tier")
plt.ylabel("Aptitude Test Score")
plt.title("K-Means Clustering with Outliers")
plt.legend()

plt.show()

# -----------------------------------------------------
# Elbow Method
# -----------------------------------------------------

wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_clean)

    wcss.append(model.inertia_)

# -----------------------------------------------------
# Elbow graph
# -----------------------------------------------------

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

# -----------------------------------------------------
# Save the results
# -----------------------------------------------------

data.to_csv(
    "placement_kmeans_outliers.csv",
    index=False
)

print("\nK-Means clustering with outliers completed successfully!")
print("Result saved as: placement_kmeans_outliers.csv")