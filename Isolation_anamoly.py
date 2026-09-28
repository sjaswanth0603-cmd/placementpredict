import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Load dataset
data = pd.read_csv(
    r"C:\Acadamics\AI ML\PythonProject5\placement_predict_50k Dataset (3)(in).csv"
)

print("Dataset loaded successfully!")
print("Shape:", data.shape)

# 2. Select numerical features
features = [
    col for col in data.select_dtypes(include="number").columns
    if col.lower() != "placement status"
]

x = data[features]

print("\nFeatures used for Isolation Forest:")
print(features)

# 3. Create Isolation Forest model
model = IsolationForest(
    random_state=42,
    contamination=0.02,
    n_estimators=100
)

# 4. Train the model
model.fit(x)

# 5. Predict anomalies
data["Anomaly"] = model.predict(x)

# 6. Calculate anomaly score
data["Anomaly_Score"] = model.decision_function(x)

# 7. Display first 10 results
print("\nAnomaly Detection Results:")
print(
    data[features + ["Anomaly", "Anomaly_Score"]].head(10)
)

# 8. Extract anomalies
anomalies = data[data["Anomaly"] == -1]

# 9. Display number of anomalies
print("\nNumber of anomalies:")
print(len(anomalies))

# 10. Display anomaly records
print("\nAnomaly Records:")
print(
    anomalies[features + ["Anomaly", "Anomaly_Score"]]
)

# 11. Save anomaly results
anomalies.to_csv("placement_anomalies.csv", index=False)

print("\nAnomaly results saved as: placement_anomalies.csv")