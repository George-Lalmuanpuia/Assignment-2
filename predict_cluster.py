import sys
import joblib
import pandas as pd

# Check if the file path was provided
if len(sys.argv) < 2:
    print("Usage: python predict_cluster.py <path_to_input_csv>")
    sys.exit(1)

# Load all saved models and values from joblib 
bundle = joblib.load("models/kmeans_lga.joblib")
scaler = bundle["scaler"]
kmeans = bundle["kmeans"]
features = bundle["features"]
cluster_names = bundle["cluster_names"]

# Read the new csv
df = pd.read_csv(sys.argv[1])

# Scale the data to make sure it is even
X_scaled = scaler.transform(df[features].values)

# Now predict kmean clusters and add name
df["Cluster"] = kmeans.predict(X_scaled)
df["Cluster_Name"] = df["Cluster"].map(cluster_names)

# Save it to the processed folder
df.to_csv("data/processed/cluster_predictions.csv", index=False)

print("Cluster predictions saved to data/processed/cluster_predictions.csv")
show = [c for c in ["LGA", "Year", "Cluster", "Cluster_Name"] if c in df.columns]
print(df[show].head())