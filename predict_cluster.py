import sys
import joblib
import pandas as pd


# Load all saved saved models and values from joblib and notebook 
bundle = joblib.load("models/kmeans_lga.joblib")
scaler = bundle["scaler"]
kmeans = bundle["kmeans"]
features = bundle["features"]
cluster_names = bundle["cluster_names"]

# read the new csv
df = pd.read_csv(sys.argv[1])

# scaler the data with make ure it is even

X_scaled = scaler.transform(df[features].values)

# Now predict  kmean clusters and add name
df["Cluster"] = kmeans.predict(X_scaled)
df["Cluster_Name"] = df["Cluster"].map(cluster_names)


# Save it 
df.to_csv("cluster_predictions.csv", index=False)
print("Cluster predictions saved to cluster_predictions.csv")
show = [c for c in ["LGA", "Year", "Cluster", "Cluster_Name"] if c in df.columns]
print(df[show].head())