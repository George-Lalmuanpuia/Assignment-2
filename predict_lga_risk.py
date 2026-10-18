import joblib, pandas as pd, sys

# Check if the file path was provided
if len(sys.argv) < 2:
    print("Usage: python predict_lga_risk.py <path_to_input_csv>")
    sys.exit(1)

# Load the model
bundle = joblib.load("models/lga_risk_model.joblib")

# Read the new data
df = pd.read_csv(sys.argv[1])

# Predict
X = df[bundle["features"]].values
proba = bundle["pipeline"].predict_proba(X)[:, 1]  # P(high-risk)

# Assign predictions and probabilities
df["predicted_high_risk"] = (proba >= bundle["threshold"]).astype(int)
df["p_high_risk"] = proba

# Save to the processed folder
df.to_csv("data/processed/lga_risk_predictions.csv", index=False)

print("Predictions saved to data/processed/lga_risk_predictions.csv")
print(df[["LGA", "predicted_high_risk", "p_high_risk"]].head())